############## common imports ##############
import torch
import copy
import os, sys
import pickle
from loaders import load_policy, load_critic, _load_state_dict_trained
############## environment imports ##############
from environment.tensorboard_logger import Tensorboard_Plotter_Obj
from environment.racedemia_env_v1 import Racedemia_Env
from environment.normalized_env_wrapper import NormalizedObservation
from evaluate_main import evaluate
import gymnasium as gym
from argparse import ArgumentParser
############## hydra imports ##############
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
############## reinforcement learning imports ##############
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.sac_style import Style_SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm
from rlkit.torch.pytorch_util import set_gpu_mode

# class RuntimeArgs:
#     def __init__(self):
#         self.exp_name = "debug"
        
        
        
def user_input_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    
    args = parser.parse_args()
    return args

def get_optimizer_class(name):
    if isinstance(name, str) == False:
        return name
    if name.lower() == "adam":
        return torch.optim.Adam
    elif name.lower() == "sgd":
        return torch.optim.SGD
    elif name.lower() == "rmsprop":
        return torch.optim.RMSprop
    elif name.lower() == "adamw":
        return torch.optim.AdamW
    
def get_obs_space_dim(obs_space):
    obs_dim = 0
    if isinstance(obs_space, gym.spaces.Dict):
        for key, value in obs_space.items():
            obs_dim += value.low.size
    elif isinstance(obs_space, gym.spaces.Box):
        obs_dim = obs_space.low.size
    else:
        raise UserWarning("Not supported observation space")
    
    return obs_dim



def _set_seeds(seed:int):
    import torch
    import numpy as np
    import random
    ### set random seeds (for reproducible results) ###
    torch.manual_seed(seed) #manual seed for both CPU and GPU operations
    torch.cuda.manual_seed(seed) #random seed for CUDA operations
    torch.cuda.manual_seed_all(seed) #random seed for multiple-GPU CUDA operations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True #deterministic flag for CuDNN
    
    np.random.seed(seed)
    random.seed(seed)
    ########################
    
    print(f"========Finished setting seed to : {seed}========")

#########################################################################################################



@hydra.main(version_base=None, 
            config_path="conf",
            config_name="config")
def experiment(cfg: DictConfig) -> None:
    
    set_gpu_mode(mode=True, gpu_id=0)
    
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    new_dict_cfg = copy.deepcopy(dict_cfg)
    do_continue_training = dict_cfg['environment']['continue_training']['usage']
    ct_epoch_mode = dict_cfg['environment']['continue_training']['epoch_mode']
    
    """get the trained dict_cfg -> once loaded, the experiment and network can be loaded as well"""
    if dict_cfg['environment']['continue_training']['usage'] == True:
        dict_cfg_path = f"{dict_cfg['agent']['test']['test_log_path']}/nam_c_logs/{dict_cfg['agent']['exp_name']}/conf_dict.pkl"
        dict_cfg = pickle.load(open(dict_cfg_path, 'rb'))
        dict_cfg['environment']['continue_training']['usage'] = do_continue_training
        dict_cfg['environment']['continue_training']['epoch_mode'] = ct_epoch_mode
        LOG_ROOT = os.path.dirname(os.path.dirname(dict_cfg['agent']['test']['test_log_path']))
        test_log_path = dict_cfg['agent']['test']['test_log_path']
    else:
        LOG_ROOT = dict_cfg['agent']['test']['test_log_path']
        TEST_LOG_ROOT = f"{LOG_ROOT}/nam_c_logs";os.makedirs(TEST_LOG_ROOT, exist_ok=True)
        test_log_path = f"{TEST_LOG_ROOT}/{dict_cfg['agent']['exp_name']}";os.makedirs(test_log_path, exist_ok=True)
        dict_cfg['agent']['test']['test_log_path'] = test_log_path
    
    dict_cfg['agent']['algorithm']['max_path_length'] = int(dict_cfg['agent']['algorithm']['max_path_length'])
    dict_cfg['agent']['test']['test_max_path_length'] = int(dict_cfg['agent']['test']['test_max_path_length'])
    
    _set_seeds(seed=dict_cfg['environment']['random_seed'])
    
    ####deprecate Tensorboard usage..
    expl_plotter = Tensorboard_Plotter_Obj(log_dir=f"{LOG_ROOT}/tensorboard/{dict_cfg['agent']['exp_name']}/expl",
                                      action_dim=dict_cfg['environment']['action']['action_dim'])
    eval_plotter = Tensorboard_Plotter_Obj(log_dir=f"{LOG_ROOT}/tensorboard/{dict_cfg['agent']['exp_name']}/eval", action_dim=dict_cfg['environment']['action']['action_dim'])
    test_plotter = Tensorboard_Plotter_Obj(log_dir=f"{LOG_ROOT}/tensorboard/{dict_cfg['agent']['exp_name']}/test", action_dim=dict_cfg['environment']['action']['action_dim'])
    # expl_plotter, eval_plotter, test_plotter = None, None, None
    
    if dict_cfg['style']['style_mode']['type'] in ['medium', 'aggressive', 'defensive']:
        modes = ['common', 'defensive', 'aggressive']
        for mode in modes:
            for key, value in dict_cfg['style'][mode].items():
                if 'reward' in key:
                    dict_cfg['reward'][key] = value
                elif 'penalty' in key:
                    dict_cfg['penalty'][key] = value
        # breakpoint()
        
    dict_cfg['agent']['trainer']['optimizer_class'] = get_optimizer_class(dict_cfg['agent']['trainer']['optimizer_class'])
    is_nam_expl = dict_cfg['environment']['track']['use_nam_only']
    use_style = dict_cfg['agent']['style']['usage']
    
    pickle.dump(
        dict_cfg,
        open(f"{test_log_path}/conf_dict.pkl", "wb")
    )
    
    
    expl_env = Racedemia_Env(
        environment_config=dict_cfg['environment'],
        penalty_config=dict_cfg['penalty'],
        reward_config=dict_cfg['reward'],
        render_config=dict_cfg['simulate'],
        agent_config=dict_cfg['agent'],
        is_nam=is_nam_expl, # False,
        plotter=expl_plotter,
        mode='exploration',
        exp_name=dict_cfg['agent']['exp_name'],
        style_config= dict_cfg['style'] if 'style' in dict_cfg else None,
        terminated_fig_path=None
    )
    eval_env = Racedemia_Env(
        environment_config=dict_cfg['environment'],
        penalty_config=dict_cfg['penalty'],
        reward_config=dict_cfg['reward'],
        agent_config=dict_cfg['agent'],
        render_config=None,
        is_nam=is_nam_expl, #False,
        plotter=eval_plotter,
        mode='evaluation',
        exp_name=dict_cfg['agent']['exp_name'],
        style_config= dict_cfg['style'] if 'style' in dict_cfg else None,
        terminated_fig_path=None
    )
    test_env = Racedemia_Env(
        environment_config=dict_cfg['environment'],
        penalty_config=dict_cfg['penalty'],
        reward_config=dict_cfg['reward'],
        render_config=dict_cfg['simulate'],
        agent_config=dict_cfg['agent'],
        is_nam=True,
        plotter=test_plotter,
        mode='test',
        exp_name=dict_cfg['agent']['exp_name'],
        style_config= dict_cfg['style'] if 'style' in dict_cfg else None,
        terminated_fig_path=None
    )
    #use the wrappers for post-processing actions and observation states
    expl_env = NormalizedObservation(env=expl_env)
    eval_env = NormalizedObservation(env=eval_env)
    '''[0418] test용인 남양 트랙에서 NormalizedObservation wrapper로 싸지 않았음!!!!
    그러니까 random track에서는 잘 돌아도 test인 남양 트랙에서는 계속 후진했던 영향도 있음.'''
    test_env = NormalizedObservation(env=test_env)
    
    obs_dim = get_obs_space_dim(expl_env.unwrapped.observation_space)
    action_dim = expl_env.unwrapped.action_space.low.size
    # style_dim = dict_cfg['agent']['style']['size'] if dict_cfg['agent']['style']['usage'] else 0
    style_dim = dict_cfg['agent']['style']['num_nets'] if dict_cfg['agent']['style']['usage'] else 0
    
    '''network를 다 따로 loading하기 때문에 나중에 학습 중에 continue_learning 하기에도 쉬움
    Q-function network에서 ConcatMlp를 사용하는 이유는 observation state와 action value를 모두 입력으로 받아서 Q-value를 계산하기 때문.'''
    ##Q-function의 경우에는 optimistic critic으로 특정 (state, action) pair에 대해서 너무 높은 Q-value를 부여하기도 하는데, 그래서 이를 방지하려면?
    ##Exploration이 부족하다면 policy distribution을 바꿔야 할수도 (beta distribution / normalizing flow / gaussian mixture model / diffusion policy)
    
    qf1, qf2, target_qf1, target_qf2 = load_critic(dict_cfg, obs_dim=obs_dim, action_dim=action_dim, 
                                                   style_dim=style_dim)
    policy = load_policy(dict_cfg, obs_dim=obs_dim, action_dim=action_dim, 
                         style_dim=style_dim)
    
    
    # if do_continue_training:
    #     last_epoch, qf1, qf2, target_qf1, target_qf2, policy = _load_state_dict_trained(qf1, qf2, target_qf1, target_qf2, policy, 
    #                                                                         exp_root=dict_cfg['agent']['test']['test_log_path'],
    #                                                                         epoch_mode=dict_cfg['environment']['continue_training']['epoch_mode'])
        
        
    
    eval_policy = MakeDeterministic(policy)
    
    eval_path_collector = MdpPathCollector(
        eval_env, eval_policy, plotter=eval_plotter
    )
    expl_step_collector = MdpStepCollector(
        expl_env, policy, plotter=expl_plotter
    )
    replay_buffer = EnvReplayBuffer(
        max_replay_buffer_size=dict_cfg['agent']['replay_buffer_size'],
        env=expl_env
    )

    train_log_save_path = f"{test_log_path}/statistics"
    os.makedirs(train_log_save_path, exist_ok=True)
    if use_style:
        trainer = Style_SACTrainer(
            env=expl_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            plotter=expl_plotter,
            train_log_save_path=train_log_save_path,
            **dict_cfg['agent']['trainer']
        )
    else:
        trainer = SACTrainer(
            # env=eval_env,
            env=expl_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            plotter=expl_plotter,
            train_log_save_path=train_log_save_path,
            # train_log_save_path=test_log_path,
            **dict_cfg['agent']['trainer']
        )
    
    if do_continue_training:
        trainer, recent_epoch = _load_state_dict_trained(trainer,
                                            exp_root=dict_cfg['agent']['test']['test_log_path'],
                                            epoch_mode=dict_cfg['environment']['continue_training']['epoch_mode'])
        print(f"Finished Loading Optimizer/Network Weights for Continued Training on exp: {dict_cfg['agent']['exp_name']}")
    else:
        recent_epoch = 0
        
    algorithm = TorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        evaluation_data_collector=eval_path_collector,
        exploration_data_collector=expl_step_collector,
        replay_buffer=replay_buffer,
        ############################### added customized function for TESTING on NAM-C Track ###############################
        customized_test_fn=evaluate,
        test_env=test_env,
        test_policy=eval_policy, #test policy와 동일하게 사용해도 됨
        test_kwargs=dict_cfg['agent']['test'],
        # warmup_actor_step=dict_cfg['agent']['algorithm']['warmup_actor_step'],
        **dict_cfg['agent']['algorithm']
    ) 
    setattr(algorithm, '_start_epoch', recent_epoch)
    algorithm.to(ptu.device)
    algorithm.train(start_epoch=recent_epoch)
    
    
if __name__ == "__main__":
    # runtime_args = user_input_args()
    # experiment(runtime_args)

    experiment()