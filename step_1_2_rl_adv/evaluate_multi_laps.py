import torch
import numpy as np
import os
from glob import glob
import pickle
from tqdm import tqdm
from natsort import natsort
import math

from train_main import get_obs_space_dim
from environment.racedemia_env_v1 import Racedemia_Env
from environment.normalized_env_wrapper import NormalizedObservation
from rlkit.torch.sac.policies import MakeDeterministic
from evaluate_main import _get_single_rollout
from loaders import load_policy
from evaluate_logger import RacerLogger



def _get_shortest_lap(exp_path:str, dt:float)->str:
    success_folders = natsort.natsorted(glob(f"{exp_path}/success_*"))
    success_car_state = [pickle.load(open(f"{p}/car_state.pkl", "rb")) for p in success_folders]
    success_lap_time = np.array([float(np.array(s).shape[0] * dt) for s in success_car_state])
    fastest_index = np.argmin(success_lap_time)
    fastest_epoch = int(success_folders[fastest_index].split('/')[-1].split('_')[0])
    
    fastest_ckpt_path = f"{exp_path}/ckpt/epoch_{fastest_epoch}"
    
    return fastest_ckpt_path, fastest_epoch
    
def _load_everything(exp_path,
                     truncate_lap:int=1,
                     dt:float=0.05,
                     epoch_num:int=None):
    ##[step0] load the configuration file##
    conf_dict = pickle.load(open(f"{exp_path}/conf_dict.pkl", "rb"))
    
    ##[step1] load the environment##
    eval_env = Racedemia_Env(
        environment_config=conf_dict['environment'],
        penalty_config=conf_dict['penalty'],
        reward_config=conf_dict['reward'],
        agent_config=conf_dict['agent'],
        plotter=None,
        render_config=conf_dict['simulate'],
        mode='test',
        exp_name=EXP_NAME,
        style_config=conf_dict['style'] if 'style' in conf_dict else None,
        terminated_fig_path=None,
        num_laps_for_truncate=truncate_lap
    )
    eval_env = NormalizedObservation(env=eval_env)

    ##[step2] load the policy##
    if epoch_num is not None:
        ckpt_path = f"{exp_path}/ckpt/epoch_{epoch_num}"
        assert os.path.isdir(ckpt_path)
        epoch = epoch_num
    else:
        ckpt_path, epoch = _get_shortest_lap(exp_path=exp_path, dt=dt)
    
    policy_ckpt_path = f"{ckpt_path}/policy.pth"
    policy = load_policy(
        dict_cfg=conf_dict,
        obs_dim=get_obs_space_dim(eval_env.unwrapped.observation_space),
        action_dim=eval_env.unwrapped.action_space.low.size,
        style_dim=conf_dict['agent']['style']['num_nets'] if conf_dict['agent']['style']['usage'] else 0
    )
    policy.load_state_dict(torch.load(policy_ckpt_path)).cuda()
    policy = MakeDeterministic(policy)
    

    return policy, eval_env, epoch

def _evaluate_multi_laps(
    trained_policy, 
    eval_env,
    test_max_path_length,
    style:float=None,
):
    SUCCESS=False
    path_length = 0
    
    o, _ = eval_env.reset(style=style)
    observations = []
    actions = []
    rewards = []
    terminal = []
    
    logger = RacerLogger(env=eval_env)
    
    
    while path_length < test_max_path_length:
        single_rollout_dict = _get_single_rollout(env=eval_env,
                            agent=policy,
                            obs=o,
                            style_setting=style)
        observations.append(single_rollout_dict['next_observation'])
        actions.append(single_rollout_dict['action'])
        is_done = single_rollout_dict['terminal']
        terminal.append(is_done)
        
        o = single_rollout_dict['next_observation']
        
        logger._update_logs()
        
        if is_done:
            if single_rollout_dict['success']:
                SUCCESS=True
            break
        path_length += 1
        
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    rewards = np.array(rewards)
    
    if len(rewards.shape) == 1:
        rewards = rewards = rewards.reshape(-1, 1) #보통 상수이기 때문에 reshape 사용

    test_ret_dict = {
            'observations': observations,
            'rewards': rewards,
            'actions': actions,
            'terminal': terminal
        }   

    return logger, test_ret_dict, SUCCESS

if __name__ == "__main__":
    EXP_ROOT = r'D:\meow_racer_experiments\nam_c_logs'.replace('\\', '/')
    EXP_NAME = '0605_SingleFeet_DTx50_NAM_Aggressive_Act3_timePen01_minVPen_offCnt_NegVelAPSReward_PrgReward1000_PrgRewardVel1_KappaV1_Ephi01_KappaWeightVelReward3'
    EPOCH = None
    DT = 0.05
    TRUNCATE_LAP = 3
    TEST_MAX_PATH_LENGTH = 1e+9
    STYLE = None
    EXP_PATH = f"{EXP_ROOT}/{EXP_NAME}"
     
    
    policy, eval_env, epoch_num = _load_everything(exp_path=EXP_PATH, 
                                        dt=DT, 
                                        epoch_num=EPOCH, 
                                        truncate_lap=TRUNCATE_LAP)
    
    logger, test_ret_dict, did_success = _evaluate_multi_laps(
        trained_policy=policy,
        eval_env=eval_env,
        test_max_path_length=TEST_MAX_PATH_LENGTH,
        style=STYLE
    )
    new_test_log_path = f"{EXP_PATH}/multi_laps";os.makedirs(new_test_log_path, exist_ok=True)
    success_prefix = "success" if did_success else "failed"
    
    logger._dump_actions(log_file_root = new_test_log_path, log_file_prefix = f"{success_prefix}_{epoch_num}")
    logger._dump_trajectory(log_file_root=new_test_log_path, log_file_prefix=f"{success_prefix}_{epoch_num}")
    logger._dump_sarsa(file = test_ret_dict, log_file_root=new_test_log_path, log_file_prefix=f"{success_prefix}_{epoch_num}")
    
    

    
    
