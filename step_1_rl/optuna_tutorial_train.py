from typing import Any
from typing import Dict
from functools import partial

import pickle
from datetime import datetime
import os, sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root);sys.path.append(os.path.dirname(root))
sys.path.append(f"{root}/rl_src")

import configparser #for .ini file

### gym (local) ###
from stable.sac import SAC
from stable.common.vec_env import DummyVecEnv
from stable.common.callbacks import EvalCallback
from stable.common.monitor import Monitor
# from stable.common.utils import set_random_seed
from stable.common.noise import NormalActionNoise

### optuna (library) ###
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import (
    TPESampler, # Tree-structured Parzen Estimator algorithm
       
)
### torch ###
import torch
import torch.nn as nn

### race driving (local) ###
from tutorial_train import load_args, INP_ARGS, prepare_args, vectorize_env
from envs.gym_car_racing import Toy_CarRacing
from envs.gym_car_racing_nam import Toy_CarRacing_NAM
from envs.gym_car_constants import *

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for SAC hyperparameters"""
    ##learning rate는 SAC 논문의 알고리즘에서는 Actor, Value function, Soft Q-value각각에 다른 lambda 변수를 쓰긴 하지만, 본 구현체에서는 같은 값을 사용한다.
    # 근데 이런 경우에 각 learning rate가 다 같은 값을 쓰는게 제일 학습이 안정적인걸까?
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-3, log=True) ## learning rate of both the actor and critic (근데 만약에 actor critic의 learning rate를 다르게 한다면?)
    gamma = 1. - trial.suggest_float("gamma", 1e-4, 1e-1, log=True) ## discount factor (0.99 default, 0.95 for HER)
    # '''theta_{target} <- \tau * theta_{current} + (1-\tau) * theta_{target}'''
    tau = trial.suggest_float("tau", 5e-3, 1, log=True) ## target Q-value smoothing factor
    # '''J(\pi) = E[Q(s_t,a_t) - \alpha * log(\pi(a_t|s_t))]   =>  이 식에서 \alpha가 ent_coef임'''
    ent_coef = trial.suggest_float("ent_coef", 1e-7, 1e-1, log=True) ## entropy maximization을 위해서 추가한 entropy term의 weight
    
    n_steps  = int(10 ** 4)
    # n_steps = 10 ** trial.suggest_int("exponent_n_timesteps", 4, 6)  ##하나의 episode당 최대 timestep의 개수##
    target_update_interval = trial.suggest_int("target_update_interval", 1, 10, log=True) ##SAC는 polyac_update를 사용하기 때문에 이 interval이 1로 자주 반복되어도 tau값에 의해서 current theta반영율이 조절이 됨. 만약 tau값을 키운다면 interval도 커지는게 맞음.
    
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "big"])
    # use_sde = trial.suggest_categorical("use_sde", [True, False])
    use_sde = True
    ## sample a new noise matrix every n steps when using gSDE (default value is 1 - only sample at the beginning of the rollout) ##
    sde_sample_freq = trial.suggest_int("sde_sample_freq", 1, 10, log=True)
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # 당연히 action space를 SAC에서 default로 -1~1 사이의 값으로 squashing을 하기 때문에 ReLU가 유리할 수 밖에 없다.
    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]
    activation_fn = nn.ReLU
    
    net_arch = [
        dict(pi=[64, 64], qf=[64, 64]) if net_arch == "small" else 
        dict(pi=[64, 128, 64], qf=[64, 128, 64]) if net_arch == "big" else
        dict(pi=[64], qf=[64]) #if "tiny"
    ][0]
    """Sampler for Environment hyperparameters
    -> might not be used
    """
    # num_vecs = trial.suggest_categorical("num_vecs", [10, 15, 20])
    # theta_diff = trial.suggest_categorical("theta_diff", [5, 10])
    # lidar_deg = trial.suggest_categorical("lidar_deg", [5, 10, 15])
    
    
    return [
        {
        'learning_rate': learning_rate,
        'gamma': gamma,
        'tau': tau,
        'ent_coef': ent_coef,
        'n_steps': n_steps,
        'target_update_interval': target_update_interval,
        'net_arch': net_arch,
        'activation_fn': activation_fn,
        'use_sde': use_sde,
        'sde_sample_freq': sde_sample_freq
        },
        # {
        # 'num_vecs': num_vecs,
        # 'theta_diff': theta_diff,
        # 'lidar_deg': lidar_deg
        # }   
    ]
    

class TrialEvalCallback(EvalCallback):
    def __init__(
        self, 
        eval_env, 
        log_path:str, 
        best_model_save_path:str,
        trial: optuna.Trial,
        n_eval_episodes:int,
        eval_freq:int,
        deterministic:bool,
        verbose:int=0,
    ):
        super().__init__(
            eval_env=eval_env, n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq, deterministic=deterministic,
            verbose=verbose,
            
            log_path=log_path,
            best_model_save_path=best_model_save_path
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # prune trial if needed #
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
 
def _load_vec_env(args, use_args, obs_conf):
    # breakpoint()
    if hasattr(use_args, "env_type"):
        env_type = use_args.env_type
        if hasattr(use_args, "n_envs"):
            num_env = use_args.n_envs
            vec_env = DummyVecEnv(
                    [vectorize_env(env_type=env_type, args=args, obs_config=obs_conf, rank=i) for i in range(num_env)]
                )
            return vec_env
    return None

def objective(trial: optuna.Trial,
              args, 
              use_args,
              obs_conf) -> float:
    TENSORBOARD_ROOT = "D:/rl_tensorboard" ## base root folder for the tensorboard logging of rl training ##
    LOG_ROOT = "D:/rl_log"; os.makedirs(LOG_ROOT, exist_ok=True)
    TENSORBOARD_LOG = f"{TENSORBOARD_ROOT}/{args.tensorboard_folder}";os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    LOG_PATH = f"{LOG_ROOT}/{args.tensorboard_folder}";os.makedirs(LOG_PATH, exist_ok=True)
    from glob import glob
    from natsort import natsorted
    
    user_optuna_kwargs = configparser.ConfigParser().read('OPTUNA.ini', encoding='utf-8')
 
    # user_agent_kwargs = user_optuna_kwargs['agent'] 
    
    base_kwargs = sample_sac_params(trial)
    base_agent_kwargs = base_kwargs[0]
    # base_env_kwargs = base_kwargs[1]
    
    # base_agent_kwargs.update(user_agent_kwargs)
    # obs_conf.update(base_env_kwargs)
    
    '''[TODO] 
    (1) tensorboard (using SAC setting)
    (2) SQL Alchemy (DB setup for logs)
    (3) Optuna-Dashboard setup & test
    (4) Optuna hyperparameter tuning for both with JW vehicle and original race car (F1)
    (5) implementation on which files to log (image, lr, reward, dynamic variables etc..)'''
    vec_env = _load_vec_env(args, use_args, obs_conf)
    assert vec_env is not None
    if base_agent_kwargs['use_sde'] == False:
        n_actions = vec_env.action_space.shape[-1]
    
        action_noise = NormalActionNoise(
            mean = np.zeros(n_actions),
            sigma = 0.1 * np.ones(n_actions)
        ) #action noise를 충분히 더해줘야 초반에 exploration을 충분히 할 수 있음.#
    else:
        action_noise = None
    today = datetime.now()
    trial_folder_name = today.strftime("%Y_%m_%d__%H_%M_%S")
    tensorboard_path = f"{TENSORBOARD_LOG}/{trial_folder_name}";os.makedirs(tensorboard_path, exist_ok=True)
    model = SAC(policy="MlpPolicy",
                env=vec_env,
                verbose=1, # verbose=1로 둬야 logging이 (tensorboard에도) 가능
                action_noise=action_noise,
                buffer_size=int(1e+6),
                ent_coef=base_agent_kwargs['ent_coef'],
                gamma=base_agent_kwargs['gamma'],
                learning_rate=base_agent_kwargs['learning_rate'],
                train_freq=1,
                tau=base_agent_kwargs['tau'],
                target_update_interval=base_agent_kwargs['target_update_interval'],
                policy_kwargs={
                    'net_arch': base_agent_kwargs['net_arch'],
                    'activation_fn': base_agent_kwargs['activation_fn'],
                },
                use_sde=base_agent_kwargs['use_sde'],
                sde_sample_freq=base_agent_kwargs['sde_sample_freq'],
                # tensorboard_log=TENSORBOARD_LOG
                tensorboard_log=tensorboard_path,
                )
    folders = natsorted(glob(TENSORBOARD_LOG + "/*"))

    log_path = f"{LOG_PATH}/trial_{len(folders)+1}";os.makedirs(log_path, exist_ok=True)

    pickle.dump(user_optuna_kwargs, open(f'{log_path}/optuna_args.pkl', 'wb'))
    pickle.dump(args, open(f'{log_path}/args.pkl', 'wb'))
    
    '''[TODO] environment configurations loading방법 -> solved with additional params with functools.partial'''
    eval_env = Monitor(Toy_CarRacing_NAM(observation_config=obs_conf, 
                                         do_zoom=3., 
                                         args=args))
    eval_callback = TrialEvalCallback(
        eval_env=eval_env,
        trial=trial,
        n_eval_episodes=1,
        eval_freq=5000,
        # eval_freq = 1,
        # n_eval_episodes=int(user_optuna_kwargs['main']['N_EVAL_EPISODES']),
        # eval_freq=int(user_optuna_kwargs['main']['EVAL_FREQ']),
        deterministic=True,
        # log_path=folders[-1]
        log_path=log_path,
        best_model_save_path=log_path
    )
    
    nan_encountered = False
    try:
        # 하나의 optuna trial당 여러개의, 즉 각 episode마다 folder이 
        # {알고리즘 이름}_{episode 번호}이런식이라 헷갈렸으나, tb_log_name을 특정지어주고 모든 episode마다 같게 하면 문제 X
        # --> 다시 수정..각 trial 마다 datetime으로 폴더를 새로 tensorboard_path로 만들어 주고, 그 안에서 SAC_1, SAC_2이런식으로 생성되게 하는것이 낫다.
        # --> 마지막 수정.. 폴더를 새로 만들어도 관리는 편해지지만 그 안에서 또 SAC_1, SAC_2이런 식으로 파일을 만들었기 때문에, learn 함수 안에서의 argument로 reset_num_timesteps=False로 두면 같은 폴더에 log 파일이 저장된다.
        # [참고 코드] C:\Users\7459985\Desktop\2024\24001_Visual_Track_Instructor\008_car_data_gen\toy_proj\step_1_rl\rl_src\stable\common\base_class.py
       

        for episode in range(use_args.max_episodes):
            model.learn(
                base_agent_kwargs['n_steps'],
                callback=eval_callback,
                # tb_log_name=trial_folder_name
                tb_log_name=f"EP{episode}",
                reset_num_timesteps=False 
                )
    except AssertionError as e:
        print(e)
        nan_encountered = True
    finally:
        # model.env.close()
        vec_env.close()
        eval_env.close()
    
    if nan_encountered:
        return float('nan')
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    return eval_callback.last_mean_reward
    
    

if __name__ == "__main__":
    args, use_args, obs_conf = prepare_args() ## 여기서 반환되는 args와 obs_conf를 environment 설계에 사용해야 함.
    optuna_config = configparser.ConfigParser()
    optuna_config.read('OPTUNA.ini')
    
    sampler = TPESampler(n_startup_trials=int(optuna_config['sampler']['N_STARTUP_TRIALS']))
    pruner = MedianPruner(n_startup_trials=int(optuna_config['pruner']['N_STARTUP_TRIALS']))
    
    study = optuna.create_study(sampler=sampler, 
                                pruner=pruner,
                                direction="minimize")#RL의 actor / critic loss를 결국에는 최소화 해야 하니까 minimize로 두는게 맞음
    try:
        objective = partial(objective, args=args, use_args=use_args, obs_conf=obs_conf)
        study.optimize(objective, 
                       n_trials=int(optuna_config['main']['N_TRIALS']) )#지정한 N_TRIAL의 개수만큼 hyperparameter을 바꾸면서,,
    except KeyboardInterrupt:
        pass
    
    
    print("Best trial:")
    trial = study.best_trial
    
    print("   Value: ", trial.value)
    
    print("   Params:  ")
    for key, value in trial.param.items():
        print("    {}:   {}".format(key, value))
    
    for key, value in trial.user_attrs.items():
        print("    {}:   {}".format(key, value))
    