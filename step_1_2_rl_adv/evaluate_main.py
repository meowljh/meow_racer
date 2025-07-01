import torch

import numpy as np
import pickle
import os, sys
import copy
from evaluate_logger import RacerLogger
from environment.racedemia_env_v1 import Racedemia_Env
from rlkit.torch.sac.policies import TanhGaussianPolicy

def _get_single_rollout(env, agent, obs, preprocess_obs_for_policy_fn=None,
                        style_setting:float=None):
    '''[TODO]
    1. 어차피 environment normalization wrapper을 사용하는데, 혹시 observation_normalization_fn이 있어야 하는지 확인 필요
    2. evaluate 단계에서는 모든 style setting에 대해서 주행 궤적을 얻어야 하기 때문에 style_setting 입력을 임의로 받을 수 있어야 함
    '''
    rollout_dict = {}
    
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
        
    o_for_agent = preprocess_obs_for_policy_fn(obs)
    # get distribution from observation -> samples from the distribution
    # when evaluation mode, the distribution is Delta distribution (deterministic)
    action, _ = agent.get_action(o_for_agent) # action, empty dict(agent info)
    
    
    env_step_ret = env.step(copy.deepcopy(action),
                            style_setting=style_setting)
    
    if len(env_step_ret) == 5:
        next_o, reward, terminated, truncated, env_info = env_step_ret
        done = terminated | truncated
    elif len(env_step_ret) == 4:
        next_o, reward, done, env_info = env_step_ret
    
    
    rollout_dict['next_observation'] = next_o
    rollout_dict['action'] = action
    rollout_dict['terminal'] = done
    rollout_dict['reward'] = reward
    rollout_dict['failed'] = terminated
    rollout_dict['success'] = truncated
    
    
    return rollout_dict
    
    
def evaluate(eval_policy:TanhGaussianPolicy, 
             eval_env:Racedemia_Env,
             epoch_num:int,
             test_max_path_length:int=1000000,
             test_log_path:str=None,
             style:float=None,
             **kwargs
             ):
    '''evaluation code referenced from the MdpPathCollector in samplers/data_collector/path_collector.py,
    and rollout function from samplers/rollout_functions.py
    (하나의 track만을 사용해서 )'''
    os.makedirs(test_log_path, exist_ok=True)
    new_test_log_path = f"{test_log_path}/test";os.makedirs(new_test_log_path, exist_ok=True)
    
    logger = RacerLogger(env=eval_env)
    
    SUCCESS = False
    
    path_length = 0
    # eval_env.reset(style=style)
     
    
    # o, env_info_dict = eval_env.reset()
    o, env_info_dict = eval_env.reset(style=style)
    observations = []
    actions = []
    rewards = []
    terminal = []
    
    # count_reverse = 0
    # reverse_patience = 100
    # breakpoint()
    
    while path_length < test_max_path_length:
        single_rollout_dict = _get_single_rollout(env=eval_env, agent=eval_policy, obs=o,
                                                  style_setting=style)
        
        observations.append(single_rollout_dict['next_observation'])
        actions.append(single_rollout_dict['action'])
        is_done = single_rollout_dict['terminal']
        terminal.append(is_done)
        
        o = single_rollout_dict['next_observation']
        
        logger._update_logs()
        # logger._log_trajectory_screen(log_file_root=test_log_path, log_file_prefix=f"epoch_{epoch_num}", num_step=path_length)
        
        # is_failed = single_rollout_dict['failed'] #failed=True when car is out of track or timeout#
        if is_done: #어쨌든 끝난 상황에서는 무조건 brake를 해야 함.
            # if not is_failed: #if is_failed=False and is_done=True then the car has finished the track#
            if single_rollout_dict['success']:
                SUCCESS = True
            break
        
        # if eval_env.unwrapped.car.bicycle_model.dTheta <= 0:
        #     count_reverse += 1
        # else:
        #     count_reverse = 0
            
        path_length += 1
        
        # if count_reverse >= reverse_patience:
        #     SUCCESS = False
        #     is_done = True
        #     break
        
    if SUCCESS:
        print(f"Succeeded in Finishing Nam-C Track at epoch {epoch_num}")
    else:
        print(f"Failed in Finishing Nam-C Track at epoch {epoch_num}")
        
        
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

    # if SUCCESS: #완주에 성공한 경우에만 logging 하기
    success_prefix = f"success_style_{style}_epoch" if style else "success_epoch"
    fail_prefix = f"fail_style_{style}_epoch" if style else "fail_epoch"
    if SUCCESS:
        logger._dump_actions(log_file_root = new_test_log_path, log_file_prefix = f"{success_prefix}_{epoch_num}")
        logger._dump_trajectory(log_file_root=new_test_log_path, log_file_prefix=f"{success_prefix}_{epoch_num}")
        logger._dump_sarsa(file = test_ret_dict, log_file_root=new_test_log_path, log_file_prefix=f"{success_prefix}_{epoch_num}")
    else:
        if epoch_num % 10 == 0:
            logger._dump_actions(log_file_root = new_test_log_path, log_file_prefix = f"{fail_prefix}_{epoch_num}")
            logger._dump_trajectory(log_file_root=new_test_log_path, log_file_prefix=f"{fail_prefix}_{epoch_num}")
            logger._dump_sarsa(file = test_ret_dict, log_file_root=new_test_log_path, log_file_prefix=f"{fail_prefix}_{epoch_num}")

    """[0519] Debug evaluation step (seems to only take maximum 100 steps for each evaluation)"""
    # breakpoint()
    
    return test_ret_dict, SUCCESS
        