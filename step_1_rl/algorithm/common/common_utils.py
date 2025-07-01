import torch
import torch.nn as nn
import copy
import math
import os, sys
import random
import numpy as np
from abc import *

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #algorithm
sys.path.append(os.path.dirname(root))
sys.path.append(f"{os.path.dirname(root)}/envs")
from gym_car_constants import *

 

'''common_utils.py
- dataset 관련 코드 (전처리 용)

'''
DYNAMICS_BOUND_DICT = {
    'theta': [], # 곡선상에서의 진행 거리 #
    'e_phi': [-2 * math.pi, 2 * math.pi],
    'e_c': [-TRACK_WIDTH, TRACK_WIDTH],
    'v_x': [-30., 80.],
    'v_y': [-20., 20.],
    'yaw_omega': [], # 차량의 조향각의 각속도 (rad/s) #
}

CAR_BOUND_DICT = {
    'omega': [0, 360.],
    'steer': [-1., 1.],
    'gas': [0., 1.],
    'brake': [0., 1.], #
    'force': [-1000000, 1000000], # 차체에 가해지는 힘 #
    'delta': [-math.pi*2, math.pi*2], # 조향각 #
    'forward': [], # 설정해둔 길이로 확인 #
    'curvature': [-0.1, 0.1]
}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
   
def load_activation(name):
    if name.lower() == "tanh":
        return nn.Tanh()
    elif name.lower() == "relu":
        return nn.ReLU()
    else:
        raise UserWarning(f"Activation function {name.lower()} NOT supported!")
    
    
'''[25.01.03] (add_state_normalizer) 
- 입력 state에 대해서 [Multi-policy Soft Actor-Critic Reinforcement Learning for Autonomous Racing] 논문에 적혀 있는 것 처럼 
min-max norm을 취해 주었음.
- 일부 동역학과 관련된 (차량의 limit이 관여하는) state들에 대해서는 min, max value들을 다시 계산해 볼 필요가 있음.
- 우선은 (0,1) 사이의 값으로 정규화
'''
def preprocess_state(env, state):
    obs_conf = env.observation_config
    # breakpoint()
    assert state.shape[-1] == sum([len(v) for v in obs_conf.values()])
    
    preprocessed = copy.deepcopy(state)
    state_idx = 0

    
    ## 1. preprocess for <dynamic> states ##
    #theta, e_phi, e_c, v_x, v_y, yaw_omega#
    if 'dynamic' in obs_conf:
        for i, key in enumerate(obs_conf['dynamic']):
            m, M = DYNAMICS_BOUND_DICT[key]
            v = state[state_idx]
            v = (v - m) / (M - m)
            preprocessed[state_idx] = v
            state_idx += 1
            
    ## 2. preprocess for <lidar> states ##
    # 그냥 lidar의 최대 길이로 나눠주면 됨 (min length는 0) #
    min_lidar, max_lidar = -1., env.input_args.lidar_length
    if 'lidar' in obs_conf:
        # for i, _ in enumerate(obs_conf['lidar']):
        for _ in obs_conf['lidar']:
            v = state[state_idx]
            v = (v - min_lidar) / (max_lidar - min_lidar)
            preprocessed[state_idx] = v
            state_idx += 1
            
    ## 3. preprocess for <car> states ##
    if 'car' in obs_conf:
        for key in obs_conf['car']:
            v = state[state_idx]
            if 'forward' in key:
                M = env.input_args.num_vecs * env.input_args.theta_diff
                m = -M
            elif 'omega' in key:
                m, M = CAR_BOUND_DICT['omega']
            elif 'force' in key:
                m, M =  CAR_BOUND_DICT['force']
            elif 'curvature' in key:
                m, M = CAR_BOUND_DICT['curvature']
            else:
                m, M = CAR_BOUND_DICT[key]
            v = (v - m) / (M - m)
            preprocessed[state_idx] = v
            state_idx += 1  
 
    return preprocessed