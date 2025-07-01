import numpy as np
import math

from abc import ABC, abstractmethod

from gym_car_constants import OUT_TRACK_LIMIT, TRACK_WIDTH, NAM_TRACK_WIDTH

class RewardShaping(ABC):
    def __init__(self) -> None:
        super().__init__()
        

###################################################################################################
###################################################################################################
def _calculate_reward(env):
    '''reward version from the paper
    [Multi-policy Soft Actor-Critic Reinforcement Learning for Autonomous Racing]'''
    # reward = env.reward ## 이미 PenaltyDetector에 의해서 트랙 벗어남이랑 이동 거리 등 확인 했을 것임.##
    add_reward = 0
        
    car_x, car_y = env.car.hull.position
    car_vx, car_vy = env.car.hull.linearVelocity
    car_vel = math.sqrt(car_vx**2 + car_vy**2)

    dynamic_state = env.dynamics_obj.dynamic_state #dict -> needed for theta value#
    
    is_valid_progress = env.dynamics_obj._check_progress(
        env=env,
        time_interval=env.input_args.progress_time_interval,
        minimum_diff=env.input_args.progress_min_diff
    )
     
    add_reward += is_valid_progress
    
    ## attitude reward (penalty) ##
    e_phi = dynamic_state['e_phi'] # car_yaw - ref_phi #
    e_c = dynamic_state['e_c']
    
    ## TRACK_WIDTH가 트랙 너비 반쪽임
    if abs(e_c) >= TRACK_WIDTH: #트랙 밖을 조금이라도 넘어간 경우#
        add_reward += -env.input_args.limit_weight * (abs(e_c) - TRACK_WIDTH)
        '''우선은 그냥 트랙 밖으로 나간 경우에는 무조건 terminate 되도록 -> 그래야 slip이 없을 듯'''
        env.car_left_track = True

    if abs(e_c) >= OUT_TRACK_LIMIT: #트랙 밖 허용 범위를 넘어간 경우#
        env.car_left_track = True 
    
    '''add reward
    - movement check -> deprecated (all zero for is_valid_process)
    - car_velocity weight (for higher speed)
    - car_theta_diff weight (for more movement forward) 
    - r_attitude
    '''
    add_reward += car_vel * env.input_args.vel_weight
    add_reward += env.dynamics_obj.theta_diff * env.input_args.theta_weight

    d_axis = env.dynamics_obj.center_dist
    
    r_attitude = (math.cos(e_phi) - math.sin(abs(e_phi)) - d_axis)
    v_x = env.car.hull.linearVelocity[0]
    r_attitude *= v_x
    
    add_reward += r_attitude 
    return add_reward
    # return r_attitude + is_valid_progress