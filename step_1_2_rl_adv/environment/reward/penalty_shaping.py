import math
import numpy as np
from .vehicle_status_check import OffCourse_Checker

def curvature_velocity_penalty(
    car_obj,
    penalty_weight:float,
    norm_kappa:bool,
    norm_vel:bool,
    normed_kappa_clip_value:float=None,
    **kwargs
):
    car_vx = car_obj.bicycle_model.Vx
    ref_kappa = abs(car_obj.bicycle_model.ref_arr_dict['kappa'][-1])
    if norm_kappa:
        ref_kappa /= max(abs(car_obj.track_dict['kappa']))
        if normed_kappa_clip_value is not None:
            ref_kappa = np.clip(ref_kappa, 0, normed_kappa_clip_value)
    if norm_vel:
        car_vx /= car_obj.bicycle_model.vehicle_constraints.vx_max
    
    penalty = ref_kappa * car_vx * penalty_weight
    penalty *= -1
    
    return penalty
    
    
def smooth_control_penalty(
    car_obj, 
    aps_penalty_value:float,
    bps_penalty_value:float, 
    steer_penalty_value:float,
    **kwargs
):
    steer_t2, aps_t2, bps_t2 = car_obj.actions['steer'][-1], car_obj.actions['throttle'][-1], car_obj.actions['brake'][-1]
    if len(car_obj.actions['steer']) > 1:
        steer_t1, aps_t1, bps_t1 = car_obj.actions['steer'][-2], car_obj.actions['throttle'][-2], car_obj.actions['brake'][-2]
    else:
        penalty = 0
        return penalty
    weight_arr = np.array([steer_penalty_value, aps_penalty_value, bps_penalty_value])
    diff_arr = np.array([steer_t2, aps_t2, bps_t2]) - np.array([steer_t1, aps_t1, bps_t1])
    weighted_diff_arr = diff_arr**2 * weight_arr
    penalty = weighted_diff_arr.sum() * -1
    
    return penalty

def min_velocity_penalty(
    car_obj, penalty_value:float, 
    min_velocity:float, **kwargs
):
    car_vx = car_obj.bicycle_model.Vx
    if car_vx <= min_velocity:
        car_obj.bicycle_model.min_vel_pen_count += 1
    else:
        car_obj.bicycle_model.min_vel_pen_count = 0
    
    penalty = penalty_value * car_obj.bicycle_model.min_vel_pen_count
    penalty *= -1
     
    return penalty
    
def neg_velocity_bps_penalty(
    car_obj,
    penalty_value:float, **kwargs):
    if car_obj.terminate_neg_vel_bps:
        return -penalty_value
    return 0
 

def curvature_vel_penalty(car_obj, 
                          penalty_value:float, 
                          target_theta:int,
                          future_mode:str='mean', 
                          **kwargs):
    '''급커브 구간에 대해서 속도 조절 (감속 유도)
    slip이 발생하지 않을 때까지의 속도 사용'''
    bm = car_obj.bicycle_model
    ref_kappa = bm.ref_arr_dict['kappa'][-1]
    ref_theta = bm.ref_arr_dict['theta'][-1]
    kappa_spline = car_obj.kappa_spline
    
    if future_mode=='single':
        target_kappa = abs(kappa_spline(ref_theta + target_theta))
    elif future_mode == 'max':
        max_kappa = 0
        for theta in range(target_theta):
            max_kappa = max(max_kappa, abs(kappa_spline(ref_theta + theta )))
        target_kappa = max_kappa
    else:
        sum_kappa = 0
        for theta in range(target_theta):
            sum_kappa += abs(kappa_spline(ref_theta + theta))
        target_kappa = sum_kappa / target_theta
        
    # if abs(ref_kappa) < 1e-5:
    if target_kappa < 1e-5:
        recommend_vel = bm.vehicle_constraints.vx_max
    else:
        # recommend_vel = np.sqrt((bm.vehicle_model.mu * bm.vehicle_model.gravity) / abs(ref_kappa))
        recommend_vel = np.sqrt((bm.vehicle_model.mu * bm.vehicle_model.gravity) / target_kappa)
    
    vehicle_speed = np.sqrt(bm.Vx**2 + bm.Vy**2)
    
    if vehicle_speed > recommend_vel:
        penalty = penalty_value * (vehicle_speed / recommend_vel - 1.)
    else:
        penalty = 0
    
    penalty *= -1

    return penalty

        
def E_phi_penalty(car_obj, penalty_value:float, **kwargs):
    ''''''
    E_phi = car_obj.bicycle_model.E_phi
    if -math.pi/2 > E_phi or math.pi/2 < E_phi: ## if wrong direction
        return -penalty_value
    penalty = 1 - math.cos(E_phi) 
    penalty *= penalty_value
    penalty *= -1
    
    return penalty

def E_c_penalty(car_obj, penalty_value:float, 
                normalize_E_c:bool=True,
                as_reward:bool=False,
                **kwargs):
    ''''''
    E_c = car_obj.bicycle_model.E_c
    if normalize_E_c:
        max_e_c = car_obj.bicycle_model.vehicle_model.dx_max
        penalty = abs(E_c) / max_e_c
    else:
        penalty = abs(E_c)
        
    if as_reward:
        reward = np.exp(penalty)
        reward *= penalty_value
        return reward
     
    penalty *= penalty_value
    penalty *= -1
    
    return penalty

def fast_brake_change_penalty(car_obj, penalty_value:float, **kwargs):
    '''이전의 steering action에 비해서 많이 변화가 없어야 하는게 목적임. 그럼 penalty로 줘야지?'''
    brake_actions = car_obj.actions['brake']
    if len(brake_actions) == 1:
        penalty = 0
    else:
        steer_diff = brake_actions[-1] - brake_actions[-2] 
        penalty = abs(steer_diff) * penalty_value 
    penalty *= -1
    
    return penalty

def fast_throttle_change_penalty(car_obj, penalty_value:float, **kwargs):
    '''이전의 steering action에 비해서 많이 변화가 없어야 하는게 목적임. 그럼 penalty로 줘야지?'''
    throttle_actions = car_obj.actions['throttle']
    if len(throttle_actions) == 1:
        penalty = 0
    else:
        steer_diff = throttle_actions[-1] - throttle_actions[-2] 
        penalty = abs(steer_diff) * penalty_value 
    penalty *= -1
    
    return penalty


def fast_steer_change_penalty(car_obj, penalty_value:float, **kwargs):
    '''이전의 steering action에 비해서 많이 변화가 없어야 하는게 목적임. 그럼 penalty로 줘야지?'''
    steer_actions = car_obj.actions['steer']
    if len(steer_actions) == 1:
        penalty = 0
    else:
        steer_diff = steer_actions[-1] - steer_actions[-2] #최대 격차의 절댓값이 2임
        penalty = abs(steer_diff) / 2. * penalty_value #위와 같은 이유로 0-1로 절댓값의 범위를 바꿔야 함.
    penalty *= -1
    
    return penalty

def time_penalty(penalty_value:float, **kwargs):
    return -1 * penalty_value

def off_course_penalty(car_obj, penalty_value:float, 
                       condition: str, 
                       ratio_usage:bool,
                       **kwargs):
    checker = OffCourse_Checker(car_obj=car_obj)
    if 'com' in condition:
        is_off = checker._off_course_com()
        # thresh_Ec = 3.5
    elif 'all' in condition:
        is_off = checker._off_course_all()
        # thresh_Ec = 3.5 + car_obj.bicycle_model.vehicle_model.body_width / 2
    elif 'instance' in condition:
        is_off = checker._off_course_instance()
        # thresh_Ec = 3.5 - car_obj.bicycle_model.vehicle_model.body_width / 2
    elif 'tire' in condition:
        is_off = checker._off_course_tire()
        # thresh_Ec = 3.5 + car_obj.bicycle_model.vehicle_model.body_width / 2
    elif 'count' in condition:
        '''트랙 밖으로 나갔을 때의 penalty를 주기 위해서 트랙 밖으로 나간 tire의 개수에 비례하도록 함.'''
        status_checker = car_obj.vehicle_status_checker
        num_tire_out = status_checker.off_track_tire_cnt
        penalty = penalty_value * (num_tire_out / 4)
        penalty *= -1
        return penalty
    
    thresh_Ec = 3.5 + car_obj.bicycle_model.vehicle_model.body_width / 2
        
        
    penalty = 0
    if is_off:
        if not ratio_usage:
            penalty = -1 * penalty_value
        else:
            E_c = car_obj.bicycle_model.E_c #center of mass기준으로 거리 측정함
            penalty = -1 * ((abs(E_c) / thresh_Ec) ** 2) #제곱을 해서 트랙 끝으로 갈수록 penalty를 더 키움
            penalty *= penalty_value
        
    return penalty
    
def reverse_penalty(car_obj, penalty_value, 
                    add_vel:bool,
                    use_cummulative:bool,
                    max_kph:float=200,
                    **kwargs):
    bicycle_model = car_obj.bicycle_model
    vx = bicycle_model.Vx
    dtheta = bicycle_model.dTheta
    max_ms = max_kph / 3.6
    penalty = 0.
    # if vx <= 0:
    if dtheta <= 0: #뒤로 이동하고 있으면
        if use_cummulative:
            penalty = -1 * penalty_value * car_obj.bicycle_model.backward_counter
        else:
            penalty = -1 * penalty_value 
    ##속도가 양수인 것에 대한 reward이기도 함
    if add_vel:
        penalty += min(vx, max_ms) #55m/s, 즉 200kph를 상한선으로 둔다고 볼 수 있음.
    return penalty
 

def wrong_direction_penalty(car_obj, penalty_value, **kwargs):
    bicycle_model = car_obj.bicycle_model

    heading_angle = bicycle_model.E_phi
    
    penalty = 0 if -math.pi / 2 <= heading_angle <= math.pi / 2 else -1 * penalty_value
    
    
    return penalty
    
def tire_slip_penalty(car_obj, penalty_weight, **kwargs):
    bicycle_model = car_obj.bicycle_model
    slip_angle_f, slip_ratio_f = bicycle_model.alpha_f, bicycle_model.sigma_f
    slip_angle_r, slip_ratio_r = bicycle_model.alpha_r, bicycle_model.sigma_r
    
    slip_angle_arr = [slip_angle_f, slip_angle_r]
    slip_ratio_arr = [slip_ratio_f, slip_ratio_r]
    
    penalty = 0
    
    for _, (angle, ratio) in enumerate(zip(slip_angle_arr, slip_ratio_arr)):
        penalty += min(1, abs(ratio)) * abs(angle)
    penalty *= penalty_weight
    return -penalty
