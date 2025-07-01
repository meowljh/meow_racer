import numpy as np
import math
from scipy.signal import find_peaks
'''사실상 이 reward들 중 하나만 선택해서 "RACING"을 위한 reward로 부여를 해야 함.
전부 다 짧은 lap time으로 완주를 할 수 있도록 빠른 속도와 한번의 time step에 많은 이동량에 대한 보상을 크게 부여할 수 있게 reward shaping이 되어 있다.

+ E_c, E_phi에 대해서는 penalty vs reward 중 어떤 식으로 학습을 시키면 좋을지
+ Reward의 범위와 terminate 될 때의 값의 차이가 어느 정도까지여야 제대로된 reward shaping이라고 할 수 있는지

-----------------------------------------[0510]
+ 우선은 중앙선을 따라가는 policy를 학습하는 것이 목적 -> 무조건 완주를 하고, 급커브 구간에서 감속을 하며 통과를 할 수 있어야 함. -> 고속이 중요한게 아님.
+ curriculum learning을 기반으로 트랙을 고속으로 빠르게 통과할 수 있도록
    -> speed / -time_per_step / +progress_ratio등을 추가해주고자 한다.
''' 

def straight_line_vel_reward(
    car_obj,
    reward_weight,
    normalize_vel:bool,
    corner_kappa_thresh: float,
    **kwargs
):
    car_vx = car_obj.bicycle_model.Vx
    if normalize_vel:
        car_vx /= car_obj.bicycle_model.vehicle_constraints.vx_max
    
    ref_kappa = car_obj.bicycle_model.ref_arr_dict['kappa'][-1]
    if abs(ref_kappa) <= corner_kappa_thresh: #if not corner
        heading_error = car_obj.bicycle_model.E_phi
        reward = car_vx * math.cos(heading_error) * reward_weight
    else:
        reward = 0
    return reward
    
def vel_joint_reward(
    car_obj,
    corner_reward_weight:float,
    straight_reward_weight:float,
    corner_kappa_thresh:float,
    normalize_vel:bool,
    normalize_kappa:bool,
    track_align_vel:bool,
    **kwargs
):
    car_vx = car_obj.bicycle_model.Vx
    if normalize_vel:
        car_vx /= car_obj.bicycle_model.vehicle_constraints.vx_max
        
    ref_kappa = car_obj.bicycle_model.ref_arr_dict['kappa'][-1]
    kappa_arr = car_obj.track_dict['kappa']

    if abs(ref_kappa) > corner_kappa_thresh: #is considered as corner
        # reward = abs(ref_kappa) * car_vx * corner_reward_weight
        if normalize_kappa:
            reward = abs(ref_kappa) / max(abs(kappa_arr)) 
            reward *= car_vx * corner_reward_weight
        else:
            reward = abs(ref_kappa) * car_vx * corner_reward_weight
        # reward = (abs(ref_kappa) / corner_kappa_thresh) * car_vx * corner_reward_weight
        # reward = (abs(ref_kappa)-corner_kappa_thresh) * car_vx * corner_reward_weight
    else: #is considered as straight track
        reward = car_vx * straight_reward_weight
    
    if track_align_vel:
        heading_error = car_obj.bicycle_model.E_phi
        reward *= math.cos(heading_error)
    return reward
        
def hard_corner_curvature_weighted_vel_reward(
            car_obj,
            reward_weight:float, #max reward value that will be available
            normalize_kappa:bool, #if true, will process min-max scaling on the curvature
            corner_kappa_thresh:float, #the curvature's threshold value to consider it as the corner,
            normalize_vel:bool,
            normed_kappa_to_percent:bool,
            **kwargs
        ):
    kappa_arr = car_obj.track_dict['kappa']
    abs_kappa_arr = abs(kappa_arr)
    ref_kappa = car_obj.bicycle_model.ref_arr_dict['kappa'][-1]
    car_vx = car_obj.bicycle_model.Vx
    if normalize_vel:
        car_vx /= car_obj.bicycle_model.vehicle_constraints.vx_max
    
    if abs(ref_kappa) >= corner_kappa_thresh: #is hard corner
        if normalize_kappa:
            kappa_normed = abs(ref_kappa) / max(abs_kappa_arr)
            if normed_kappa_to_percent:
                kappa_normed *= 100
            reward = kappa_normed * car_vx * reward_weight
        else:
            reward = abs(ref_kappa) * car_vx * reward_weight
    else:
        reward = 0
    heading_error = car_obj.bicycle_model.E_phi
    
    reward *= math.cos(heading_error)
    return reward
    

    
    
    
def curvature_weighted_vel_reward(car_obj,
                                  kappa_weight_value:float,
                                  reward_weight:float,
                                  normalize_vel:bool,
                                  **kwargs
                                  ):
    vx, vy = car_obj.bicycle_model.Vx, car_obj.bicycle_model.Vy
    car_vel = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    if normalize_vel:
        bm = car_obj.bicycle_model
        max_speed = np.sqrt(bm.vehicle_constraints.vx_max**2 + bm.vehicle_constraints.vy_max**2)
        car_vel /= max_speed
        
    ref_track_kappa = car_obj.bicycle_model.ref_arr_dict['kappa'][-1]
    ref_track_kappa = abs(ref_track_kappa)
    
    reward = car_vel / (1 + kappa_weight_value * ref_track_kappa)
    reward *= reward_weight
    
    return reward

def progress_reward(car_obj,
                    reward_value: float,
                    **kwargs):
    ref_theta_arr = car_obj.bicycle_model.ref_arr_dict['theta']
    ref_theta = ref_theta_arr[-1]
    ref_theta_prev = ref_theta_arr[-2] if len(ref_theta_arr) > 1 else 0
    
    moved = ref_theta - ref_theta_prev
    progress = moved / max(car_obj.track_dict['theta'])
    
    return progress * reward_value * 100

def curvature_vel_reward(car_obj,  
                          target_theta:int,
                          reward_value:float,
                          future_mode:str='mean', 
                          continuous:bool=True,
                        
                          ############[debug-may fix possible error]#############
                          fix_possible_error:bool=False,
                          vehicle_speed_vx:bool=False,
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
    
    if not fix_possible_error:
        if target_kappa < 1e-5:
            recommend_vel = bm.vehicle_constraints.vx_max
        else: 
            recommend_vel = np.sqrt((bm.vehicle_model.mu * bm.vehicle_model.gravity) / target_kappa)
    else:
        recommend_vel = np.sqrt((bm.vehicle_model.mu * bm.vehicle_model.gravity) / target_kappa)
        recommend_vel = min(recommend_vel, bm.vehicle_constraints.vx_max)
    
    vehicle_speed = np.sqrt(bm.Vx**2 + bm.Vy**2) if not vehicle_speed_vx else bm.Vx
    
    if continuous:
        speed_ratio = vehicle_speed / recommend_vel
        reward = np.exp(-abs(speed_ratio - 1))
        reward *= reward_value
    else:
        if vehicle_speed > recommend_vel:
            reward = -1 * reward_value
        else:
            reward = vehicle_speed / recommend_vel
            reward *= reward_value
 

    return reward


def E_phi_reward(
    car_obj,
    reward_value:float,
    **kwargs
):
    E_phi = car_obj.bicycle_model.E_phi
    
    reward_heading = math.cos(E_phi)
    
    return reward_value * reward_heading

def movement_reward(
    car_obj,
    reward_value:float,
    reward_min_vel:float,
    reward_max_vel:float,
    overspeed_pen_value:float,
    underspeed_pen_value:float,
    **kwargs
):
    '''
    최대-최소 속도를 기준으로 그 안으로 들어오는 경우에만 movement에 대한 reward를 주고,
    그 이상을 넘는 경우에는 
    '''
    Vx = car_obj.bicycle_model.Vx
    Vy = car_obj.bicycle_model.Vy
    speed = math.sqrt(Vx**2 + Vy**2)
    if reward_min_vel <= speed <= reward_max_vel:
        # mean_vel = (reward_min_vel + reward_max_vel) / 2
        # gauss_sigma = reward_max_vel - mean_vel
        # reward = np.exp(
        #     -0.5 * ((speed - mean_vel) / gauss_sigma) ** 2
        # )
        # reward *= reward_value
        reward = reward_value
    elif speed > reward_max_vel:
        ratio = ((speed / reward_max_vel) - 1)
        reward = -overspeed_pen_value * ratio
    elif speed < reward_min_vel:
        ratio = (1 - (speed / reward_min_vel))
        reward = -underspeed_pen_value * ratio
    return reward

def neg_velocity_aps_reward(car_obj, reward_weight:float, **kwargs):
    if car_obj.reward_neg_vel_aps:
        return reward_weight
    return 0


def tile_step_reward(
    car_obj,
    max_reward_weight:float, 
    **kwargs,
):
    track_dict = car_obj.track_dict
    num_tiles = len(track_dict['x'])
    
    reward = max_reward_weight / num_tiles
    reward *= car_obj.bicycle_model.current_step_tile_add
    
    return reward

def target_velocity_reward(
    car_obj, 
    reward_weight:float,
    target_vel:float,
    gauss_sigma:float=1.,
    give_penalty_to_out_dist:bool=True,
    use_hard_penalty:bool=False,
    **kwargs
):
    '''PID제어로 테스트를 해 봤을 때 약 10 ~ 12m/s를 유지하면서 주행을 하게 되면 트랙을 온전히 완주를 하는 것을 알 수 있었음.
    그래서 gaussian distribution 기반으로 target
    '''
    bm = car_obj.bicycle_model
    vehicle_speed = np.sqrt(bm.Vx**2 + bm.Vy**2)
    vehicle_speed *= np.cos(bm.E_phi) #이동 방향에 대한 차속 계산
    
    reward = 0
    
    if target_vel - gauss_sigma <= vehicle_speed <= target_vel + gauss_sigma:
        reward = np.exp(
            -0.5 * ((vehicle_speed - target_vel) / gauss_sigma) ** 2
        ) 
    ##reward인 만큼, 목표 target velocity부터 차이가 많이 나는 값인 경우에 penalty를 크게 주거나, reward를 그냥 0으로 두는 것도 방법.
    ##대신 동시에 movement에 대한 reward도 추가로 주면 이동 + 목표 속도 도달을 할 때 reward가 크다는 것을 학습할 수 있을 것.
    else:
        if give_penalty_to_out_dist:
            if vehicle_speed > target_vel + gauss_sigma: #가속하는 경우
                thresh_speed = target_vel + gauss_sigma
                if use_hard_penalty:
                    ratio = (vehicle_speed / thresh_speed) - 1
                else:
                    # ratio = 1 - (thresh_speed / vehicle_speed)
                    diff = vehicle_speed - thresh_speed
                    ratio = diff / thresh_speed
                reward = -1 * ratio
            # else: #감속하는 경우
            elif vehicle_speed < target_vel - gauss_sigma:
                thresh_speed = target_vel - gauss_sigma
                if use_hard_penalty:
                    ratio = (thresh_speed / vehicle_speed) - 1
                    if vehicle_speed == 0:
                        ratio = thresh_speed
                else:
                    diff = thresh_speed - vehicle_speed
                    ratio = diff / thresh_speed
                    # if vehicle_speed == 0:
                    #     ratio = thresh_speed
                    # else:
                    #     ratio = (thresh_speed / vehicle_speed) - 1
                    # if vehicle_speed == 0:
                    #     ratio = 1
                    # else:
                    #     ratio = 1 - (vehicle_speed / thresh_speed)
                        
                reward = -1 * ratio 

    reward *= reward_weight 
    
    return reward
    
    
    
def track_align_velocity_reward(
    car_obj, 
    reward_weight:float, 
    normalize_vel:bool,
    **kwargs
):
    '''
    - speed만 고려: 빠른 주행 유도하지만 방향 무관
    - cos(E_phi): 방향 정렬 유도하지만 속도는 고려 X
    - speed * cos(E_phi): 속도와 정렬을 모두 고려 가능
    '''
    bm = car_obj.bicycle_model
    vehicle_speed = np.sqrt(bm.Vx**2 + bm.Vy**2)
    heading_error = bm.E_phi
    speed_forward = vehicle_speed * np.cos(heading_error)
    if normalize_vel:
        max_speed = np.sqrt(bm.vehicle_constraints.vx_max**2 + bm.vehicle_constraints.vy_max**2)
        speed_forward /= max_speed
    reward = speed_forward * reward_weight
    
    return reward
    
def center_consist_reward(car_obj,
                          max_reward:float, 
                          E_c_thresh:float,
                          grad:float,
                          **kwargs):
    E_c = car_obj.bicycle_model.E_c
    
    if abs(E_c) < E_c_thresh:
        reward = max_reward
    else:
        b = max_reward - E_c_thresh * grad
        reward = grad * abs(E_c) + b
        
    # elif abs(E_c) < 1:
    #     reward = reward_weight * (1-abs(E_c))
    # else:
    #     reward = 1 / abs(E_c) 
    
    return reward

def correct_steering_reward(car_obj, forward_theta_range:int,
                            reward_weight:float,
                            **kwargs):
    '''현재 차량의 위치로부터 제일 가까운 peak corner의 곡률 방향과 맞게 
    steering 방향이 조절된 경우 reward를 부여하도록 함.''' 
    steer_action = car_obj.action['steer'][-1]
    #좌회전: steer, kappa > 0
    #우회전: steer, kappa < 0 
    ref_theta = car_obj.bicycle_model.ref_arr_dict['theta'][-1] 
    #전방의 100m 앞에서 제일 절댓값이 큰 kappa 값
    kappa_spline = car_obj.kappa_spline
    max_kappa = 0 
    for i in range(forward_theta_range):
        kappa_val = kappa_spline(ref_theta+i)
        if abs(max_kappa) < abs(kappa_val):
            max_kappa = kappa_val
            
    if max_kappa * steer_action >= 0: #방향이 같으면 그냥 reward weight 추가
        return reward_weight
    
    return 0
    

def alignment_reward(car_obj, reward_weight:float, **kwargs):
    '''중앙선과의 heading 방향의 정렬이 맞는지에 대한 reward'''
    e_phi = car_obj.bicycle_model.E_phi
    reward = math.cos(e_phi) * reward_weight
    
    return reward  
    
def center_line_reward(car_obj,
                       e_phi_weight:float,
                       e_c_weight:float, 
                       **kwargs):
    bm = car_obj.bicycle_model
    track_half_width = 3.5
    e_phi = bm.E_phi #부호와 상관 없이 e_phi의 절대적인 크기를 줄이도록 하는 것이 목표
    e_c = bm.E_c #중앙선으로부터의 거리도 줄이도록 해야 함.
    reward = 0
    e_c_ratio = 1. - (abs(e_c) / track_half_width)
    
    e_phi_ratio = 1. - (abs(e_phi) / (np.pi/2)) #이렇게 하면 방향이 반대로 이동하는 중이라면 음수가 됨
    
    reward = (e_c_ratio * e_c_weight) + (e_phi_ratio * e_phi_weight)
    
    return reward


 

def progress_reward_euclidian(car_obj, reward_weight:float, **kwargs):
    '''직선 거리 기반으로 progress reward'''
    bicycle_model = car_obj.bicycle_model
    traj_arr = bicycle_model.car_traj_arr
    
    if len(traj_arr) == 1:
        reward = 0
    else:
        x1, y1 = traj_arr[-2][:2] # time step (t-1)
        x2, y2 = traj_arr[-1][:2] # time step (t)
        reward = math.dist([x1, y1], [x2, y2])
    return reward * reward_weight

def curvature_reward(car_obj, n_points, d_theta,
                     reward_weight:float, **kwargs):
    '''-steer * curvature * reward_weight = final_reward
    curvature은 전방의 몇 m이상 전진했을떄의 평균 곡률임.
    steer > 0: 좌회전
    steer < 0: 우회전
    
    우회전: kappa < 0
    좌회전: kappa > 0
    '''
    
    steer_val = car_obj.actions['steer'][-1]
    ref_arr_dict = car_obj.bicycle_model.ref_arr_dict
    ref_theta = ref_arr_dict['theta'][-1]
    ref_kappa = ref_arr_dict['kappa'][-1]
    kappa = ref_kappa
    for i in range(n_points):
        kappa += car_obj.kappa_spline(ref_theta + d_theta * (i+1))
    mean_kappa = kappa / n_points
    reward = steer_val * mean_kappa * reward_weight # 두개의 부호가 같아야 동일 방향의 curvature에 대해서 구할 수 있음.
    
    return reward
        
    
def progress_reward_curve(car_obj, reward_weight:float,
                          scale_progress:bool=False,
                          **kwargs):
    '''theta distance error (theta_t - theta_(t-1))'''
    bicycle_model = car_obj.bicycle_model
    ref_theta_arr = bicycle_model.ref_arr_dict['theta']
    
    if len(ref_theta_arr) == 1:
        reward = ref_theta_arr[0] 
    else:
        reward = ref_theta_arr[-1] - ref_theta_arr[-2]
    
    if scale_progress:
        reward_scale = max(car_obj.track_dict['theta'])
        reward = reward / reward_scale
        reward *= 100
        
    return reward * reward_weight

def velocity_reward(car_obj, reward_weight:float, **kwargs):
    bicycle_model = car_obj.bicycle_model
    v_x = bicycle_model.Vx
    # ref_track_phi = bicycle_model.ref_arr_dict['phi'][-1]
    # car_heading = bicycle_model.car_phi
    
    # heading_angle_diff = ref_track_phi - car_heading
    heading_angle_diff = bicycle_model.E_phi
    
    reward = v_x * math.cos(heading_angle_diff) #트랙의 진행 방향으로의 종방향 속도
    
    return reward * reward_weight

def distance_based_reward(car_obj, max_theta:float, theta_weight, **kwargs):
    '''
    @max_theta: 새로 랜덤하게 트랙을 만들고, 그 트랙의 전체 theta가 다름. 최대 theta값을 의미'''
    theta_dist = progress_reward_curve(car_obj=car_obj)
    reward = theta_weight * (theta_dist / max_theta)
    
    return reward

def vel_dist_balanced_reward(car_obj, vel_weight, dist_weight, **kwargs):
    bicycle_model = car_obj.bicycle_model
    vel = bicycle_model.Vx
    dist = progress_reward_euclidian(car_obj=car_obj)
    
    reward = dist * dist_weight + vel * vel_weight
    
    return reward

def vel_error_balanced_reward(car_obj, vel_weight, c_error_weight,
                              do_scale:bool=True, **kwargs):
    bicycle_model = car_obj.bicycle_model
    vel_max_constraint = car_obj.vehicle_constraints.vx_max if do_scale else 1.
    track_half_width = 7. / 2 if do_scale else 1.
    
    reward = vel_weight * velocity_reward(car_obj=car_obj) / vel_max_constraint
    reward += c_error_weight * abs(bicycle_model.E_c) / track_half_width
    
    return reward

def attitude_reward(car_obj, **kwargs):
    bicycle_model = car_obj.bicycle_model
    v_x = bicycle_model.Vx
    d_axis = abs(bicycle_model.E_c) #단순히 중앙선으로부터의 떨어진 거리의 절댓값
    
    ref_track_phi = bicycle_model.ref_arr_dict['phi'][-1]
    car_heading = bicycle_model.car_phi
    # ref_track_phi %= np.pi #후진하는 상황에 대비 -> 왜냐면 트랙의 phi값은 누적식으로 증가하는 추세를 보이게 설계 되어 있기 때문에 주기가 있는 형태로 바꿔야 함.
    '''변수 자체에 in-space로 나눠서 값을 할당해주는 operation은 최대한 피하자. 그 변수 뿐 아니라 배열 속의 해당 값이 함께 바뀌는 듯 함'''
    # theta = (ref_track_phi % np.pi) - car_heading
    theta = ref_track_phi %  (np.pi/2) - car_heading
    
    reward = (math.cos(theta) - math.sin(abs(theta)) - d_axis) * v_x
    
    return reward