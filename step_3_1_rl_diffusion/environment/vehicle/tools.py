import math
import numpy as np

def dist_to_center_simple(xy_spline, theta, X, Y):
    p = xy_spline(theta)
    return math.dist(p, (X, Y))

def dist_to_center(xy_spline, theta, X, Y):
    p = xy_spline(theta)
    dp = xy_spline(theta, nu=1)
    psi = np.arctan2(dp[1], dp[0])
    el = np.cos(psi)*(X-p[0]) + np.sin(psi) * (Y-p[1])
    
    return el**2


def calculate_error_c(ref_x, ref_y, ref_phi, car_x, car_y):
    R = np.array([
        [math.cos(ref_phi), -math.sin(ref_phi)],
        [math.sin(ref_phi), math.cos(ref_phi)]
    ])
    
    v = np.array([car_x-ref_x, car_y-ref_y])
    
    vv = R.T@v
    
    error_c = vv[1]
    
    return error_c

'''REVISED VEHICLE DYNAMICS
(1) APS, BPS action 기반으로 차량 선속도 (velocity) 계산
(2) 이전의 차량의 heading(=phi) 값을 바탕으로 Vx, Vy 계산 (V*cos(phi), V*sin(phi)))
(3) 차량의 Yaw각의 속도 계산 
    -> 횡방향 힘만을 필요로 함
    domega = (Ffy * vm.lf * cos_delta - Fry * vm.lr) / vm.Iz
(4) 바퀴의 회전 속도 계산 (선속도 / 휠반지름)
(5) dX, dY 계산해서 차량의 위치 업데이트
(6) 차량의 위치와 phi 기반으로 E_c, E_phi 계산
'''

#####################################################################################################################
def calculate_friction_force(vm, #vehicleModel object#
                             torque_value, #tau_r#
                             steer_value, #delta#
                             omega,
                             vx,
                             vy,
                             allow_neg_torque:bool=False
                            ): 
    # slip angles: 후륜 구동 bicycle model이기 때문에 전륜에만 steering 각 부여 -> 전륜의 슬립각에만 steer_value을 더해줌 (슬립각) #
    alpha_f = -math.atan2(vm.lf * omega + vy, vx) + steer_value
    alpha_r = math.atan2(vm.lr * omega - vy, vx)
    
    # slip ratio
    sigma_f = math.atan2(vm.lf * omega - vx, vm.lf * omega) if torque_value > 0 else math.atan2(vm.lf * omega - vx, vx)
    sigma_r = math.atan2(vm.lr * omega - vx, vm.lr * omega) if torque_value > 0 else math.atan2(vm.lr * omega - vx, vx)
    
    # sigma_f = (vm.lf * omega - vx) / (vm.lf * omega) if torque_value > 0 else (vm.lf * omega - vx) / vx
    # sigma_r = (vm.lr* omega - vx) / (vm.lr* omega) if torque_value > 0 else (vm.lr* omega - vx) / vx
    
    
    # lateral force (횡방향) #
    Ffy = vm.Df * math.sin(vm.Cf * math.atan(vm.Bf * alpha_f))
    Fry = vm.Dr * math.sin(vm.Cr * math.atan(vm.Br * alpha_r))
 
    '''브레이크가 낼 수 있는 최대 감속력 == 타이어와 지면 사이의 마찰력'''
    F_rolling = vm.mu * vm.m * vm.gravity # 마찰계수 x 차의 질량 x 중력가속도
    

    
    #longitudinal force (종방향) #
    ##차속이 음수(또는 정지 상태)이면서 APS/BPS action에 의한 torque 값도 음수가 되면 자동으로 종방향으로 가해지는 힘은 음수가 되도록 해야 함.
    # if vx <= 0 and torque_value <= 0:
    #     Ffx, Frx = 0, 0
    # else:
    #     Ffx = vm.Cb * min(torque_value, 0) / vm.rw - \
    #             vm.rolling_coef_fx * F_rolling
    #     Frx = max(torque_value, 0) / vm.rw + \
    #             (1. - vm.Cb) * np.min(torque_value, 0) / vm.rw - \
    #             vm.rolling_coef_rx * F_rolling
    
    # breakpoint()
    ##-113.4225 값을 설정해 둔 이유는 그 값보다 토크가 작아지면 앞으로 전진을 할 수 없기 때문
    '''
    BPS로 인한 최대 제동력 = 마찰계수 x 차의 질량(Kg) x 중력가속도 -> BPS x MAX_BRAKE_FORCE (Brake_F_fx, Brake_F_rx)
    APS로 인한 driving Force = APS x MAX_tau / Wheel Radians (Drive_F_fx, Drive_F_rx)
    
    '''
    if not allow_neg_torque:
        Ffx = vm.Cb * min(torque_value, 0.) / vm.rw - \
                vm.rolling_coef_fx * F_rolling #0.: tau_min ## longitudinal tire force at front tires
        #APS로 인한 driving force (N) + BPS로 인한 제동력 (N) - 마찰력
        Frx = max(torque_value, 0.) / vm.rw + \
                (1. - vm.Cb) * min(torque_value, 0.) / vm.rw - \
                vm.rolling_coef_rx * F_rolling ## longitudinal tire force at rear tires
    else:
        Ffx = vm.Cb * min(torque_value, -113.4225) / vm.rw - \
                    vm.rolling_coef_fx * F_rolling #-113.4225: tau_min ## longitudinal tire force at front tires
        #APS로 인한 driving force (N) + BPS로 인한 제동력 (N) - 마찰력
        Frx = max(torque_value, -113.4225) / vm.rw + \
                    (1. - vm.Cb) * min(torque_value, -113.4225) / vm.rw - \
                    vm.rolling_coef_rx * F_rolling ## longitudinal tire force at rear tires
     
    
    return Ffy, Fry, Ffx, Frx, alpha_f, alpha_r, sigma_f, sigma_r

def calculate_friction_force_2(vm, #vehicleModel object#
                             x, 
                             u,
                             allow_neg_torque:bool=False
                            ): 
    torque_value, steer_value = u
    vx, vy, omega = x[3:]
    alpha_f = -math.atan2(vm.lf * omega + vy, vx) + steer_value
    alpha_r = math.atan2(vm.lr * omega - vy, vx)
    Ffy = vm.Df * math.sin(vm.Cf * math.atan(vm.Bf * alpha_f))
    Fry = vm.Dr * math.sin(vm.Cr * math.atan(vm.Br * alpha_r))
    F_rolling = vm.mu * vm.m * vm.gravity # 마찰계수 x 차의 질량 x 중력가속도
    if not allow_neg_torque:
        Ffx = vm.Cb * min(torque_value, 0.) / vm.rw - \
                vm.rolling_coef_fx * F_rolling #0.: tau_min ## longitudinal tire force at front tires
        #APS로 인한 driving force (N) + BPS로 인한 제동력 (N) - 마찰력
        Frx = max(torque_value, 0.) / vm.rw + \
                (1. - vm.Cb) * min(torque_value, 0.) / vm.rw - \
                vm.rolling_coef_rx * F_rolling ## longitudinal tire force at rear tires
    else:
        Ffx = vm.Cb * min(torque_value, -113.4225) / vm.rw - \
                    vm.rolling_coef_fx * F_rolling #-113.4225: tau_min ## longitudinal tire force at front tires
        #APS로 인한 driving force (N) + BPS로 인한 제동력 (N) - 마찰력
        Frx = max(torque_value, -113.4225) / vm.rw + \
                    (1. - vm.Cb) * min(torque_value, -113.4225) / vm.rw - \
                    vm.rolling_coef_rx * F_rolling ## longitudinal tire force at rear tires
    return Ffy, Fry, Ffx, Frx

def calculate_curvilinear_derivatives(vm,
                            e_c, e_phi, vx, vy, omega,
                            Ffy, Fry, Ffx, Frx, 
                            steer_value,
                            kappa, 
                        ):
    # vxnew = max(vx, 1)
    vxnew = vx
    
    cos_e_phi = math.cos(e_phi)
    sin_e_phi = math.sin(e_phi)
    
    cos_delta = math.cos(steer_value)
    sin_delta = math.sin(steer_value)
    
    dtheta = (1./(1. - kappa * e_c)) * (vxnew * cos_e_phi - vy * sin_e_phi)
    de_c = vxnew * sin_e_phi + vy * cos_e_phi
    de_phi = omega - (kappa / (1. - kappa * e_c)) * (vxnew * cos_e_phi - vy * sin_e_phi)
    
    dvx = ((Frx + Ffx * cos_delta - Ffy * sin_delta) / vm.m) + vy * omega
    dvy = ((Fry + Ffx * sin_delta + Ffy * cos_delta) / vm.m) - vxnew * omega
    # dvx = ((Frx - Ffy * sin_delta) / vm.m) + vy * omega
    # dvy = ((Fry + Ffy * cos_delta) / vm.m) - vxnew * omega
    
    domega = (Ffy * vm.lf * cos_delta - Fry * vm.lr) / vm.Iz
 
    return np.array(
        [dtheta, de_c, de_phi, dvx, dvy, domega]
    )
    
def calculate_cartesian_derivatives(vx, vy, phi, omega):
    rot_mat = np.array([
        [math.cos(phi), -math.sin(phi)],
        [math.sin(phi), math.cos(phi)]
    ])
    
    # vxnew = max(vx, 1)
    vxnew = vx
    vel_mat = np.array([vxnew, vy])
    # dCar_pos = rot_mat.T@vel_mat #vel_mat@rot_mat.T
    # dCar_x, dCar_y = dCar_pos[0], dCar_pos[1]
    dCar_x = vxnew*math.cos(phi) - vy*math.sin(phi)
    dCar_y = vxnew*math.sin(phi) + vy*math.cos(phi)
    
    dPhi = omega #Omega는 차량 
    
    return np.array(
        [dCar_x, dCar_y, dPhi]
    ) 
    
import os, sys
# sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reward import OffCourse_Checker

def sin_simulation(num_step,
                   freq, 
                   dt, 
                   max_deg,
                   race_car,
                   debug_oc:bool=False):
    t = 0
    A = np.deg2rad(max_deg)
    # A = 3.5
    oc_checker = OffCourse_Checker(car_obj=race_car)
    
    com_off, all_off, instance_off = [], [], []
    Kp = 0.01
    
    for i in range(num_step):
        rad = np.sin(2 * np.pi * freq * t)  ##목표로 하는 조향각을 임의로 지정을 해줌
        # rad = math.radians(deg)
        steering = rad * 2 ## 원래는 범위를 맞춰주려면 steering value를 이렇게 두면 안되는데, sin 곡선으로 이동하는게 동작을 하는지 확인하고 싶었던 것임.
        # print(rad, steering)
        # rad = np.sin(i)
        # x_current = race_car.bicycle_model.car_x - race_car.bicycle_model.init_x
        # y_current = race_car.bicycle_model.car_y - race_car.bicycle_model.init_y
        # x_ref = A * np.sin(freq * y_current)  # 현재 x에 대한 목표 y
        # x_error = x_ref-x_current
        # steering = Kp * x_error  # 단순 비례 제어

        # print(steering, x_error)
        # deg = math.degrees(rad)
        # print(deg, rad)
        # print(rad)
        '''-1 ~ 1 사이의 action 값에 대해서 VALUE 자체를 계산하고자 할 때
        steering action에는 0.4를 곱함
        torque action에는 양수일때 max_tau, 음수일때 min_tau를 곱해줌'''
        # torque_action = np.random.randint(-10, 10) / 100
        torque_action = 0.2 + t * 0.01
        race_car._step([steering, torque_action]) 
        t += dt
        
        com_off.append(oc_checker._off_course_com())
        all_off.append(oc_checker._off_course_all())
        instance_off.append(oc_checker._off_course_instance())
    
    if debug_oc:
        return race_car, com_off, all_off, instance_off
    
    return race_car