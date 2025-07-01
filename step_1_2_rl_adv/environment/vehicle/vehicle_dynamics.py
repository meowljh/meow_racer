from abc import ABC, abstractmethod

import math
import numpy as np
from collections import defaultdict

from scipy.optimize import minimize

from .vehicle_model import vehicleModel
from .vehicle_constraints import PP_MPCC_Params
from .tools import (
    dist_to_center, 
    calculate_error_c,
    calculate_friction_force,
    calculate_friction_force_2,
    calculate_curvilinear_derivatives,
    calculate_cartesian_derivatives
)

'''
angular velocity : omega (1,) - 후륜 구동의 bicycle model이기 때문에 바퀴 1개의 각속도만                 을 고려하면 됨
velocity : (vx vy) (2,)
acceleration : (ax, ay) (2,)
tire slip angles : (alpha_f, alpha_r) (2,)
load on each tire : (Ffy, Fry, Ffx, Frx) (4,)


'''
class RaceCar_Dynamics(ABC):
    def __init__(self, dt):
        super().__init__()
        
        self.dt = dt # 이 값을 알아야 derivative로 역으로 변화 결과를 알 수 있음
        
        
        
    def _reset(self, config_yaml_path, init_x:float, init_y:float, init_phi:float, init_kappa:float,
               init_vx:float):
        # self.new_start = True
        
        self.vehicle_constraints = PP_MPCC_Params() ## vehicle constraint class ##
        self.vehicle_model = vehicleModel(config_yaml_path) ## vehicle parameter configuration class ##
        
        ### CURVILINEAR ###
        self.dTheta = 0 # derivation of theta value (차량이 트랙의 center line을 따라 이동한 거리)
        self.dE_c = 0
        self.dE_phi = 0
        self.dVx = 0 # derivation of vx (종방향 차속의 미분값)
        self.dVy = 0 # derivation of vy (횡방향 차속의 미분값)
        self.dOmega = 0 # derivation of omega (yaw rate)
        
        #### CARTESIAN ###
        self.dCar_x = 0
        self.dCar_y = 0
        self.dCar_phi = 0
        
    
        #### TIRE MODELING ####
        self.alpha_f = 0 # front wheel tire-slip angle
        self.alpha_r = 0 # rear wheel tire-slip angle
        self.sigma_f = 0 # front wheel tire-slip ratio
        self.sigma_r = 0 # rear wheel tire-slip ratio
        
        self.backward_counter = 0
        # self.total_car_moved_curve = 0
        
        self.Ffx, self.Ffy, self.Frx, self.Fry = 0, 0, 0, 0 # Load on each Tire
        
        # self.Theta, self.Vx, self.Vy, self.Omega = 0, 0, 0, 0
        self.Theta, self.Vx, self.Vy, self.Omega = 0, init_vx, 0, 0
        
        self.E_c, self.E_phi = 0, 0
        self.dx, self.dy = 0, 0
        self.kappa = init_kappa
        
        self.init_phi = init_phi
        self.init_x, self.init_y = init_x, init_y
        
        self.car_x, self.car_y = init_x, init_y # 초기 차량의 위치 지정
        self.car_phi = init_phi # 초기 차량의 조향각 지정 (yaw angle)
        
        self.continuous_e_c_arr = [] # 트랙의 center line에 대한 좌/우 이격 정도
        self.continuous_e_phi_arr = [] # 트랙의 접선에 대한 yaw각의 오차 
        
        self.e_c_arr = []
        self.e_phi_arr = []
        
        self.theta_arr = []
        self.omega_arr = []
        
        self.force_dict = defaultdict(list)
        self.alpha_dict = defaultdict(list)
        self.deriv_dict = defaultdict(list)
        
        self.Ec_w_spline_arr = []
        self.Ec_w_spline = 0
        self.Ephi_w_spline_arr = []
        self.Ephi_w_spline = 0
        self.is_right = False
        
        self.current_step_tile_add = 0
        self.prev_car_index = 0
        
        self.tile_step_cnt = 0
        
        self.car_traj_arr = [[self.car_x, self.car_y, self.car_phi, self.Vx, self.Vy]]
        
        self.heading_diff = 0 # 초기에는 트랙의 중앙선과 같은 phi를 차량의 heading으로 둠
        
        # self.ref_arr_dict = defaultdict(lambda : [0])
        self.ref_arr_dict = defaultdict(list)
        
        self.torque_action2value = 0
        self.steer_action2value = 0
        
        self.neg_vel_count_time = 0
        self.min_vel_pen_count = 0
        
    def state_equations(self):
        deriv_arr = np.array([
            self.dTheta, self.dE_c, self.dE_phi, self.dVx, self.dVy, self.dOmega, 
        ])
        
        prev_step_arr = np.array([
            self.Theta, self.E_c, self.E_phi, self.Vx, self.Vy, self.Omega,  
        ])
        
        new_step_arr = (deriv_arr * self.dt) + prev_step_arr
        return new_step_arr
    
    def _calculate_Ec_w_spline(self, theta_center_spline, theta_left_spline, theta_right_spline):
        ref_theta = self.ref_arr_dict['theta'][-1]
        
        ref_x, ref_y = theta_center_spline(ref_theta)
        ref_lx, ref_ly = theta_left_spline(ref_theta)
        ref_rx, ref_ry = theta_right_spline(ref_theta)
        
        left_dist = math.dist([self.car_x, self.car_y], [ref_lx, ref_ly])
        right_dist = math.dist([self.car_x, self.car_y], [ref_rx, ref_ry])
        
        Ec_w_spline = math.dist([self.car_x, self.car_y], [ref_x, ref_y])
        
        if left_dist < right_dist: #중앙선을 기준으로 왼쪽에 차량이 위치해서 왼쪽 벽면과의 거리가 더 가까우면 e_c < 0
            '''ppt에서 보면 왼쪽이랑 가까우면 e_c>0라고 되어 있는데, 사용하는 식 기준으로 보면 위와 같이 나오는게 맞다.'''
            Ec_w_spline *= -1
        
        self.Ec_w_spline = Ec_w_spline
        self.Ec_w_spline_arr.append(self.Ec_w_spline)
        
        is_right = left_dist > right_dist
        self.is_right = is_right
            
    def _calculate_Ephi_w_spline(self, phi_spline):
        ref_theta = self.ref_arr_dict['theta'][-1]
        
        ref_phi = phi_spline(ref_theta) #% (np.pi)
        ref_phi %= (2 * np.pi)
        phi_diff = abs(ref_phi - self.car_phi)
        
        if self.is_right:
            phi_diff *= -1
        
        self.Ephi_w_spline = phi_diff
        self.Ephi_w_spline_arr.append(self.Ephi_w_spline)

    
    def tire_modeling(self, torque_action2value, steer_action2value, 
                      zero_force_neg_vel:bool=False,
                      allow_neg_torque:bool=False):
         
        
        Ffy, Fry, Ffx, Frx, alpha_f, alpha_r, sigma_f, sigma_r = calculate_friction_force(
            vm=self.vehicle_model,
            # u=np.array([torque_action2value, steer_action2value]),
            # x=np.array([self.car_x, self.car_y, self.car_phi, self.Vx, self.Vy, self.Omega]),
            torque_action=torque_action2value,
            steer_action=steer_action2value,
            omega=self.Omega,
            vx=self.Vx,
            vy=self.Vy,
            allow_neg_torque=allow_neg_torque
        )
        
        if zero_force_neg_vel:
            #음의 종방향 속도를 가질 때 종방향 힘은 무조겅 0 이상이 되어야 함.
            if self.Vx <= 0:
                Ffx = max(Ffx, 0)
                Frx = max(Frx, 0)
                
                
        self.force_dict['Ffy'].append(Ffy);self.force_dict['Fry'].append(Fry);self.force_dict['Ffx'].append(Ffx);self.force_dict['Frx'].append(Frx)
        # self.alpha_dict['f'].append(alpha_f);self.alpha_dict['r'].append(alpha_r)
         
        self.alpha_f = alpha_f # 전륜 타이어 슬립 각
        self.alpha_r = alpha_r # 후륜 타이어 슬립 각
        self.sigma_f = sigma_f # 전륜 타이어 슬립율
        self.sigma_r = sigma_r # 후륜 타이어 슬립율
        
        self.Ffy, self.Fry, self.Ffx, self.Frx = Ffy, Fry, Ffx, Frx # 전륜 횡방향 / 후륜 횡방향 / 전륜 종방향 / 후륜 종방향
        
        return Ffy, Fry, Ffx, Frx
        
    def _calculate_discrete_error(self):
        close_ref_x, close_ref_y = self.ref_arr_dict['center'][-1]
        close_ref_phi = self.ref_arr_dict['phi'][-1]
        dX, dY = self.car_x - close_ref_x, self.car_y - close_ref_y
        discrete_E_c = math.sin(close_ref_phi) * dX - math.cos(close_ref_phi) * dY
        
        self.E_c = discrete_E_c
        self.e_c_arr.append(discrete_E_c)
        
        # close_ref_phi %= math.pi # (math.pi / 2)
        # discrete_E_phi = self.car_phi - (close_ref_phi % math.pi) #phi - phi_ref
        '''car_phi는 어떻게 해도 2pi를 넘을 수는 없을 것. (정상적으로 동작한다면)'''
        discrete_E_phi = self.car_phi - (close_ref_phi % (2*math.pi))
        self.E_phi = discrete_E_phi #후진을 해서 트랙의 끝쪽에 간다면 
        self.e_phi_arr.append(discrete_E_phi)
         
        
    def curvilinear_dynamics(self, 
                             torque_action2value, steer_action2value,
                             zero_force_neg_vel:bool=False,
                             always_pos_vel:float=-1.,
                             allow_neg_torque:bool=False):
        self.torque_action2value = torque_action2value
        self.steer_action2value = steer_action2value
        '''자동차 기준의 좌표계에서 계산된 값.
        - 운전하는 것은 결국에는 차량 자체이기 때문에, 자신의 position 및 angle등을 기준으로 값들이 계산되어야 한다.
        - 차량이 이동하는 방향의 속도가 Vx, Vy로 계산됨. 즉, 전방의 이동 방향으로의 속도가 Vx가 되는 것이고 그에 수직인 좌표계의 횡방향 속도가 Vy가 되는 것이다.'''
        Ffy, Fry, Ffx, Frx = self.tire_modeling(torque_action2value, steer_action2value,
                                                zero_force_neg_vel=zero_force_neg_vel,
                                                allow_neg_torque=allow_neg_torque) 
        
        # vel = math.sqrt(self.Vx**2 + self.Vy**2)
        # kappa = math.atan2(abs(self.Omega), vel)
        # self.kappa = kappa
        
        self.dTheta, self.dE_c, self.dE_phi, self.dVx, self.dVy, self.dOmega = calculate_curvilinear_derivatives(
            vm=self.vehicle_model,
            e_c=self.E_c, #그냥, 어떻게 보면 매 step마다 별도로 처리하는 느낌
            e_phi=self.E_phi,
            vx=self.Vx,
            vy=self.Vy,
            omega=self.Omega,
            Ffy=Ffy, Fry=Fry, Ffx=Ffx, Frx=Frx,
            steer_value=steer_action2value,
            kappa=self.kappa
            # kappa=kappa
        )
        new_step_arr = self.state_equations()
        
        self.Theta, E_c, E_phi, self.Vx, self.Vy, self.Omega = new_step_arr
        if always_pos_vel >= 0:
            self.Vx = max(self.Vx, always_pos_vel)
            
        
        '''continuously calculated error values with state equation (seems to have a serious bug)
        but, if the previous value of self.E_c, self.E_phi, self.Omega etc are used from the 
        discretely calculated version, the state equation seems to work, which means the other values has no certain bugs.
        NOW THE ONLY ISSUE IS THE REVERSE MOVEMENT OF THE VEHICLE'''
        self.continuous_e_c_arr.append(E_c)
        self.continuous_e_phi_arr.append(E_phi)
        
        self.theta_arr.append(self.Theta)
        self.omega_arr.append(self.Omega)

        self.deriv_dict['dTheta'].append(self.dTheta);self.deriv_dict['dE_c'].append(self.dE_c);self.deriv_dict['dE_phi'].append(self.dE_phi)
        self.deriv_dict['dVx'].append(self.dVx);self.deriv_dict['dVy'].append(self.dVy);self.deriv_dict['dOmega'].append(self.dOmega)
        
    def cartesian_dynamics(self):
        '''global 좌표계에서의 값들이기 때문에,
        만약에 일반적인 X, Y 좌표계에서 트랙과 함께 차량을 시각화 하고 싶다면 이 역학식을 사용해야 함.
        '''
        self.dCar_x, self.dCar_y, self.dCar_phi = calculate_cartesian_derivatives(
            vx=self.Vx, vy=self.Vy, phi=self.car_phi, omega=self.Omega
        )
        
        self.car_x += self.dCar_x * self.dt 
        self.car_y += self.dCar_y * self.dt
        self.car_phi += self.dCar_phi * self.dt
        
        self.car_traj_arr.append([self.car_x, self.car_y, self.car_phi, self.Vx, self.Vy])
        
        self.deriv_dict['dCar_x'].append(self.dCar_x);self.deriv_dict['dCar_y'].append(self.dCar_y);self.deriv_dict['dCar_phi'].append(self.dCar_phi)
    
    def tire_modeling_2(self, x, u, allow_neg_torque:bool=False):
        return calculate_friction_force_2(vm=self.vehicle_model, x=x, u=u, allow_neg_torque=allow_neg_torque)
        
    def cartesian_dynamics_2(self, x, u, zero_force_neg_vel:bool=True, allow_neg_torque:bool=False, 
                             always_pos_vel:float=-1):
        '''
        x = [X, Y, phi, Vx, Vy, omega]
        ''' 
        vxnew = x[3] 
        dCar_x = vxnew*math.cos(x[2]) - x[4]*math.sin(x[2])
        dCar_y = vxnew*math.sin(x[2]) + x[4]*math.cos(x[2])
        
        dPhi = x[5] 
        
        self.torque_action2value = u[0]
        self.steer_action2value = u[1]
        
        Ffy, Fry, Ffx, Frx = self.tire_modeling_2(
                                                  x=x,
                                                  u=u,
                                                  allow_neg_torque=allow_neg_torque
                                                  )
        self.force_dict['Ffy'].append(Ffy);self.force_dict['Fry'].append(Fry)
        self.force_dict['Ffx'].append(Ffx);self.force_dict['Frx'].append(Frx)
        
        cos_delta = math.cos(self.steer_action2value)
        sin_delta = math.sin(self.steer_action2value)

        self.dVx = ((Frx + Ffx * cos_delta - Ffy * sin_delta) / self.vehicle_model.m) + x[4] * x[5]
        self.dVy = ((Fry + Ffx * sin_delta + Ffy * cos_delta) / self.vehicle_model.m) - vxnew * x[5]

        self.dOmega = (Ffy * self.vehicle_model.lf * cos_delta - Fry * self.vehicle_model.lr) / self.vehicle_model.Iz
 
        
        new_x = x + self.dt * np.array([dCar_x, dCar_y, dPhi, self.dVx, self.dVy, self.dOmega])
        
        # self.Theta += self.dt * self.dTheta
        
        self.car_x = new_x[0]
        self.car_y = new_x[1]
        self.car_phi = new_x[2]
        
        if self.Vx <= 0:
            self.neg_vel_count_time += self.dt
        else:
            self.neg_vel_count_time = 0
            
        if always_pos_vel > -1:
            self.Vx = max(new_x[3], always_pos_vel) #  new_x[3]
        else:
            self.Vx = new_x[3]
            
        self.Vy = new_x[4]
        self.Omega = new_x[5]
        
        self.car_traj_arr.append([self.car_x, self.car_y, self.car_phi, self.Vx, self.Vy])
        
      
        return new_x
        
        
    def update_track_tile_stats(self, track_dict):
        cnt = 0
        self.tile_step_cnt = 0
        
        car_index = self.prev_car_index
        updated = False
        
        for i, poly in enumerate(track_dict['vertices']):
            X = [p[0] for p in poly];Y = [p[1] for p in poly]
            mx, Mx, my, My = min(X), max(X), min(Y), max(Y)
            if mx <= self.car_x <= Mx and my <= self.car_y <= My:
                if not updated:
                    car_index = i #현재 차량의 track상에서의 위치
                updated = True
                
                if track_dict['passed'][i] == False:
                    self.tile_step_cnt += 1
                    cnt += 1
                track_dict['passed'][i] = True
        
        # if self.new_start:
        #     car_index = 0
            
        index_diff = car_index - self.prev_car_index ##이렇게 하면 방향이 다르거나 해서 후진하는 경우에 대해서는 negative reward를 받게 될 것임.
        # if index_diff < 0:
        #     breakpoint()
        self.prev_car_index = car_index
        # self.current_step_tile_add = cnt 
        self.current_step_tile_add = index_diff
        
        # self.new_start = False
              
        return track_dict
        
    def track_reference_error(self, 
                              track_dict,
                              theta_center_spline, 
                              theta_left_spline, theta_right_spline,
                              phi_spline, kappa_spline):
        cx, cy = self.car_x, self.car_y

        '''워낙에 track center line이 sparse한 분포를 갖고 좌표가 구성되어 있다보니,
        interpolate를 해서 e_c, e_phi, e_theta를 구해야 더 정확하게 찾을 수 있다.
        '''
        initial_guess = track_dict['theta'][np.argmin((track_dict['x']- cx)**2 + (track_dict['y'] - cy)**2)]
        if isinstance(initial_guess, int):
            initial_guess = track_dict[initial_guess]
        
        close_theta = minimize(lambda x:
                dist_to_center(theta_center_spline, x[0], cx, cy), [initial_guess]
            ).x[0] # 첫 theta값으로 minimize optimization을 초기화
        self.Theta = close_theta
        close_phi = phi_spline(close_theta) # 제일 차량의 위치와 가까운 중앙의 theta를 기반으로 트랙의 phi값 계산
        close_refX, close_refY = theta_center_spline(close_theta) # 제일 차량의 위치와 가까운 중앙의 theta를 기반으로 중앙선에서의 좌표 계산
        self.dx = close_refX - self.car_x
        self.dy = close_refY - self.car_y
        close_ref_kappa = kappa_spline(close_theta)
        close_lX, close_lY = theta_left_spline(close_theta)
        close_rX, close_rY = theta_right_spline(close_theta)
        
        # self.close_ref_kappa = close_ref_kappa
        self.kappa = close_ref_kappa
        '''차량의 heading angle과 가까운 트랙의 접선 방향의 phi값의 차이인 heading_diff 계산'''
    
        # self.heading_diff = self.car_phi - close_phi ##discrete E_phi와는 달리, 트랙의 진행 방향과 반대와 같은지 아닌지 판단할 수 있게 계산됨
        self.heading_diff = self.car_phi - (close_phi % (2*math.pi))
        # self.heading_diff = self.E_phi
        
        self.ref_arr_dict['initial_guess'].append(initial_guess)
        self.ref_arr_dict['phi'].append(close_phi)
        self.ref_arr_dict['theta'].append(close_theta)
        self.ref_arr_dict['kappa'].append(close_ref_kappa)
        self.ref_arr_dict['center'].append([close_refX, close_refY])
        self.ref_arr_dict['left'].append([close_lX, close_lY])
        self.ref_arr_dict['right'].append([close_rX, close_rY])
         
        # self.total_car_moved_curve += self.dTheta
        # if self.Vx < 0:
        if self.dTheta < 0:
            self.backward_counter += 1
             
    @property
    def car_speed_ms(self):
        '''speed는 velocity와 달리 방향 개념이 없기 때문에 그냥 값 자체만 계산하면 됨'''
        return math.sqrt(self.Vx ** 2 + self.Vy ** 2)
    
    @property
    def car_speed_kph(self):
        speed_ms = self.car_speed_ms()
        speed_kph = speed_ms * 3.6
        return speed_kph
        
    @property
    def acc_x(self):
        if len(self.car_traj_arr) <= 1:
            return 0
        return (self.car_traj_arr[-1][-2] - self.car_traj_arr[-2][-2]) / self.dt

    @property
    def acc_y(self):
        if len(self.car_traj_arr) <= 1:
            return 0
        return (self.car_traj_arr[-1][-1] - self.car_traj_arr[-2][-1]) / self.dt
    
    
    @property
    def acc_x_arr(self):
        acc_x_arr = [0]
        assert len(self.car_traj_arr) >= 2
        v_x_arr = np.array(self.car_traj_arr).T[-2]
        diff_v_x_arr = np.diff(v_x_arr)
        acc_x_arr.extend(list(diff_v_x_arr / self.dt))
        
        return acc_x_arr
    
    @property
    def acc_y_arr(self):
        acc_y_arr = [0]
        assert len(self.car_traj_arr) >= 2
        v_y_arr = np.array(self.car_traj_arr).T[-1]
        diff_v_y_arr = np.diff(v_y_arr)
        acc_y_arr.extend(list(diff_v_y_arr / self.dt))
        
        return acc_y_arr