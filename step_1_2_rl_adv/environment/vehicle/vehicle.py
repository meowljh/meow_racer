from abc import ABC
import yaml
import numpy as np
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from collections import defaultdict
import math
from scipy.interpolate import CubicSpline, interp1d

from .vehicle_dynamics import RaceCar_Dynamics
from .vehicle_constraints import PP_MPCC_Params 
from .vehicle_aps_bps_mapping import APS_Mapping, BPS_Mapping
from reward.vehicle_status_check import OffCourse_Checker
class RaceCar(ABC):
    def __init__(self,
                 action_dim: int,
                 dt: float,
                 world_dt: float,
                 aps_bps_weight: float,
                 allow_both_feet:bool,
                 brake_on_pos_vel:bool,
                 normalize_aps_bps:bool,
                 schedule_brake_ratio:bool,
                 schedule_brake_ratio_scale:float,
                 schedule_brake_episode:float,
                 ################################################
                 zero_force_neg_vel:bool,
                 always_pos_vel:bool,
                 allow_neg_torque:bool,
                 use_continuous_bps:bool,
                 cfg_file_path:str,
                 initial_vx:float,
                 use_aps_bps_diff:bool,
                 ):
        super().__init__()
        
        self.cfg_file_path = cfg_file_path
        self.dt = dt
        self.world_dt = world_dt
        self.aps_bps_weight = aps_bps_weight
        self.action_dim = action_dim
        ############### for torque calculation ####################
        self.allow_both_feet = allow_both_feet
        self.brake_on_pos_vel = brake_on_pos_vel
        self.normalize_aps_bps = normalize_aps_bps
        self.schedule_brake_ratio = schedule_brake_ratio
        self.schedule_brake_episode = schedule_brake_episode
        self.schedule_brake_ratio_scale = schedule_brake_ratio_scale
        self.use_aps_bps_diff = use_aps_bps_diff
        ############### for preventing backward movement ################
        self.zero_force_neg_vel = zero_force_neg_vel
        self.always_pos_vel = always_pos_vel
        self.allow_neg_torque = allow_neg_torque
          
        self.bicycle_model = RaceCar_Dynamics(dt=self.dt)
        self.vehicle_constraints = PP_MPCC_Params() 
        
        self.use_continuous_bps = use_continuous_bps
        
        ################ for calculating the torque values based on the APS / BPS actions relative to the longitude vehicle velocity ################
        self.aps_mapper = APS_Mapping()
        self.bps_mapper = BPS_Mapping()
        
        self.episode_cnt = 0
        
        self.initial_vx = initial_vx

    def _calculate_brake_ratio(self):
        if self.schedule_brake_ratio:
            self.aps_bps_weight = min(1., self.schedule_brake_ratio_scale * (self.episode_cnt / self.schedule_brake_episode))
    
    def _reset(self, track_dict):
        self.episode_cnt += 1
        self.track_dict = track_dict
        
        self.bicycle_model._reset(
            config_yaml_path=self.cfg_file_path,
            
            init_x=track_dict['x'][0],
            init_y=track_dict['y'][0],
            init_phi=track_dict['phi'][0],
            init_kappa=track_dict['kappa'][0],
            
            init_vx = self.initial_vx
        )

        track_centerX, track_centerY = np.array(track_dict['x']), np.array(track_dict['y']) #남양 트랙만 center point는 이미 처음과 끝이 매칭 되어 있음
        track_centerX[-1] = track_centerX[0];track_centerY[-1] = track_centerY[0]
        
        track_leftX, track_leftY = np.array(track_dict['left']).T
        track_leftX[-1] = track_leftX[0];track_leftY[-1] = track_leftY[0]
        
        track_rightX, track_rightY = np.array(track_dict['right']).T
        track_rightX[-1] = track_rightX[0];track_rightY[-1] = track_rightY[0]
        
        track_phi = np.array(track_dict['phi'])
        track_theta = np.array(track_dict['theta'])
        track_kappa = np.array(track_dict['kappa'])
        
        #because of the periodic Cubic Spline interpolation function#
        # track_phi[-1] = track_phi[0]
        # track_kappa[-1] = track_kappa[0]
        
        self.theta_center_spline = CubicSpline(track_theta, np.vstack([track_centerX, track_centerY]).T, bc_type='periodic')
        self.theta_left_spline = CubicSpline(track_theta, np.vstack([track_leftX, track_leftY]).T, bc_type='periodic')
        self.theta_right_spline = CubicSpline(track_theta, np.vstack([track_rightX, track_rightY]).T, bc_type='periodic')
        
        # self.phi_spline = CubicSpline(track_theta, track_phi, bc_type="periodic")
        # self.kappa_spline = CubicSpline(track_theta, track_kappa, bc_type="periodic")
        '''[0421] error fix'''
        self.phi_spline = interp1d(track_theta, track_phi, kind='linear', fill_value='extrapolate')
        self.kappa_spline = interp1d(track_theta, track_kappa, kind='linear', fill_value='extrapolate')
        
        self.actions = defaultdict(list)
        self.action2control = defaultdict(list)

        self.vehicle_status_checker = OffCourse_Checker(car_obj=self)
        
        self.terminate_neg_vel_bps  = False
        self.reward_neg_vel_aps =  False
         
    def _postprocess_action(self, action,
                            continuous_bps:bool=False):
        action = np.array(action) 
        assert action.shape[-1] == self.action_dim

        #(1) obtain steer_action / throttle_action / brake_action from the raw actor output
        if self.action_dim == 2: #steering, torque
            steer_action = action[0]
        
            torque_action = action[1]
            brake_action = abs(torque_action) if torque_action <= 0 else 0
            throttle_action = torque_action if torque_action > 0 else 0
        elif self.action_dim == 3: #steering, throttle, brake
            steer_action = action[0]
            throttle_action = action[1]
            brake_action = action[2]
            #scale throttle, brake actions to APS, BPS ratio (-1~1)->(0,1)
            throttle_action = (throttle_action+1)/2
            brake_action = (brake_action+1)/2
        #(2) calculate the Torque values from the APS / BPS actions, regarding the longitudinal velocity and mapping values
 
                
        if not self.allow_both_feet:
            if brake_action > throttle_action: #apply BPS
                throttle_action_applied = 0
                if self.use_aps_bps_diff:
                    brake_action_applied = brake_action - throttle_action
                else:
                    brake_action_applied = brake_action
            else: #apply APS
                brake_action_applied = 0
                if self.use_aps_bps_diff:
                    throttle_action_applied = throttle_action - brake_action
                else:
                    throttle_action_applied = throttle_action
        else:
            throttle_action_applied = throttle_action
            brake_action_applied = brake_action
        
        if self.brake_on_pos_vel:
            if self.bicycle_model.Vx <= 0:
                brake_action_applied = 0
            
            
            
        self.actions['steer'].append(steer_action)
        self.actions['brake'].append(brake_action)
        self.actions['throttle'].append(throttle_action)
        self.actions['brake_applied'].append(brake_action_applied)
        self.actions['throttle_applied'].append(throttle_action_applied)
        
        aps_torque = self.aps_mapper._get_APS_Torque(
            # throttle_action=throttle_action,
            throttle_action=throttle_action_applied,
            max_tau=self.vehicle_constraints.tau_max,
            car_vx=self.bicycle_model.Vx
        ) ##engine torque
        if continuous_bps:
            bps_torque = self.bps_mapper._get_BPS_Torque_Continuous(
                # brake_action=brake_action,
                brake_action=brake_action_applied,
                min_tau=self.vehicle_constraints.tau_min,
                car_vx=self.bicycle_model.Vx
            ) ##brake torque
        else:
            bps_torque = self.bps_mapper._get_BPS_Torque(
                # brake_action=brake_action,
                brake_action=brake_action_applied,
                min_tau=self.vehicle_constraints.tau_min,
                car_vx=self.bicycle_model.Vx
            )
        # assert bps_torque >= 0
        #(3) calculate the Joint Torque values 
        # if self.bicycle_model.Vx <= 0 and abs(bps_torque) > 0:
        if self.bicycle_model.Vx <= 0 and brake_action_applied > 0:
            self.terminate_neg_vel_bps=True
        else:
            self.terminate_neg_vel_bps = False
        
        if self.bicycle_model.Vx <= 0 and throttle_action_applied > 0:
            self.reward_neg_vel_aps = True
        else:
            self.reward_neg_vel_aps = False
            
        # if self.brake_on_pos_vel:
        #     if self.bicycle_model.Vx <= 0:
        #         bps_torque = 0
        # if self.allow_both_feet:
        #     # torque_action2value = aps_torque + bps_torque
        #     torque_action2value = aps_torque - bps_torque
        # else:
        #     if throttle_action > brake_action:
        #         torque_action2value = aps_torque
        #     else:
        #         # torque_action2value = bps_torque
        #         torque_action2value = -bps_torque
        torque_action2value = aps_torque - bps_torque
        #(4) calculate the Steering Radius from the steering action
        steer_action2value = steer_action * self.vehicle_constraints.delta_max if steer_action > 0 \
                                                else abs(steer_action) * self.vehicle_constraints.delta_min
        self.action2control['steer'].append(steer_action2value)
        self.action2control['torque'].append(torque_action2value)
  
        return steer_action2value, torque_action2value
        
        
        
    # def _postprocess_action(self, action):
    #     action = np.array(action)
    #     self._calculate_brake_ratio() ##여기서 scheduling된 aps_bps_weight를 계산함.
        
    #     assert action.shape[-1] == self.action_dim
    #     if self.action_dim == 2:
    #         steer_action = action[0]
    #         torque_action = action[1]
    #         self.actions['steer'].append(steer_action)
            
    #         brake_action = abs(torque_action) if torque_action <= 0 else 0
    #         throttle_action = torque_action if torque_action > 0 else 0
            
    #         self.actions['brake'].append(brake_action)
    #         self.actions['throttle'].append(throttle_action)
            
    #         if self.brake_on_pos_vel:
    #             vel = self.bicycle_model.Vx
    #             if vel <= 0:
    #                 brake_action = 0
    #             #     torque_action2value = throttle_action * self.vehicle_constraints.tau_max
    #             # else:
    #             #     torque_action2value = brake_action * self.vehicle_constraints.tau_min * self.aps_bps_weight + throttle_action * self.vehicle_constraints.tau_max
    #         # else:
    #         #     torque_action2value = throttle_action * self.vehicle_constraints.tau_max + brake_action * self.vehicle_constraints.tau_min * self.aps_bps_weight
                            
                            
    #         torque_action2value = throttle_action * self.vehicle_constraints.tau_max + brake_action * self.vehicle_constraints.tau_min * self.aps_bps_weight
            
    #     else:
    #         steer_action = action[0]
    #         ##agent로 하여금 양발 운전이 허용 가능한 상황이라고 볼 수 있음##
    #         throttle_action = action[1]
    #         brake_action = action[2]
    #         ##(-1~1) 사이의 값으로 tanh function에 의해서 squashing되는데, 원하는 action은 0~1사이의 값이기 때문에 직접 mapping을 해주면 됨
    #         throttle_action = (throttle_action + 1) / 2
    #         brake_action = (brake_action + 1) / 2
            
    #         self.actions['steer'].append(steer_action)
    #         self.actions['brake'].append(brake_action)
    #         self.actions['throttle'].append(throttle_action)

    #         if self.normalize_aps_bps:
    #             sum_action = brake_action + throttle_action
    #             brake_action /= sum_action
    #             throttle_action /= sum_action
                            
    #         if self.brake_on_pos_vel:
    #             vel = self.bicycle_model.Vx
    #             if vel <= 0:
    #                 brake_action = 0
    #             # torque_action2value = throttle_action * self.vehicle_constraints.tau_max
    #             # if vel > 0:
    #                 # if allow_both_feet:
    #                     # torque_action2value += brake_action * self.vehicle_constraints.tau_min * self.aps_bps_weight
            
    #         if self.allow_both_feet:
    #             torque_action2value = throttle_action * self.vehicle_constraints.tau_max + \
    #                                     brake_action * self.vehicle_constraints.tau_min * self.aps_bps_weight
    #         else:
    #             if throttle_action > brake_action:
    #                 torque_action2value = throttle_action * self.vehicle_constraints.tau_max
    #             else:
    #                 torque_action2value = brake_action * self.vehicle_constraints.tau_min * self.aps_bps_weight 
    #     '''apply constraint on the torque value when the longitudinal velocity is below zero.
    #     Same as manually setting the gear mode''' 
            
    #     steer_action2value = steer_action * self.vehicle_constraints.delta_max if steer_action > 0 else abs(steer_action) * self.vehicle_constraints.delta_min
        
    #     self.action2control['steer'].append(steer_action2value)
    #     self.action2control['torque'].append(torque_action2value)
        
    
        
        
    #     return steer_action2value, torque_action2value
    
    def _step(self, action):
        steer_action2value, torque_action2value = self._postprocess_action(action=action,
                                                                           continuous_bps=self.use_continuous_bps)
        assert int(self.world_dt % self.dt) == 0
        num_dynamics_iter = int(self.world_dt / self.dt)
 
        # ## step1: calculate the longitude/lateral force and derivatives ##
        # self.bicycle_model.curvilinear_dynamics(torque_action2value, 
        #                                         steer_action2value, 
        #                                         zero_force_neg_vel=self.zero_force_neg_vel, 
        #                                         always_pos_vel=self.always_pos_vel,
        #                                         allow_neg_torque=self.allow_neg_torque)
        # ## step2: calculate the vehicle location and yaw angle from the cartesian coordinates ##
        # self.bicycle_model.cartesian_dynamics()
        for _ in range(num_dynamics_iter):
            x = np.array([self.bicycle_model.car_x, self.bicycle_model.car_y, self.bicycle_model.car_phi,
                          self.bicycle_model.Vx, self.bicycle_model.Vy, self.bicycle_model.Omega])
            u = np.array([torque_action2value, steer_action2value])

            self.bicycle_model.cartesian_dynamics_2(x=x,
                                                    u=u,
                                                    zero_force_neg_vel=self.zero_force_neg_vel,
                                                    allow_neg_torque=self.allow_neg_torque,
                                                    always_pos_vel=self.always_pos_vel)
            ## step3: calculate the reference point on the track and the reference phi, theta, kappa ##
            self.bicycle_model.track_reference_error(self.track_dict,
                                                     theta_center_spline=self.theta_center_spline, 
                                                     theta_left_spline=self.theta_left_spline,
                                                     theta_right_spline=self.theta_right_spline,
                                                     phi_spline=self.phi_spline, 
                                                     kappa_spline=self.kappa_spline,
                                                     )
            ## step4: update track tile status ##
            self.track_dict = self.bicycle_model.update_track_tile_stats(self.track_dict)
            ## step5: calculate E_c with spline curve manually ##
            self.bicycle_model._calculate_Ec_w_spline(
                theta_center_spline=self.theta_center_spline,
                theta_left_spline=self.theta_left_spline,
                theta_right_spline=self.theta_right_spline
            )
            ## step 6: calculate E_phi with spline curve manually ##
            self.bicycle_model._calculate_Ephi_w_spline(
                phi_spline=self.phi_spline
            )
            ## step 7: calculate the discrete error values ##
            self.bicycle_model._calculate_discrete_error()

        return self.track_dict
        
        
    
    
        