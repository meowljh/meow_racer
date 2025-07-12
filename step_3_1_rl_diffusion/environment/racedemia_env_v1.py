'''racedemia_env_v1.py
- first version of the Racedemia (Race driving + Academia) Reinforcement Learning Environment
'''
######## library imports ########
import numpy as np
from typing import Optional
import math
import pickle
from PIL import Image
import itertools
from collections import deque, OrderedDict, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import random
import os, sys
import gc
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = ROOT.replace('\\', '/')
BASE = os.path.dirname(ROOT)
sys.path.append(ROOT);sys.path.append(BASE) ##어차피 여기서 경로를 추가해 주기 때문에 .track / .vehicle등을 사용해서 처리하는게 더 나을 것임.
from simulator.main.pygame_sim import _render_all_pygame

######## gym imports ########
import gymnasium as gym
# from gym import Env
# from gym.spaces import Box

######## local imports ########
from .track import (
        Random_TrackGenerator, \
        Nam_TrackGenerator, \
        Bezier_TrackGenerator
    )
from .vehicle import RaceCar
from .observation import (
        Observation_Lidar_State, \
        Observation_ForwardVector_State,\
        Observation_Lookahead_State
    )
from .reward import OffCourse_Checker
from .reward import *
from .reward import reward_shaping
from .reward import penalty_shaping
from .style_cond_reward import styleReward_Obj
class Racedemia_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,
                 plotter, ##plotter이 Summary Writer을 사용해서 Tensorboard에 로깅을 하게 되는데, 전부 log 경로가 다르기 때문에 다른 곳에 저장이 될 것임.
                 environment_config:dict,
                 agent_config:dict,
                 penalty_config:dict,
                 reward_config:dict,
                 render_config:dict,
                 is_nam:bool,
                 mode:str,
                 terminated_fig_path:str=None,
                 style_config:dict=None,
                 exp_name:str=None,
                 num_laps_for_truncate:int=1,
                 ):
        super().__init__()
        self.is_nam = is_nam
        
        self.mode = mode
        self.exp_name=exp_name
        self.plotter = plotter
        
        self.environment_config = environment_config
        self.penalty_config = penalty_config
        self.agent_config = agent_config
        self.reward_config = reward_config 
        self.render_config = render_config
        self.style_config = style_config
        
        self.initial_start = True
        
        self.terminated_fig_path = terminated_fig_path
        
        """for driving level (style) conditioning"""
        if self.style_config is not None and self.style_config['style_mode']['type'] in ['medium', 'aggressive', 'defensive']:
            self.use_style = self.agent_config['style']['usage']
            self.style_size = self.agent_config['style']['size']
            self.num_style_nets = self.agent_config['style']['num_nets']
        else:
            self.use_style = False
            
        if self.use_style:
            if self.num_style_nets == 2:
                self.style_setting_candidates = [0, 0.5, 1]
            elif self.num_style_nets == 3:
                self.style_setting_candidates = [0, 1, 2]
            else:
                self.style_setting_candidates = None
        
        self.num_laps_for_truncate = num_laps_for_truncate
        
        self.reset()
    
    def _get_rotated_car_patch(self, ax):
        car_x, car_y, car_phi = self.car.bicycle_model.car_x, self.car.bicycle_model.car_y, self.car.bicycle_model.car_phi
        car_vis_height = self.car.bicycle_model.vehicle_model.body_width
        car_vis_width = self.car.bicycle_model.vehicle_model.body_height
        rect = patches.Rectangle((car_x-car_vis_width/2, car_y-car_vis_height/2), car_vis_width, car_vis_height, linewidth=2, facecolor='red', edgecolor='red')
        rotate = transforms.Affine2D().rotate_deg_around(car_x, car_y, math.degrees(car_phi))
        rect.set_transform(rotate + ax.transData)
        
        return rect
    
    def _local_car_on_track(self, return_image:bool=False):
        def _get_figsize(dx, dy):
            return (8*2, 8 * (dy / dx))

        car_ref_theta = self.car.bicycle_model.ref_arr_dict['theta'][-1]
        car_x, car_y = self.car.bicycle_model.car_x, self.car.bicycle_model.car_y
        
        mx, Mx, my, My = math.inf, -math.inf, math.inf, -math.inf
        mi, Mi = -1, -1
        for i, theta in enumerate(self.track_dict['theta']):
            if car_ref_theta - 20 <= theta <= car_ref_theta + 100:
                if mi == -1:
                    mi = i
                Mi = i

                mx = min(mx, self.track_dict['x'][i]);Mx = max(Mx, self.track_dict['x'][i])
                my = min(my, self.track_dict['y'][i]);My = max(My, self.track_dict['y'][i])
        
        dx, dy = Mx-mx, My-my
        figsize = _get_figsize(dx, dy)
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        
        
        for j in range(2):
            for i in range(mi, Mi):
                ax[j].scatter(self.track_dict['left'][i][0], self.track_dict['left'][i][1], s=1, alpha=0.5, c='k')
                ax[j].scatter(self.track_dict['right'][i][0], self.track_dict['right'][i][1], s=1, alpha=0.5, c='k')
                ax[j].scatter(self.track_dict['x'][i], self.track_dict['y'][i], s=1, alpha=0.1, c='k')
        
        ## (1) render the forward vector ##
        forward_vector_dict = self.forward_vector_obj.vector_dict
        forward_vector = forward_vector_dict['inertia']
        for i, (vx, vy) in enumerate(forward_vector):
            nc_x, nc_y = vx + car_x, vy + car_y
            ax[0].plot([car_x, nc_x], [car_y, nc_y], c='g', linewidth=1)
        ax[0].set_title("Forward Vector")
        
        ## (2) render the lidar sensor ##
        lidar_sensor_arr = self.lidar_sensor_obj.lidar_results
        for i, result_dict in enumerate(lidar_sensor_arr):
            px, py = result_dict['point']
            dist = result_dict['distance']
            if dist == -1:continue
            ax[1].plot([car_x, px], [car_y, py], c='g', linewidth=1)
        ax[1].set_title("Lidar Sensor")
        
        rect = self._get_rotated_car_patch(ax=ax[0])
        ax[0].add_patch(rect);ax[0].set_aspect(True)
        
        rect = self._get_rotated_car_patch(ax=ax[1])
        ax[1].add_patch(rect);ax[1].set_aspect(True)
        
        fig.tight_layout()
    
        
        canvas = fig.canvas
        canvas.draw()
        
        width, height = canvas.get_width_height()
        image_array = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
        image_array = image_array.reshape(height, width, 4)
        if return_image:
            if self.plotter is not None:
                self.plotter._log_simulation(simulate_arr = image_array[:, :, 1:], tag_name = 'local_view')
                plt.cla();plt.clf();plt.close();gc.collect()
                return image_array[:, :, 1:]
        ################### Log Simulation Screen to TensorBoard ###################
        if self.plotter is not None:
            self.plotter._log_simulation(simulate_arr = image_array[:, :, 1:], tag_name = 'local_view')
            plt.cla();plt.clf();plt.close();gc.collect()
        ############################################################################
        
    def _global_car_on_track(self, return_image:bool=False):
        def _get_figsize(dx, dy):
            return (8, 8 * (dy / dx))

        mx, Mx = self.track_dict['x'].min(), self.track_dict['x'].max();dx = Mx - mx
        my, My = self.track_dict['y'].min(), self.track_dict['y'].max();dy = My - my
        fig, ax = plt.subplots(figsize=_get_figsize(dx, dy))

        for i, (x, y) in enumerate(zip(self.track_dict['x'], self.track_dict['y'])):
            plt.scatter(x, y, c='k', alpha=0.1, s=1)
            left_x, left_y = self.track_dict['left'][i]
            right_x, right_y = self.track_dict['right'][i]
            plt.scatter(left_x, left_y, c='k', s=1, alpha=0.1);plt.scatter(right_x, right_y, c='k', s=1, alpha=0.1)
        
        rect = self._get_rotated_car_patch(ax=ax)
        ax.add_patch(rect)
        ax.set_aspect(True)
        
        canvas = fig.canvas
        canvas.draw()
        
        width, height = canvas.get_width_height()
        image_array = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
        image_array = image_array.reshape(height, width, 4)

        if return_image:
            if self.plotter is not None:
                self.plotter._log_simulation(simulate_arr = image_array[:, :, 1:], tag_name = 'global_view')
                plt.cla();plt.clf();plt.close();gc.collect()
                return image_array
        ################### Log Simulation Screen to TensorBoard ###################
        if self.plotter is not None:
            self.plotter._log_simulation(simulate_arr = image_array[:, :, 1:], tag_name = 'global_view')
            plt.cla();plt.clf();plt.close();gc.collect()
        ############################################################################
    
    
        

    def _create_observation_space(self):
        observation_dict = self.environment_config['observation']
        observation_space = OrderedDict()
        self.forward_vector_obj = None
        self.lidar_sensor_obj = None
        self.lookahead_obj = None
        for key, value_dict in observation_dict.items():
            if value_dict['usage'] == False:
                continue
            if key == 'forward_vector': #벡터이기 때문에 num_vecs*2개#
                self.forward_vector_obj = Observation_ForwardVector_State(
                    car_obj=self.car,
                    track_dict=self.track_dict,
                    theta_diff=value_dict['theta_diff'],
                    num_vecs=value_dict['num_vecs']
                )
                observation_space['forward_vector'] = gym.spaces.Box(
                    low = value_dict['min_val'],
                    high = value_dict['max_val'],
                    shape = (value_dict['num_vecs'] * 2,)
                )
                
            elif key == 'lidar_sensor': #길이이기 떄문에 num_lidar#
                self.lidar_sensor_obj = Observation_Lidar_State(
                    car_obj=self.car,
                    track_dict=self.track_dict,
                    lidar_angle_min=value_dict['lidar_angle_min'],
                    lidar_angle_max=value_dict['lidar_angle_max'],
                    num_lidar=value_dict['num_lidar'],
                    max_lidar_length=value_dict['max_lidar_length']
                )
                observation_space['lidar_sensor'] = gym.spaces.Box(
                    low = value_dict['min_val'],
                    high = value_dict['max_val'],
                    shape = (value_dict['num_lidar'],)
                )
            elif key == 'lookahead': #curvature만 쓰면 num_states, coordinate도 포함하면 num_states * 2또는3
                self.lookahead_obj = Observation_Lookahead_State(
                    car_obj=self.car,
                    lookahead_time=value_dict['lookahead_time'],
                    lookahead_theta=value_dict['lookahead_theta'],
                    num_states=value_dict['num_states'],
                    track_dict=self.track_dict,
                    fixed=value_dict['fixed']
                )
                use_coords, use_kappa = value_dict['coords'], value_dict['curvature']
                if use_coords:
                    observation_space['lookahead_coords'] = gym.spaces.Box(
                        low = -self.environment_config['track']['track_radius']*4,
                        high = self.environment_config['track']['track_radius']*4,
                        shape = (value_dict['num_states'] * 2 * 3,) # left - center - right 모두 추가
                    )
                if use_kappa:
                    observation_space['lookahead_kappa'] = gym.spaces.Box(
                        low = 0.,
                        high = self.environment_config['track']['max_kappa'],
                        shape = (value_dict['num_states'],)
                    )
            elif key == 'e_phi':
                observation_space['e_phi'] = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,))
            elif key == 'e_c':
                car_half_width = self.car.bicycle_model.vehicle_model.body_width / 2
                track_half_width = self.environment_config['track']['track_width'] / 2
                if 'all' in self.penalty_config['terminate']['off_course_condition'] or 'tire' in self.penalty_config['terminate']['off_course_condition']:
                    val = car_half_width + track_half_width
                elif 'com' in self.penalty_config['terminate']['off_course_condition']:
                    val = track_half_width
                elif 'instance' in self.penalty_config['terminate']['off_course_condition']:
                    val = track_half_width - car_half_width
                observation_space['e_c'] = gym.spaces.Box(low=-val, high=val, shape=(1,))
            elif key == 'vx':
                observation_space['vx'] = gym.spaces.Box(low=self.car.vehicle_constraints.vx_min, high=self.car.vehicle_constraints.vx_max, shape=(1,))
            elif key == 'vy':
                observation_space['vy'] = gym.spaces.Box(low=self.car.vehicle_constraints.vy_min, high=self.car.vehicle_constraints.vy_max, shape=(1,))
            elif key == 'ax':
                observation_space['ax'] = gym.spaces.Box(low=self.car.vehicle_constraints.ax_min, high=self.car.vehicle_constraints.ax_max, shape=(1,))
            elif key == 'ay':
                observation_space['ay'] = gym.spaces.Box(low=self.car.vehicle_constraints.ay_min, high=self.car.vehicle_constraints.ay_max, shape=(1,))
            elif key == 'is_off_track':
                observation_space['is_off_track'] = gym.spaces.Box(low=0, high=1, shape=(1,))
            elif key == 'alpha_f':
                observation_space['alpha_f'] = gym.spaces.Box(low=self.car.vehicle_constraints.alpha_min, high=self.car.vehicle_constraints.alpha_max, shape=(1,))
            elif key == 'alpha_r':
                observation_space['alpha_r'] = gym.spaces.Box(low=self.car.vehicle_constraints.alpha_min, high=self.car.vehicle_constraints.alpha_max, shape=(1,))
            elif key == 'dx':
                observation_space['dx'] = gym.spaces.Box(low=self.car.bicycle_model.vehicle_model.dx_min, high=self.car.bicycle_model.vehicle_model.dx_max, shape=(1,))
            elif key == 'dy':
                observation_space['dy'] = gym.spaces.Box(low=self.car.bicycle_model.vehicle_model.dy_min, high=self.car.bicycle_model.vehicle_model.dy_max, shape=(1,))
                
            #tire force는 추가하지는 않았음.
        self.observation_space = gym.spaces.Dict(observation_space)
                
    def _create_action_space(self):
        action_dim = self.environment_config['action']['action_dim']
        
        if action_dim == 2:
            steer_range = self.environment_config['action']['steer_range']
            torque_range = self.environment_config['action']['torque_range']
            action_space = gym.spaces.Box(
                np.array([steer_range[0], torque_range[0]]).astype(np.float32),
                np.array([steer_range[1], torque_range[1]]).astype(np.float32)
            ) # steer, torque
            
        elif action_dim == 3:
            steer_range = self.environment_config['action']['steer_range']
            throttle_range = self.environment_config['action']['throttle_range']
            brake_range = self.environment_config['action']['brake_range']
            action_space = gym.spaces.Box(
                np.array([steer_range[0], throttle_range[0], brake_range[0]]).astype(np.float32),
                np.array([steer_range[1], throttle_range[1], brake_range[1]]).astype(np.float32)
            ) # steer, throttle, brake
            
        else:
            action_space = None
        
        self.action_space = action_space

    def _process_reward_penalty_medium(self, reward_dict:dict, penalty_dict:dict):
        common_rp, aggressive_rp, defensive_rp = 0, 0, 0
        common_dict, aggressive_dict, defensive_dict = self.style_config['common'], self.style_config['aggressive'], self.style_config['defensive']
        
        for key, value in reward_dict.items():
            if key in common_dict:
                common_rp += value
            elif key in aggressive_dict:
                aggressive_rp += value
            elif key in defensive_dict:
                defensive_rp += value
        
        for key, value in penalty_dict.items():
            if key in common_dict:
                common_rp += value
            elif key in aggressive_dict:
                aggressive_rp += value
            elif key in defensive_dict:
                defensive_rp += value
        
        final_rp = np.sum(
            np.array([self.style_config['style_mode']['common_weight'], self.style_config['style_mode']['aggressive_weight'], self.style_config['style_mode']['defensive_weight']]) \
                 * np.array([common_rp, aggressive_rp, defensive_rp])
        )
        
        return float(final_rp)
    
    def step(self, 
             action,
             style_setting=None):
        if action is None:
            action = np.zeros(self.action_space.low.size)
            
        ## (1) update car state (동시에 트랙의 status도 업데이트) ##
        self.track_dict = self.car._step(action=action) #SAC policy의 squashing function에 의해서 -1~1 사이의 값으로 action이 매핑 될 예정
        ## (2) update observation state ## 
        
        observation_arr = np.array([])
        
        for key, value in self.observation_space.items():
            if key == 'forward_vector' and hasattr(self, 'forward_vector_obj'):
                forward_vector_arr = self.forward_vector_obj._step(theta_center_spline=self.car.theta_center_spline)
                observation_arr = np.hstack([observation_arr, forward_vector_arr])
                self.observation_values[key].append(forward_vector_arr)
                
            elif key == 'lidar_sensor' and hasattr(self, 'lidar_sensor_obj'):
                lidar_scan_results = self.lidar_sensor_obj._step()
                lidar_distance = np.array([dict['distance'] for dict in lidar_scan_results])
                observation_arr = np.hstack([observation_arr, lidar_distance])
                self.observation_values[key].append(lidar_distance)
            
            elif key == 'lookahead_coords' and hasattr(self, 'lookahead_obj'):
                lookahead_coord_arr = self.lookahead_obj._lookahead_coords(
                    theta_center_spline=self.car.theta_center_spline,
                    theta_right_spline=self.car.theta_right_spline,
                    theta_left_spline=self.car.theta_left_spline,
                    normalize=False,
                    debug=False
                )
                lookahead_coord_arr = lookahead_coord_arr.T.reshape(-1) #(x1_left, y1_left, x1_center, y1_center, x1_right, y1_right,..)
                observation_arr = np.hstack([observation_arr, lookahead_coord_arr])
                self.observation_values[key].append(lookahead_coord_arr)
                
            elif key == 'lookahead_kappa' and hasattr(self, 'lookahead_obj'):
                lookahead_kappa_arr = self.lookahead_obj._lookahead_curvature(
                    theta_center_spline= self.car.theta_center_spline,
                    kappa_spline=self.car.kappa_spline,
                    normalize = False,
                    debug = False
                )
                
                observation_arr = np.hstack([observation_arr, lookahead_kappa_arr])
                self.observation_values[key].append(lookahead_kappa_arr)
                
                
            elif key == 'e_phi':
                e_phi = np.array([self.car.bicycle_model.E_phi])
                observation_arr = np.hstack([observation_arr, e_phi])
                self.observation_values[key].append(e_phi)
                
            elif key == 'e_c':
                e_c = np.array([self.car.bicycle_model.E_c])
                observation_arr = np.hstack([observation_arr, e_c])
                self.observation_values[key].append(e_c)
                
            elif key == 'vx':
                vx = np.array([self.car.bicycle_model.Vx])
                observation_arr = np.hstack([observation_arr, vx])
                self.observation_values[key].append(vx)
                
            elif key == 'vy':
                vy = np.array([self.car.bicycle_model.Vy])
                observation_arr = np.hstack([observation_arr, vy])
                self.observation_values[key].append(vy)
                
            elif key == 'ax':
                ax = np.array([self.car.bicycle_model.acc_x])
                observation_arr = np.hstack([observation_arr, ax])
                self.observation_values[key].append(ax)
                
            elif key == 'ay':
                ay = np.array([self.car.bicycle_model.acc_y])
                observation_arr = np.hstack([observation_arr, ay])
                self.observation_values[key].append(ay)
                
            elif key == 'is_off_track':
                condition = self.penalty_config['off_course_penalty']['condition']
                if condition == 'off_course_com':
                    status = self.offcourse_checker._off_course_com()
                elif condition == 'off_course_instance':
                    status = self.offcourse_checker._off_course_instance()
                elif condition == 'off_course_all':
                    status = self.offcourse_checker._off_course_all()
                elif condition == 'off_course_tire':
                    status = self.offcourse_checker._off_course_tire()
                elif condition == 'off_course_count':
                    status = self.offcourse_checker._off_course_tire()
                    
                binary_val = np.array([status], dtype=int)
                observation_arr = np.hstack([observation_arr, binary_val])
                self.observation_values[key].append(binary_val)
                
            elif key == 'alpha_f':
                alpha_f = np.array([self.car.bicycle_model.alpha_f])
                observation_arr = np.hstack([observation_arr, alpha_f])
                self.observation_values[key].append(alpha_f)
                
            elif key == 'alpha_r':
                alpha_r = np.array([self.car.bicycle_model.alpha_r])
                observation_arr = np.hstack([observation_arr, alpha_r])
                self.observation_values[key].append(alpha_r)

            elif key == 'dx':
                dx = np.array([self.car.bicycle_model.dx])
                observation_arr = np.hstack([observation_arr, dx])
                self.observation_values[key].append(dx)
            elif key == 'dy':
                dy = np.array([self.car.bicycle_model.dy])
                observation_arr = np.hstack([observation_arr, dy])
                self.observation_values[key].append(dy)
                
        STEP_REWARD = 0
        TERMINATED = False #주행 지속 조건에 위배되는 경우가 있을 때에 TERMINATED
        TRUNCATED = False #완주를 한 경우에 TRUNCATED
        
        REWARD_DICT = {}
        PENALTY_DICT = {}
        
        ## (3) calculate reward value from the reward functions ##
        if not self.use_style:
            current_reward = 0
            for reward_func_name, config in self.reward_config.items():
                if not config['usage']:
                    continue
                reward_func = getattr(reward_shaping, reward_func_name)
                reward_value = reward_func(car_obj = self.car, **config)

                current_reward += reward_value

                REWARD_DICT[reward_func_name] = reward_value

            ## (4) calculate penalty value from the penalty functions ##
            current_penalty = 0
            for penalty_func_name, config in self.penalty_config.items():
                # if not getattr(config, 'usage', False):
                #     continue
                if not config.get('usage', False):
                    continue
                penalty_func = getattr(penalty_shaping, penalty_func_name)
                penalty_value = penalty_func(car_obj=self.car, **config)
                current_penalty += penalty_value
                PENALTY_DICT[penalty_func_name] = penalty_value
        

        if self.use_style:
            num_style = self.agent_config['style']['size'] 
            """매 step마다 style setting을 다르게 하도록 함.
            Aggressive / Medium / Defensive의 3개의 class로 나누고자 할 때에는 2개의 style로 나타내는 것으로 생각하면 됨.
            Medium Style은 2개의 style의 중간 값을 갖개 된다.
            """
            if style_setting is not None:
                if self.num_style_nets == 2:
                    self.style_setting = np.random.randint(0, num_style+1) / num_style #0, 0.5, 1 중에서 하나의 값
                    
                elif self.num_style_nets == 3:
                    self.style_setting = np.random.randint(0, num_style+1) #0 , 1, 2 중에서 하나의 값
                else:
                    raise UserWarning(f"{self.num_style_nets} is not supported for number of style networks")
                ## 3이면 [0, 1, 2]중 하나 -> [0, 0.5, 1]중 하나로 정규화 (num_net==2인 경우)
                ##-> [0,1] / [0.5,0.5] / [1,0]
                ## num_net == 3이면 [0, 1, 2]중 style_setting이 설정 되어서 indexing으로 one-hot condition vector 사용
            else:
                # self.style_setting = style_setting / num_style
                self.style_setting = style_setting
                assert style_setting in self.style_setting_candidates
        else:
            self.style_setting = None
            
        if self.use_style:
            self.SR_Obj = styleReward_Obj(
                style_config=self.style_config,
                agent_config=self.agent_config
            )
            STEP_REWARD, style_weight = self.SR_Obj._condition_w_style(
                style_setting=self.style_setting
            )
        
        else:
            if self.style_config is not None and self.style_config['style_mode']['type'] == 'medium':
                STEP_REWARD  = self._process_reward_penalty_medium(reward_dict=REWARD_DICT, penalty_dict=PENALTY_DICT)
            else:    
                STEP_REWARD = current_penalty + current_reward
                
        REWARD_DICT['step_reward'] = STEP_REWARD
        self.reward_arr.append(STEP_REWARD)
        
        ## (5) check if status matches terminate condition ##
        terminated_status = False
        terminate_config = self.penalty_config['terminate'] 
        off_course_cond = terminate_config['off_course_condition']
        if 'all' in off_course_cond:
            terminated_status = self.offcourse_checker._off_course_all()
        elif 'com' in off_course_cond:
            terminated_status = self.offcourse_checker._off_course_com()
        elif 'instance' in off_course_cond:
            terminated_status = self.offcourse_checker._off_course_instance()
        elif 'tire' in off_course_cond:
            terminated_status = self.offcourse_checker._off_course_tire()
        elif 'count' in off_course_cond:
            terminated_status = self.offcourse_checker._off_course_tire()
        
        vel_cond = terminate_config['vel_condition']
        # if vel_cond is not 'none':
        if vel_cond != 'none':
            if vel_cond == 'count_time_neg_vel':
                if self.car.bicycle_model.neg_vel_count_time >= terminate_config['neg_vel_patience']:
                    terminated_status = True
            elif vel_cond == 'instant_neg_vel':
                if self.car.bicycle_model.Vx <= 0:
                    terminated_status = True
        
        ### 일정 시간
        if terminated_status:
            TERMINATED = True
            self.terminated_cnt += 1
            STEP_REWARD = -1 * terminate_config['penalty_value']  
            print("Car off track... Will start new episode!!")  
        
        ## (6) check truncated (lap finish) ##
        car_moved_tile = sum(self.track_dict['passed'])
        total_tile = len(self.track_dict['passed'])
        
        car_ref_theta = self.car.bicycle_model.ref_arr_dict['theta'][-1]
        total_theta = max(self.track_dict['theta'])
        # breakpoint()
        # if car_moved_tile >= total_tile * 0.95: ##트랙을 완주를 했다고 간주가 된다면##
        # if car_moved_tile >= total_tile * 0.99:
        if car_ref_theta >= total_theta * 0.99:
            self.num_lap += 1
            if self.num_lap == self.num_laps_for_truncate:
                TRUNCATED= True
                print(f"Car successfully finished the track for {self.num_laps_for_truncate} LAPS... Will start new episode!!")
        
            # breakpoint()
        ## (6) return for next step ##
        if self.use_style:
            # if self.agent_config['style']['equal_weight_medium']:
            """반드시 One-Hot-Encoding 된 style conditioning vector을 observation과 concat해야 함.
            weighting은 STEP_REWARD를 계산하기 위해서 styleReward_Obj에서 사용하게 될 값이고, 
            multi-style critic의 경우에는 각각 defensive / medium / aggressive style policy에 대한 value를 예측하도록 학습을 시킬 것이기 때문엘
            나중에 업데이트 할 때에도 이 값을 사용하게 될 것임."""
            if self.num_style_nets == 2:
                return_observation = np.concatenate(
                    (
                        [self.style_setting, 1-self.style_setting],
                        observation_arr
                    )
                )
            elif self.num_style_nets == 3: #indexing을 여기서 편하게 하기 위해서
                temp = [0, 0, 0];temp[self.style_setting] = 1
                return_observation = np.concatenate(
                    (
                        temp,
                        observation_arr
                    )
                )
            else:
                raise UserWarning(f"Unsupported Number of Style Nets: {self.num_style_nets}")
            # else:
            #     return_observation = np.concatenate(
            #         (
            #             style_weight,
            #             observation_arr
            #         )
            #     )
            
        else:
            return_observation = observation_arr
        return_reward = STEP_REWARD
        return_terminated = TERMINATED
        return_truncated = TRUNCATED
        return_info = {}
        
        ## (7) update attributes for the env object ##
        self.reward = STEP_REWARD
        # self.t += self.car.dt
        self.t += self.environment_config['vehicle']['dt']
        self.state = observation_arr
        
        rlkit_terminate = return_terminated | return_truncated #하나라도 True이면 True이기 때문에 OR 연산자를 사용해야 함.
        
        ######################################################### debugging ################################################################
        # if rlkit_terminate:
            # breakpoint()
        ####################################################################################################################################
        ## (8) log global and local views ##
        '''[TODO] 2개의 화면 (local / global view) 각각을 시각화 하는 과정을 병렬 처리를 하거나, 속도의 향상을 위해서 dash로 시각화 하는 것도 고려해 볼만 하다.'''
        # self._global_car_on_track(return_image=False) ##이건 항상 만들기에 시간이 너무 오래 걸리는 느낌이 들어서,, 계속 쓸지 말지 고민이 되는 상황임.
        # image_arr = self._local_car_on_track(return_image=True)
        # if self.terminated_fig_path is not None and TERMINATED:
        #     Image.fromarray(image_arr).save(f"{self.terminated_fig_path}/{self.terminated_cnt}.png")
        # if self.is_nam:
        #     image_arr = self._local_car_on_track(return_image=True)
        #     breakpoint()
        # else:
        #     self._local_car_on_track(return_image=False)
        

        ## (9) Render simulation
        if self.render_config is not None:
            
            self.screen, self.clock = _render_all_pygame(
                render_cfg=self.render_config,
                car_obj=self.car,
                track_dict=self.track_dict,
                t=self.t,
                lidar_obj=self.lidar_sensor_obj,
                fvec_obj=self.forward_vector_obj,
                screen=self.screen,
                clock=self.clock,
                
                action=action,
                reward_dict=REWARD_DICT,
                penalty_dict=PENALTY_DICT,
                render_car_state=True,
                
                exp_name=self.exp_name
            )
        return return_observation, return_reward, return_terminated, return_truncated, return_info
        # return return_observation, return_reward, rlkit_terminate, return_info
    
    def _nam_rand_track(self):
        if self.environment_config['track']['nam_ratio'] > 0 and np.random.random(1)[0] <= self.environment_config['track']['nam_ratio']:
            track_generator = Nam_TrackGenerator(
                    track_width=self.environment_config['track']['track_width'],
                    nam_track_path = f"{os.path.dirname(ROOT)}/statics/nam_c_track.pkl",
                    min_num_ckpt=self.environment_config['track']['min_num_ckpt'],
                    max_num_ckpt=self.environment_config['track']['max_num_ckpt'],
                )
        else:
            track_generator = Bezier_TrackGenerator(
                min_num_ckpt=self.environment_config['track']['min_num_ckpt'],
                max_num_ckpt=self.environment_config['track']['max_num_ckpt'],
                min_kappa=self.environment_config['track']['min_kappa'],
                max_kappa=self.environment_config['track']['max_kappa'],
                track_width=self.environment_config['track']['track_width'],
                track_density=self.environment_config['track']['track_density'],
                track_radius=self.environment_config['track']['track_radius'],
                scale_rate=self.environment_config['track']['scale_rate']
            )
        return track_generator
    
    # def _set_style_level(self, s):
    #     self.style_level = s
        
    def reset(self, seed=None, options=None, style=None):
        super().reset(seed=seed)
        ####################################################################
        self.reward = 0.
        self.t = 0.
        self.num_lap = 0
        self.state = None
        self.reward_arr = []
        self.observation_values = defaultdict(list)
        """SAC 알고리즘은 off-policy이라서 replay buffer을 쓰기 때문에 buffer에 저장할 때에 observation에 이미 style conditioning이 concat되어 있어야 한다."""
        # if style is not None:
        #     self._set_style_level(s=style)
        # else:
        #     self.style_level = random.choice(np.arange(0, 1+self.style_step_size, self.style_step_size)) #어차피 reset을 할 때에, 즉 하나의 새로운 episode를 학습 시키는 단계에서 style type를 바꿀 수 있어야 한다.
        ####################################################################
        if self.initial_start:
            self.screen = None
            self.clock = None
            if self.is_nam:
                self.track_generator = Nam_TrackGenerator(
                    track_width=self.environment_config['track']['track_width'],
                    nam_track_path = f"{os.path.dirname(ROOT)}/statics/nam_c_track.pkl",
                    min_num_ckpt=self.environment_config['track']['min_num_ckpt'],
                    max_num_ckpt=self.environment_config['track']['max_num_ckpt'],
                )
            else:
                self.track_generator = self._nam_rand_track()
                
            self.track_generator._generate()
            self.track_dict = self.track_generator._calculate_track_dict()
            
            self.car = RaceCar(
                action_dim=self.environment_config['action']['action_dim'],
                dt=self.environment_config['vehicle']['dt'],
                world_dt=self.environment_config['vehicle']['world_dt'],
                allow_both_feet=self.environment_config['vehicle']['allow_both_feet'],
                aps_bps_weight=self.environment_config['vehicle']['aps_bps_weight'],
                brake_on_pos_vel=self.environment_config['vehicle']['brake_on_pos_vel'],
                normalize_aps_bps=self.environment_config['vehicle']['normalize_aps_bps'],
                schedule_brake_episode=self.environment_config['vehicle']['schedule_brake_episode'],
                schedule_brake_ratio=self.environment_config['vehicle']['schedule_brake_ratio'],
                schedule_brake_ratio_scale=self.environment_config['vehicle']['schedule_brake_ratio_scale'],
                cfg_file_path=f"{ROOT}/vehicle/{self.environment_config['vehicle']['cfg_file_path']}",
                zero_force_neg_vel=self.environment_config['vehicle']['zero_force_neg_vel'],
                always_pos_vel=self.environment_config['vehicle']['always_pos_vel'],
                allow_neg_torque=self.environment_config['vehicle']['allow_neg_torque'],
                use_continuous_bps=self.environment_config['vehicle']['use_continuous_bps'],
                initial_vx=self.environment_config['vehicle']['initial_vx'],
                use_aps_bps_diff=self.environment_config['vehicle']['use_aps_bps_diff']
            )
            self.car._reset(track_dict=self.track_dict)
                
            # self.offcourse_checker = OffCourse_Checker(car_obj=self.car)
            self.offcourse_checker = self.car.vehicle_status_checker
            self._create_observation_space()
            self._create_action_space()
            self.initial_start = False
                                            
            self.terminated_cnt = 0
            import pickle
            #""dump the object itself (car object, environment object)"""
            pickle.dump(self.car.__dict__, open(f"{self.agent_config['test']['test_log_path']}/{self.mode}_car.pkl", "wb"))  
            # pickle.dump(self.__dict__, open(f"{self.agent_config['test']['test_log_path']}/{self.mode}_env.pkl", "wb"))          
                 
        
        else:
            '''[0421] - Debugging (Save all the values for current episode)
            '''
            ###############################################################################
            if self.environment_config['do_debug_logs'] > 0 and self.terminated_cnt <= self.environment_config['do_debug_logs']:
                import pickle, pygame
                from pathlib import Path
                DEBUG_LOG_ROOT=r'D:\meow_racer_debug'.replace('\\', '/')
                BM = self.car.bicycle_model
                save_path = f"{DEBUG_LOG_ROOT}/{self.exp_name}";os.makedirs(save_path, exist_ok=True)
                save_path = f"{save_path}/{self.mode}/{self.terminated_cnt}";Path(save_path).mkdir(parents=True, exist_ok=True)
                #(1) save bicycle model attributes for trajectory and dynamics logging
                pickle.dump(BM.__dict__, open(f"{save_path}/bicycle_model.pkl", "wb"))
                #(2) save car object (for action logging and splines etc)
                pickle.dump(self.car.__dict__, open(f"{save_path}/car.pkl", "wb"))
                #(3) save track dict for logging 
                pickle.dump(self.track_dict, open(f"{save_path}/track.pkl", "wb"))
                #(4) save the main rendered screen
                pygame.image.save(self.screen, f"{save_path}/final_screen.jpg")
            ###############################################################################
            # if self.is_nam == False:
            if self.is_nam == True:
                self.track_generator = Nam_TrackGenerator(
                    track_width=self.environment_config['track']['track_width'],
                    nam_track_path = f"{os.path.dirname(ROOT)}/statics/nam_c_track.pkl",
                    min_num_ckpt=self.environment_config['track']['min_num_ckpt'],
                    max_num_ckpt=self.environment_config['track']['max_num_ckpt'],
                )
            else:
                self.track_generator = self._nam_rand_track()
 
            self.track_generator._generate()
            self.track_dict = self.track_generator._calculate_track_dict()
            self.car._reset(track_dict=self.track_dict)
            
            if hasattr(self, 'forward_vector_obj') and self.forward_vector_obj is not None:
                self.forward_vector_obj._reset(car_obj=self.car, track_dict=self.track_dict)
            
            if hasattr(self, 'lidar_sensor_obj') and self.lidar_sensor_obj is not None:
                self.lidar_sensor_obj._reset(car_obj=self.car, track_dict=self.track_dict)
            
            if hasattr(self, 'lookahead_obj') and self.lookahead_obj is not None:
                self.lookahead_obj._reset(car_obj=self.car)
            
            # self.offcourse_checker = OffCourse_Checker(car_obj=self.car)
            self.offcourse_checker = self.car.vehicle_status_checker
        

        
        return self.step(None, style_setting=style)[0], {}
    
    
    
    



