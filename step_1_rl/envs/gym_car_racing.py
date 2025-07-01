import os, sys
import numpy as np
import math
import re
from typing import Optional
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import pygame
# import gym
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding
import Box2D
from Box2D.b2 import fixtureDef, polygonShape
import pickle
##############################################
from envs.gym_car_constants import *
from envs.gym_car import Toy_Car
from envs.gym_car_jw import JW_Toy_Car
from envs.utils import (
    calculate_theta, 
    # calculate_phi,
    calculate_track_phi,
    # calculate_curvature,
    calculate_track_curvature,
    get_track_boundary, return_track_center,
    Lidar,
    get_track_state, get_feature_vec_state,
    
    
    Dynamics,
    gen_beta,
    Forward_Observation,
    
    create_checkpoints, connect_checkpoints, check_connected_loop,
    create_tiles,
    
    resplit,
    find_hard_corner,
    gen_alpha,
    
    gen_straight_track
)

from envs.reward import _calculate_reward

from envs.detectors import (
    FrictionDetector,  # contactListener is necessary to check how many tiles the vehicle has passed #
    PenaltyDetector,
    TileLocation_Detector
)

from envs.org_env import (
    CarRacing, 
    PYGAME_STATE_DICT
)
from algorithm.common.common_utils import preprocess_state

class Toy_CarRacing(CarRacing):
    """
    ### Action Space ###
    Steer: [-1, +1]   -1 is full left, +1 is full right
    Gas: [0, +1]
    Brake: [0, +1]
    
    ### Observation Space ###
    Dynamics (e_c, e_phi, v_x, v_y)
    Sensory (LiDAR)
    Car (omega value of 4 wheels, forward obseraation vector)
    """
    metadata = {"render_modes" : ["human"], "render_fps": FPS}
    
    def __init__(self, 
                 observation_config:dict,
                 lap_complete_percent:float=0.95,
                 do_zoom:bool=True,
                 simple_circle_track:bool=False,
                 args:dict=None
                 ):
        super(CarRacing, self).__init__(
            render_mode='human', lap_complete_percent=lap_complete_percent, verbose=False,
            domain_randomize=False, continuous=True
        )
        EzPickle.__init__(
            self, observation_config, lap_complete_percent, simple_circle_track, do_zoom, args
        )
        self.input_args = args
        self.track_turn_rate = float(np.clip(np.random.rand(1) * TRACK_TURN_RATE,
                                        a_min=math.pi * 0.75, 
                                        a_max=TRACK_TURN_RATE))
        self.do_reverse = False
        self.min_track_turn_rate = self.track_turn_rate - 0.01
        self.use_beta_dist = args.use_beta_dist
        
        self.do_zoom = do_zoom
        self.continuous = True
        self.domain_randomize = False
        self.verbose = False
        self.car_left_track = False
        self.is_backward = False
        self.render_mode = 'human'
        self.observation_config = observation_config
        self.simple_circle_track = simple_circle_track
        self.num_episode = 0
        
        self.status_queue = []
        
        self.track_width = TRACK_WIDTH
        
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()
        
        if self.input_args.oscillation_penalty > 0: 
            self.contactListener_keepref = TileLocation_Detector(self, self.lap_complete_percent, 
                                                                 max_reward_tile=self.input_args.max_reward_tile)
        else:
            self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent, max_reward_tile=5000.)
        # self.contactListener_keepref = PenaltyDetector(self, self.lap_complete_percent, max_reward_tile=self.input_args.max_reward)
        ## the world class manages all physic entities, dynamic simulation, and asynchronous queries ##
        self.world = Box2D.b2World(gravity=(0,0), 
                                   contactListener=self.contactListener_keepref)
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0,0), (1,0), (1,-1), (0,-1)])
        )
        
        self.is_nam = False
        self.screen = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None

        self.reward = 0.0
        self.backward_count = 0
        self.same_tile_count = 0 # 동일한 tile에 몇번째 계속 위치하는지 -> 속도 및 정체 개선을 위함 #
        self.on_same_tile_penalty = False # 동일한 tile에 input_args.same_tile_penalty번 이상 위치하면 True -> TERMINATE #
        self.prev_reward = 0.0 # 이전 reward 값 자체 #
        self.out_track_count = 0 # 트랙 밖에 나간 상태로 몇번의 time step만큼 지속중인지 횟수 count, 200번째 time step이상 밖에 위치하면 True -> TERMINATE #
        self.new_lap = False
        
        ## set up the environment space from the Gymnasium library ##
        if args.action_num == 2:
            self.action_space = spaces.Box(
                np.array([-1., -1.]).astype(np.float32), #MIN (steer, gas/brake) 
                np.array([1., 1.]).astype(np.float32) #MAX (steer, gas/brake)
            )
        elif args.action_num == 3:
            self.action_space = spaces.Box(
                np.array([-1., 0., 0.]).astype(np.float32), #MIN (steer, gas, brake)#
                np.array([+1., +1., +1.]).astype(np.float32) #MAX (steer, gas, brake)#
            )
        else:
            raise UserWarning(f"{args.action_num} is not supported for action noise")
        ## the observation space will vary depending on what features to use for representing the state of the vehicle ##
        self.observation_space = self._load_observation_space(mode=observation_config)        
        
        self.car_state = defaultdict(list)
        self.action_dict = defaultdict(list)
        self.actor_state_dict = defaultdict(list)
        
        self.status = PYGAME_STATE_DICT['RUNNING']
        
        self.neg_reward_check = 0
        self.neg_reward_limit = self.input_args.neg_reward_limit
    
    def __call__(self):
        return self
    
    def _setup_additional(self):
        if 'lidar' in self.observation_config:
            self.sensory_obj = Lidar(car_obj=self.car, track_dict=self.track_dict,
                                 road_poly=self.road_poly, 
                                 degree=self.input_args.lidar_deg,
                                 is_nam=self.is_nam,
                                 lidar_length=self.input_args.lidar_length,
                                )
        else:
            self.sensory_obj = None
            
        if 'dynamic' in self.observation_config:
            self.dynamics_obj = Dynamics(car_obj=self.car, track_dict=self.track_dict)
        else:
            self.dynamics_obj = None
        
        if 'car' in self.observation_config:
            self.forward_obj = Forward_Observation(
                car_obj=self.car, track_dict=self.track_dict,
                theta_diff=self.input_args.theta_diff, num_vecs=self.input_args.num_vecs
            )
        else:
            self.forward_obj = None
            
            
    def _load_observation_space(self, mode):
        min_arr = []
        max_arr = []
        '''observation space의 범위를 지정해 주는 것은 사실상 어딘가에 제약이 걸리게 하는 등의 문제는 아님.
        그냥 형식상으로 정의 해주는 개념이라고 볼 수 있다.''' 
        if 'dynamic' in mode: # theta, e_c, e_phi, v_x, v_y, omega #
            if 'theta' in mode['dynamic']: #곡선 방향으로의 이동 거리#
                min_arr.append(0.);max_arr.append(1.)
            if 'e_c' in mode['dynamic']: #distance between the vehicle and track axis#
                min_arr.append(-TRACK_WIDTH);max_arr.append(TRACK_WIDTH)
            if 'e_phi' in mode['dynamic']: #angle between the vehicle's orientation and centerline of the track#
                min_arr.append(-2. * math.pi);max_arr.append(2. * math.pi)
            if 'v_x' in mode['dynamic']: ## m/s -> KPH (gym car racing uses m/s) ##
                # min_arr.extend([0. for _ in range(4)]);max_arr.extend([300. for _ in range(4)]) 
                # min_arr.append(0.);max_arr.append(300.)
                min_arr.append(-5 * 3.6);max_arr.append(80 * 3.6)
            if 'v_y' in mode['dynamic']: ## m/s -> KPH (gym car racing uses m/s) ##
                # min_arr.extend([0. for _ in range(4)]);max_arr.extend([300. for _ in range(4)]) 
                # min_arr.append(0.);max_arr.append(300.)
                min_arr.append(-10 * 3.6);max_arr.append(10 * 3.6)
            if 'yaw_omega' in mode['dynamic']:
                min_arr.append(-10.);max_arr.append(10.)

        if 'lidar' in mode: # lidar sensory data #
            # degs = np.arange(0, 180, self.input_args.lidar_deg)
            min_arr.extend([-1 for _ in range(len(mode['lidar']))])
            # max_arr.extend([max(WINDOW_W, WINDOW_H) for _ in range(len(degs))])
            max_arr.extend([self.input_args.lidar_length for _ in range(len(mode['lidar']))])
            
        if 'car' in mode:
            check = re.compile("omega_*") #차량의 바퀴의 각속도#
            if len(list(filter(check.match, self.observation_config['car']))) > 0:
                n_wheel = 4
                min_arr.extend([0 for _ in range(n_wheel)])
                max_arr.extend([math.pi for _ in range(n_wheel)])

            if 'delta' in mode['car']: #차량의 조향각#
                min_arr.append(0)
                max_arr.append(math.pi)
            if 'steer' in mode['car']:
                min_arr.append(-1);max_arr.append(1)
            if 'gas' in mode['car']:
                min_arr.append(0);max_arr.append(1)
            if 'brake' in mode['car']:
                min_arr.append(0);max_arr.append(1)
            if 'force' in mode['car']:
                min_arr.append(-10000);max_arr.append(10000)
  
            # if 'forward' in mode['car']: #forward observation vector#
            check = re.compile("forward_*")
            if len(list(filter(check.match, mode['car']))) > 0:
                min_arr.extend([0 for _ in range(self.input_args.num_vecs * 2)])
                max_arr.extend([self.input_args.theta_diff for _ in range(self.input_args.num_vecs * 2)])

            check = re.compile("curvature_*")
            if len(list(filter(check.match, mode['car']))) > 0:
                ##정확히 이 값이 맞는지는 모르겠으나,, 사실 곡률이라고 하면 최댓값은 inf까지 갈 수있을 듯
                min_arr.extend([-0.1 for _ in range(self.input_args.num_vecs)])
                max_arr.extend([0.1 for _ in range(self.input_args.num_vecs)])
                
            # check = re.compile("force_*")
            # if len(list(filter(check.match, mode['car']))) > 0:
            #     n_wheel = 4
            #     min_arr.extend([-FRICTION_LIMIT for _ in range(n_wheel)])
            #     max_arr.extend([FRICTION_LIMIT for _ in range(n_wheel)])
                
        if 'track' in mode: # actual track data #
            pass
        
        '''이렇게 track 자체의 이미지를 넣는 방법이 좋을수는 있으나, 결과적으로는 cnn_policy를 
        사용하는 것과 큰 차이가 없을 것 같기 때문에..'''
        if 'feature_vec' in mode: # feature vector from the pretrained cnn using the track BEV image as the input #
            pass
         
        # for i, (min_v, max_v) in enumerate(zip(np.array(min_arr), np.array(max_arr))):
        #     dict_obs[f"state{i}"] = spaces.Box(low=min_v, high=max_v)
        # breakpoint()
        if (self.input_args.replay_buffer_class is not None) and (self.input_args.replay_buffer_class !='None'):
            # return spaces.Dict(dict_obs) 
            return spaces.Dict({
                'obs': spaces.Box(
                    low=np.array(min_arr).astype(np.float32),
                    high=np.array(max_arr).astype(np.float32)
                    )
                }
            )
        
        return spaces.Box(
            low=np.array(min_arr).astype(np.float32), 
            high=np.array(max_arr).astype(np.float32),
        )
    
    def main_reset(self, seed):
        self.super_reset(seed=seed)
        

    def render_all(self):
        if self.do_zoom is None:
            self.render()
        else:
            self._render(mode=self.render_mode, 
                         zoom=self.do_zoom,
                         W=WINDOW_W,
                         H=WINDOW_H)
            
        # self.sensory_obj.draw(screen=self.surf, zoom=self.zoom, translation=self.translation, angle=self.angle)
        # self._update_human_screen()

    def _create_nam_track(self):
        self.track_width = NAM_TRACK_WIDTH
        nam_track = pickle.load(open(f'{ROOT}/statics/nam_c_track.pkl', 'rb'))
        # x, y, phi = np.array(nam_track['x'])[:-1], np.array(nam_track['y'])[:-1], np.array(nam_track['phi'])[:-1]
        # x, y, phi = np.array(nam_track['x'])[:-1][::self.input_args.skip_rate], np.array(nam_track['y'])[:-1][::self.input_args.skip_rate], np.array(nam_track['phi'])[:-1][::self.input_args.skip_rate]
        x, y, phi = np.array(nam_track['x'])[::self.input_args.skip_rate], np.array(nam_track['y'])[::self.input_args.skip_rate], np.array(nam_track['phi'])[::self.input_args.skip_rate]
    
        # if self.do_reverse:
        #     # track = track[::-1]
        #     x = x[::-1]
        #     y = y[::-1]
        #     phi = phi[::-1]
        
        # if self.input_args.random_start:
        #     idx = np.random.randint(0, len(x)-1)
        #     # breakpoint()
        #     if idx == len(x)-1:
        #         x = np.hstack([[x[-1]], x[:-1]])
        #         y = np.hstack([[y[-1]], y[:-1]]) # [y[-1]] + y[:-1]
        #         phi =  np.hstack([[phi[-1]], phi[:-1]])  # [phi[-1]] + phi[:-1]
        #     if idx != 0:
        #         x = np.hstack([x[:idx], x[idx:]]) #  x[:idx] + x[idx:]
        #         y = np.hstack([y[:idx], y[idx:]]) #  y[:idx] + y[idx:]
        #         phi = np.hstack([phi[:idx], phi[idx:]]) # phi[:idx] + phi[idx:]
                
                
        x -= x[0]
        y -= y[0]
        
        x, y, phi = resplit(x=x,y=y,phi=phi)
        
        beta = gen_beta(phi)
        
        self.phi = phi  
        # breakpoint()
        self.track_turn_rate = NAM_TRACK_TURN_RATE
        
        border = find_hard_corner(beta_arr=phi, track_turn_rate=self.track_turn_rate, is_nam=True)
        
        road_poly, road  = create_tiles(
            box_world=self.world, box_tile=self.fd_tile,
            X=x, Y=y, beta=beta, border_arr=border,
            is_nam=True,
            width = None if self.input_args.nam_width_change is False else TRACK_WIDTH
        )
        self.road_poly = road_poly
        self.road = road
        
        alpha_arr = gen_alpha(x, y)
        track = np.vstack((alpha_arr, beta, x, y)).T # (N, 4)
        theta_arr = calculate_theta(x, y)
        kappa_arr = calculate_track_curvature(x, y)
        # phi_arr = calculate_phi(x, y)
        # phi_arr = self.phi
        left_bound, right_bound = get_track_boundary(cX=x, cY=y, phi_arr=self.phi)
        # right_bound, left_bound = get_track_boundary(cX=x, cY=y, phi_arr=self.phi)
        
        left_X, left_Y = np.array(left_bound).T[0], np.array(left_bound).T[1]
        right_X, right_Y = np.array(right_bound).T[0], np.array(right_bound).T[1]
        straight_arr = gen_straight_track(kappa_arr=kappa_arr, theta_arr=theta_arr, 
                                          theta_length=self.input_args.theta_diff,
                                          vec_num=self.input_args.num_vecs,
                                          is_nam=True,
                                          kappa_limit=self.input_args.straight_kappa_limit,
                                          consider_forward_vec=self.input_args.consider_forward_vec)
        
        ## update track tiles (with detector objects) ##
        if self.input_args.oscillation_penalty > 0:
            for ti, tile in enumerate(self.road):
                if straight_arr[ti]:
                    setattr(tile, "is_straight", True)
                else:
                    setattr(tile, "is_straight", False)
                self.road[ti] = tile
        ###################################################
        
        track_dict = {
            'theta' : theta_arr,
            'phi': self.phi,  # # phi_arr,
            'beta': beta,
            'x': x, 'y': y,
            'left_x': left_X, 'left_y': left_Y,
            'right_x': right_X, 'right_y': right_Y,
            'kappa': kappa_arr,
            
            'straight': straight_arr
            
        }
        # breakpoint()
        # self.do_reverse = self.input_args.do_reverse_nam > 0.
        self.do_reverse = False
        
        return road_poly, road, track, track_dict
    
    def _create_track_both(self):
        ratio = np.random.rand(1)
        if self.input_args.max_both_track_ratio > self.input_args.both_track_ratio:
            ratio_diff = self.input_args.max_both_track_ratio - self.input_args.both_track_ratio
            both_track_ratio = (self.num_episode / self.input_args.max_episodes) * ratio_diff + self.input_args.both_track_ratio
        else:
            both_track_ratio = self.input_args.both_track_ratio
        # if ratio > self.input_args.both_track_ratio:
        if ratio > both_track_ratio:
            self.is_nam = False
            self.track_width = TRACK_WIDTH
            return self._create_track()
        else:
            self.is_nam = True
            self.track_width = NAM_TRACK_WIDTH
            road_poly, road, track, track_dict = self._create_nam_track()
            self.road = road
            self.track_dict = track_dict
            self.track = track
            self.road_poly = road_poly
            self.checkpoints = None
            return True
            
    def reset(self, *, seed:Optional[int] = None, options:Optional[dict]=None):
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        #world, pygame display등 모두 지우기#
        self._destroy()
        self.car_state = defaultdict(list)
        self.action_state_dict = defaultdict(list)
        
        if self.input_args.oscillation_penalty > 0:
            self.world.contactListener_bug_workaround = TileLocation_Detector(
                self, self.lap_complete_percent, self.input_args.max_reward_tile
            )
        else:
            self.world.contactListener_bug_workaround = FrictionDetector(
                self, self.lap_complete_percent
            )
        self.world.contactListener = self.world.contactListener_bug_workaround
        
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.backward_count = 0
        self.new_lap = False
        self.car_left_track = False
        self.is_backward = False
        self.same_tile_count = 0
        self.on_same_tile_penalty = False 
        
        self.out_track_count = 0
        self.track_dict = {}
        self.road_poly = [] #새로운 racing track을 만들 것이기 때문#
        
        while True:
            if self.simple_circle_track:
                success = self._create_circle_track()
            elif self.input_args.both_track_ratio > 0:
                success = self._create_track_both()
            else:
                success = self._create_track()
            if success:
                break

        
        cX, cY = return_track_center(road_poly=self.road_poly)
        
        # if self.do_reverse:
        #     cX = [cX[0]] + cX[1:][::-1]
        #     cY = [cY[0]] + cY[1:][::-1]
            
        # phi_arr = calculate_phi(X=cX, Y=cY)
        phi_arr = calculate_track_phi(X=cX, Y=cY) ## 이렇게 계산을 해야 중앙선의 좌표들을 바탕으로 원하는대로 (1, 0)으로부터의 회전 각을 계산할 수 있음.
        beta_arr = gen_beta(phi_arr=phi_arr)
        kappa_arr = calculate_track_curvature(X=cX, Y=cY)
        theta_arr = calculate_theta(X=cX, Y=cY)
        straight_arr = gen_straight_track(kappa_arr=kappa_arr, theta_arr=theta_arr, theta_length=self.input_args.theta_diff, vec_num=self.input_args.num_vecs,
                                          kappa_limit=self.input_args.straight_kappa_limit,
                                          consider_forward_vec=self.input_args.consider_forward_vec)
        
        # left_limit_poly, left_limit_road, right_limit_poly, right_limit_road = self._make_track_limit(
        #     X=cX, Y=cY, PHI=phi_arr, BETA=beta_arr
        # )
        
        # self.left_limit_poly = left_limit_poly
        # self.right_limit_poly = right_limit_poly
        
        left_bound, right_bound = get_track_boundary(cX=cX, cY=cY, phi_arr=phi_arr)
        left_X, left_Y = np.array(left_bound).T[0], np.array(left_bound).T[1]
        right_X, right_Y = np.array(right_bound).T[0], np.array(right_bound).T[1]
        
    
        ## update track tiles (with detector objects) ##
        if self.input_args.oscillation_penalty > 0:
            for ti, tile in enumerate(self.road):
  
                if straight_arr[ti]: 
                    setattr(tile, "is_straight", True)
                else:
                    setattr(tile, "is_straight", False)
 
                self.road[ti] = tile
        
            # for ti, tile in enumerate(self.road):
            #     if tile.is_straight:
            #         print(f"straight #{ti}")
        ###################################################
        
        
        self.track_dict = {
            'theta': theta_arr,
            'phi': phi_arr , 
            'beta': beta_arr,
            'x': np.array(cX), 
            'y': np.array(cY),
            'kappa': kappa_arr,
            
            'straight': straight_arr,
            
            'lx': left_X, 'ly': left_Y, 'rx': right_X, 'ry': right_Y,
            'checkpoints': self.checkpoints
        }
        import pickle
        pickle.dump(self.track_dict, open('check_1.pkl', 'wb'))
        pickle.dump(self.track, open('check_track_1.pkl', 'wb'))
        pickle.dump(self.track_dict, open('check_debug_1.pkl', 'wb'))

        '''[0213]reverse 추가
        -> 이유는, 학습 되는 과정과 남양 트랙 주행하는 것을 봤는데, 2번째 헤어핀 코너는 잘 돌아가는데 코너를 탈출하고 나올 때 항상 멈추게 된다.
        -> 랜덤하게 생성되는 트랙들이 항상 앞으로 전진을 하기 때문에 그 부분에 대해서 학습이 더 많이 될 수 밖에 없다. 따라서 역방향으로 트랙의 방향을 시작할 수 있도록 하였다.'''
        init_angle = self.track[0][1]
        if self.input_args.use_jw:
            self.car = JW_Toy_Car(world=self.world, init_angle=init_angle, init_x=self.track[0][2], init_y=self.track[0][3], do_reverse=self.do_reverse, 
                                  use_beta_dist=self.use_beta_dist)
        else:
            self.car = Toy_Car(world=self.world, init_angle=init_angle, init_x=self.track[0][2], init_y=self.track[0][3], do_reverse=self.do_reverse)
        


        self._setup_additional()
        self.render_all()
    
        return self.step(None)[0], {}
    
    
    def _create_circle_track(self):

        num_checkpoints = 36
        track_rad = np.random.uniform(TRACK_RAD / 3, TRACK_RAD)
        checkpoints, start_alpha = create_checkpoints(num_checkpoints, track_rad, is_circle=True)
        track = connect_checkpoints(checkpoints=checkpoints, 
                                    track_turn_rate=self.track_turn_rate,
                                    min_track_turn_rate=self.min_track_turn_rate,
                                    track_rad=track_rad, detail_step=3/SCALE, is_circle=True)
        is_glued, i1, i2, track = check_connected_loop(track=track, start_alpha=0.0000001)
        if not is_glued:
            return False
        
        road_poly, road = create_tiles(self.world, self.fd_tile, 
                                       X=np.array(track).T[2],
                                        Y=np.array(track).T[3],
                                        beta=np.array(track).T[1],
                                        border_arr=[],
                                        is_nam=False)
        self.road_poly = road_poly
        self.road = road
        self.track = track
        return True
    
    def _log_car_states(self):
        vel = self.car.hull.linearVelocity
        self.car_state['true_speed'].append(np.sqrt(np.square(vel[0]) + np.square(vel[1])))
        for i in range(len(self.car.wheels)):
            w = self.car.wheels[i]
            self.car_state[f'omega_{i}'].append(w.omega)
            if i == 0:
                self.car_state['joint_angle'].append(w.joint.angle)
        # self.car_state['angular_vel'].append(self.car.hull.angularVelocity)
        # self.car_state['angle'].append(math.degrees(self.car.hull.angle))
        # self.car_state['angle'].append(math.degrees(self.car.hull.angle) % 360)

        self.car_state['time_reward'].append(self.input_args.max_reward / len(self.road_poly))
        
    def _closest_tile(self):
        import heapq
        car_x, car_y = self.car.hull.position
        track_x, track_y = self.track_dict['x'], self.track_dict['y']
        distances = [[math.dist((car_x, car_y), (tx, ty)), i] for i, (tx, ty) in enumerate(zip(track_x, track_y))]
        heapq.heapify(distances)
        min_dist, min_tile_idx = heapq.heappop(distances)
        return min_tile_idx
    
    def _calc_oscillation_value(self):
        e_c_arr = self.dynamics_obj.e_c_arr
        if len(e_c_arr) == 1:
            return False, -1
        prev, cur = e_c_arr[-2], e_c_arr[-1]
        if prev * cur >= 0:
            return True, abs(prev - cur)
        else:
            if prev > cur:
                return True, prev - cur
            else:
                return True, cur - prev
    
    def _preprocess_action(self, action):
        # if len(self.action_space) == 2:
        if action is None:
            return action
        if self.input_args.action_num == 2:
            new_action = np.zeros(3)
            new_action[0] = action[0]
            '''양발 운전이 불가능하게 하도록'''
            new_action[1] = action[1] if action[1] > 0 else 0 ##gas
            new_action[2] = abs(action[1]) if action[1] < 0 else 0 ##brake
            return new_action
        
        return action
    
    def step(self, action):
        assert self.car is not None 
        
        action = self._preprocess_action(action)
        
        if action is not None:
            if self.use_beta_dist:
                # action[0] = (action[0]-0.5) * 2
                new_action = (action[0] - 0.5) * 2
                self.car.steer(-new_action)
            else:
                self.car.steer(-action[0])
            # self.car.steer(0) ##for debugging the physics engine##
            self.car.gas(action[1])
            self.car.brake(action[2])
        
        self.car.step(1. / FPS, 
                      new_friction=self.input_args.new_friction_limit,) 
        # self.sensory_obj.step(car_yaw_rad = self.car.hull.angle)
        self.world.Step(1. / FPS, 6 * 30, 2 * 30)
        self.t += 1. / FPS
        
        self.state = self._get_state() # should be implemented #
        
        if isinstance(self.observation_space, spaces.Dict):
            self.state = {'obs': self.state}
        step_reward = 0
        terminated = False
        truncated = False
        
        d_axis = abs(self.dynamics_obj.dynamic_state['e_c'])
        if action is not None:
            if self.tile_visited_count == len(self.track) or self.new_lap:
                truncated = True
            '''(1)-1. Common Reward - Time Pass Penalty'''
            self.reward -= self.input_args.time_penalty # 시간이 지남에 따른 penalty #

            '''(1)-2. Common Reward - Same Tile Count Penalty'''
            if self.input_args.same_tile_penalty > 0:
                self.reward -= self.same_tile_count # 동일한 tile을 계속 밟고 있으면 penalty #
                if self.on_same_tile_penalty:
                    terminated = True
                    step_reward = -500

            '''(1)-3. Common Reward - Minimum Movement Penalty'''
        
            if self.input_args.use_theta_diff: 
                if self.input_args.min_theta_movement == 0:
                    if self.dynamics_obj.theta_diff < self.input_args.min_movement:
                        self.reward -= 10.
                else:        
                    if self.dynamics_obj.theta_diff_val < self.input_args.min_theta_movement:
                        self.reward -= 10.
                    if self.dynamics_obj.dist_diff < self.input_args.min_movement:
                        self.reward -= 10.
            else:
                if self.dynamics_obj.dist_diff < self.input_args.min_movement:
                    self.reward -= 10.
            
            '''(1)-4. Common Reward - Minimum Velocity Penalty'''
            if (self.dynamics_obj.vel < self.input_args.min_vel) and (self.input_args.min_vel > 0):
                self.reward -= 5.

            '''(1)-5. Common Reward - Car body leave track Penalty
            - 현재는 Center of Mass 기준으로 트랙 밖을 벗어났을 때 environment를 terminate 하게 된다.
            - 만약에 차량이 조금이라도 밖으로 나가게 되었을 때 reward를 조금씩 깎는다면??
            '''
            half_car_width = JW_CAR_WIDTH / 2
            if (self.input_args.body_left_penalty > 0) and d_axis >= (TRACK_WIDTH - half_car_width):
                
                car_left_length = half_car_width - (TRACK_WIDTH - d_axis)
                if self.input_args.body_left_mode == "constant":
                    self.reward -= self.input_args.body_left_penalty
                    
                elif self.input_args.body_left_mode == "percent":
                    leave_percentile = car_left_length / half_car_width 
                    self.reward -= self.input_args.body_left_penalty * leave_percentile
                
                else:
                    raise UserWarning(f"{self.input_args.body_left_mode} is not supported!!")
            
            if self.input_args.center_line_far_max_reward > 0:
                e_c_abs = d_axis
                percentage = e_c_abs / TRACK_WIDTH
                # close_tile_idx = self._closest_tile()
                close_tile_idx = self.dynamics_obj.ref_index
                ## corner 구간을 통과할 때에는 oscillation에 대한 error도 없기 떄문에 corner을 돌 때에 track의 양 끝단에서 움직이도록 하기 위해서
                # state에 따라서 다른 reward와 penalty weight를 줄 수 있도록 한다.
                ### corner에 대해서 center line으로부터 떨어진 거리 reward = 10
                ### 직선로에 대해서 center line으로부터 떨어진 거리 reward = 1
                ###### [corner]: reward 10
                if (self.input_args.center_line_far_max_reward_corner > 0) and (self.track_dict['straight'][close_tile_idx] == False):
                    self.reward += percentage * self.input_args.center_line_far_max_reward_corner
                ###### [straight]: reward 1
                else: 
                    self.reward += percentage * self.input_args.center_line_far_max_reward
                
            if self.input_args.oscillation_max_penalty > 0.:
                is_oscillate, oscillate_val = self._calc_oscillation_value()
                #gym_car_constants에 정의된 TRACK_WIDTH는 트랙의 너비의 절반
                #따라서 트랙 너비 전체가 최댓값이 oscillate_val은 트랙의 전체 너비로 나눠주어야 함.
                oscillate_percent = oscillate_val / (TRACK_WIDTH * 2)
                close_x, close_y = self.dynamics_obj.ref_x, self.dynamics_obj.ref_y
                close_idx = self.dynamics_obj.ref_index
                ####### [straight]: penalty with oscillation (-5)
                if is_oscillate and self.track_dict['straight'][close_idx] == True:
                    self.reward -= self.input_args.oscillation_max_penalty * oscillate_percent
                ####### [corner]: reward with oscillation (1)
                if is_oscillate and (self.input_args.oscillation_max_reward_corner > 0) and (self.track_dict['straight'][close_idx] == False):
                    self.reward += self.input_args.oscillation_max_reward_corner * oscillate_percent
                    
                '''(2) AM Reward (reward on velocity)'''
            if self.input_args.reward_type.lower() == "am":
                velocity = self.car.hull.linearVelocity
                normal_vector = self.car.hull.GetWorldVector((0, 1)) # 종방향 벡터 ## forward velocity ##
                lateral_v, longitude_v = velocity # (횡속 / 종속)
                # longitude_v_proj = np.dot(np.array(normal_vector), np.array(velocity))
                '''https://github.com/pybox2d/pybox2d/blob/master/library/Box2D/examples/top_down_car.py
                위 코드 링크의 내용을 참고했는데, 내가 생각한 대로 (0, 1)이 종방향 벡터 / (1, 0)이 횡방향 벡터의 방향이 맞았다.'''
                
                # longitude_v_proj = velocity.dot(normal_vector) * normal_vector
                longitude_v_proj = velocity.dot(normal_vector)
                # longitude_velocity = np.sqrt(longitude_v_proj[0]**2 + longitude_v_proj[1]**2)
                # longitude_v_proj = np.array(longitude_v_proj)
                
                 
                self.dynamics_obj.dynamic_state['longitude_velocity'] = longitude_v_proj
                
                self.dynamics_obj.dynamic_state['lateral_v'] = lateral_v
                self.dynamics_obj.dynamic_state['longitude_v'] = longitude_v
                # self.dynamics_obj.dynamic_state['longitude_v_proj_x'] = longitude_v_proj[0]
                # self.dynamics_obj.dynamic_state['longitude_v_proj_y'] = longitude_v_proj[1]
                
                
                
                heading_angle = self.forward_obj.heading_angle ## 결국에는 이 값이 e_phi에 해당하는 값이 된다.
                self.dynamics_obj.dynamic_state['heading_angle'] = heading_angle
                
                # step_reward = math.cos(heading_angle) * abs(longitude_v) ## track의 진행 방향으로 얼마나 이동을 했는지를 velocity에 cos(phi)를 곱해서 계산하게 됨.
                # self.reward += math.cos(heading_angle) * longitude_v_proj
                e_phi = self.dynamics_obj.dynamic_state['e_phi']
                self.reward += math.cos(e_phi) * longitude_v_proj
                
                
                if self.input_args.am_with_theta_reward:
                    self.reward += self.dynamics_obj.dynamic_state['theta_diff'] #center line의 reference point의 theta의 이전 time step과의 차이#
                    
                # breakpoint()
                if heading_angle <= -np.pi / 2 or heading_angle >= np.pi / 2:
                    self.reward += -1 * self.input_args.backward_penalty
                
                step_reward = self.reward - self.prev_reward
                if d_axis >= TRACK_WIDTH:
                    terminated = True
                    step_reward = -1 * self.input_args.terminate_penalty
                    # breakpoint()
                
                self.prev_reward = self.reward
                
                '''(2)-0 Centerline Reward'''
            elif self.input_args.reward_type.lower() == "center" and self.input_args.center_line_max_penalty != 0:
                pen_weight = d_axis / TRACK_WIDTH ## 중앙선으로부터 떨어진 거리 / 트랙 너비의 절반
                self.reward -= self.input_args.center_line_max_penalty * pen_weight
                step_reward = self.reward - self.prev_reward
                
                if d_axis >= TRACK_WIDTH:
                    terminated = True
                    step_reward = -1 * self.input_args.terminate_penalty
                    
                self.prev_reward = self.reward
                    
                """(2)-1 Baseline Reward - Nothing modified except for termination when leaving track"""
                
            elif self.input_args.reward_type.lower() == "baseline": 
                # 그냥 제일 기본적인 reward 제공 #
                step_reward = self.reward - self.prev_reward
                # print(d_axis, TRACK_WIDTH)
                if d_axis >= TRACK_WIDTH: # 트랙 밖으로 벗어난 순간 #
                    # breakpoint()
                    terminated = True
                    step_reward = -1 * self.input_args.terminate_penalty
                self.prev_reward = self.reward
            
            elif self.input_args.reward_type.lower() == "mpc":
                '''기존 CarRacing에서 설계했던 reward function이 이전 step에 비해서 reward가 얼마나 증/감 하였는지를 
                현재 action policy의 reward로 주는 이유는 ""누적되는 값"이기 때문이다.
                시간 개념이 들어가기 때문에 reward는 "빠른 lap time의 policy" 확보를 위해서 "시간이 지남에 따라 current reward 감소"에 의미를 둔다.
                하지만 지금같은 경우에는 현재 (state, action)을 수행했을떄의 누적 보상이 아닌 즉각 보상을 학습하고자 한다.
                따라서 이전과 비교한 step_reward를 고려할 필요가 없다.
                '''
                
                ec_weight = self.input_args.ec_weight # alpha
                etheta_weight = self.input_args.etheta_weight # beta
                e_c = self.dynamics_obj.dynamic_state['e_c'] # perpendicular distance from center line (should use absolute value?)
                ## "theta" is the theta value of the coordinate point of the closest center line from the current point
                e_theta = self.dynamics_obj.theta_diff # (property of dynamics object) movement difference from previous step regarding the curve coordinate
                # mpc_reward = (1/math.pow(e_c, 2)) * ec_weight + e_theta * etheta_weight
                ### 만약에 e_c가 중앙선과 가까울때 0에 수렴하는 값인데 -1제곱승을 구하면 무한대로 초반 값이 치솟게 되는 경향이 있을 수 있어서, 그냥 절댓값에 음수 처리를 하기로 함.
                mpc_reward = (-abs(e_c) * ec_weight) + (e_theta * etheta_weight)
                mpc_reward -= self.input_args.time_penalty
                
                mpc_reward_weight = self.input_args.mpc_reward_scaler
                if d_axis >= TRACK_WIDTH:
                    terminated = True
                    step_reward = -1 * self.input_args.terminate_penalty
                else:
                    step_reward = mpc_reward - self.prev_reward
                
                step_reward *= mpc_reward_weight
                    
                self.reward = mpc_reward
                self.prev_reward = self.reward
                
                
                '''(3) Small Margin Reward - Terminated when car leaves the track with a very small margin'''
            elif self.input_args.reward_type.lower() == "small":
                # print(d_axis, TRACK_WIDTH)
                if (d_axis >= TRACK_WIDTH):
                    diff = TRACK_WIDTH - d_axis
                    self.reward += -1 * self.input_args.tile_leave_weight * diff

                step_reward = self.reward - self.prev_reward

                if d_axis >= (TRACK_WIDTH * 1.2):
                    terminated = True
                    step_reward = -1 * self.input_args.terminate_penalty
                    
                self.prev_reward = self.reward

                '''(4) Border Reward - Terminated when car leaves a certain width of the border'''
            elif self.input_args.reward_type.lower() == "border": # 일정 border 이상으로 넘어가면 terminate #
                # print(d_axis, TRACK_WIDTH)

                if d_axis >= (TRACK_WIDTH + OUT_TRACK_LIMIT): # 트랙 밖에서의 state인 경우에는 안쪽으로 들어와야 reward가 커지게 됨#
                # if d_axis >= TRACK_WIDTH:
                    self.out_track_count += 1
                    diff_from_track_border = d_axis - TRACK_WIDTH
                    
                    if self.prev_diff_from_track_border is None:
                        self.prev_diff_from_track_border = diff_from_track_border
                    ##트랙쪽으로 다시 돌아올 수 있도록##
                    else: # 만약에 밖에 있는데 어쨌든 변화가 있는 경우라면, 더 트랙쪽으로 이동을 했다면 reward의 증가가 그만큼은 있어야 함 #
                        self.reward += (self.prev_diff_from_track_border - diff_from_track_border) * 10.
                        self.prev_diff_from_track_border = diff_from_track_border 
                
                    self.reward -= diff_from_track_border # 가까워질수록 reward에서 까이는 값은 적어지겠지? 하지만 동일한 tile에 대한 penalty를 이길 수 있을지는 모름 #
                    self.reward -= self.out_track_count
                    # if d_axis >= (TRACK_WIDTH + OUT_TRACK_LIMIT): # (절대 안되는 state임을 알기 위해서) 완전 한계 border 밖에까지 나간 경우에는 reward에서 100씩 뺌 #
                    #     self.reward -= 10 * self.out_track_count
                    # else:
                    #     self.reward -= self.out_track_count


                else: # 트랙의 밖에 있다가 다시 돌아온 경우 #
                    self.out_track_count = 0
                    self.prev_diff_from_track_border = None

                # 속도에 대한 reward (일정 속도 이상을 유지해야 함) #
                self.reward += self.dynamics_obj.vel * self.input_args.vel_weight

                step_reward = self.reward - self.prev_reward

                if self.out_track_count > 200:
                    step_reward = -1000 
                    terminated = True  
            else:
                raise UserWarning(f"NOT SUPPORTED REWARD TYPE : {self.input_args.reward_type.lower()}")
 
            

        self.prev_reward = self.reward
        self.step_reward = step_reward

        self._log_car_states()
        self._log_car_actions(action)
        self._log_actor_states()
        
        self.render_all()
        
        return self.state, step_reward, terminated, truncated, {}


        

    def _log_actor_states(self):
        self.actor_state_dict[self.t] = self.state
        
    def _log_car_actions(self, action):
        if action is not None:
            if self.use_beta_dist:
                new_action = (action[0] - 0.5) * 2
                self.action_dict['steer'].append(-new_action)
            else:
                self.action_dict['steer'].append(-action[0])
            self.action_dict['gas'].append(action[1])
            self.action_dict['brake'].append(action[2])
        
    def _get_state(self):
        """function that gets the necessary state that matches the observation state
        1) dynamic: input state parameters the same as the MPCC gets as input 
        2) lidar: distance from the track sides to the location of the car's center of mass
        3) car: actual car states
        ______________________________________
        4) track: track meta information (kappa, x, y etc) from the front view of the car
        5) feature_vec: feature vector of the top-down view BEV image of the track
        """
        state_array = np.array([])
        # state_dict = {}
        state_dict_for_debug = {}
        
        if 'dynamic' in self.observation_config:
            dynamic_state = self.dynamics_obj.run_dynamics(dt=1./FPS)
            # state_dict = copy.deepcopy(dynamic_state)
            for key in self.observation_config['dynamic']:
                state_array = np.hstack((state_array, np.array([dynamic_state[key]])))
            # vals = np.array(list(dynamic_state.values()))
            # state_array = np.hstack((state_array, vals))
            state_dict_for_debug['dynamic'] = dynamic_state
            
        if 'lidar' in self.observation_config:
            ##현재 위치로부터 lidar의 거리 정보만 반영
            #
            # sensory_state = get_sensory_state(car_obj=self.car, track_dict=self.track_dict)
            self.sensory_obj.step(car_yaw_rad=self.car.hull.angle, do_reverse=self.do_reverse) ## distances from all the lidar sensors to the wall of the track ##
            sensory_state = self.sensory_obj.sensor
            dist_arr = np.array([s[0] for s in sensory_state])
            state_array = np.hstack((state_array, dist_arr))
            
            state_dict_for_debug['lidar'] = sensory_state
        
        if 'car' in self.observation_config:
            '''
            vehicle's longitudinal velocity (v_x)
            turning rate
            front wheel steering angle
            forward observation vectors
            '''
            # car_x, car_y = self.car.hull.position # position in vector format # 
            force_arr = np.array([w.force for w in self.car.wheels])
            omega_arr = np.array([w.omega for w in self.car.wheels]) # omega value of all 4 wheels (vehicle's angular velocity) #
            
            # delta = self.car.wheels[0].steer # car heading angle in radians (the front two wheels have the same steering value) #
            delta = self.car.wheels[0].joint.angle # car front wheel steering angle 
            self.forward_obj.step()
            if self.input_args.use_rotated_forward:
                fo_vector_state = self.forward_obj.rotated_vectors
            else:
                fo_vector_state = self.forward_obj.vectors
            vec_arr = np.hstack(np.array(fo_vector_state))
            curvature_arr = np.array(self.forward_obj.curvature_arr)
            state_dict_for_debug['car_forward_vector'] = vec_arr
            state_dict_for_debug['car_omega_arr'] = omega_arr
            state_dict_for_debug['car_curvature'] = curvature_arr
            
            '''TabNet에서의 버그랑 비슷한데, 어쨌든 configuration list에 feature 배열이 입력으로 들어오는데
            이것들의 순서를 좀 맞춰 줘야 함.
            매번 이것 때문에 입력 state에 문제가 생길수는 없기 때문에...'''
            check = re.compile("omega_*")
            if len(list(filter(check.match, self.observation_config['car']))) > 0:
                state_array = np.hstack((state_array, omega_arr))

            check = re.compile("forward_*")
            if len(list(filter(check.match, self.observation_config['car']))) > 0:
                state_array = np.hstack((state_array, vec_arr)) 
        
            for val in self.observation_config['car']:

                if val == 'delta':
                    state_array = np.hstack( (state_array, np.array([delta])))
                ##### steer, gas, brake의 경우에는 이전의 action에 대한 정보를 입력으로 넣어줌.
                elif val == "steer": #전륜만 steer을 적용하기 때문에#
                    # state_array = np.hstack( (state_array, np.array([self.car.wheels[0].steer])))
                    state_array = np.hstack( (state_array, np.array([self.car.prev_steer])))
                    
                elif val == "gas": #후륜만 gas를 적용하기 때문에#
                    # state_array = np.hstack( (state_array, np.array([self.car.wheels[2].gas])))
                    state_array = np.hstack( (state_array, np.array([self.car.prev_gas])))
                    
                elif val == "brake":
                    # state_array = np.hstack( (state_array, np.array([self.car.wheels[0].brake])))
                    state_array = np.hstack( (state_array, np.array([self.car.prev_brake])))
                    
                elif val == "force":
                    state_array = np.hstack( (state_array, np.array([self.car.wheels[2].force])))
                
            # check = re.compile("force_*")
            # if len(list(filter(check.match, self.observation_config['car']))) > 0:
            #     state_array = np.hstack((state_array, force_arr)) 
            '''무조건 min-max value 정한거랑 state array define 순서를 맞춰주어야 함'''
            check = re.compile("curvature_*")
            if len(list(filter(check.match, self.observation_config['car']))) > 0:
                state_array = np.hstack((state_array, curvature_arr))
            # breakpoint()
        if 'track' in self.observation_config:
            track_state = get_track_state(car_obj=self.car, track_dict=self.track_dict)
            state_array = np.hstack((state_array, track_state))
            
        if 'feature_vec' in self.observation_config:
            feature_vec_state = get_feature_vec_state(car_obj=self.car, track_dict=self.track_dict)
            state_array = np.hstack((state_array, feature_vec_state))

       
        # if state_array.shape[0] != 33:
        #     breakpoint()
        
        preprocessed_state_array = preprocess_state(env=self, state=state_array)
    
        # return state_array
        # print(preprocessed_state_array)
        car_x, car_y = self.car.hull.position
        # breakpoint()
        state_dict_for_debug['car_x'] = car_x
        state_dict_for_debug['car_y'] = car_y
   
        pickle.dump(state_dict_for_debug, open('observation_space_debug.pkl', 'wb'))
        
        return preprocessed_state_array.astype(float)
    
    # def _get_obs(self):
    #     ## translates the environment's state into an observation ##
    #     return
    
    # def _get_info(self):
    #     ## implements auxiliary information returned by the 'step' and 'reset' function ##
    #     return
    
    def close(self):
        if hasattr(self, "window") and (self.window is not None):
            pygame.display.quit()
            self.isopen=False
            pygame.quit()
            

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from algorithm.pid.run_pid import PID_Controller, load_args, main_control
    
    args = load_args()
    args.steer_p = 1. /  TRACK_WIDTH
    args.steer_d = args.steer_p / 2
    args.steer_i = 0.
    args.gas_p = 1. / args.target_vel
    args.gas_d = 0.
    args.gas_i = 0.
    args.num_runs = 1
    args.log_path = "random_track"
    
    
    class inp_args:
        def __init__(self):
            self.lidar_deg = 15 #  30
            self.lidar_length = NAM_TRACK_WIDTH * 5
            
            self.num_vecs = 5
            self.theta_diff = 20.
            self.do_view_angle = False # True
            
    
            
            
    controller = PID_Controller(
        dt=1/NAM_FPS,
        steer_p=args.steer_p, steer_i=args.steer_i, steer_d=args.steer_d,
        gas_p=args.gas_p, gas_i=args.gas_i, gas_d=args.gas_d
    )
    
    INP_ARGS = inp_args()
    
    observation_config = {
        'dynamic': ['theta', 'e_c', 'e_phi', 'v_x', 'v_y', 'yaw_omega'], ##(1,1,1,1,1,1)## -> yaw_omega: 조향각의 각속도##
        'lidar': [f"lidar_{i}" for i in range(int(180 / INP_ARGS.lidar_deg))], ##(180 / lidar_deg)##
        'car': ['omega', 'delta', 'forward'] ##(4, 1, num_vecs)## -> omega: 바퀴 4개의 각속도
    }
    nam_race_env = Toy_CarRacing(observation_config=observation_config, 
                                 do_zoom=4.,
                                args=INP_ARGS)
    
    main_control(
        env=nam_race_env,
        controller=controller,  
        pid_args=args
    )    
# if __name__ == "__main__":
#     observation_config = {
#         'dynamic': {'theta': [], 'e_c':[], 'e_phi': [],
#                      'v_x': [], 'v_y': [], 'omega': []}
#     }
#     env = Toy_CarRacing(observation_config=observation_config, simple_circle_track=False)
#     env.reset()
#     while True:
#         ## give dummy values ##
#         steer = np.random.uniform(-1., 1.)
#         gas = np.random.uniform(0., 1.)
#         brake = np.random.uniform(0., 1.)
#         state, _, _, _, _ = env.step(action=(steer, gas, brake))
        
        
#         print(state)