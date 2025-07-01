"""gym_car_racing_nam.py
- Used for testing the RL racing agent.
- Agent will be trained on random tracks as the CarRacing environment of the openAI gym, and will be tested it's general-ness on the Nam-C track

"""
from collections import defaultdict
import numpy as np
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(ROOT)))  
sys.path.append(ROOT)

import pickle
from envs.gym_car_constants import *
from envs.utils import (
    calculate_theta, 
    # calculate_phi,
    calculate_track_phi,
    # calculate_curvature, 
    calculate_track_curvature,
    get_track_boundary, 
    find_hard_corner, create_tiles,
    gen_alpha, gen_beta, resplit,
    
)
import pygame
from pygame import gfxdraw


# import gym
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding# objects that are pickled and unpickled via their constructor arguments (쉽게 environment object를 객체 그 자체로 picklizing이 가능하게 함.) #
from envs.gym_car import Toy_Car
import Box2D
from Box2D.b2 import fixtureDef, polygonShape, contactListener
from envs.org_env import (
    CarRacing, FrictionDetector, # contactListener is necessary to check how many tiles the vehicle has passed #
    PYGAME_STATE_DICT,
    TileLocation_Detector
)
from envs.gym_car_racing import Toy_CarRacing
from envs.gym_car_jw import JW_Toy_Car

class Toy_CarRacing_NAM(Toy_CarRacing):
    def __init__(self, 
                 observation_config:dict, # observation state로 어떤 값을 사용할지 #
                 lap_complete_percent:float=0.95,
                 do_zoom:float=None,
                 args:dict=None):
        super(CarRacing, self).__init__(
            render_mode='human', lap_complete_percent=lap_complete_percent, verbose=False,
            domain_randomize=False, continuous=True, observation_config=observation_config
        )
        EzPickle.__init__(self, observation_config,  lap_complete_percent)
        
        self.input_args = args
        
        self.use_beta_dist = args.use_beta_dist
        
        self.do_reverse = False
        # self.do_reverse = self.input_args.do_reverse_nam > 0.
        
        self.num_episode = 0
        
        self.track_width = NAM_TRACK_WIDTH
        
        self.observation_config = observation_config
        self.domain_randomize = False
        self.random_track = False
        self.lap_complete_percent = lap_complete_percent
        self.do_zoom = do_zoom
        self.simple_circle_track = False
        
        self.continuous = True
        self.verbose = False
        self.render_mode = 'human'
        self._init_colors()
        
        
        self.status_queue = []
        
        if self.input_args.oscillation_penalty > 0:
            self.contactListener_keepref = TileLocation_Detector(self,
                                                                 self.lap_complete_percent,
                                                                 max_reward_tile=self.input_args.max_reward_tile)
        else:
            self.contactListener_keepref = FrictionDetector(self, 
                                                        self.lap_complete_percent,
                                                        max_reward_tile=self.input_args.max_reward_tile)
        
        ## the world class manages all physic entities, dynamic simulation, and asynchronous queries ##
        self.world = Box2D.b2World(gravity=(0,0), contactListener=self.contactListener_keepref)
        self.screen = None
        self.surf = None
        self.road = None
        self.clock = None
        self.car = None
        
        self.is_nam = True
        
        self.reward = 0.0
        self.prev_reward = 0.0
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape = polygonShape(vertices=[(0,0), (1,0), (1,-1), (0,-1)])
        )
         
        ## added attributes for reward & penalty ##
        self.same_tile_count = 0
        self.on_same_tile_penaly: bool = False
        self.out_track_count:int = 0
        # self.prev_diff_from_track_border = None

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
        self.actor_state_dict = {}
            
        self.state = PYGAME_STATE_DICT['RUNNING']
    
    def render_all(self, render_mode:str='human'):
        self._render(render_mode='render_mode')
        
    def _render(self, render_mode:str='human'):
        pygame.font.init()
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((NAM_WINDOW_W, NAM_WINDOW_H))
            pygame.display.set_caption(self.input_args.screen_title)
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        if "t" not in self.__dict__:
            return # the reset() function is not called yet #
        
        self.surf = pygame.Surface((NAM_WINDOW_W, NAM_WINDOW_H))

        assert self.car is not None
        
        # compute transformation and zoom #
        scale = NAM_SCALE  
        if self.do_zoom is not None:
            zoom = self.do_zoom
        else:
            ## self.t는 계속 증가하고, 1.0 / FPS만큼 증가한다. 따라서 처음에만 zoom = 0.1 * scale임. ##
            zoom = 0.1 * scale * max(1 - self.t, 0) + ZOOM * scale * min(self.t, 1)
            
        if self.input_args.do_view_angle:
            angle = -self.car.hull.angle
        else:
            angle = 0
        
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
 
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (NAM_WINDOW_W / 2 + trans[0], NAM_WINDOW_H / 4 + trans[1])
        
        self._render_road(zoom, trans, angle)
        self.car.draw(
                surface=self.surf, zoom=zoom, translation=trans, 
                angle=angle,
                draw_particles=True)
        self.sensory_obj.draw(
            screen=self.surf,
            zoom=zoom,
            translation=trans,
            angle=angle,
        )
        self.forward_obj.draw(
            screen=self.surf,
            zoom= zoom,
            translation=trans,
            angle=angle
        ) 
        self.surf = pygame.transform.flip(self.surf, False, True)
        
        self._render_actions(W=NAM_WINDOW_W, H=NAM_WINDOW_H)
        self._render_text(W=NAM_WINDOW_W, H=NAM_WINDOW_H)
        self._display()
    

        
    def _display(self):
        pygame.event.pump()
        self.clock.tick(FPS)
        assert self.screen is not None
        self.screen.fill(0)
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        
        
    def _render_road(self, zoom, translation, angle):
        bounds = NAM_PLAYFIELD
        field = [
            (bounds, bounds), (bounds, -bounds), (-bounds, -bounds), (-bounds, bounds)
        ] 
        # draw background #
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )
        # draw grass patches #
        grass = []
        grass_dim = NAM_GRASS_DIM
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (grass_dim * x + grass_dim, grass_dim * y + 0),
                        (grass_dim * x + 0, grass_dim * y + 0),
                        (grass_dim * x + 0, grass_dim * y + grass_dim),
                        (grass_dim * x + grass_dim, grass_dim * y + grass_dim),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )
        # draw road track #
        for poly, color in self.road_poly:
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
 
        
    def _draw_colored_polygon(self, surface, poly, color, zoom, translation, angle, clip:bool=True):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ] 
        max_shape_dim = NAM_MAX_SHAPE_DIM 
        if not clip or any(
            (-max_shape_dim <= coord[0] <= NAM_WINDOW_W + max_shape_dim) and
            (-max_shape_dim <= coord[1] <= NAM_WINDOW_H + max_shape_dim) for coord in poly
        ):
            gfxdraw.aapolygon(surface, poly, color)
            gfxdraw.filled_polygon(surface, poly, color)
            

 
    def _destroy(self):
        if not self.road:
            return
        for t in self.road: # tiles in road #
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy() # must be implemented in the Car object #
        
        
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

        self._destroy() 
        # self.main_reset(seed=seed) 
        if self.input_args.oscillation_penalty > 0:
            self.world.contactListener_bug_workaround = TileLocation_Detector(
                self, self.lap_complete_percent, self.input_args.max_reward_tile
            )
        else:
            self.world.contactListener_bug_workaround = FrictionDetector(
                self, self.lap_complete_percent,
                max_reward_tile=self.input_args.max_reward_tile
            )
        self.world.contactListener = self.world.contactListener_bug_workaround
        
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.car_state = defaultdict(list)
        self.action_state_dict = defaultdict(list)
        self.same_tile_count = 0
        self.out_track_count = 0
        self.do_reverse =False
        # self.do_reverse = self.input_args.do_reverse_nam > 0.
        
        self.new_lap = False
        self.car_left_track = False
        self.is_backward = False
        self.on_same_tile_penalty = False
        self.track_dict = {}
        self.road_poly = []
        
        road_poly, road, track, track_dict = self._create_nam_track()
        
        self.track_dict = track_dict
        self.track = track
        self.road = road
        self.road_poly = road_poly
        
        ## added attributes for reward & penalty ##
        self.on_same_tile_penaly: bool = False
        self.out_track_count:int = 0

        ## [minor typo fix] -> JW evaluation할 때 매번 실패한다 싶었는데 보니까 use_jw인데 Toy_Car 객체를 불러오고 있었음. 수정 완 ##
        if self.input_args.use_jw:
            self.car = JW_Toy_Car(world=self.world,
                       init_angle=self.track_dict['beta'][0], # if self.random_track else self.phi[0],
                        # init_angle=nam_init_angle,
                       init_x=self.track_dict['x'][0],
                       init_y=self.track_dict['y'][0],
                       use_beta_dist=self.use_beta_dist)
        else:
            self.car = Toy_Car(world=self.world,
                       init_angle=self.track_dict['beta'][0], # if self.random_track else self.phi[0],
                        # init_angle=nam_init_angle,
                       init_x=self.track_dict['x'][0],
                       init_y=self.track_dict['y'][0])
        # if self.hasattr("frames"):
        if hasattr(self, "frames"):
            self.frames = []
            
        self._setup_additional()
        
        self.render_all()
        
        return self.step(None)[0], {}
 
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
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])
        
        self.car.step(1. / FPS, 
                      new_friction=self.input_args.new_friction_limit) 
        self.world.Step(1. / NAM_FPS, 6*30, 2*30) 
        
        self.t += 1. / NAM_FPS
        
        self.state = self._get_state()
        
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
            self.reward -= self.input_args.time_penalty  

            '''(1)-2. Common Reward - Same Tile Count Penalty'''
            if self.input_args.same_tile_penalty > 0:
                self.reward -= self.same_tile_count  
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
                if self.dynamics_obj.dist_diff < self.input_args.min_movement: ## 최소한의 움직임에 대한 threshold를 남양 트랙에 대해서는 바꿔줘야 할지도 모름 
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
                if (self.input_args.center_line_far_max_reward_corner > 0) and (self.track_dict['straight'][close_tile_idx] == False):
                    self.reward += percentage * self.input_args.center_line_far_max_reward_corner
                else:
                    self.reward += percentage * self.input_args.center_line_far_max_reward
                
            if self.input_args.oscillation_max_penalty > 0.:
                is_oscillate, oscillate_val = self._calc_oscillation_value()
                oscillate_percent = oscillate_val / (TRACK_WIDTH * 2)
                close_x, close_y = self.dynamics_obj.ref_x, self.dynamics_obj.ref_y
                close_idx = self.dynamics_obj.ref_index
                if is_oscillate and self.track_dict['straight'][close_idx] == True:
                    self.reward -= self.input_args.oscillation_max_penalty * oscillate_percent
                    
                if is_oscillate and (self.input_args.oscillation_max_reward_corner > 0) and (self.track_dict['straight'][close_idx] == False):
                    self.reward += self.input_args.oscillation_max_reward_corner * oscillate_percent
                    
                '''(2) AM Reward (reward on velocity)'''
            if self.input_args.reward_type.lower() == "am":
                velocity = self.car.hull.linearVelocity
                normal_vector = self.car.hull.GetWorldVector((0, 1)) # 횡방향 벡터
                lateral_v, longitude_v = velocity # (횡속 / 종속)
                # longitude_v_proj = np.dot(np.array(normal_vector), np.array(velocity))
                # longitude_v_proj = b2Dot(velocity, normal_vector)
                
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
                self.reward += math.cos(heading_angle) * longitude_v_proj
                
                if self.input_args.am_with_theta_reward:
                    self.reward += self.dynamics_obj.dynamic_state['theta_diff'] #center line의 reference point의 theta의 이전 time step과의 차이#
                    
                # breakpoint()
                """반대로 이동하는 경우"""
                if heading_angle <= -np.pi / 2 or heading_angle >= np.pi / 2:
                    self.reward += -1 * self.input_args.backward_penalty
                
                step_reward = self.reward - self.prev_reward
                if d_axis >= TRACK_WIDTH:
                    terminated = True
                    step_reward = -1 * self.input_args.terminate_penalty
                
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
                
                '''(2) Baseline Reward - Nothing modified except for termination when leaving track'''
            elif self.input_args.reward_type.lower() == "baseline":  
                step_reward = self.reward - self.prev_reward 
                if d_axis >= TRACK_WIDTH:  
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
                if (d_axis >= TRACK_WIDTH):
                    diff = TRACK_WIDTH - d_axis
                    self.reward += -1 * self.input_args.tile_leave_weight * diff

                step_reward = self.reward - self.prev_reward

                if d_axis >= (TRACK_WIDTH * 1.2):
                    terminated = True
                    step_reward = -1 * self.input_args.terminate_penalty
                    
                self.prev_reward = self.reward

                '''(4) Border Reward - Terminated when car leaves a certain width of the border'''
            elif self.input_args.reward_type.lower() == "border":  

                if d_axis >= (TRACK_WIDTH + OUT_TRACK_LIMIT): 
                    self.out_track_count += 1
                    diff_from_track_border = d_axis - TRACK_WIDTH
                    
                    if self.prev_diff_from_track_border is None:
                        self.prev_diff_from_track_border = diff_from_track_border
 
                    else:  
                        self.reward += (self.prev_diff_from_track_border - diff_from_track_border) * 10.
                        self.prev_diff_from_track_border = diff_from_track_border 
                
                    self.reward -= diff_from_track_border  
                    self.reward -= self.out_track_count 

                else: 
                    self.out_track_count = 0
                    self.prev_diff_from_track_border = None
 
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
        
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from algorithm.pid.run_pid import PID_Controller, load_args, main_control
    
    args = load_args()
    args.steer_p = 1. / NAM_TRACK_WIDTH
    args.steer_d = args.steer_p / 2
    args.steer_i = 0.
    args.gas_p = 1. / args.target_vel
    args.gas_d = 0.
    args.gas_i = 0.
    args.num_runs = 1
    args.log_path = "nam_track"
    
    class inp_args:
        def __init__(self):
            self.lidar_deg = 15
            self.lidar_length = NAM_TRACK_WIDTH * 5 # max(NAM_WINDOW_W, NAM_WINDOW_H)
            self.theta_diff = 20.
            self.do_view_angle = False # True
            
            
    controller = PID_Controller(
        dt=1/NAM_FPS,
        steer_p=args.steer_p, steer_i=args.steer_i, steer_d=args.steer_d,
        gas_p=args.gas_p, gas_i=args.gas_i, gas_d=args.gas_d
    )
    
    observation_config = {
        'dynamic': ['theta', 'e_c', 'e_phi', 'v_x', 'v_y', 'omega'],
        'lidar': [], 
        'car': []
    }
    nam_race_env = Toy_CarRacing_NAM(observation_config=observation_config, do_zoom=4.,
                                     args=inp_args())
    
    main_control(
        env=nam_race_env,
        controller=controller,  
        pid_args=args
    )