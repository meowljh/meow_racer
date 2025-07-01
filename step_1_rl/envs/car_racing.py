import numpy as np
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(ROOT))) #008_
sys.path.append(ROOT)

from envs.deprecated_env_constants import *
from dynamic_params import Vehicle_Parameters_JW
from utils import (
    create_checkpoints, connect_checkpoints, check_connected_loop,
    find_hard_corner, create_tiles,
    gen_alpha, gen_beta,
    reverse_beta
)

import pygame
from pygame import gfxdraw


# import gym
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle # objects that are pickled and unpickled via their constructor arguments (쉽게 environment object를 객체 그 자체로 picklizing이 가능하게 함.) #
from car_dynamics import Car
import Box2D
from Box2D.b2 import fixtureDef, polygonShape, contactListener



class CarRacing(gym.Env, EzPickle):
    def __init__(self, 
                 observation_factors=['lidar'], # observation state로 어떤 값을 사용할지 #
                 random_track:bool=False, #트랙을 매번 랜덤하게 새롭게 생성 할지 말지#
                 lap_complete_percent:float=0.95):
        EzPickle.__init__(self, observation_factors, random_track, lap_complete_percent)
        self.observation_factors = observation_factors
        self.random_track = random_track
        self.lap_complete_percent = lap_complete_percent
        
        self._init_colors()
        
        self.world = Box2D.b2World((0, 0))
        self.screen = None
        self.surf = None
        self.road = None
        self.clock = None
        self.car = None
        
        self.reward = 0.0
        self.prev_reward = 0.0
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape = polygonShape(vertices=[(0,0), (1,0), (1,-1), (0,-1)])
        )
        
        self.vehicle_param = Vehicle_Parameters_JW()
        
        self.action_space = spaces.Box(
            low = np.array([self.vehicle_param.delta_min, self.vehicle_param.torque_min]),
            high = np.array([self.vehicle_param.delta_max, self.vehicle_param.torque_max]),
            dtype=np.float64
        ) # delta(조향각), torque(토크값) #
        self.observation_space = self._set_observation_space()
    
    def _set_observation_space(self):
        pass
    
    def _init_colors(self):
        self.grass_color = GRASS_COLOR
        self.road_color = ROAD_COLOR
        self.bg_color = BG_COLOR
        
    def _render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        if "t" not in self.__dict__:
            return # the reset() function is not called yet #
        
        self.surf = pygame.Surface((SCREEN_W, SCREEN_H))

        # compute transformation and zoom #
        scale = NAM_SCALE if not self.random_track else SCALE
        ## self.t는 계속 증가하고, 1.0 / FPS만큼 증가한다. 따라서 처음에만 zoom = 0.1 * scale임. ##
        zoom = 0.1 * scale * max(1 - self.t, 0) + ZOOM * scale * min(self.t, 1)
        angle = -self.car.hull.angle
        
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
 
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (SCREEN_W / 2 + trans[0], SCREEN_H / 4 + trans[1])
        
        self._render_road(zoom, trans, angle)
        self.car.draw(
                surface=self.surf, zoom=zoom, translation=trans, 
                angle=angle,
                draw_particles=False
            )
        self.surf = pygame.transform.flip(self.surf, False, True)
        
        self._display()
        
    def _display(self):
        pygame.event.pump()
        self.clock.tick(FPS)
        assert self.screen is not None
        self.screen.fill(0)
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        
        
    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD if self.random_track else NAM_PLAYFIELD
        field = [
            (bounds, bounds), (bounds, -bounds), (-bounds, -bounds), (-bounds, bounds)
        ] 
        # draw background #
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )
        # draw grass patches #
        grass = []
        grass_dim = GRASS_DIM if self.random_track else NAM_GRASS_DIM
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
        max_shape_dim = NAM_MAX_SHAPE_DIM if not self.random_track else MAX_SHAPE_DIM
        if not clip or any(
            (-max_shape_dim <= coord[0] <= SCREEN_W + max_shape_dim) and
            (-max_shape_dim <= coord[1] <= SCREEN_H + max_shape_dim) for coord in poly
        ):
            gfxdraw.aapolygon(surface, poly, color)
            gfxdraw.filled_polygon(surface, poly, color)
            
    def _create_nam_track(self):
        nam_track = pickle.load(open(f'{ROOT}/statics/nam_c_track.pkl', 'rb'))
        x, y, phi = np.array(nam_track['x']), np.array(nam_track['y']), np.array(nam_track['phi'])
        beta = gen_beta(phi)
        
        self.phi = phi  
        
        border = find_hard_corner(beta_arr=phi, is_nam=True)
        road_poly, road  = create_tiles(
            box_world=self.world, box_tile=self.fd_tile,
            X=x, Y=y, beta=beta, border_arr=border,
            is_nam=True
        )
        alpha_arr = gen_alpha(x, y)
        track = np.vstack((alpha_arr, beta, x, y)).T # (N, 4)
 
        return road_poly, road, track
    
    def _create_track_random(self):
        done = False
        while not done:
            checkpoints, start_alpha = create_checkpoints()
            track = connect_checkpoints(checkpoints=checkpoints)
            is_valid, i1, i2, track = check_connected_loop(track=track, START_ALPHA=start_alpha)

            if is_valid:
                beta_arr = np.array(track).T[1]
                border = find_hard_corner(beta_arr, is_nam=False)
                road_poly, road  = create_tiles(
                    box_world = self.world, box_tile=self.fd_tile,
                    X=np.array(track).T[2],
                    Y=np.array(track).T[3],
                    beta=beta_arr,
                    border_arr=border,
                    is_nam=False
                )
                done = True
                return road_poly, road, track
            else:
                continue
 
    
    def _destroy(self):
        if not self.road:
            return
        for t in self.road: # tiles in road #
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy() # must be implemented in the Car object #
        
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy() # removes all the road tiles in the Box2D World and the car object #
        
        self.reward = 0.0
        self.prev_reward = 0.0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        
        if self.random_track:
            road_poly, road, track = self._create_track_random()
        else:
            road_poly, road, track = self._create_nam_track()

        self.track = track
        self.road = road
        self.road_poly = road_poly
        
        self.car = Car(world=self.world,
                       init_angle=self.track[0][1], # if self.random_track else self.phi[0],
                       init_x=self.track[0][2],
                       init_y=self.track[0][3])
        
        self.render()
        
        # return self.step(None)[0], {}

    def step(self, action):
        assert self.car is not None
        if action is not None:
            self.car.apply_dynamics(action)
        
    
    def render(self):
        return self._render()
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
    
if __name__ == "__main__":
    race_env = CarRacing(random_track=False)
    # race_env = CarRacing(random_track=True)
    race_env.reset() 
    while True:
        race_env.render()
