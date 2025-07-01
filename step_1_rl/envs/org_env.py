__credits__ = ["Andrea PIERRÉ"]

import math
from typing import Optional, Union
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

# import gym
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

from gym_car_constants import *
from envs.utils import get_track_border_limit, create_tiles

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gym[box2d]`"
    )

from detectors import (
    FrictionDetector,
    TileLocation_Detector
)
PYGAME_STATE_DICT = {
    'RUNNING': 1,
    'PAUSE': 2,
    'QUIT': 3
}


        
class CarRacing(gym.Env, EzPickle):
    """
    ### Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gym/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ### Action Space
    If continuous:
        There are 3 actions: steering (-1 is full left, +1 is full right), gas, and breaking.
    If discrete:
        There are 5 actions: do nothing, steer left, steer right, gas, brake.

    ### Observation Space
    State consists of 96x96 pixels.

    ### Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited,
    where N is the total number of tiles visited in the track. For example,
    if you have finished in 732 frames, your reward is
    1000 - 0.1*732 = 926.8 points.

    ### Starting State
    The car starts at rest in the center of the road.

    ### Episode Termination
    The episode finishes when all of the tiles are visited. The car can also go
    outside of the playfield - that is, far off the track, in which case it will
    receive -100 reward and die.

    ### Arguments
    `lap_complete_percent` dictates the percentage of tiles that must be visited by
    the agent before a lap is considered complete.

    Passing `domain_randomize=True` enables the domain randomized variant of the environment.
    In this scenario, the background and track colours are different on every reset.

    Passing `continuous=False` converts the environment to use discrete action space.
    The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    ### Reset Arguments
    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.
    Example usage:
    ```py
        env = gym.make("CarRacing-v1", domain_randomize=True)

        # normal reset, this changes the colour scheme by default
        env.reset()

        # reset with colour scheme change
        env.reset(options={"randomize": True})

        # reset with no colour scheme change
        env.reset(options={"randomize": False})
    ```

    ### Version History
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ### References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ### Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self, 
        
        render_mode: Optional[str] = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        
    
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self.sensory_obj = None
    
        self._init_colors()
        self.do_reverse = False

        self.track_turn_rate = float(np.clip(np.random.rand(1) * TRACK_TURN_RATE,
                                        a_min=math.pi * 0.75, 
                                        a_max=TRACK_TURN_RATE))

        self.min_track_turn_rate = self.track_turn_rate - 0.01

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Optional[Car] = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.status_queue = []
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.status = PYGAME_STATE_DICT['RUNNING']
        
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.render_mode = render_mode
        
        self.neg_reward_check = 0
    
    def _save_frame(self,):
        W, H = self.surf.get_width(), self.surf.get_height()
        arr = np.zeros((W, H, 3)).astype(np.uint8)
        pygame.pixelcopy.surface_to_array(
            array=arr, surface=self.surf, kind='P'
        )
        
        
        if 'frames'  in self.__dict__:
            self.frames.append(arr)
        else:
            self.frames = [arr]
            
    def _make_track_limit(self, X, Y, PHI, BETA):
        left_limit, right_limit = get_track_border_limit(cX=X, cY=Y, phi_arr=PHI)
        
        left_limit_poly, left_limit_road = create_tiles(
            box_world=self.world, box_tile=self.fd_tile,
            X=left_limit[:, 0], Y=left_limit[:, 1],
            beta=BETA, border_arr=[], is_nam=self.is_nam, width=OUT_TRACK_LIMIT/5, color=np.array([0,0,0]),
            additional_dict = {'name': 'left_limit', 'limit_visit_count': 0}
        )
        right_limit_poly, right_limit_road = create_tiles(
            box_world=self.world, box_tile=self.fd_tile,
            X=right_limit[:, 0], Y=right_limit[:, 1],
            beta=BETA, border_arr=[], is_nam=self.is_nam, 
            width=OUT_TRACK_LIMIT/5, color=np.array([0,0,0]),
            additional_dict = {'name': 'right_limit', 'limit_visit_count': 0}
        )
        
        return left_limit_poly, left_limit_road, right_limit_poly, right_limit_road
    
    def _check_terminate(self):
        if self.neg_reward_check > self.neg_reward_limit:
            self.neg_reward_check = 0
            return True
        return False
    
    def _pause(self):
        pause_text = pygame.font.SysFont(pygame.font.get_default_font(), 300).render('PAUSE', True, pygame.color.Color('White'))
        self.screen.fill(0) 
        self.screen.blit(pygame.Surface.convert_alpha(self.surf ), (0, 0))
        self.screen.blit(pause_text, (WINDOW_W / 4, WINDOW_H / 2))
        pygame.display.flip()
        
    def _quit(self):
        pygame.display.quit()
        pygame.quit()
        sys.exit()
        
    def _track_events(self, status_dict):
        # if len(status_list) == 0:
        #     return None
        
        # e = status_list[-1]
            
        # if e == "EXIT": #keyboard (f3)
        if status_dict['EXIT'] > 0:
            self.status = PYGAME_STATE_DICT['QUIT']
            self._quit()
            return -1
        
        # elif e == 'PAUSE': #keyboard (f1)
        elif status_dict['PAUSE'] > 0:
            # print("Key pressed <PAUSE> !!! ") 
            self.status = PYGAME_STATE_DICT['PAUSE'] 
            return 0
        
        # elif e == "RUNNING": #keyboard (f2)
        elif status_dict['RUNNING'] > 0:
            # print("Key pressed <START> !!! ")
            self.status = PYGAME_STATE_DICT['RUNNING'] 
     
            return 1 
        
        return None
    
    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _connect_reverse_track(self):
        x = 1.5 * TRACK_RAD
        y = 0
        beta = 0
        laps = 0
        dest_i = 0  
        reverse_track = []
        no_freeze = 2500
        visited_other_side = False

        while True:
            alpha = math.atan2(y, x)    
    
            #원의 반대편 방문했고 다시 alpha값이 0, 혹은 음수가 된 경우에는 새로운 lap 시작
            if visited_other_side and alpha <= 0:
                laps += 1
                visited_other_side = False
        
            #원의 반대편을 방문한 상태
            if alpha > 0:
                visited_other_side = True
        
    
            while True:
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = self.checkpoints[dest_i % len(self.checkpoints)]
                    if dest_alpha >= math.pi:
                        dest_alpha = dest_alpha - 2 * math.pi
                    if alpha * dest_alpha < 0:
                        dest_alpha -= 2 * math.pi
                    if alpha >= dest_alpha:
                        failed = False
                        break
                    dest_i -= 1
                    if dest_i % len(self.checkpoints) == 0:
                        break
                if not failed:
                    break
        
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
    
            p1x = r1y
            p1y = -r1x
    
            dest_dx = x - dest_x  # dest_x - x
            dest_dy = y - dest_y # dest_y - y
    
            ## proj는 현재 위치로부터 목적지 checkpoint까지의 방향의 벡터
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi 
        
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi

            prev_beta = beta
            proj *= SCALE
    
            if proj > self.min_track_turn_rate:
                beta -= min(self.track_turn_rate,
                    abs(0.001 * proj))
            if proj < -self.min_track_turn_rate:
                beta += min(self.track_turn_rate,
                    abs(0.001 * proj))
        
    
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
    
            reverse_track.append(
                (alpha, 0.5*(prev_beta + beta), x, y)
            )
        
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break
            
        return reverse_track

    def _create_track(self): 
        ##########################################################################
        ###################### STEP 1: Create checkpoints ########################
        #checkpoints will be the corner peak points of the generated track #######
        
        checkpoints = []
        from step_1_rl.envs.gym_car_constants import CHECKPOINTS
        p = np.random.rand(1)
        
        if self.input_args.do_reverse > 0:
            if p < self.input_args.do_reverse:
                self.do_reverse = True
            else:
                self.do_reverse = False
                
                
        if self.input_args.random_checkpoints:
            if self.input_args.num_random_checkpoints == -1:
                m, M = -7, 7
            else:
                m, M = -self.input_args.num_random_checkpoints, self.input_args.num_random_checkpoints
            ##생각보다 적은 수의 checkpoint를 사용해서 트랙을 구성할 때 남양 트랙과 비슷한 모습을 띌 수 있을 듯
            # ckpts = np.random.choice([i for i in range(CHECKPOINTS-7, CHECKPOINTS+7)])
            ckpts = np.random.choice([i for i in range(CHECKPOINTS+m, CHECKPOINTS+M)])
            # ckpts = np.random.choice([i for i in range(CHECKPOINTS-5, CHECKPOINTS+5)])
            CHECKPOINTS = ckpts
        else:
            CHECKPOINTS = CHECKPOINTS # number of checkpoints
            
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
        
            # rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            #### randomly chooses the radius from the center point to the c번째 checkpoint ####
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD * 2.)

            if c == 0:
                alpha = 0
                # rad = 1.5 * TRACK_RAD
                rad = self.np_random.uniform(TRACK_RAD * (3/4), TRACK_RAD * 1.5)
                start_rad = rad
 
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                # rad = 1.5 * TRACK_RAD
                rad = start_rad

        
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        
        self.checkpoints = checkpoints
        self.road = []

        if self.do_reverse:
            track = self._connect_reverse_track()
            print("reverse!")
            # breakpoint()
        else:
            # Go from one checkpoint to another to create track
            x, y, beta = 1.5 * TRACK_RAD, 0, 0
            dest_i = 0
            laps = 0
            track = []
            no_freeze = 2500
            visited_other_side = False
            while True:
                alpha = math.atan2(y, x)
                if visited_other_side and alpha > 0:
                    laps += 1
                    visited_other_side = False
                if alpha < 0:
                    visited_other_side = True
                    alpha += 2 * math.pi

                while True:  # Find destination from checkpoints
                    failed = True

                    while True:
                        dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                        if alpha <= dest_alpha:
                            failed = False
                            break
                        dest_i += 1
                        if dest_i % len(checkpoints) == 0:
                            break

                    if not failed:
                        break

                    alpha -= 2 * math.pi
                    continue

                r1x = math.cos(beta)
                r1y = math.sin(beta)
                p1x = -r1y
                p1y = r1x
                dest_dx = dest_x - x  # vector towards destination
                dest_dy = dest_y - y
                # destination vector projected on rad:
                proj = r1x * dest_dx + r1y * dest_dy
                while beta - alpha > 1.5 * math.pi:
                    beta -= 2 * math.pi
                while beta - alpha < -1.5 * math.pi:
                    beta += 2 * math.pi
                prev_beta = beta
                proj *= SCALE
                if proj > self.min_track_turn_rate:
                    beta -= min(self.track_turn_rate, abs(0.001 * proj))
                if proj < -self.min_track_turn_rate:
                    beta += min(self.track_turn_rate, abs(0.001 * proj))
   
                x += p1x * TRACK_DETAIL_STEP
                y += p1y * TRACK_DETAIL_STEP
                track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
                if laps > 4:
                    break
                no_freeze -= 1
                if no_freeze == 0:
                    break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        if self.do_reverse:
            track = track[::-1]
        
        if self.input_args.random_start:
            idx = np.random.randint(0, len(track)-1)
            if idx == len(track)-1:
                track = [track[-1]] + track[:-1]
            if idx != 0:
                track = track[:idx] + track[idx:]
                
            
        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        # if well_glued_together > TRACK_DETAIL_STEP:
        if well_glued_together > TRACK_DETAIL_STEP_CHECK:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                # good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                good &= abs(beta1 - beta2) > self.track_turn_rate * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
            
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]
    
 
        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.num_visited = 0
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track
        return True

    def _save_frames(self, fname):
        pygame.image.save(self.screen, fname)
        
    def super_reset(self, seed):
        super().reset(seed=seed)
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        # if self.do_reverse:
        #     self.car = Car(self.world, init_angle = 180 + self.track[0][1], init_x=self.track[0][2], init_y=self.track[0][3])
        # else:
        #     self.car = Car(self.world, *self.track[0][1:4])
        self.car = Car(self.world, *self.track[0][1:4], do_reverse=self.do_reverse)
        
        self.frames = []
        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        step_reward = 0
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
                truncated = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        else:
            return self._render(W=WINDOW_W, H=WINDOW_H, mode=self.render_mode)
        
    def _render_text(self,W=WINDOW_W, H=WINDOW_H):
        font = pygame.font.Font(pygame.font.get_default_font(), 25)
        # text = font.render("%.2f" % self.reward, True, (255, 255,255), (0,0,0))
        if hasattr(self, "step_reward"):
            reward_render = self.step_reward
        else:
            reward_render = self.reward
        text = font.render("step " + "%.4f" % reward_render , True, (255,255,255), (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (80, H - W * 2.5 / 40.)
        self.surf.blit(text, text_rect)

        text = font.render("%.4f" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (80, H-W*5./40.)
        self.surf.blit(text, text_rect)
        
        # for key, value in self.car_states.items():
        if len(self.car_state) == 0:
            return
        diff = 100 / (len(self.car_state))
        font = pygame.font.Font(pygame.font.get_default_font(), int(diff))
        
        for i, (key, value) in enumerate(self.car_state.items()):
            text = font.render(key + "    %.3f" %value[-1], True, (255, 255, 255), (0, 0, 0))
            text_rect = text.get_rect()
            # text_rect.center = (60, NAM_WINDOW_H - NAM_WINDOW_W * 2.5 / 20.)
            text_rect.center = (200, H-50-(diff * (i+1)))
            self.surf.blit(text, text_rect)
        
        for i, (key, value) in  enumerate(self.dynamics_obj.dynamic_state.items()):
            text = font.render(key + "    %.3f" %value, True, (255,255,255), (0,0,0))
            text_rect = text.get_rect()
            text_rect.center = (600, H-50-(diff* (i+1)))
            self.surf.blit(text, text_rect)        
 
    def _render_actions(self, W, H):
        ## action, speed, reward 등 정보 나타낼 수 있도록 검은색 직사각형을 우선 blit로 그려 넣음 ##
        s = W / 30.0 # 40.0
        h = H / 30.0 # 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon) 
        
        action_color_dict = {
            'gas': (255,0,0), 'brake': (0, 255,0), 'steer': (0, 0, 255)
        }
        font = pygame.font.Font(pygame.font.get_default_font(), 10)
        place = 12
        for key, value in self.action_dict.items():
            
            points = self.vertical_ind(place=place, val=value[-1],
                                       s=s, 
                                       h=h, 
                                       H=H)
            pygame.draw.polygon(self.surf, points=points, color=action_color_dict[key])
            text = font.render(f"{key}", True, (255,255,255), (0,0,0))    
            text_rect = text.get_rect()
            cx = (points[0][0] + points[2][0]) / 2
            cy = min(np.array(points).T[1]) - 20.
            
            text_rect.center = (cx, cy)
            self.surf.blit(text, text_rect)
            text = font.render("%.3f" %value[-1], True, (255,255,255), (0,0,0))
            text_rect = text.get_rect()
            cy = min(points[0][1], points[2][1]) -  30.
            text_rect.center = (cx, cy)
            self.surf.blit(text, text_rect)
            place += 2
            
            
    def _render(self, W, H, mode: str, zoom:int=None):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption(self.input_args.screen_title)
            self.screen = pygame.display.set_mode((W, H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.status_queue.extend(pygame.event.get())
        # print(self.status_queue)
        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((W, H))

        assert self.car is not None
        # computing transformations
        if self.input_args.do_view_angle:
            angle = -self.car.hull.angle
        else:
            angle = 0
        
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1) if zoom is None else zoom
        
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (W / 2 + trans[0], H / 4 + trans[1])
        
        self.zoom = zoom
        self.translation = trans
        self.angle = angle

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            # self.screen,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )
        if self.sensory_obj is not None:
            self.sensory_obj.draw(screen=self.surf, zoom=self.zoom, translation=self.translation, angle=self.angle)
            # self.sensory_obj.draw(screen=self.screen, zoom=self.zoom, translation=self.translation, angle=self.angle)
            
        if self.forward_obj is not None:
            self.forward_obj.draw(screen=self.surf, zoom=self.zoom,translation=self.translation, angle=self.angle)
            # self.forward_obj.draw(screen=self.screen, zoom=self.zoom,translation=self.translation, angle=self.angle)
            
        '''surface flip을 해야 원하는대로 보이는 이유
        우리가 pygame에서 볼 때 (0, 0)에 screen을 blit하기 때문이 좌측 상단이 (0, 0)의 좌표를 갖는다.
        그리고 pygame에서는 가로가 X축, 세로가 Y축인 것은 동일하지만
        오른쪽으로 +, 아래쪽으로 + (y값 증가) 하기 때문에 의도한 화면과 차이가 있다.
        따라서 원하는대로 보기 위해서는 Y축에 대해서 뒤집어야 우리가 의도한 트랙의 그림을 볼 수 있다.'''
        self.surf = pygame.transform.flip(self.surf, flip_x=False, flip_y=True)

        # showing stats
        # self._render_indicators(W, H)
        self._render_actions(W=W, H=H)
        self._render_text()
 

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            
            pygame.display.flip()

        if mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen
        
    def _update_human_screen(self):
        pygame.event.pump()
        self.clock.tick(self.metadata['render_fps'])
        self.screen.fill(0) ## 눈에 보여지는 화면은 전부 0으로 채워 넣음 ##
        self.screen.blit(self.surf, (0, 0)) ## 빈 화면에 road, car등 그려 넣은 surface를 붙여 넣음 ##
        pygame.display.flip()
        
    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
            
        # draw limit boundary 
        if hasattr(self, "left_limit_poly") and hasattr(self, "right_limit_poly"):
            for arr in [self.left_limit_poly, self.right_limit_poly]:
                for poly, color in arr:
                    poly = [(p[0], p[1]) for p in poly]
                    color = [int(c) for c in color]
                    self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
            
    def vertical_ind(self, place, val, s, h, H):
        return [
            (place * s, H - (h + h * val)),
            ((place + 1) * s, H - (h + h * val)),
            ((place + 1) * s, H - h),
            ((place + 0) * s, H - h),
        ]
         
    def horiz_ind(self, place, val, s, h, H):
        return [
            ((place + 0) * s, H - 4 * h),
            ((place + val) * s, H - 4 * h),
            ((place + val) * s, H - 2 * h),
            ((place + 0) * s, H - 2 * h),
        ]


    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)


        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, self.vertical_ind(5, 0.02 * true_speed,  s, h, H), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            self.vertical_ind(7, 0.01 * self.car.wheels[0].omega, s, h, H),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            self.vertical_ind(8, 0.01 * self.car.wheels[1].omega, s, h, H),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            self.vertical_ind(9, 0.01 * self.car.wheels[2].omega, s, h, H),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            self.vertical_ind(10, 0.01 * self.car.wheels[3].omega, s, h, H),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            self.horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, s, h, H),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            self.horiz_ind(30, -0.8 * self.car.hull.angularVelocity, s, h, H),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        W, H = surface.get_size()
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

 
if __name__ == "__main__":
    env = CarRacing(render_mode='human', verbose=True, domain_randomize=False)
    env.reset()
    env.render()

# if __name__ == "__main__":
#     a = np.array([0.0, 0.0, 0.0])

#     def register_input():
#         global quit, restart
#         for event in pygame.event.get():
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_LEFT:
#                     a[0] = -1.0
#                 if event.key == pygame.K_RIGHT:
#                     a[0] = +1.0
#                 if event.key == pygame.K_UP:
#                     a[1] = +1.0
#                 if event.key == pygame.K_DOWN:
#                     a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
#                 if event.key == pygame.K_RETURN:
#                     restart = True
#                 if event.key == pygame.K_ESCAPE:
#                     quit = True

#             if event.type == pygame.KEYUP:
#                 if event.key == pygame.K_LEFT:
#                     a[0] = 0
#                 if event.key == pygame.K_RIGHT:
#                     a[0] = 0
#                 if event.key == pygame.K_UP:
#                     a[1] = 0
#                 if event.key == pygame.K_DOWN:
#                     a[2] = 0

#             if event.type == pygame.QUIT:
#                 quit = True

#     env = CarRacing(render_mode="human")

#     quit = False
#     while not quit:
#         env.reset()
#         total_reward = 0.0
#         steps = 0
#         restart = False
#         while True:
#             register_input()
#             s, r, terminated, truncated, info = env.step(a)
#             total_reward += r
#             if steps % 200 == 0 or terminated or truncated:
#                 print("\naction " + str([f"{x:+0.2f}" for x in a]))
#                 print(f"step {steps} total_reward {total_reward:+0.2f}")
#             steps += 1
#             if terminated or truncated or restart or quit:
#                 break
#     env.close()