import os, sys, math
import numpy as np
import pygame
from pygame import gfxdraw
sys.path.append("..")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)
from simulator.test.pygame_vehicle_render import (
    _render_road, _render_text, _render_car,
    _render_lidar, _render_fvec,
    _render_action, 
    _render_car_state,
    _render_reward,
    _render_penalty
)

def _render_all_pygame(
                       render_cfg:dict,
                       car_obj, 
                       track_dict:dict,
                       t:int, #새로운 episode일때 t=0 (environment['vehicle']['dt']만큼씩 environment 안에서 계속 더해 나가야 함)
                       lidar_obj=None, 
                       fvec_obj=None,
                       screen=None,
                       clock=None,
                       #####################################################################
                       action=None,
                       reward_dict=None,
                       penalty_dict=None,
                       render_car_state:bool=False,
                       
                       exp_name:str=None
                    ):
    '''during the simulation, all the lidar states and forward vector states will have been "stepped", so the features will all be calculated already. 
    NO NEED TO "_step" HERE
    [TODO]
    - add function to render the reward values / car dynamics etc
    
    '''
    pygame.font.init()
    if screen is None:
        pygame.init()
        pygame.display.init()
        screen = pygame.display.set_mode((render_cfg['window_w'], render_cfg['window_h']))
        clock = pygame.time.Clock()

    surface = pygame.Surface((render_cfg['window_w'], render_cfg['window_h']))
    
    zoom = 0.3 * max(1-t, 0) + render_cfg['zoom_rate'] * render_cfg['scale_rate'] * min(t, 1)
    car_x = car_obj.bicycle_model.car_x
    car_y = car_obj.bicycle_model.car_y
    car_w = car_obj.bicycle_model.vehicle_model.body_width
    car_h = car_obj.bicycle_model.vehicle_model.body_height
    
    scroll_x = -(car_x) * zoom
    scroll_y = -(car_y) * zoom
    angle = car_obj.bicycle_model.car_phi #heading angle of the vehicle's body
    trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(-angle)
    translation = [render_cfg['window_w']/2 + trans[0], render_cfg['window_h']/2 + trans[1]]
    
    #(1) render the road on the screen
    road_added_surface = _render_road(
        zoom=zoom, translation=translation, angle=-angle,
        track_dict=track_dict,
        surface=surface,
        bounds=render_cfg['bounds'],
        bg_color=render_cfg['bg_color'],
        road_color=None, #render_cfg['road_color'], 
        continuous_road_color=False, #False
    )
    #(2) render the car on the screen
    car_added_surface = _render_car(
        surface=road_added_surface,
        zoom=zoom, translation=translation,
        car_x=car_x, car_y=car_y,
        car_w=car_w, car_h=car_h,
        car_phi=angle,
        car_color=render_cfg['car_color']
    )
    ret_surface = car_added_surface
    #(3) render the lidar sensors on the screen
    if lidar_obj is not None:
        ret_surface = _render_lidar(
            surface=ret_surface, 
            lidar_obj=lidar_obj,
            zoom=zoom, translation=translation,
            angle=angle, car_x=car_x, car_y=car_y,
            ray_color=render_cfg['lidar_color']
        )
    
    #(4) render the forward vector on the screen
    if fvec_obj is not None:
        ret_surface = _render_fvec(
            surface=ret_surface,
            fvec_obj=fvec_obj,
            zoom=zoom, translation=translation,
            angle=angle, car_x=car_x, car_y=car_y,
            vec_color=render_cfg['fvec_color']
        ) 
    ret_surface = pygame.transform.flip(ret_surface, 0, 1) #Y축에 대해서 뒤집음
    #(5) render actions of the agent
    if action is not None:
        ret_surface = _render_action(
            surface=ret_surface,
            action_arr=action,
            cx=render_cfg['action_cx'],
            cy=render_cfg['action_cy'],
            font_size=render_cfg['action_font_size']
        )
    #(6) render the reward values
    if reward_dict is not None:
        ret_surface = _render_reward(
            surface=ret_surface,
            reward_dict=reward_dict,
            text_color=render_cfg['reward_text_color'],
            bg_color=render_cfg['reward_bg_color'],
            cx=render_cfg['reward_cx'],
            cy=render_cfg['reward_cy'],
            font_size=render_cfg['reward_font_size']
        )
    
    #(7) render the penalty values
    if penalty_dict is not None:
        ret_surface = _render_penalty(
            surface=ret_surface,
            penalty_dict=penalty_dict,
            text_color=render_cfg['penalty_text_color'],
            bg_color=render_cfg['penalty_bg_color'],
            cx=render_cfg['penalty_cx'],
            cy=render_cfg['penalty_cy'],
            font_size=render_cfg['penalty_font_size']
        )
        
    #(8) render the car state values
    if render_car_state:
        ret_surface = _render_car_state(
            surface=ret_surface,
            car_obj=car_obj,
            state_name_arr=render_cfg['render_car_state'],
            cx=render_cfg['car_state_cx'],
            cy=render_cfg['car_state_cy'],
            font_size=render_cfg['car_state_font']
        )
    #(9) write the name of the experiment
    if exp_name is not None:
        ret_surface = _render_text(
            surface=ret_surface,
            t=f"<{exp_name}>",
            cx = 450,
            cy = 300,
            font_size=10,
        )
    #(10) update all the screen
    pygame.event.pump()
    clock.tick(render_cfg['render_fps'])
    screen.fill(0)
    screen.blit(ret_surface, (0, 0))
    pygame.display.flip() 
    
    return screen, clock
    