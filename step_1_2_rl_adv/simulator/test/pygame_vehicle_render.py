import os, sys
import math
import numpy as np
import pygame
from pygame import gfxdraw
import math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from environment.vehicle import RaceCar
from environment.vehicle.vehicle_model import vehicleModel
from environment.track import Bezier_TrackGenerator, Nam_TrackGenerator
from environment.observation.lidar_sensor import Observation_Lidar_State
from environment.observation.forward_vector import Observation_ForwardVector_State

def define_track_car(is_nam:bool=False):

    if is_nam:
        track_generator = Nam_TrackGenerator(
            track_width =7.,
            nam_track_path=f"{ROOT}/statics/nam_c_track.pkl",
            min_num_ckpt=4,
            max_num_ckpt=16,
            track_radius=200,
            scale_rate=1.,
        )
    else:
        track_generator = Bezier_TrackGenerator(
            min_num_ckpt=4,
            max_num_ckpt=16,
            min_kappa=0.04,
            max_kappa=0.1,
            track_width=7.,
            track_density=1,
            track_radius=200.,
            scale_rate=1., 
        )

    car = RaceCar(
        action_dim=2,
        dt=1/60,
        cfg_file_path=f"{ROOT}/environment/vehicle/jw_config.yaml"
    )
    

    return track_generator, car

def define_states(track_dict, car_obj):
    lidar_obj = Observation_Lidar_State(
        car_obj=car_obj,
        track_dict=track_dict,
        lidar_angle_min=-90,
        lidar_angle_max=90,
        num_lidar=20,
        max_lidar_length=20.
    )
    
    fvec_obj = Observation_ForwardVector_State(
        car_obj=car_obj,
        track_dict=track_dict,
        theta_diff=5,
        num_vecs=15
    )
    
    return lidar_obj, fvec_obj

def _render_mini_map(surface, track_dict):
    pass

def _zoom_and_trans_vec(vec, zoom, translation, angle):
    ret_vec = pygame.math.Vector2(vec).rotate_rad(angle)
    if len(np.array(ret_vec).shape) == 1:
        ret_vec = [ret_vec[0]*zoom + translation[0], ret_vec[1]*zoom + translation[1]]
    else:
        ret_vec = [(c[0]*zoom + translation[0], c[1]*zoom + translation[1]) for c in ret_vec]
    return ret_vec

def _render_lidar(surface, lidar_obj, zoom, translation, angle, car_x, car_y,
                ray_color=None):
    lidar_results = lidar_obj.lidar_results
    car_vec = _zoom_and_trans_vec([car_x,car_y], zoom, translation, -angle)
    ray_color = (0, 255, 0) if ray_color is None else ray_color
    
    for i in range(len(lidar_results)):
        ray_x, ray_y = lidar_results[i]['point']
        if lidar_results[i]['distance'] == -1:
            continue
        '''ray의 시작과 끝 좌표를 전부 따로 변환을 시켜서 pygame.draw.line으로 lidar ray를 그려야 함'''
        ray_vec = _zoom_and_trans_vec([ray_x, ray_y], zoom, translation, -angle)
        pygame.draw.line(surface, color=ray_color, start_pos=ray_vec, end_pos=car_vec)
        
    return surface

def _render_fvec(surface, fvec_obj, zoom, translation, angle, car_x, car_y, 
                 vec_color=None):
    fvec_arr = fvec_obj.vector_dict['inertia']
    car_vec = _zoom_and_trans_vec([car_x,car_y], zoom, translation, -angle)
    vec_color = (0, 0, 255) if vec_color is None else vec_color
    for i in range(len(fvec_arr)):
        vec_x, vec_y = fvec_arr[i]
        end_x, end_y = vec_x + car_x, vec_y + car_y
        end_vec = _zoom_and_trans_vec([end_x, end_y], zoom, translation, -angle)
        pygame.draw.line(surface, color=vec_color, start_pos=end_vec, end_pos=car_vec)
    
    return surface
        
def _render_text(surface, t, cx, cy,
                 font_size:int,
                 number:bool=False,
                 text_color=None,
                 text_bg_color=None):
    font = pygame.font.Font(pygame.font.get_default_font(), font_size)
    text_bg_color = (0, 0, 0) if text_bg_color is None else text_bg_color
    text_color = (255, 255, 255) if text_color is None else text_color
    
    if number:
        text = font.render("%04i" % t, True, text_color, text_bg_color)
    else:
        text = font.render(t, True, text_color, text_bg_color)
        
        
    text_rect = text.get_rect()
    text_rect.center = (cx, cy)
    surface.blit(text, text_rect)
    
    return surface

def _render_reward(surface, reward_dict, text_color, bg_color, cx, cy, font_size):
    for key, value in reward_dict.items():
        if key == 'step_reward':
            font_size += 5
        text = f"{key}:" + f"{value:.3f}"
        surface = _render_text(surface, text, cx, cy, font_size, text_color=text_color, text_bg_color=bg_color)
        cy += font_size
    return surface

def _render_penalty(surface, penalty_dict, text_color, bg_color, cx, cy, font_size):
    for key, value in penalty_dict.items():
        text = f"{key}:" + f"{value:.3f}"
        surface = _render_text(surface, text, cx, cy, font_size, text_color=text_color, text_bg_color=bg_color)
        cy += font_size
    return surface
    
def _render_car_state(surface, car_obj, state_name_arr,  cx, cy, font_size, text_color=None, bg_color=None):
    bicycle_model = car_obj.bicycle_model
    for name in state_name_arr:
        value = getattr(bicycle_model, name, -9999)
        text = f"{name}:" + f"{value:.3f}"
        surface = _render_text(surface, text, cx, cy, font_size, text_color=text_color, text_bg_color=bg_color)
        cy += font_size

    return surface
    
    

def _render_action(surface, action_arr, cx, cy, font_size:int):
    if len(action_arr) == 3:
        steer, throttle, brake = action_arr
        throttle = (throttle + 1) / 2
        brake = (brake + 1) / 2
        action_dict = {"steer": steer, "throttle": throttle, "brake": brake}
        for key, value in action_dict.items():
            text = f"{key}:" + f"{value:.3f}"
            surface = _render_text(surface, t=text, cx=cx, cy=cy, font_size=font_size, number=False)
            cy += font_size
            
    elif len(action_arr) == 2:
        steer, torque = action_arr
        if torque > 0:
            throttle = torque;brake = 0
        else:
            throttle = 0;brake=torque
        action_dict = {"steer": steer, "throttle": throttle, "brake": brake}
        for key, value in action_dict.items():
            text = f"{key}:" + f"{value:.3f}"
            surface = _render_text(surface, t=text, cx=cx, cy=cy, font_size=font_size, number=False)
            cy += font_size
    return surface

def _check_car_in_tile(cx, cy, poly):
    X = [p[0] for p in poly];Y = [p[1] for p in poly]
    mx, Mx = min(X), max(X);my, My = min(Y), max(Y)
    
    return mx <= cx <= Mx and my <= cy <= My
        
def _render_road(zoom, translation, angle, track_dict, surface, bounds,
                 fill_bg:bool=False, 
                 road_color=None,
                 bg_color=None,
                 continuous_road_color:bool=False):
    init_road_color = road_color
    field = [(bounds, bounds), (bounds, -bounds), (-bounds, -bounds), (-bounds, bounds)]
    bg_color = [0, 0, 0] if bg_color is None else bg_color
    if fill_bg:
        gfxdraw.aapolygon(surface, field, bg_color)
        gfxdraw.filled_polygon(surface, field, bg_color)
    
    vertices = track_dict['vertices']
    # mx, Mx = min(track_dict['x']), max(track_dict['x'])
    # my, My = min(track_dict['y']), max(track_dict['y'])
    N = len(vertices)
    for i, poly in enumerate(vertices):
        poly = [(p[0], p[1]) for p in poly]
        alpha = [1 - (0.5 * i / N) for _ in range(3)]
        if init_road_color is None:
            if continuous_road_color:
                road_color = [int(255 * alpha[0]), int(255 * alpha[1]), int(255 * alpha[2])]
            else:
                ##지나간 트랙의 타일인 경우에는 해당 타일을 노란색으로 나타내도록 함.
                if int(track_dict['passed'][i]) == 1: 
                    road_color = [255, 255, 0]
                else:
                    road_color = [255,255,255]
        # road_color = [255,255,255] if road_color is None else road_color
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [(c[0]*zoom + translation[0], c[1]*zoom + translation[1]) for c in poly]
        gfxdraw.aapolygon(surface, poly, road_color)
        gfxdraw.filled_polygon(surface, poly, road_color)
    
    
    return surface


def _car_rotate_inplace(coords, rad):
    R = np.array([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad), math.cos(rad)]
    ]) 
    rot_X, rot_Y = np.array(coords).T[0], np.array(coords).T[1]
    mid_X, mid_Y = np.mean(rot_X), np.mean(rot_Y)
    shifted = np.array([rot_X-mid_X, rot_Y-mid_Y])
    rot_shifted = R@shifted
    rot_shifted_new = rot_shifted + np.array([[mid_X], [mid_Y]])
    
    return rot_shifted_new.T

def _render_car(surface, 
                zoom,
                translation,
                car_x, car_y, 
                car_w, car_h,
                car_phi,
                car_color=None):
    car_poly = [
            (car_x-car_h/2, car_y-car_w/2),
            (car_x+car_h/2, car_y-car_w/2),
            (car_x+car_h/2, car_y+car_w/2),
            (car_x-car_h/2, car_y+car_w/2)
        ]
    rotated_poly = [pygame.math.Vector2(c).rotate_rad(-car_phi) for c in car_poly] #트랙의 방향을 따라가도록
    pygame.draw.polygon(surface, color=(0, 255, 0), points=rotated_poly)
    car_color = (255, 0, 0) if car_color is None else car_color
    ##"확대"를 하는 측면에서는 zoom만이 그 역할을 해준다고 볼 수 있음,
    translated_poly = [(c[0]*zoom + translation[0], c[1]*zoom + translation[1] ) for c in rotated_poly] #zoom, translation은 단순 비율 키우고 평행이동하는 것
    new_translated_poly = _car_rotate_inplace(coords=translated_poly, rad=car_phi) 
    pygame.draw.polygon(surface, color=(255, 0, 0), points=new_translated_poly)
    
    return surface
    
###########################################################################
WINDOW_W = 1000 #1200
WINDOW_H = 1000 #1200
RENDER_FPS=60
ZOOM_RATE=4
SCALE_RATE=1 #애초에 coordinate 계산할 때 scale rate를 1로 뒀었기 때문에 굳이 scaling을 할 필요는 없음
vehicle_model = vehicleModel(config_yaml_path=f"{ROOT}/environment/vehicle/jw_config.yaml")

def simple_pygame_test(): 
    track_generator, car = define_track_car(is_nam=True)
    track_generator._generate()
    track_dict = track_generator._calculate_track_dict()
    car._reset(track_dict=track_dict)
    lidar_obj, fvec_obj = define_states(track_dict=track_dict, car_obj=car)
    
    fvec_obj._step(car.theta_center_spline)
    lidar_obj._step()
    
    #(0) initialize pygame and the screen for the "human visualization"
    pygame.init()
    pygame.display.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock() #essential for updating the pygame screen
    
    #(1) define the surface to render the simulation on
    ##처음에는 zoom-out을 해서 전체 궤적을 보여주는 화면으로 시작을 하고, 이후에는 zoom-in을 해서 현재 차량의 위치와 yaw angle을 기준으로 회전을 해서
    #트랙의 전방의 형상을 차량 내부에서 보는 시야와 동일하게 맞춰서 시각화를 할 수 있도록 한다.
    for t in range(4):
        surface = pygame.Surface((WINDOW_W, WINDOW_H))
        
        # zoom = 0.1 * SCALE_RATE * max(1-t, 0) + ZOOM_RATE * SCALE_RATE * min(t, 1)
        zoom = 0.5 * max(1-t, 0) + ZOOM_RATE * SCALE_RATE * min(t, 1)
        car_x = car.bicycle_model.car_x # track_dict['x'][0]
        car_y = car.bicycle_model.car_y # track_dict['y'][0]
        #차량의 현재 위치가 원점이 되도록 scroll
        #화면에서 원점이 좌측 상단이기 때문에 WINDOW_W/2, WINDOW_H/2를 각각 더해줌
        scroll_x = -(car_x) * zoom
        scroll_y = -(car_y) * zoom 
        angle = -car.bicycle_model.car_phi # -track_dict['phi'][0] 
        #시계 -phi == 반시계 +phi // 반시계 -phi == 시계 +phi
        print(f"ANGLE : {math.degrees(angle)}")
        
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        translation = [WINDOW_W/2+trans[0], WINDOW_H/2+trans[1]]

        bounds = 1000 # 1200
        road_added_surface = _render_road(zoom, translation, angle=0, track_dict=track_dict, surface=surface, bounds=bounds, fill_bg=True,
                                          color=(0, 0, 255)) #blue track (no rotation)
        '''This road_added_surface with the rotation will be used'''
        road_added_surface = _render_road(zoom, translation, angle=angle, track_dict=track_dict, surface=road_added_surface, 
                                          bounds=bounds, color=None, random_color=False) #white track (yes rotation)
        
        #화면의 중앙 좌표(변환 후에는 차량의 위치가 화면의 중앙이 되도록 해야 함)
        road_added_surface = _render_text(road_added_surface, t=f"center", 
                                          font_size=40,
                                          cx=WINDOW_W/2, #화면의 X축의 중간 좌표
                                          cy=WINDOW_H/2, #화면의 Y축의 중간 좌표
                                          number=False)
        road_added_surface = _render_text(road_added_surface, t=f"{math.ceil(math.degrees(-angle))} deg",
                                          font_size=20,
                                          cx=100, cy=30, number=False) 
        road_added_surface = _render_text(road_added_surface, t="translation",
                                          font_size=10,
                                          cx=translation[0], cy=translation[1], number=False)
        road_added_surface = _render_text(road_added_surface, t="track_start",
                                          font_size=10,
                                          cx=car_x*zoom+translation[0], cy=car_y*zoom+translation[1], number=False) #starting point on the blue track
        
        #(2) simply draw a polygon
        '''차량을 정의 할 때, 차량의 phi값이 X축을 기준으로 반시계 방향으로 측정된 값인데,
        rotate_rad를 사용하면 pygame의 coordinate system의 Y축이 뒤집혀 있기 때문에 원래는 반시계 회전인데 시계 방향으로 회전한 것처럼 보일 것임.'''
        car_w, car_h = vehicle_model.body_width, vehicle_model.body_height

        '''이걸로 car_polygon을 정의 해야지 track 정의 할 때와 동일한 효과를 가져올 수 있음.'''
        car_poly = [
            (car_x-car_h/2, car_y-car_w/2),
            (car_x+car_h/2, car_y-car_w/2),
            (car_x+car_h/2, car_y+car_w/2),
            (car_x-car_h/2, car_y+car_w/2)
        ]
        car_poly_hull = [
            (-car_h/2, -car_w/2),
            (car_h/2, -car_w/2),
            (car_h/2, car_w/2),
            (-car_h/2, car_w/2)
        ]
        print(car_poly)
        
        '''when rotating, the -1 * phi_angle_of_car should be used as rotation angle'''
        car_angle = car.bicycle_model.car_phi
        '''[TODO] change the vertices of the car correctly with rotation'''
        
        rotated_poly = [pygame.math.Vector2(c).rotate_rad(0) for c in car_poly_hull] #트랙의 방향을 따라가도록
        rotated_poly = [(c[0] + car_x, c[1]+car_y) for c in rotated_poly]
        pygame.draw.polygon(road_added_surface, color=(0, 0, 255), points=rotated_poly)
        
        
        rotated_poly = [pygame.math.Vector2(c).rotate_rad(-car_angle) for c in car_poly] #트랙의 방향을 따라가도록
        pygame.draw.polygon(road_added_surface, color=(0, 255, 0), points=rotated_poly)
        print(rotated_poly)

        
        
        ##"확대"를 하는 측면에서는 zoom만이 그 역할을 해준다고 볼 수 있음,
        translated_poly = [(c[0]*zoom + translation[0], c[1]*zoom + translation[1] ) for c in rotated_poly] #zoom, translation은 단순 비율 키우고 평행이동하는 것
        # translated_poly = [pygame.math.Vector2(c).rotate_rad(-car_angle) for c in translated_poly]
        print(translated_poly)
        new_translated_poly = _car_rotate_inplace(coords=translated_poly, rad=car_angle) #track이 회전을 했으니, 사실상 차량은 회전을 할 필요가 없음. 때문에 제자리에서 차량을 회전 시킬 수 있어야 함.
        pygame.draw.polygon(road_added_surface, color=(255, 0, 0), points=new_translated_poly)
         
        # gfxdraw.aapolygon(surface, rotated_poly, color)
        # gfxdraw.filled_polygon(surface, rotated_poly, color)
        #(3) draw the lidar sensors on the surface
        road_added_surface = _render_lidar(
            surface=road_added_surface, lidar_obj=lidar_obj, zoom=zoom,
            translation=translation, angle=car_angle, car_x=car_x, car_y=car_y,
            ray_color=(0, 255, 0)
        )
        #(4) draw the forward vectors on the surface
        road_added_surface = _render_fvec(
            surface=road_added_surface, fvec_obj=fvec_obj, zoom=zoom,
            translation=translation, angle=car_angle, car_x=car_x, car_y=car_y,
            vec_color=(0, 0, 255)
        )
        #(5) pump pygame to update all changes
        pygame.event.pump()
        clock.tick(RENDER_FPS)
        screen.fill(0)
        road_added_surface = pygame.transform.flip(road_added_surface, 0, 1)
        screen.blit(road_added_surface, (0, 0)) #좌측 상단
        pygame.display.flip() #updates the whole screen (MUST BE NEEDED)

    
        breakpoint()
    
    


if __name__ == "__main__":
    simple_pygame_test()
