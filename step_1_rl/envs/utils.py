import numpy as np
import os, sys
import math
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root);sys.path.append(os.path.dirname(root))
from envs.gym_car_constants import *
from sympy import Line, Point
from collections import defaultdict

import heapq
import pygame
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from pygame.math import Vector2
import keyboard

def _calc_yaw_of_car(car_x, car_y):
    '''car_x, car_y should be the car_vx and car_vy
    차량의 yaw 각을 계산한다는게 조향각을 계산하는건데,
    여기서 차량의 위치 coordinate 좌표를 쓰면 절대 안됨. 시작 점으로부터의 차량의 위치에 불과하기 때문에
    조향각을 구하고 싶다면 vx, vy즉 속도 벡터의 단위 값들을 사용해야 함.'''
    rad = math.atan2(car_y, car_x)
    deg = math.degrees(rad)
    if deg < 0:
        deg += 360
    
    return math.radians(deg)

def dist_to_center(xy_spline, theta, X, Y):
    p = xy_spline(theta)
    dp = xy_spline(theta, nu=1)
    psi = np.arctan2(dp[1], dp[0])
    el = np.cos(psi)*(X-p[0]) + np.sin(psi) * (Y-p[1])
    return el**2


class Forward_Observation(object):
    def __init__(self, car_obj, track_dict, theta_diff:float=20.,
                 num_vecs:int=5):
        super().__init__()
        self.car_obj = car_obj
        self.tx = track_dict['x']
        self.ty = track_dict['y']
        self.theta = track_dict['theta']
        self.phi = track_dict['phi']
        self.kappa = track_dict['kappa']
        
        self.num_vecs = num_vecs
        self.theta_diff = theta_diff
        self.points = []
        self.rotated_vectors = []
        self.vectors = []
        
        self.curvature_arr = []
        
        self.heading_angle = 0 # 초기에는 track의 첫번째 지점의 phi와 align이 맞도록 할테니까, 0일 것임.
        
    def closest_center(self, x, y):
        distance = []
        for ci, (cx, cy) in enumerate(zip(self.tx, self.ty)):
            heapq.heappush(distance, (math.dist((x, y), (cx, cy)), ci))
        _, i = heapq.heappop(distance)
        
        return self.tx[i], self.ty[i], i
    
    def rotate_clock(self, x, y, a):
        x1 = x * np.cos(a) - y * np.sin(a)
        y1 = x * np.sin(a) + y * np.cos(a)
            
        return [x1, y1]
    
    def rotate_counterclock(self, x, y, a):
        x1 = x * np.cos(a) + y * np.sin(a)
        y1 = -x * np.sin(a) + y * np.cos(a)
        
        return [x1, y1]
    
    def step(self):
        '''현재 차량의 위치에서 제일 가까운 center line의 reference point로부터
        차량의 이동 방향으로부터 일정 거리 (theta_diff)만큼 이동했을 때의 center line에서의 위치까지의 벡터들'''

        self.vectors = []
        self.rotated_vectors = []
        self.points = []
        self.curvature_arr = []
        
        car_x, car_y = self.car_obj.hull.position
        self.start_point = [car_x, car_y]
        
        # angle_rad = self.car_obj.hull.angle
        # car_vx, car_vy = self.car_obj.hull.linearVelocity
        # angle_rad =  _calc_yaw_of_car(car_x=car_vx, car_y=car_vy) # degrees to radians #
        forward_vector = self.car_obj.hull.GetWorldVector((0, 1))
        angle_rad = math.atan2(forward_vector[1], forward_vector[0])
        # angle_rad += math.pi / 2
        
        center_x, center_y, idx = self.closest_center(car_x, car_y)
        self.vectors.append([center_x - car_x, center_y - car_y])
        v1, v2 = center_x-car_x, center_y-car_y
        ref_phi = self.phi[idx]
        # heading_angle = angle_rad - np.pi/2
        heading_angle = ref_phi - angle_rad
        self.heading_angle = heading_angle
        
        # self.rotated_vectors.append(self.rotate(v1, v2, angle_rad))
        self.rotated_vectors.append(self.rotate_clock(v1, v2, heading_angle))
        self.curvature_arr.append(self.kappa[idx])
        self.points.append([center_x, center_y])
        center_phi = self.phi[idx] % (math.pi * 2)
        
        '''[TODO]
        - 차량의 방향도 함께 고려 해주기.
        - 예를 들어서 차량이 도로의 정주행 방향과 반대로 이동하고 있는 경우에는 "forward observation vector"으로서의 조건을 만족하기 위해서
        vehicle direction 또한 고려해 주어야 한다는 뜻이다.'''
        
        '''[250101] error fix - 1
        차량이 도로의 역방향으로 이동하고자 할 때 end_idx의 범위 수정
        정방향으로 이동할 때에도 state의 dimension을 무조건 맞춰주어야 하기 때문에 end_idx 범위 수정

        [250107] error fix - 2
        차량과 center line의 각도의 차이가 90도 ~ 135도 사이인 경우에 "역방향"으로 간주할 수 있다.
        그리고 역방향인 경우의 end_idx도 올바르게 변경해줌
        '''

        # if ((0 <= angle_rad <= math.pi) and (math.pi <= center_phi <= math.pi * 2)) or ((0 <= center_phi <= math.pi) and (math.pi <= angle_rad <= math.pi*2)):
        if math.pi * 1.5 <= abs(angle_rad - center_phi) <= math.pi * 0.5: 
            start_idx = idx
            end_idx = idx - len(self.tx) # -(idx + len(self.tx)) # -idx # -1
            diff = -1
        # else:
        #     start_idx, end_idx, diff = idx, idx + len(self.tx), 1
        else:
            start_idx = idx
            end_idx = len(self.tx) + idx
            diff = 1
            
        # start_idx = idx;end_idx = len(self.tx);diff= 1
        '''중앙선에 있는 좌표들중 다음에 좌표까지의 거리가 theta_diff보다 커지는 순간 그 중앙선의 좌표를 forward vector의 벡터의 끝 점으로 잡는다.
        때문에 forward vector들 사이의 간격이 완벽하게 동일한 것이 아니다.'''
        prev_theta = self.theta[idx]
        cnt = 1   
        for i in range(start_idx, end_idx, diff):
            if i >= len(self.tx):
                i %= len(self.tx)
            
            theta = self.theta[i]
            kappa = self.kappa[i]
            
            
            if abs(theta - prev_theta) > self.theta_diff:
                # print(theta, prev_theta, i)
                cnt += 1
                prev_theta = theta
                v1, v2 = self.tx[i]-car_x, self.ty[i]-car_y
                self.vectors.append([v1, v2])
                '''rotate the forward vector as the heading angle to match the object space equally'''
                # self.rotated_vectors.append(self.rotate(v1, v2, a=angle_rad))
                # 차량의 위치에서 계산한 heading angle 그대로 사용 (당연)
                self.rotated_vectors.append(self.rotate_clock(v1, v2, a=heading_angle))
                self.points.append([self.tx[i], self.ty[i]])
                self.curvature_arr.append(kappa)
            
            if cnt == self.num_vecs:
                # breakpoint()
                return
             
        
         
    def draw(self, screen, zoom, translation, angle): 
        if self.points == []:
            return
        '''draw the lidar information on the pygame track'''
        start = Vector2(self.start_point[0], self.start_point[1]).rotate_rad(angle) 
        sx, sy = (start[0]*zoom + translation[0],start[1]*zoom + translation[1])

        # for vec in self.points: 
        for i, vec in enumerate(self.points):
            if vec is None:
                continue
            vx, vy = vec 
            rotated = Vector2(vx, vy).rotate_rad(angle)
            transformed = (rotated[0]*zoom + translation[0], rotated[1]*zoom + translation[1]) 
            ex, ey = transformed 
            if i == 0:
                color = (255, 0, 0)
            elif i == len(self.points)-1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            pygame.draw.line(screen, color=color, start_pos=(sx, sy), end_pos=(ex,ey), width=2)            
                        
            
        
        
def key_press_check(status_dict):   
    # for event in pygame.event.get():
    done = False
    
    while not done:  
        event = keyboard.read_event() 
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'f3':
                # status_list.append("EXIT");done = True
                status_dict['EXIT'] += 1;done=True
            elif event.name == 'f2':
                status_dict['RUNNING'] += 1 
                done=True
            elif event.name == 'f1': 
                status_dict['PAUSE'] += 1 
                done=True

    
class Lidar(object):
    def __init__(self, car_obj, track_dict, road_poly,
                 is_nam:bool=False,
                 degree:int=10, 
                 lidar_length:float=100.,
                 do_reverse:bool=False):
        super().__init__()
        self.car = car_obj
        ##이 두개의 값만 있으면 됨##
        self.tx = track_dict['x']
        self.ty = track_dict['y']
        self.track_theta = track_dict['theta']
        self.track_phi = track_dict['phi']
        self.road_poly = road_poly
        self.is_nam = is_nam 
        self.degree = degree
        self.lidar_length = lidar_length
        self.start_point = (self.car.init_x, self.car.init_y)
        self.sensor = []
        
        self.do_reverse = do_reverse
        
        
        phi_arr = self.track_phi
        phi_arr[-1] = phi_arr[0]
        
        xx, yy = self.tx, self.ty
        xx[-1] = xx[0];yy[-1]=  yy[0]
        self.theta_spline = CubicSpline(self.track_theta, np.vstack([xx, yy]).T, bc_type='periodic')
        self.phi_spline = CubicSpline(self.track_theta, phi_arr, bc_type="periodic")

    def check_collision(self, wall, lidar_start_point, lidar_end_point):
        x1, y1 = wall[0] 
        x2, y2 = wall[1] 
        x3, y3 = lidar_start_point
        x4, y4 = lidar_end_point
        # x1, y1 = lidar_start_point
        # x2, y2 = lidar_end_point
        # x3, y3 = wall[0]
        # x4, y4 = wall[1]
        
        denominator = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4)
        numerator = (x1-x3) * (y3-y4) - (y1-y3) * (x3-x4)
        if denominator == 0:
            return None
        t= denominator / numerator
        u = -((x1-x2) * (y1-y3) - (y1-y2) * (x1-x3)) / denominator

        if  0 < t < 1 and u > 0:
            x = x1 + t * (x2-x1)
            y = y1 + t * (y2-y1)
            # breakpoint()    
            return (x, y)
        return None
     
    def check_div(self, x1, y1, x2, y2, x3, y3, x4, y4):
        f1 = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        f2 = (x2-x1)*(y4-y1) - (y2-y1)*(x4-x1)
        if f1 * f2 < 0:
            return True
        return False
     
    def check_cross(self, x1,y1, x2, y2, x3, y3, x4, y4):
        b1 = self.check_div(x1, y1, x2, y2,x3,y3,x4,y4)
        b2 = self.check_div(x3,y3, x4,y4, x1,y1, x2,y2)
        if b1 and b2:
            return True
        return False
    
    # def check_intersection_simple(self, wall, lidar_start_point, lidar_end_point):
    #     x1, y1 = wall[0]
    #     x2, y2 = wall[1]
    #     x3, y3 = lidar_start_point
    #     x4, y4 = lidar_end_point
        
    #     r = (x2-x1, y2-y1)
    #     s = (x4-x3, y4-y3)
    #     diff = (x1-x3, y1-y3)
    #     cross_r_s = r[0] * s[1] - r[1] * s[0]
    #     if cross_r_s == 0:
    #         return None
        
    #     ## 교점이 두 선분 내부에 존재하는지 확인 ##
    #     t = (diff[0] * s[0] - diff[1] * s[0]) / cross_r_s
    #     u = (diff[0] * r[1] - diff[1] * r[0]) / cross_r_s
    #     # if 0 <= t <= 1 and 0 <= u <= 1:
    #     if 0 <= t <= 1 and 0 <= u <= 1:
    #         # inter_p = (x1 + t * r[0], y1 + t * r[1])
    #         inter_p = (x3 + u*s[0], y3 + u*s[1])
    #         return inter_p
    #     return None
    
    def _calc_inter_p(self, x1, y1, x2, y2, x3, y3, x4, y4): 
        vec_1 = np.array([x1, y1, 1]);vec_2 = np.array([x2, y2, 1])
        vec_3 = np.array([x3, y3, 1]);vec_4 = np.array([x4, y4, 1])
        cross_vec = np.cross(np.cross(vec_1, vec_2), np.cross(vec_3, vec_4))
        inter_p_x = cross_vec[0] / cross_vec[-1]
        inter_p_y = cross_vec[1] / cross_vec[-1]
        return (inter_p_x, inter_p_y)
    
    def check_intersection_simple(self, wall, lidar_start_point, lidar_end_point):
        x1, y1 = wall[0];x2, y2 = wall[1]
        x3, y3 = lidar_start_point;x4, y4 = lidar_end_point
        is_inter = self.check_cross(x1, y1, x2, y2, x3, y3, x4, y4)
        
        if is_inter:
            return self._calc_inter_p(x1,y1,x2,y2,x3,y3,x4,y4)
        else:
            return None
    
    def check_intersection_basic(self, wall, lidar_start_point, lidar_end_point):
        x1, y1 = wall[0];x2, y2 = wall[1]
        x3, y3 = lidar_start_point;x4, y4 = lidar_end_point
        
        n = (y2-y1) / (x2-x1);p = (y4-y3) / (x4-x3) ## 선분의 직선의 기울기 ##
        m = y2-n*x2;q = y4-p*x4 ## 선분의 직선의 Y 절편 ##
        
        a = (q-m) / (m-p) # 교점의 x좌표 #
        b = a*n + m # a*p + q로 해도 구할 수 있음 (교점의 y좌표) #

        if min([x1, x2]) <= a <= max([x1, x2]) and min([x3, x4]) <= a <= max([x3, x4]):
            if min([y1, y2]) <= b <= max([y1, y2]) and min([y3, y4]) <= b <= max([y3, y4]):
                return (a, b)
        return None
                
    def check_intersection_sympy(self, wall, lidar_start_point, lidar_end_point):
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        x3, y3 = lidar_start_point
        x4, y4 = lidar_end_point
        
        from sympy.geometry.util import intersection
        from sympy import Ray
        
        r1 = Ray((x1, y1), (x2, y2))
        r2 = Ray((x3, y3), (x4, y4))
        inter_p = intersection(r1, r2)
        print(inter_p, len(inter_p))
        if len(inter_p) == 1:
            x = float(inter_p[0][0])
            y = float(inter_p[0][1])
            return (x, y)
        return None
        # breakpoint()
        
        
        
    #     return (inter_x, inter_y)
        
    def step(self, car_yaw_rad, do_reverse:bool=False):
        self.sensor = []
        car_x, car_y = self.car.hull.position #current position of the vehicle (same world coordinate as the track) #
        start_point = (car_x, car_y)
        self.start_point = start_point
        car_yaw_deg = math.degrees(car_yaw_rad)
        min_deg = int(car_yaw_deg)
        max_deg = min_deg + 180
        # breakpoint()
        # if do_reverse == False:
        #     min_deg = int(car_yaw_deg)
        #     max_deg = int(car_yaw_deg) + 180
        # else:
        #     max_deg = int(car_yaw_deg)
        #     min_deg = max_deg - 180
        # for deg in range(0, 360, self.degree):
        # for deg in range(-90, 90, self.degree):
        # for deg in range(0, 180, self.degree):
        
        ## theta를 입력으로 넣으면 그 시점에서 제일 가까운 (x,y) 좌표를 얻을 수 있음.
        
        # breakpoint()
        for deg in range(min_deg, max_deg, self.degree): 
            
        # for deg in range(int(car_yaw_deg), int(car_yaw_deg) + 90, self.degree): ## 1사분면
        # for deg in range(int(car_yaw_deg)+ 90, int(car_yaw_deg) + 180, self.degree): ## 2사분면
        # for deg in range(int(car_yaw_deg) + 180, int(car_yaw_deg) + 270, self.degree): ## 3사분면
        # for deg in range(int(car_yaw_deg)+270, int(car_yaw_deg)+360, self.degree): ## 4사분면
            # if do_reverse:
            #     deg += 180
            
            end_point = start_point + Vector2(self.lidar_length, 0).rotate(deg) # uses degree value for rotation (if rotate_rad, should use the radian value)
 
            closest_dist = float('inf')
            closest_point = None
            if self.is_nam:
                step = 1
            else:
                step = 1
            # for poly, color in self.road_poly:
            for j in range(0, len(self.road_poly), step):
                poly, color = self.road_poly[j]
     
                if list(color) in [[255,0,0], [255,255,255]]:
                    continue
                left_wall = [poly[0], poly[3]]
                right_wall = [poly[1], poly[2]]
                # wall = [left_wall, right_wall]
                for wall in [left_wall, right_wall]:
                    # collide_pos = self.check_collision(wall, start_point, end_point)
                    # collide_pos = self.check_intersection(wall, start_point, end_point)
                    collide_pos = self.check_intersection_simple(wall, start_point, end_point)
                    
                    if collide_pos is not None: 
                        cpx, cpy = collide_pos
                        ## (1) (x,y) 좌표에 대해서 theta array를 interpolate한 것이 theta_spline임. 따라서 이 spline function에 theta값을 __call__() function으로 불러주면 해당하는 제일 가까운 center line의 (x,y) 좌표를 반환해 준다.
                        # (cpx, cpy)와 제일 가까운 좌표의 theta값을 dist_to_center 함수를 optimize하며 찾음.
                        close_theta = minimize(lambda x: dist_to_center(self.theta_spline, x[0], cpx, cpy), [0]).x[0]
                        ## (2) close_theta 값에 해당하는 center line에서의 phi 값을 찾을 수 있음.
                        close_phi = self.phi_spline(close_theta)
                        # if deg == min_deg:
                        #     breakpoint()
                        # if math.radians(abs(math.degrees(close_phi) - car_yaw_deg)) > math.pi / 2:
                        #     closest_dist=-1;closest_point=None
                            
                        sensor_dx = start_point[0] - collide_pos[0]
                        sensor_dy = start_point[1] - collide_pos[1]
                        dist = math.sqrt(sensor_dx**2 + sensor_dy**2)
                        if (dist < closest_dist): # and dist < self.lidar_length:
                            closest_dist = dist
                            closest_point = collide_pos
 
            if closest_point is not None: 
                self.sensor.append([closest_dist, closest_point])
                # self.sensor.append([closest, end_point]) ## -> used for debugging the lidar length ##
            else:
                self.sensor.append([-1, None]) # 무한대이기 때문에 없는 경우가 있을 수 있는데 이런 경우에는 그냥 None이나 -1로 처리를 해야 함.
        # breakpoint()
        # if self.do_reverse:
        #     self.sensor = self.sensor[::-1]
            
    def draw(self, screen, zoom, translation, angle):
 
        if self.sensor == []:
            return
        '''draw the lidar information on the pygame track'''
        start = Vector2(self.start_point[0], self.start_point[1]).rotate_rad(angle)
        # start = Vector2(self.start_point[0], self.start_point[1])
        sx, sy = (start[0]*zoom + translation[0],start[1]*zoom + translation[1])
   
        # sx, sy = start
        # breakpoint()
        # for (close_dist, close_point) in self.sensor:
        for i, (close_dist, close_point) in enumerate(self.sensor):
            if close_point is None:
                continue
            cx, cy = close_point
            # rotated = Vector2(cx, cy)
            rotated = Vector2(cx, cy).rotate_rad(angle)
            transformed = (rotated[0]*zoom + translation[0], rotated[1]*zoom + translation[1])
            # rotated = Vector2(transformed[0], transformed[1]).rotate_rad(angle)
            ex, ey = transformed
            # ex,  ey = transformed 
            # breakpoint() 
            # print("lidar ",close_dist, sx, sy, ex, ey)
            if i == 0:
                color = (255,0, 0)
            elif i == len(self.sensor)-1:
                color = (0,255, 0)
            else:
                color = (0, 125, 255)
                
            pygame.draw.line(screen, color=color, start_pos=(sx, sy), end_pos=(ex,ey), width=2)            
             

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
class Dynamics(object):
    '''simplified bicycle model 대신 Box2D physics engine의 차속 등 계산 결과를 그대로 사용하기 때문에,
    별도로 굴림 계수 등의 값을 반영한 계산을 통해서 구할 필요는 없다.'''
    def __init__(self, car_obj, track_dict):
        super().__init__()
        self.dt = 1/60.
        
        self.car = car_obj
        self.tx = track_dict['x']
        self.ty = track_dict['y']
        self.phi = track_dict['phi']
        self.k = track_dict['kappa']
        self.track_theta = track_dict['theta']

        self.prev_car_pos = (self.tx[0], self.ty[0])
        
        ## initialize values with the attributes saved in the car object ##
        ## 앞바퀴 2개의 yaw angle, yaw rate는 모두 동일함. ##
        self.joint_angle = car_obj.wheels[0].joint.angle # 0.
        self.joint_speed = car_obj.wheels[0].joint.speed # 0.
        self.prev_dict = defaultdict(list)

        self.prev_yaw = self.calc_yaw_with_box2d() # self.car.hull.angle # 0.
        self.yaw_omega = self.car.hull.angularVelocity # 0.
        
        self.e_c = 0.
        self.e_phi = 0.
        self.theta = 0.
        self.v_x = 0.
        self.v_y = 0.
        
        self.d_e_phi = 0 
        
        self.e_c_arr = [0]

        self.theta_diff_val = 0.
        self.theta_val = 0.
        self.prev_theta = 0.
        
        self.kappa = 0.
        
        import copy
        xx = copy.deepcopy(self.tx);xx[-1] = xx[0]
        yy = copy.deepcopy(self.ty);yy[-1] = yy[0]
        # breakpoint()
        self.xy_spline = CubicSpline(self.track_theta,
                                   np.vstack((xx, yy)).T, 
                                   bc_type='periodic')
        self.phi[-1] = self.phi[0]
        self.phi_spline = CubicSpline(
            self.track_theta,
            self.phi, bc_type="periodic"
        )
        
        self.k[-1] = self.k[0]
        self.kappa_spline = CubicSpline(
            self.track_theta,
            self.k, bc_type="periodic"
        )
        
    @property
    def theta_diff(self):
        if len(self.prev_dict['theta']) == 1:
            return self.prev_dict['theta']
        return self.prev_dict['theta'][-1] - self.prev_dict['theta'][-2]
    
    def _check_progress(self, env, time_interval, minimum_diff):
        if time_interval >= len(self.prev_dict['theta']):
            return 0 #OK라는 뜻#
        
        prev_theta = self.prev_dict['theta'][-time_interval]
        recent_theta = self.prev_dict['theta'][-1]
        
        theta_diff = recent_theta - prev_theta
    
        # breakpoint()
        if theta_diff < 0:
            env.is_backward = True 
            return 0 #NO라는 뜻 - 왜냐면 backward로 이동한 셈이 되니까#
        else:
            env.is_backward = False
            if theta_diff < minimum_diff: 
                return -10 #NO라는 뜻 - 왜냐면 최소한으로 이동해야 하는 거리도 이동하지 않았기 때문에#
        return 0 #OK라는 뜻#

    def calc_dist_to_center(self, theta, X, Y):
        p =self.xy_spline(theta)
        dp = self.xy_spline(theta, nu=1)
        psi = np.arctan2(dp[1], dp[0])
        el = np.cos(psi)*(X-p[0]) + np.sin(psi) * (Y-p[1])
        return el**2
    
    def calc_de_c(self):
        vx, vy = self.car.hull.linearVelocity
        de_c = vx * math.sin(self.e_phi) + vy * math.cos(self.e_phi)
        return de_c
    
    def calc_de_phi(self):
        vx, vy = self.car.hull.linearVelocity
        de_phi = self.yaw_omega - (self.kappa / (1-self.kappa * self.e_c) * (vx*math.cos(self.e_phi) - vy*math.sin(self.e_phi)))
        return de_phi
    
    def calc_yaw_with_box2d(self):
        '''결국에는 self.car.hull.angle과 동일한 값이 나옴.
        rad = 0: X축과 평행
        rad < 0: 반시계 방향으로 회전
        rad > 0: 시계 방향으로 회전
        
        ==> GetWorldVector에서 차량의 실제 전진 방향인 Y축 (0, 1)을 기준으로 구해줘야지 차량의 Yaw Angle을 구할 수 있다.
        이렇게 되었을 때 결과적으로 원하는 +X축으로부터의 회전각을 구할 수 있게 되는 것이다.
        '''
        # forward_vec = self.car.hull.GetWorldVector((1, 0))
        forward_vec = self.car.hull.GetWorldVector((0, 1))
        rad = math.atan2(forward_vec[1], forward_vec[0])
        
        return rad
    
    def run_dynamics(self, dt):

        car_x, car_y = self.car.hull.position
        self.dist_diff = math.dist((car_x, car_y), self.prev_car_pos) # 그냥 무지성으로 이전이랑 이동 거리 계산하는 것도 가능 
        self.prev_car_pos = (car_x, car_y)

        car_vx, car_vy = self.car.hull.linearVelocity ## m/s ##
        # center_dist, ref_index, _, ref_val = calculate_error_c(x=car_x, y=car_y, 
        #                                                        track_x=self.tx, track_y=self.ty, 
        #                                                        phi_arr=self.phi, kappa_arr=self.k, 
        #                                                        debug=True)
        # ref_x, ref_y, ref_phi, ref_kappa = ref_val
        
    
        from scipy.optimize import minimize
        close_theta = minimize(
            lambda x: dist_to_center(self.xy_spline, x[0], car_x, car_y), [0]
        ).x[0]
        ref_phi = self.phi_spline(close_theta)
        ref_x, ref_y = self.xy_spline(close_theta)
        
        ##트랙의 중앙선의 reference point의 접선까지의 수직 거리를 center_dist, 즉 e_c로 간주한다.
        R = np.array([[math.cos(ref_phi), -math.sin(ref_phi)], [math.sin(ref_phi), math.cos(ref_phi)]])
        v = np.array([car_x-ref_x, car_y-ref_y])

        vv = R.transpose()@v
        center_dist = vv[1] 
        
        self.theta_diff_val = close_theta - self.theta_val
        self.theta_val = close_theta
        self.kappa = self.kappa_spline(close_theta)
        
        self.d_e_phi =  self.calc_de_phi()
        e_phi = self.d_e_phi * self.dt + self.e_phi
        self.e_phi = e_phi
          
        self.ref_x = ref_x  
        self.ref_y = ref_y
        # self.ref_index = ref_index
        self.ref_phi = ref_phi
        self.center_dist = center_dist
        
        # breakpoint()
        # self.prev_yaw = self.car.hull.angle
        # self.yaw_omega = self.car.hull.angularVelocity
        car_yaw = self.calc_yaw_with_box2d()
        
        # car_yaw = math.radians(_calc_yaw_of_car(car_x=car_vx, car_y=car_vy)) # degrees to radians #
        # car_yaw = _calc_yaw_of_car(car_x=car_vx, car_y=car_vy) ## _calc_yaw_of_car 함수 안에서 이미 radians로 바꿔줌
        ### (1) theta (곡선 상에서의 이동 거리) ###
        # theta_deriv = car_vx*math.cos(self.e_phi) - car_vy*math.sin(self.e_phi)
        # theta_deriv /= (1 - ref_kappa*self.e_c)
   
        # self.theta = (theta_deriv*dt) + self.theta
        
        # self.theta_val = minimize(
        #     lambda x : self.calc_dist_to_center(x[0], car_x, car_y), [0]
        # ).x[0]
        
        # self.theta_diff_val = self.theta_val - self.prev_theta

        # self.theta_diff_val = (theta_deriv*dt)
        # self.theta_diff_val = theta_deriv
   
        ### (2) e_phi (center line의 reference point와의 phi 오차) ###
        # 트랙 접선에 대한 yaw각의 오차
        
        # self.e_phi = car_yaw - ref_phi 
        # self.e_phi = car_yaw - ref_phi
        ### e_phi: angle between heading and tangent to the centerline
        # self.e_phi = ref_phi - car_yaw ##이게 AM에서 말하는 heading angle
        
        # breakpoint()
        
        
        ### (3) e_c (center line의 reference point으로부터의 수직 거리) ###
        self.e_c = center_dist
        self.e_c_arr.append(self.e_c)
        
        
        ### (4) v_x (차량의 종방향 속도) ###
        self.v_x = car_vx
        
        ### (5) v_y (차량의 횡방향 속도) ###
        self.v_y = car_vy
        
        ### (6) 차량의 yaw의 각속도 ###
        self.yaw_omega = (car_yaw - self.prev_yaw) / dt
        self.prev_yaw = car_yaw
        
        self.vel = math.sqrt(self.v_x**2 + self.v_x**2)
         
        
        dynamic_state = {
            # 'theta': self.theta, 
            'theta': self.theta_val,
            'e_phi': self.e_phi, 
            'e_c': self.e_c,
            'v_x': self.v_x, 'v_y': self.v_y,
            'heading_angle': ref_phi - car_yaw,
            'theta_diff': self.theta_diff_val,
            'theta_val': self.theta_val,
            'dist_diff': self.dist_diff,
            'yaw': car_yaw,
            'ref_phi': ref_phi,
            'yaw_omega': self.yaw_omega,
        }
        
        for key, value in dynamic_state.items():
            self.prev_dict[key].append(value)
            
        self.dynamic_state = dynamic_state
        self.prev_theta = self.theta_val
        return dynamic_state
 
        
        
def calc_ratio(car_x, car_y, x1, y1, x2, y2):
    p1, p2 = Point(x1, y1), Point(x2, y2)
    p_car = Point(car_x, car_y)
    try:
        l_hori = Line(p1, p2)
    except:
        return 1, 0

    try:
        distance = l_hori.distance(p_car) # perpendicular distance from the car to the line connecting the two points on the center line #
    except:
        # breakpoint()
        return 1, 0
    
    d1 = math.dist((x1, y1), (car_x, car_y))
    d2 = math.dist((x2, y2), (car_x, car_y))
    
    inter_d1 = d1 ** 2 - distance ** 2
    inter_d2 = d2 ** 2 - distance ** 2
    
    ratio_1 = inter_d1 / (inter_d1 + inter_d2)
    ratio_2 = inter_d2 / (inter_d1 + inter_d2)
    
    return ratio_1, ratio_2

def calculate_error_c(x, y, track_x, track_y, phi_arr, kappa_arr, debug:bool=False):
    '''
    @x: x-coordinate of the car
    @y: y-coordinate of the car
    @track_x, track_y: x, and y array data for the center line of the track
    @phi_arr: phi(=yaw) array data for the center line of the track
    '''
    import heapq
    
    distance = []
    for ci, (cx, cy) in enumerate(zip(track_x, track_y)):
        heapq.heappush(distance, (math.dist((x, y), (cx, cy)), ci))
    _, i = heapq.heappop(distance) # most closest #
    _, j = heapq.heappop(distance) # 2nd most closest #
    x1, y1, x2, y2, p1, p2, k1, k2 = track_x[i], track_y[i], track_x[j], track_y[j], phi_arr[i], phi_arr[j], kappa_arr[i], kappa_arr[j]
    ratio_1, ratio_2 = calc_ratio(x, y, x1, y1, x2, y2)
    
    target = np.array([[x1, y1, p1, k1], [x2, y2, p2, k2]])
    ratio = np.array([ratio_2, ratio_1]).T
    ref = np.dot(target.T, ratio) #(4,2)(2,1)->(X_ref, Y_ref, P_ref, K_ref)

    rot_mat = np.array([[math.cos(ref[2]), -math.sin(ref[2])], 
                        [math.sin(ref[2]), math.cos(ref[2])]])
    ret = np.dot(rot_mat.T, np.array([x-ref[0], y-ref[1]]).T) # (2,1)
    
    _, error_c = ret[0], ret[1] # error_theta, error_c #
    error_c = -math.sin(ref[2]) * (x-ref[0]) + math.cos(ref[2]) * (y-ref[1])
    
    if debug:
        return float(error_c), i, j, ref
    else:
        return float(error_c)
     
def find_closest_point(x, y, track_x, track_y):
    distances = np.array([math.dist((x,y), (tx, ty)) for (tx, ty) in zip(track_x, track_y)])
    return np.argmin(distances)

def get_track_state(car_obj, track_dict):
    pass

def get_feature_vec_state(car_obj, track_dict):
    pass



def get_track_boundary(cX, cY, phi_arr):
    '''
    Inputs
    @cX: array object of X coords of the center line
    @cY: array object of Y coords of the center line
    @phi_arr: array object of the phi angle
    '''
    left_bound = np.zeros((len(cX), 2))
    right_bound = np.zeros((len(cY), 2))
                          
    for i, (x, y, phi) in enumerate(zip(cX, cY, phi_arr)):
        rot_mat = np.array([
            [math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]
        ])
        radius = TRACK_WIDTH # // 2
        right_rot_vec = np.dot(rot_mat, np.array([0, radius]).T)
        left_rot_vec = np.dot(rot_mat, np.array([0, -radius]).T)
        
        right_bound[i,0] = x + right_rot_vec[0]
        right_bound[i,1] = y + right_rot_vec[1]

        left_bound[i,0] = x + left_rot_vec[0]
        left_bound[i,1] = y + left_rot_vec[1]
        
    return left_bound, right_bound

def get_track_border_limit(cX, cY, phi_arr):
    '''
    Outputs:
    track_border_left & track_border_right: array object of (X, Y) coords of the left and right track border
    '''
    track_border_right = np.zeros((len(cX), 2))
    track_border_left = np.zeros((len(cX), 2))
    
    for i, (x, y, phi) in enumerate(zip(cX, cY, phi_arr)):
        rot_mat = np.array([
            [math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]
        ])
        radius = TRACK_WIDTH // 2
        right_rot_vec = np.dot(rot_mat, np.array([0, radius + OUT_TRACK_LIMIT]).T)
        left_rot_vec = np.dot(rot_mat, np.array([0, -(radius + OUT_TRACK_LIMIT)]).T)
        
        track_border_right[i, 0] = x + right_rot_vec[0]
        track_border_right[i, 1] = y + right_rot_vec[1]
        
        track_border_left[i, 0] = x + left_rot_vec[0]
        track_border_left[i, 1] = y + left_rot_vec[1]
        
    return track_border_left, track_border_right

### NEW FUNCTION ###
def calculate_track_curvature(X, Y):
    dx, dy = np.gradient(X), np.gradient(Y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    K = (dx*ddy-dy*ddx) / (dx**2+dy**2)**(3/2)
    return K

### DEPRECATED FUNCTION ###
def calculate_curvature(X, Y):
    '''[REFERENCE] https://github.com/peijin94/PJCurvature'''
    points = np.vstack((X, Y)).T 
    curvature = [] 
    for i in range(len(points)):
        prev_i, next_i = (i-1) % len(points), (i+1) % len(points)
        t_a = np.linalg.norm([points[i][0] - points[prev_i][0], points[i][1] - points[prev_i][1]])
        t_b = np.linalg.norm([points[next_i][0] - points[i][0], points[next_i][1] - points[i][1]])
        t_c = np.linalg.norm([points[next_i][0] - points[prev_i][0], points[next_i][1] - points[prev_i][1]])
 
        '''Curvature by the derivatives of the coordinates'''
        M = np.array([
            [1, -t_a, t_a ** 2], [1, 0, 0], [1, t_b, t_b**2]
        ])
        try:
            inv_M = np.linalg.inv(M)
        except:
            I = np.eye(M.shape[0], M.shape[0])
            inv_M = np.linalg.lstsq(M, I)[0]
    
        x = [points[j][0] for j in [prev_i, i, next_i]]
        y = [points[j][1] for j in [prev_i, i, next_i]]

        a = np.matmul(inv_M, x)
        b = np.matmul(inv_M, y) 
        kappa = 2 * (b[2]*a[1] - a[2] * b[1]) / (a[1]**2.+b[1]**2.) ** 1.5
        
        curvature.append(kappa)
        
    curvature = curvature[1:-1]
    curvature = [curvature[0]] + curvature + [curvature[-1]]
    assert len(curvature) == len(points)
    return curvature

def calculate_theta(X,  Y):
    dx, dy = np.gradient(np.array(X)), np.gradient(np.array(Y))
    ds = np.sqrt(dx**2 + dy**2)
    theta = np.cumsum(ds)
    return theta

# def calculate_theta(X, Y):
#     '''각각의 corner마다 spline interpolation을 통해서 길이를 구하기'''
#     points = np.vstack([X, Y]).T
#     I = list(np.arange(0, len(points)))
#     J = I[1:] 
#     dist_btw_points = [math.dist(points[i], points[j]) for i, j in zip(I, J)]
#     theta = np.cumsum(dist_btw_points)
#     theta = np.r_[0, theta]
#     assert len(theta) == len(points)
    
#     return theta
    
def make_unit_vec(vec):
    vec_size = math.hypot(vec[0], vec[1])
    unit_vec = [vec[0] / vec_size, vec[1] / vec_size]
    return unit_vec
    
    
def vec_length(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    
def dot_prod(vec1, vec2):
    assert len(vec1) == 2 and len(vec2) == 2
    ret =  vec1[0] *  vec2[0] + vec1[1] * vec2[1]
    return ret

def calculate_track_phi(X, Y):
    dx, dy = np.gradient(X), np.gradient(Y)
    phi = np.arctan2(dy, dx)
    phi = np.unwrap(phi)
    return phi
    '''math.atan used for calculating the radian value'''
def calculate_phi(X, Y):
 
    def phi_range_change(x, y):
        rad = math.atan2(y, x)
        deg = math.degrees(rad)
        if deg < 0:
            deg += 360
            
        return math.radians(deg)
    
    def smooth_yaw(arr):
        '''np.unwrap function
        : Unwraps a signal p by changing elements which can have an absolute difference from their
        predecessor of more than max(discount, period/2)
        - 그래서 period보다 작은 값에 대해서는 period만큼 더해줌 
        '''
        arr = np.unwrap(arr, period=math.pi)
        arr -= math.pi 
        return arr
    
    phi = []
    points = np.vstack([X, Y]).T
    YAW_OFFSET = 1
    
    for i in range(len(points)):
        next_i = ( i + YAW_OFFSET ) % len(points)

        unit_vec_j = np.array([points[i][0] - points[next_i][0],
                               points[i][1] - points[next_i][1]]) 
        y = unit_vec_j[1]
        x = unit_vec_j[0]
        rad = math.atan2(y,x) 
        rad = phi_range_change(x, y)
        phi.append(rad)
        # phi.append(rad + math.pi/2)
    phi = np.array(phi) 
    phi = smooth_yaw(phi)
    
    return phi


def return_track_center(road_poly):
    c_X, c_Y = [], []
    for pi, (border, color) in enumerate(road_poly):
        if list(color) not in [[255,0,0], [255,255,255]]:
            x, y = np.mean(border, 0)
            c_X.append(x);c_Y.append(y)
    return c_X, c_Y

def reverse_beta(rad):
    if rad > math.pi:
        return rad - math.pi * 1.5
    else:
        return rad + math.pi / 2

def gen_beta(phi_arr):
    '''phi 값을 그대로 쓰는게 아니라, phi에 수직인 벡터를 구해야 하는 것이었기 때문에
    pi / 2에서 뺀 각을 사용해야 했다.'''
    beta = np.zeros_like(phi_arr)
    for i, phi in enumerate(phi_arr):
        if phi > math.pi / 2:
            val = phi - math.pi / 2
        else:
            val = math.pi / 2 - phi #  math.pi / 2 - phi
        # if val > math.pi:
        #     val = math.pi - val
        beta[i] = val
    return beta

def gen_alpha(X, Y):
    # cx, cy = NAM_CENTER[0], NAM_CENTER[1]
    cx, cy = 0, 0

    alpha = []
    for i, (x,y) in enumerate(zip(X, Y)):
        rad = math.atan2(y-cy, x-cx)
        if rad < 0:
            rad += math.pi * 2
        alpha.append(rad)
    return alpha

def loc_before(theta_arr, idx, 
               vec_length:float=5., vec_num:float=15.):
    total_vec_dist = vec_length * vec_num
    loc_theta = theta_arr[idx]
    
    prev_theta = max(0, loc_theta - total_vec_dist)
    diff_arr = [(abs(prev_theta-theta_arr[i]), i) for i in range(len(theta_arr))]
    diff_arr.sort()
    target_idx = diff_arr[0][1]
    targets = np.arange(target_idx, idx, 1) 
    
    return targets

def gen_straight_track(kappa_arr, theta_arr, theta_length, vec_num,
                       is_nam:bool=False,
                       kappa_limit:float=1e-2,
                       consider_forward_vec:bool=True):
    # kappa_limit = 1e-2
    if is_nam:
        kappa_limit = float(1e-3)
    else:   
        if kappa_limit > 1.:
            kappa_limit = np.percentile(kappa_arr, kappa_limit)

    straight = np.where(np.abs(kappa_arr) < kappa_limit)
    N = len(kappa_arr)
    assert  N == len(theta_arr)
    # others = np.array(list(set([i for i in  range(N)]) - set(straight[0])))
    others = np.where(np.abs(kappa_arr) >= kappa_limit)[0]
    is_straight = np.zeros(N)
    
    if not consider_forward_vec:
        is_straight[straight] = 1
        return is_straight
    
    targets = []
    for si in others:
        targets.extend(list(loc_before(theta_arr, 
                                       idx=si,
                                       vec_length=theta_length,
                                       vec_num=vec_num)))
    
    for si in straight[0]:
        if si not in targets:
            is_straight[si] = 1
    
    return is_straight
        
    
def create_checkpoints(num_checkpoints, track_rad, is_circle:bool=False):
    checkpoints = []
    if is_circle:
        for c in range(num_checkpoints):
            if c == num_checkpoints-1:
                start_alpha = 2 * math.pi * (-0.5) / num_checkpoints
            alpha = 2 * math.pi * c / num_checkpoints
            checkpoints.append((alpha, track_rad * math.cos(alpha), track_rad * math.sin(alpha)))
        return checkpoints, start_alpha
    
    
    for c in range(num_checkpoints):
        noise =  np.random.uniform(0, 2 * math.pi * 1 / num_checkpoints)
        alpha = 2 * math.pi * c / num_checkpoints + noise
        rad = np.random.uniform(track_rad / 3, track_rad)
        if c == 0:
            alpha = 0
            rad = 1.5 * track_rad ## 처음 시작 점 동일하게 유지
        if c == num_checkpoints - 1:
            alpha = 2 * math.pi * c / num_checkpoints
            START_ALPHA = 2 * math.pi * (-0.5) / num_checkpoints
            rad = 1.5 * track_rad
        ## (X축으로부터의 각 in radians, x, y) ##
        checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

    return checkpoints, START_ALPHA

def connect_checkpoints(checkpoints, track_rad, detail_step, 
                        min_track_turn_rate:float,
                        track_turn_rate:float,
                        is_circle:bool=False):
    if is_circle:
        x, y, beta = track_rad, 0, 0
    else:
        x, y, beta = 1.5 * track_rad, 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0: # alpha가 양수라서 다시 track를 여는 구간의 시작이 되었는데 이전에 이미 한번 track을 완성한 경우 lap 횟수 추가 #
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True # 처음 시작한 초기 위치로부터 아래쪽에, 즉 이제 closed loop track를 닫는(?) 구간의 시작에 진입했다는 뜻 #
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
        if proj > min_track_turn_rate:
            beta -= min(track_turn_rate, abs(0.001 * proj))
        if proj < -min_track_turn_rate:
            beta += min(track_turn_rate, abs(0.001 * proj))
        # if proj > 0.3:
        #     beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
        # if proj < -0.3:
        #     beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
        x += p1x * detail_step
        y += p1y * detail_step
        track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y)) # (alpha, beta, x, y) #
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break
        
    return track

def check_connected_loop(track, start_alpha):
    # Find closed loop range i1..i2, first loop should be ignored, second is OK
    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i == 0:
            break
        pass_through_start = (
            track[i][0] > start_alpha and track[i - 1][0] <= start_alpha
        )
        if pass_through_start and i2 == -1:
            i2 = i
        elif pass_through_start and i1 == -1:
            i1 = i
            break 
        
    assert i1 != -1
    assert i2 != -1
    track = track[i1 : i2 - 1]
    first_beta = track[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)
    # Length of perpendicular jump to put together head and tail
    # 마지막 (x,y)와 첫번째 (x,y)의 좌표간의 거리가 일정 TRACK_DETAIL_STEP보다 작아야 함 #
    well_glued_together = np.sqrt(
        np.square(first_perp_x * (track[0][2] - track[-1][2]))
        + np.square(first_perp_y * (track[0][3] - track[-1][3]))
    )
    
    if well_glued_together > TRACK_DETAIL_STEP:
        return False, -1, -1, []
    return True, i1, i2, track
    
    

def find_hard_corner(beta_arr, 
                     track_turn_rate,
                     is_nam:bool=False):
    # track_turn_rate = NAM_TRACK_TURN_RATE if is_nam else TRACK_TURN_RATE
    border_min_count = NAM_BORDER_MIN_COUNT if is_nam else BORDER_MIN_COUNT
    
    border = [False] * len(beta_arr)
    for i in range(len(beta_arr)):
        good = True
        oneside = 0
        for neg in range(border_min_count):
            beta1 = beta_arr[i - neg - 0]
            beta2 = beta_arr[i - neg - 1]
            good &= abs(beta1 - beta2) > track_turn_rate * 0.2
            oneside += np.sign(beta1 - beta2)
 
        good &= abs(oneside) == border_min_count
        border[i] = good
    for i in range(len(beta_arr)):
        for neg in range(border_min_count):
            border[i - neg] |= border[i]
    return border


def create_tiles(
        box_world,
        box_tile,
        X, Y, beta, border_arr, 
        is_nam:bool=False,
        width:int=None,
        color=None,
        additional_dict:dict=None):
      
    def order_vertices(vert):
        import copy
        X, Y = np.array(vert).T[0], np.array(vert).T[1]
        new_X = copy.deepcopy(X);new_Y = copy.deepcopy(Y)
        if Y[1] > Y[2]:
            new_Y[2] = Y[1];new_Y[1] = Y[2]
            new_X[2] = X[1];new_X[1] = X[2]
            
        if Y[0] < Y[3]:
            new_Y[3] = Y[0];new_Y[0] = Y[3]
            new_X[3] = X[0];new_X[0] = X[3]
        return np.vstack((new_X, new_Y)).T
    
    track_width = NAM_TRACK_WIDTH if is_nam else TRACK_WIDTH
    if width is not None:
        track_width = width
    border = NAM_BORDER if is_nam else BORDER
    
    # caution_poly = []
    
    road_poly = []
    road = []
    
    error = []
    # offset = 1 if color is None else 5
    offset = 1
    for i in range(len(X)):
    # for i in range(0, len(X), offset//2):
        x1, y1, beta1 = X[i], Y[i], beta[i]
        x2, y2, beta2 = X[i-offset], Y[i-offset], beta[i-offset]
        road1_l = (
            x1 - track_width * math.cos(beta1),
            y1 - track_width * math.sin(beta1),
        )
        road1_r = (
            x1 + track_width * math.cos(beta1),
            y1 + track_width * math.sin(beta1),
        )
        road2_l = (
            x2 - track_width * math.cos(beta2),
            y2 - track_width * math.sin(beta2),
        )
        road2_r = (
            x2 + track_width * math.cos(beta2),
            y2 + track_width * math.sin(beta2),
        )

        try:
            vertices = [road1_l, road1_r, road2_r, road2_l]
            box_tile.shape.vertices = vertices
        except:
            ordered = order_vertices(np.array(vertices))
            box_tile.shape.vertices = [tuple(a) for a in ordered]
            error.append(vertices)
            
        t = box_world.CreateStaticBody(fixtures = box_tile)
        t.userData = t
        c = 0.01 * (i % 3) * 255
        t.color = ROAD_COLOR + c if color is None else color
        t.road_visited = False
        t.num_visited = 0
        t.road_friction = 1.0 ## tile에 대해서는 road_friction 부여 ##
        # t.road_slope = 0.0 ## road tile에 대해서 구배 정보를 반영하기 위함 ##
        t.idx = i
        t.fixtures[0].sensor = True 
        
        '''순전히 트랙의 경계를 위해서 필요함.'''
        if additional_dict is not None:
            for key, value in additional_dict.items():
                setattr(t, key, value)
        
        road_poly.append([[road1_l, road1_r, road2_r, road2_l], t.color])
        road.append(t)
        
        if len(border_arr) == 0:
            continue
        if border_arr[i]:
            side = np.sign(beta2 - beta1)
            b1_l = (
                x1 + side * track_width * math.cos(beta1),
                y1 + side * track_width * math.sin(beta1),
            )
            b1_r = (
                x1 + side * (track_width + border) * math.cos(beta1),
                y1 + side * (track_width + border) * math.sin(beta1),
            )
            b2_l = (
                x2 + side * track_width * math.cos(beta2),
                y2 + side * track_width * math.sin(beta2),
            )
            b2_r = (
                x2 + side * (track_width + border) * math.cos(beta2),
                y2 + side * (track_width + border) * math.sin(beta2),
            )
            road_poly.append(
                (
                    [b1_l, b1_r, b2_r, b2_l],
                    tuple(WHITE_COLOR) if i % 2 == 0 else tuple(RED_COLOR)
                )
            )
    
    X = np.array([arr[0][0][0] for arr in road_poly])
    Y = np.array([arr[0][0][1] for arr in road_poly])
    print(f"X: {np.min(X)}  {np.max(X)}")
    print(f"Y: {np.min(Y)}  {np.max(Y)}")
    return road_poly, road 

def resplit(x, y, phi):
    X,Y,PHI = [],[],[]
    idx = 0
    while idx < len(x):
        X.append(x[idx]);Y.append(y[idx]);PHI.append(phi[idx])
        j = idx + 1
        if j == len(x):
            break
        while True:
            if j == len(x):
                idx = j
                break
            dist = math.dist((x[idx], y[idx]), (x[j], y[j]))
            if 2.4 <= dist  <= 3.4:
                idx = j
                break 
            elif dist > 3.4:
                X.append((x[idx] + x[j])/2)
                Y.append((y[idx]+y[j])/2)
                PHI.append((phi[idx]+phi[j])/2)
                idx = j
                break
            elif dist < 2.4:
                j += 1
    return X, Y, PHI

def make_gif(image_root, dest_path):
    from PIL import Image
    from glob import glob
    from natsort import natsort
    images = []
    img_paths = natsort.natsorted(glob(image_root + "/*.png"))
    for p in img_paths:
        img = Image.open(p)
        images.append(img)
    images[0].save(dest_path,
                   save_all=True, 
                   append_images=images,
                   optimize=False,
                   duration=150)
    
if __name__ == "__main__":
    import pickle
    import Box2D
    from Box2D.b2 import fixtureDef, polygonShape, contactListener
    
    world = Box2D.b2World((0, 0))
    tile = fixtureDef(
        #top-left, top-right, bottom-right, bottom-left#
        shape = polygonShape(vertices = [(0,0), (1,0), (1,-1), (0,-1)])
    )
    
    do_nam=True # False
    if do_nam:
        NAM_DATA = pickle.load(open('statics/nam_c_track.pkl', 'rb'))
        nam_PHI = np.array(NAM_DATA['phi'])
        beta = gen_beta(nam_PHI)
        border = find_hard_corner(beta, True)
        road_poly, road, error = create_tiles(
            box_world=world, box_tile=tile,
            X=np.array(NAM_DATA['x']),
            Y=np.array(NAM_DATA['y']),
            beta=beta, border_arr=border, is_nam=True
        )
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,10))
        cmap = plt.get_cmap('Reds')
        for j, poly in enumerate(error):
            for i in range(len(poly)):
                x1, y1 = poly[i];x2, y2 = poly[(i+1) % (len(poly))] 
                ax.plot([x1, x2], [y1, y2], c=cmap(j*10))
        fig.savefig('check_error.png')
        pickle.dump(error, open('error.pkl','wb'))
    else:
        checkpoints, start_alpha = create_checkpoints()
        track = connect_checkpoints(checkpoints=checkpoints)
        is_valid, i1, i2 = check_connected_loop(track=track, START_ALPHA=start_alpha)
    
        if is_valid:
            track = track[i1:i2-1]
            
            beta_arr = np.array(track).T[1]
            border = find_hard_corner(beta_arr, is_nam=False)
            road_poly, road = create_tiles(
                box_world = world, box_tile=tile,
                X=np.array(track).T[2],
                Y=np.array(track).T[3],
                beta=beta_arr,
                border_arr=border,
                is_nam=False
            )
    # breakpoint()
    
    