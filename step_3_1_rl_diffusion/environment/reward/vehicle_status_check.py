'''vehicle_status_check.py
1. Check if the vehicle is in track
2. Check if the vehicle collided with the track wall
'''
import math
import numpy as np
from abc import ABC
from scipy.optimize import minimize
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from vehicle.tools import dist_to_center

def dist_to_center(xy_spline, theta, X, Y):
    '''same utility function in environment/vehicle/tools.py
    copied here due to circular import'''
    p = xy_spline(theta)
    dp = xy_spline(theta, nu=1)
    psi = np.arctan2(dp[1], dp[0])
    el = np.cos(psi)*(X-p[0]) + np.sin(psi) * (Y-p[1])
    
    return el**2

class Status_Checker(ABC):
    def __init__(self, car_obj, track_dict=None):
        super().__init__() 
        
    def _reset(self, car_obj, track_dict=None):
        self.car_obj = car_obj
        self.track_dict = track_dict
    
    def _calculate_penalty(self):
        pass


class OffCourse_Checker(Status_Checker):
    '''트랙 상의 bezier curve들 중에서 어디에 위치해 있는지, 차체의 테두리의 "선분들"과의 교점을 바탕으로 
    계산을 하려 했었는데, 그렇게 하려면 남양 트랙에 적용을 할 때의 bezier curve로 구성되어 있는게 아니라서
    쉽지 않을 것 같았음.
    '''
    def __init__(self, car_obj):
        super().__init__(car_obj=car_obj)
        if car_obj is not None:
            self._reset(car_obj)
    
    def _reset(self, car_obj,):
        self.car_obj = car_obj
        self.theta_center_spline = car_obj.theta_center_spline
        self.track_dict = self.car_obj.track_dict
        
        self.bicycle_model = self.car_obj.bicycle_model
        self.vehicle_model = self.bicycle_model.vehicle_model
         
        self.track_half_width = 7. / 2
        self.off_course_time = 0
        self.is_off_course = False
        self.off_track_tire_cnt = 0
        
    def _calculate_penalty(self, target:str):
        if target == 'com':
            self.is_off_course = self._off_course_com()
        elif target == 'all':
            self.is_off_course = self._off_course_all()
        elif target == 'instance':
            self.is_off_course = self._off_course_instance()
        elif target == 'tire':
            self.is_off_course = self._off_course_tire()
        
        speed = self.bicycle_model.car_speed_kph() 
        ##차속에 누적 밖에 나가 있던 시간을 곱해줌.
        ##그러면 차속을 0으로 바꾸게 되는 상황이 발생할 우려가 분명히 있기는 하다.
        return math.pow(speed, 2) * self.off_course_time
    
    def _off_course_com(self)->bool:
        '''
        Center-of-Mass가 트랙의 밖에 위치해 있을 때를 'off course'로 간주하는 경우
        - 이 경우에는 e_c로 추정하면 됨. (e_c)
        '''
        ##step1: Center-of-Mass가 트랙 밖에 나가 있는 시간 판단 로직##
        curr_e_c = self.bicycle_model.E_c
        prev_e_c = self.bicycle_model.e_c_arr[-2] if len(self.bicycle_model.e_c_arr) > 1 else 0
        prev_d_e_c = (curr_e_c - prev_e_c) / self.bicycle_model.dt
        #1-1 f(x) >= k
        diff_t_1 = math.atan2(self.track_half_width - prev_e_c, prev_d_e_c) # self.track_half_width - prev_e_c / prev_d_e_c
        #1-2 -f(x) >= k
        diff_t_2 = math.atan2(self.track_half_width + prev_e_c, -prev_d_e_c) # self.track_half_width + prev_e_c / (-prev_d_e_c)
        self.off_course_time = max([0., self.bicycle_model.dt-diff_t_1, self.bicycle_model.dt-diff_t_2])
        
        ##step2: Center-of-Mass가 트랙 밖에 나가 있음을 판단하는 로직##
        if abs(self.bicycle_model.E_c) > self.track_half_width:
            return True
        return False

        
    def _off_course_all(self):
        '''
        차체 전체가 트랙의 밖에 위치해 있을 때를 'off course'로 간주하는 경우
        '''
        car_width = self.vehicle_model.body_width 
        
        dist_threshold = self.track_half_width + car_width / 2
        
        ##step1: Center-of-Mass가 트랙 밖에 나가 있는 시간 판단 로직##
        curr_e_c = self.bicycle_model.E_c
        prev_e_c = self.bicycle_model.e_c_arr[-2] if len(self.bicycle_model.e_c_arr) > 1 else 0
        prev_d_e_c = (curr_e_c - prev_e_c) / self.bicycle_model.dt
        #1-1 f(x) >= k
        diff_t_1 = math.atan2(self.track_half_width - prev_e_c, prev_d_e_c) # self.track_half_width - prev_e_c / prev_d_e_c
        #1-2 -f(x) >= k
        diff_t_2 = math.atan2(self.track_half_width + prev_e_c, -prev_d_e_c) # self.track_half_width + prev_e_c / (-prev_d_e_c)
        self.off_course_time = max([0., self.bicycle_model.dt-diff_t_1, self.bicycle_model.dt-diff_t_2])
        
        ##step2: Center-of-Mass가 트랙 밖에 나가 있음을 판단하는 로직##
        if abs(self.bicycle_model.E_c) > dist_threshold:
            return True
        return False
    
    def _off_course_instance(self):
        '''
        차체 중 일부라도 트랙의 밖에 위치해 있을 때 그 순간에도 'off course'로 간주하는 경우
        '''
        car_width = self.vehicle_model.body_width 
        
        dist_threshold = self.track_half_width - car_width / 2
        
        ##step1: Center-of-Mass가 트랙 밖에 나가 있는 시간 판단 로직##
        curr_e_c = self.bicycle_model.E_c
        prev_e_c = self.bicycle_model.e_c_arr[-2] if len(self.bicycle_model.e_c_arr) > 1 else 0
        prev_d_e_c = (curr_e_c - prev_e_c) / self.bicycle_model.dt
        #1-1 f(x) >= k
        diff_t_1 = math.atan2(self.track_half_width - prev_e_c, prev_d_e_c) # self.track_half_width - prev_e_c / prev_d_e_c
        #1-2 -f(x) >= k
        diff_t_2 = math.atan2(self.track_half_width + prev_e_c, -prev_d_e_c) # self.track_half_width + prev_e_c / (-prev_d_e_c)
        self.off_course_time = max([0., self.bicycle_model.dt-diff_t_1, self.bicycle_model.dt-diff_t_2])
        
        ##step2: Center-of-Mass가 트랙 밖에 나가 있음을 판단하는 로직##
        if abs(self.bicycle_model.E_c) > dist_threshold:
            return True
        return False
    
    
    def _check_point_in_track(self, xx, yy) -> bool:
        '''important!! 
        when using the "scipy.optimize.minimize" function, the variable names should be different from x'''
        theta_center_spline = self.theta_center_spline # self.car_obj.theta_center_spline
        # initial_guess = self.track_dict['theta'][np.argmin((self.track_dict['x']- x)**2 + (self.track_dict['y'] - y)**2)]
        # if isinstance(initial_guess, int):
        #     initial_guess = self.track_dict[initial_guess]
        initial_guess = self.bicycle_model.ref_arr_dict['initial_guess'][-1]
        close_theta = minimize(lambda x:
            dist_to_center(theta_center_spline, x[0], xx, yy), [initial_guess]).x[0]
        close_x, close_y = theta_center_spline(close_theta)
        dist = math.sqrt((xx-close_x)**2 + (yy-close_y)**2)

        if dist > self.track_half_width:
            return False
        return True
        
    def _off_course_tire(self):
        '''
        COM을 기준으로 확인하는 것이 아니라 4개의 바퀴, 즉 차량의 4개의 body coordinate의 위치를 기반으로
        4개의 바퀴가 모두 트랙 밖에 위치하는 경우에만 TERMINATE 되도록 함.
        '''
        car_width = self.vehicle_model.body_width #차체 전체 너비
        car_height = self.vehicle_model.body_height #차체 전체 높이

        car_x = self.bicycle_model.car_x #차량의 world coordinate 상에서의 X좌표
        car_y = self.bicycle_model.car_y #차량의 world coordinate 상에서의 Y좌표
        
        car_phi = self.bicycle_model.car_phi #차량의 world coordinate 상에서의 조향 방향 (x축 기준으로의 각)
        
        
        coords = np.array([
            [car_height/2, car_width/2], [-car_height/2, car_width/2], [car_height/2, -car_width/2], [-car_height/2, -car_width/2]
        ])
        R = np.array([
            # [math.cos(-car_phi), -math.sin(-car_phi)],[math.sin(-car_phi), math.cos(-car_phi)]
            [math.cos(car_phi), -math.sin(car_phi)],[math.sin(car_phi), math.cos(car_phi)]
            
        ])
        trans = np.dot(coords, R.T)
        trans += np.array([[car_x,car_y], [car_x,car_y], [car_x,car_y], [car_x,car_y]])
        
        new_coords = trans
        
        all_out_track = True
        
        self.out_track_tire_cnt = 0
        
        for i, (xx, yy) in enumerate(new_coords):
            in_track = self._check_point_in_track(xx, yy)
            if in_track:
                all_out_track = False
            else:
                self.out_track_tire_cnt += 1
        
        return all_out_track