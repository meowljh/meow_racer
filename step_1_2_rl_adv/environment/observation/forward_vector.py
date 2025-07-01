import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
import os, sys

# sys.path.append("..")
from .base_observation import ObservationState
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vehicle.tools import dist_to_center

class Observation_ForwardVector_State(ObservationState):
    def __init__(self, 
                 car_obj,
                 track_dict,
                 theta_diff:float, 
                 num_vecs:int,
                 rotate_vec:bool=True,
                 **kwargs):
        
        super().__init__()
        self._reset(car_obj=car_obj, track_dict=track_dict)
        self.theta_diff = theta_diff
        self.num_vecs = num_vecs
        self.rotate_vec = rotate_vec
        
    def _reset(self, car_obj, track_dict):
        self.car_obj = car_obj
        self.track_dict = track_dict
        self.vector_dict = defaultdict(list) #대부분 이런거 저장하는 이유는 visualization 때문#
    
    def rotate(self, x, y, yaw_angle):
        '''vehicle의 yaw 각도만큼 시계 방향으로 회전을 해서
        차량 좌표계를 고정할 수 있도록 한다.'''
        R = np.array([
            [np.cos(-yaw_angle), -np.sin(-yaw_angle)],
            [np.sin(-yaw_angle), np.cos(-yaw_angle)]
        ])
        
        rotated = R@np.array([x, y]).T
        x1, y1 = rotated
        
        return x1, y1
    
        
    def _step(self, theta_center_spline, debug:bool=False):
        '''
        (1) get the closest point on the track center relative to the vehicle
        (2) from the closest point on track, get the center points forward using the theta spline
        '''
        bicycle_model = self.car_obj.bicycle_model
        car_x, car_y = bicycle_model.car_x, bicycle_model.car_y
        car_phi = bicycle_model.car_phi
        
        '''[0418] Troubleshooting - on reference point finding
        '''

        initial_guess = self.track_dict['theta'][np.argmin((self.track_dict['x']- car_x)**2 + (self.track_dict['y'] - car_y)**2)]
        if isinstance(initial_guess, int):
            initial_guess = self.track_dict[initial_guess]
        close_car_theta = minimize(
            lambda x: dist_to_center(self.car_obj.theta_center_spline,
                                     x[0], car_x, car_y),
            [initial_guess]
        ).x[0]
        forward_vector_arr = []
        vector_dict = defaultdict(list) #매번 vector_dict는 초기화
        
        next_theta = close_car_theta
        for i in range(self.num_vecs):
            next_theta += self.theta_diff
            next_center_x, next_center_y = theta_center_spline(next_theta)
            vec_x, vec_y = next_center_x - car_x, next_center_y - car_y
            r_vec_x, r_vec_y = self.rotate(vec_x, vec_y,  yaw_angle=car_phi)
            
            if self.rotate_vec:
                forward_vector_arr.extend([r_vec_x, r_vec_y])
            else:
                forward_vector_arr.extend([vec_x, vec_y])
                
            vector_dict['fixed_body'].append([r_vec_x, r_vec_y]) #차체 고정 좌표 사용을 위해 회전함
            vector_dict['inertia'].append([vec_x, vec_y]) #시각화 할 때는 이걸로 (이전 mdp space에서는 이 vector을 그대로 observation으로 사용했었음.)
            
        
        forward_vector_arr = np.array(forward_vector_arr)
        self.vector_dict = vector_dict
 
        
        return forward_vector_arr
