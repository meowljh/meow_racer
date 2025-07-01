import numpy as np
from scipy.optimize import minimize
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vehicle.tools import dist_to_center
from .base_observation import ObservationState

class Observation_Lookahead_State(ObservationState):
    '''According to the lookahead time initialized in the state object,
    will be able to calculate the coordinates that the car would be located when moving in current speed
    '''
    def __init__(self, car_obj, 
                 lookahead_time, num_states, lookahead_theta,
                 track_dict,
                 fixed:str=None):
        super().__init__()
        self._reset(car_obj)
        self.lookahead_time = lookahead_time
        self.lookahead_theta = lookahead_theta
        self.num_states = num_states
        self.track_dict = track_dict
     
        self.fixed = fixed
        
    def _reset(self, car_obj):
        self.car_obj = car_obj
    
    def _calculate_lookahead_location(self):
        bicycle_model = self.car_obj.bicycle_model
        car_x, car_y = bicycle_model.car_x, bicycle_model.car_y
        dCar_x, dCar_y = bicycle_model.dCar_x, bicycle_model.dCar_y
        
        future_x = car_x + self.lookahead_time * dCar_x
        future_y = car_y + self.lookahead_time * dCar_y
        
        return car_x, car_y, future_x, future_y
    
    def _get_close_theta(self, xx, yy):

        initial_guess = self.track_dict['theta'][np.argmin((self.track_dict['x']- xx)**2 + (self.track_dict['y'] - yy)**2)]
        if isinstance(initial_guess, int):
            initial_guess = self.track_dict[initial_guess]
        close_theta = minimize(
            lambda x: dist_to_center(self.car_obj.theta_center_spline,
                                     x[0], xx, yy),
            [initial_guess]
        ).x[0]
        
        return close_theta
    
    def _get_theta_range(self, 
                         theta_center_spline,
                         car_x, car_y, future_x, future_y):
        

        close_car_theta = self._get_close_theta(car_x, car_y)
        
        close_future_theta = self._get_close_theta(future_x, future_y)
        
        theta_range = np.linspace(close_car_theta, close_future_theta, self.num_states)
        return theta_range
    
    def _get_theta_range_fixed_theta(self):
        bm = self.car_obj.bicycle_model
        car_x, car_y = bm.car_x, bm.car_y

        close_car_theta = self._get_close_theta(car_x, car_y)
        
        theta_range = np.linspace(close_car_theta, close_car_theta + self.lookahead_theta, self.num_states)
        
        return theta_range
    
    def _get_theta_range_fixed_point(self):
        bm = self.car_obj.bicycle_model
        car_x, car_y = bm.car_x, bm.car_y
        close_ref_idx = np.argmin((self.track_dict['x']-car_x)**2 + (self.track_dict['y']-car_y)**2)
        
        track_length = len(self.track_dict['x'])
        theta_range = np.array([
            self.track_dict['theta'][min(i,track_length-1)] for i in range(close_ref_idx, close_ref_idx + self.num_states)
        ])
        
        return theta_range
        
    def _lookahead_coords(self, theta_center_spline, theta_left_spline, theta_right_spline,
                          normalize:bool=True,
                          debug:bool=False) -> np.ndarray:
        '''[TOOO] add normalization code & change coordinates to sin and cosine representations'''
        car_x, car_y, future_x, future_y = self._calculate_lookahead_location()
        theta_range = self._get_theta_range(theta_center_spline, car_x, car_y, future_x, future_y)
        
        
        lookahead_X_arr = []
        lookahead_Y_arr = []
        
        for i, theta in enumerate(theta_range):
            left_coord = theta_left_spline(theta)
            right_coord = theta_right_spline(theta)
            center_coord = theta_center_spline(theta)
            
            lookahead_X_arr.append([left_coord[0], center_coord[0], right_coord[0]])
            lookahead_Y_arr.append([left_coord[1], center_coord[1], right_coord[1]])
        
        # left X: arr[0][0::3], center X: arr[0][1::3], right X: arr[0][2::3]
        # left Y: arr[1][0::3], center Y: arr[1][1::3], right Y: arr[1][2::3]
        
        return np.vstack((lookahead_X_arr, lookahead_Y_arr)) ##(num points*3, 2)
        
        
    def _lookahead_curvature(self, theta_center_spline, kappa_spline, 
                             normalize:bool=True, #만약에 디버깅을 해서 그림을 그려야 하는 경우라면 normalize를 하지 않고 저장을 해야 할수도 있음.
                             debug:bool=False) ->  np.ndarray:
        '''
        used for state space generation (look ahead curvature values)
        @time: must be defined in seconds
        '''
        if not self.fixed:
            car_x, car_y, future_x, future_y = self._calculate_lookahead_location()
            theta_range = self._get_theta_range(theta_center_spline, car_x, car_y, future_x, future_y)
        elif self.fixed == 'theta':
            theta_range = self._get_theta_range_fixed_theta()
        elif self.fixed == 'point':
            theta_range = self._get_theta_range_fixed_point()
        
        lookahead_K_arr = [] 
        
        for i, theta in enumerate(theta_range):
            kappa = kappa_spline(theta)
            lookahead_K_arr.append(kappa)
        
        lookahead_K_arr = np.array(lookahead_K_arr)
        
        if debug:
            return future_x, future_y, theta_range, lookahead_K_arr
        
        return lookahead_K_arr
        
        