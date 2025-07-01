import numpy as np
import math
import sys, os
from scipy.optimize import minimize

from .base_observation import ObservationState
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vehicle.tools import dist_to_center
class Observation_Lidar_State(ObservationState):
    def __init__(self,
                 car_obj,
                 track_dict,
                 ## angle in radians ##
                 lidar_angle_min, 
                 lidar_angle_max,
                 num_lidar:int,
                 max_lidar_length:float,
                 ):
        super().__init__() 
        lidar_angles = np.linspace(lidar_angle_min, lidar_angle_max, num_lidar)
        self.lidar_angles = np.array([math.radians(angle) for angle in lidar_angles])
        self.max_range = max_lidar_length
        
        self._reset(track_dict, car_obj)
    
    def _reset(self, track_dict, car_obj):
        self.car_obj = car_obj
        self.track_dict = track_dict
        self.wall_points_arr = [
            np.array(self.track_dict['left']), np.array(self.track_dict['right'])
        ] 
        self.lidar_results = []
        
        
        
        
    def _step(self):
        car_x = self.car_obj.bicycle_model.car_x
        car_y = self.car_obj.bicycle_model.car_y
        phi = self.car_obj.bicycle_model.car_phi
        results = self.lidar_scan(car_x,
                                 car_y,
                                 phi)
        self.lidar_results  = results
        
        return results
        
        
    def get_lidar_ray(self, car_x, car_y, yaw, lidar_angle):
        angle = yaw + lidar_angle 
        x1 = car_x + self.max_range * np.cos(angle)
        y1 = car_y + self.max_range * np.sin(angle)
        
        return (car_x, car_y), (x1, y1) # lidar의 시작 point / lidar의 끝 point

    def segment_intersection(self, p1, p2, q1, q2):
    # p1-p2: ray, q1-q2: wall segment
    
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = q2[0] - q1[0]
        dy2 = q2[1] - q1[1]
    
        delta = det((dx1, dy1), (dx2, dy2))
        if delta == 0:
            return None
    
        s = det((q1[0] - p1[0], q1[1] - p1[1]), (dx2, dy2)) / delta
        t = det((q1[0] - p1[0], q1[1] - p1[1]), (dx1, dy1)) / delta
    
        if s >= 0 and 0 <= t <= 1:
            ix = p1[0] + s * dx1
            iy = p1[1] + s * dy1
            return (ix, iy)
        else:
            return None
        
    def find_closest_intersection(self, car_x, car_y, yaw, lidar_angle, wall_points, 
                                  theta_limit):
        ray_start, ray_end = self.get_lidar_ray(car_x, car_y, yaw, lidar_angle)
        closest_point = None
        min_dist = float('inf')
    
        for i in range(len(wall_points)-1):
            '''[0415] added additional limits to the lidar sensor
            -> if the theta range of the ray end point is not limited, then the ray checks on the points on the other side of the track'''
            if theta_limit[0] > self.track_dict['theta'][i] or theta_limit[1] < self.track_dict['theta'][i]:
                continue
            wall_start = wall_points[i]
            wall_end = wall_points[i+1]
            intersect = self.segment_intersection(p1=ray_start, p2=ray_end, q1=wall_start, q2=wall_end)
            if intersect is not None:
                dist = np.linalg.norm(np.array(intersect) - np.array((car_x, car_y)))
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point = intersect
    
    
        return closest_point, min_dist

    def lidar_scan(self, 
               car_x, car_y, yaw,  ):
        results = []
        '''[TODO]
        minor fixed for more concrete algorithm for track generation
        -> 차량이 초반에 자리하고 있을 때를 보면 track의 끝부분과 시작이 완벽하게 closed form이 아니기 때문인지 뒤에는 감지되는 track의 side영역이 없었음.
        확인 필요'''

        initial_guess = self.track_dict['theta'][np.argmin((self.track_dict['x']- car_x)**2 + (self.track_dict['y'] - car_y)**2)]
        if isinstance(initial_guess, int):
            initial_guess = self.track_dict[initial_guess]
        close_car_theta = minimize(
            lambda x: dist_to_center(self.car_obj.theta_center_spline,
                                     x[0], car_x, car_y),
            [initial_guess]
        ).x[0]
    
        for angle in self.lidar_angles:
            min_dist = float('inf');min_point = None
            for i, wall_point in enumerate(self.wall_points_arr): ##left wall, right wall##
                point, distance = self.find_closest_intersection(car_x, car_y, yaw, angle, wall_point, 
                                                                 theta_limit = [close_car_theta-10, 
                                                                                close_car_theta + 100])
                if min_dist > distance:
                    min_dist = distance
                    min_point = point
            
            if min_point is not None:
                results.append({
                    "angle": angle, "point": min_point, "distance": min_dist
                })
            else:
                results.append({
                    "angle": -1, "point": [car_x, car_y], "distance": -1
                }) 
        
        return results