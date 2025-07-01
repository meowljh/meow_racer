import numpy as np
import os, sys
import math
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

import heapq
from envs.gym_car_constants import *

class ERGLogger:
    def __init__(self,
                 car_obj,
                 env):
        super().__init__()
        self.car = car_obj
        self.env = env
        self.time = 0
        self.prev_vx = 0.
        self.prev_vy = 0.
        self.prev_x = 0
        self.prev_y = 0
        
        self.sroad = 0
        
        self.dt = 1./FPS
    
        self.log_data = {
            'Time': [],
            'Action.Gas': [],
            'Action.Brake': [],
            'Action.Steer.Ang': [],
            'DM.Gas': [],
            'DM.Brake': [],
            'DM.Steer.Ang': [],
            'Vhcl.sRoad': [],
            'Vhcl.tRoad': [],
            'Car.v': [],
            'Car.vx': [],
            'Car.vy': [],
            'Car.tx': [],
            'Car.ty': [],
            'Car.ax': [],
            'Car.ay': []
        }
    
    def _calc_car_velocity(self):
        velocity_vector = self.car.wheels[0].linearVelocity
        return math.sqrt(velocity_vector[0]**2 + velocity_vector[1]**2)
    
    def _closest_coord(self, car_x, car_y):
        X, Y = self.env.track_dict['x'], self.env.track_dict['y']
        PHI = self.env.track_dict['phi']
        XY = np.vstack((X, Y)).T #(N,2)
        car_pos = np.array([car_x, car_y])
        dist_q = [(math.dist(XY[i], car_pos), i) for i in range(len(XY))]
        heapq.heapify(dist_q)
        _, min_idx = heapq.heappop(dist_q)
        # print(f"carX:{car_x}   carY:{car_y}    minDist:{_}")
   
        return X[min_idx], Y[min_idx], PHI[min_idx]
        
        
    def _calc_troad(self, cx, cy):
        close_x, close_y, close_phi = self._closest_coord(car_x=cx, car_y=cy)
        
        dist = math.dist((cx, cy), (close_x, close_y))
        # print(dist)
        if close_phi > math.pi: #X값이 중앙선보다 작으면 왼쪽 -> troad 양수#
            if cx < close_x:
                return dist
            else:
                return -dist
        else: #X값이 중앙선보다 크면 왼쪽 -> troad 양수#
            if cx > close_x:
                return dist
            else:
                return -dist
                 
    
    
    def log_single_step(self, action):
        action = self.env._preprocess_action(action)
        # breakpoint()
        if len(action.shape) == 2:
            action = action.squeeze()
            
        steer_action = -action[0]
        gas_action = action[1]
        brake_action = action[2]
        cx, cy = self.car.hull.position
        vx, vy = self.car.wheels[0].linearVelocity
        self.sroad += math.dist((cx, cy), (self.prev_x, self.prev_y))
        
        self.log_data['Time'].append(self.time);self.time += self.dt #단위: s#
        self.log_data['Action.Gas'].append(np.clip(gas_action, 0., +1.))
        self.log_data['Action.Brake'].append(brake_action)
        self.log_data['Action.Steer.Ang'].append(steer_action)
        self.log_data['DM.Gas'].append(self.car.wheels[2].gas)
        self.log_data['DM.Brake'].append(self.car.wheels[0].brake)
        self.log_data['DM.Steer.Ang'].append(self.car.wheels[0].joint.angle) #단위: rad#
        self.log_data['Vhcl.sRoad'].append(self.sroad) #단위: m#(vehicle route/path coordinate)
        self.log_data['Vhcl.tRoad'].append(self._calc_troad(cx=cx, cy=cy)) #단위: m#(lateral distance to route centerline)
    
        self.log_data['Car.v'].append(self._calc_car_velocity()) #단위: m/s#
        self.log_data['Car.vx'].append(vx) #단위: m/s#
        self.log_data['Car.vy'].append(vy) #단위: m/s#

        self.log_data['Car.tx'].append(cx) #단위: m#
        self.log_data['Car.ty'].append(cy) #단위: m#
        self.log_data['Car.ax'].append((vx-self.prev_vx) / self.dt) #단위: m/s^2#
        self.log_data['Car.ay'].append((vy-self.prev_vy) / self.dt) #단위: m/s^2#
        
        self.prev_vx, self.prev_vy = vx, vy
        self.prev_x, self.prev_y = cx, cy
        