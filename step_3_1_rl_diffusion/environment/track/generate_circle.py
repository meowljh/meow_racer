import os, sys
import math
import numpy as np
from .generate_random import Random_TrackGenerator

class Circle_TrackGenerator(Random_TrackGenerator):
    def __init__(self, track_width:float,
                 min_num_ckpt:int=4, max_num_ckpt:int=16):
        super().__init__(min_num_ckpt=min_num_ckpt, max_num_ckpt=max_num_ckpt, track_width=track_width)
    
    def _generate(self, radius, d_angle_deg, 
                  straight_length=None,
                  d_straight=None,
                  initial_phi_deg=None,
                  start_deg=0):
        self._reset()
        if straight_length is not None:
            self.cX, self.cY = self._generate_oval_points(
                radius, straight_length, d_angle_deg, d_straight, 
                initial_phi_deg
            )
        else:
            self.cX, self.cY = self._generate_circle_points(radius, d_angle_deg, start_deg)
        self.phi = self._get_track_phi()
        self.beta = self._get_track_beta()
        self.kappa = self._get_track_kappa()
        self.theta = self._get_track_theta()
        
    def _calculate_track_dict(self):
        self.vertice_arr, self.left_arr, self.right_arr = self._get_track_sides()
        track_dict = {
            'theta': self.theta, 
            'phi': self.phi,
            'beta': self.beta,
            'kappa': self.kappa,
            'x': self.cX, 'y': self.cY,
            'vertices': self.vertice_arr,
            'left': self.left_arr,
            'right': self.right_arr,
            'passed': np.zeros_like(self.theta)
        }
        return track_dict

    def _generate_oval_points(self, 
                              radius, straight_length, 
                              d_angle_deg, d_straight,
                              initial_phi_deg:float=0.):

        x, y = radius, -straight_length / 2

        #### (1) Right Part ####
        center_x, center_y = [x], [y]
        straight_N = straight_length / d_straight
        for n in range(int(straight_N)):
            x += d_straight * math.cos(math.radians(initial_phi_deg))
            y += d_straight * math.sin(math.radians(initial_phi_deg))
            center_x.append(x);center_y.append(y)

        circle_cx = center_x[-1] + math.cos(math.radians(90+initial_phi_deg)) * radius
        circle_cy = center_y[-1] + math.sin(math.radians(90+initial_phi_deg)) * radius

        circle_N = 180 / d_angle_deg
        deg = initial_phi_deg - 90

        for n in range(int(circle_N)):
            rad = math.radians(deg + n * d_angle_deg)
            center_x.append(circle_cx + radius*math.cos(rad))
            center_y.append(circle_cy + radius*math.sin(rad))


        #### (2) Left Part ####
        last_x, last_y = center_x[-1], center_y[-1]
        for n in range(int(straight_N)):
            last_x += d_straight * math.cos(math.radians(180 + initial_phi_deg))
            last_y += d_straight * math.sin(math.radians(180 + initial_phi_deg))
            center_x.append(last_x)
            center_y.append(last_y) 

        circle_cx = center_x[-1] + math.cos(math.radians(270+initial_phi_deg)) * radius
        circle_cy = center_y[-1] + math.sin(math.radians(270+initial_phi_deg)) * radius
        deg = initial_phi_deg + 90

        for n in range(int(circle_N)):
            rad = math.radians(deg + n * d_angle_deg)
            center_x.append(circle_cx + radius*math.cos(rad))
            center_y.append(circle_cy + radius*math.sin(rad))

        return np.array(center_x), np.array(center_y)

    def _generate_circle_points(self, radius, d_angle_deg, start_deg:float=0):
        N = int(360 // d_angle_deg)
        center_x, center_y = [], []
        for n in range(N):
            deg = d_angle_deg * n + start_deg
            x = radius * math.cos(math.radians(deg))
            y = radius * math.sin(math.radians(deg))
            center_x.append(x)
            center_y.append(y)
        
        return np.array(center_x), np.array(center_y) 
        
    def _get_track_center(self):
        return self.cX, self.cY
    
    def _get_track_beta(self):
        phi_arr = self._get_track_phi()
        beta_arr = phi_arr - (np.pi/2)
        return beta_arr
    