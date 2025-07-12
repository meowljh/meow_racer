import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from seeding import np_random
import bezier
from .generate_random import Base_TrackGenerator, Random_TrackGenerator
class Bezier_TrackGenerator(Random_TrackGenerator):
    def __init__(self,
                 min_num_ckpt:int, 
                 max_num_ckpt:int,
                 min_kappa:float=0.04,
                 max_kappa:float=0.1,
                 track_width:float=7.,
                 track_density:int=1,
                 track_radius:float=200.,
                 scale_rate:float=1.,
                 **kwargs,
                 ):
        '''Random Track Generation with Bezier Curve Fitting
        
        '''
        super().__init__(min_num_ckpt=min_num_ckpt, max_num_ckpt=max_num_ckpt,
                         track_radius=track_radius, scale_rate=scale_rate,
                         track_width=track_width)
        self.track_density = track_density
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa
        
    def _generate(self):
        success = False
        while not success:
            self._reset()
            self._create_checkpoints()
            success = self._validate_checkpoints()
            self._connect_checkpoints()
            success &= self._check_intersection()
            success &= self._validate_kappa()
    
    def _get_track_beta(self):
        phi_arr = self._get_track_phi()
        return phi_arr - np.pi/2
     
    def _calculate_theta(self, track_coords):
        X_arr, Y_arr = self._get_track_center()
        dx = np.gradient(X_arr)
        dy = np.gradient(Y_arr)
        ds = np.sqrt(dx**2 + dy**2)
        
        theta = np.cumsum(ds)
        
        return theta
        
    def _calculate_track_dict(self):
        self.cX, self.cY = self._get_track_center()
        self.vertice_arr, self.left_arr, self.right_arr = self._get_track_sides()
        self.kappa_arr = self._get_track_kappa()
        self.theta_arr = self._get_track_theta()
        self.phi_arr = self._get_track_phi()
        self.beta_arr = self._get_track_beta()
        
        track_dict = {
            'theta': self.theta_arr,
            'phi': self.phi_arr,
            'beta': self.beta_arr,
            'kappa': self.kappa_arr,
            'x': self.cX, 'y': self.cY,
            'vertices': self.vertice_arr,
            'left': self.left_arr,
            'right': self.right_arr,
            'passed': np.zeros_like(self.theta_arr)
        }
        
        return track_dict
            
    def _get_track_center(self):
        return self.cX, self.cY
    
    def _check_intersection(self):
        for i in range(len(self.curves)-1):
            for j in range(i+1, len(self.curves)):
                inter = self.curves[i].intersect(self.curves[j])
                if len(inter.T) > 1:
                    return False

        return True
    
    def _validate_kappa(self):
        '''너무 급격한 curve는 피하도록,
        그렇지만 남양 트랙의 hair pin corner에 해당하는 난이도의 corner은 무조건 포함해야 함'''
        kappa = self._get_track_kappa()
        if len(np.where(np.abs(kappa) > self.max_kappa)[0]) > 0:
            return False
        if len(np.where(np.abs(kappa) > self.min_kappa)[0]) == 0:
            return False
        
        return True
    
    def _validate_checkpoints(self):
        for i, (alpha, _, _, _) in enumerate(self.checkpoints):
            if alpha > np.pi * 2:
                break
            
        self.checkpoints = self.checkpoints[:i]
        if len(self.checkpoints) < 3:
            return False
        return True
    
    def _connect_checkpoints(self):
        X, Y = np.array(self.checkpoints).T[1], np.array(self.checkpoints).T[2]
        self.initial_coords = np.array(self.checkpoints).T[1:3]
        
        arr = np.vstack((X, Y)).T
        mids = (arr[:-1] + arr[1:]) / 2
        last = 0.5 * (arr[0] + arr[-1])

        mid_points = np.concatenate([mids, [last]])
        
        final_points = None

        mid_X, mid_Y = mid_points.T[0],mid_points.T[1]
        all_X, all_Y = np.zeros(len(mid_X)*2), np.zeros(len(mid_Y)*2)
        np.put(all_X, [i for i in range(1, len(mid_X)*2, 2)], np.concatenate([X[1:], [X[0]]]));np.put(all_X, [i for i in range(0, len(mid_X)*2, 2)], mid_X)
        np.put(all_Y, [i for i in range(1, len(mid_Y)*2, 2)], np.concatenate([Y[1:], [Y[0]]]));np.put(all_Y, [i for i in range(0, len(mid_Y)*2, 2)], mid_Y)

        x_arr = np.hstack((all_X, all_X))
        y_arr = np.hstack((all_Y, all_Y))

        self.all_X = all_X
        self.all_Y = all_Y
        
        curves = []
        
        for start_idx in range(0, len(all_X), 2):
            end_idx = start_idx + 3
    
            nodes = np.asfortranarray([
                x_arr[start_idx:end_idx], y_arr[start_idx:end_idx]
            ])
            curve = bezier.Curve(nodes, degree=2)
    
            mx, Mx = min(x_arr[start_idx:end_idx]), max(x_arr[start_idx:end_idx])
            my, My = min(y_arr[start_idx:end_idx]), max(y_arr[start_idx:end_idx])
            diffX, diffY = Mx-mx, My-my
            diff_val = max(diffX, diffY)
            s_vals = np.linspace(0., 1., int(diff_val)*self.track_density)
            points = curve.evaluate_multi(s_vals)
       
            if final_points is None:
                final_points = points
            else:
                final_points = np.hstack((final_points, points[:, 1:]))

            curves.append(curve)
            
        self.final_points = final_points
        self.cX = final_points[0]
        self.cY = final_points[1]
        
        self.curves = curves
        
            
    