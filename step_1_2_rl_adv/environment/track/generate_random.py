'''track/generate_random.py
creates random racing tracks by generating randomized key points as the corner of the track.

'''
from abc import ABC, abstractmethod

import math
import numpy as np
import pickle as pkl

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from seeding import np_random

class Base_TrackGenerator(ABC):
    def __init__(self, 
                 min_num_ckpt:int,
                 max_num_ckpt:int,
                 track_radius_min_ratio:float=0.5,
                 track_radius_max_ratio:float=4.,
                 track_radius:float=200,
                 scale_rate:float=1.,
                 min_track_turn_rate:float=0.31,
                 track_turn_rate:float=0.31,
                 track_detail_step:float=15,
                 track_detail_step_check:float=15,
                 ):
        super().__init__()
        self.track_radius_ratio = [track_radius_min_ratio, track_radius_max_ratio]
        
        self.ckpt_range = [min_num_ckpt, max_num_ckpt]
        self.track_radius = track_radius
        self.np_random, _ = np_random()
        
        self.scale_rate = scale_rate
        
        self.min_track_turn_rate = min_track_turn_rate
        self.track_turn_rate = track_turn_rate
        self.track_detail_step = track_detail_step #굳이 따지자면 track을 구성하는 tile을 만들기 위한 것으로 tile의 최소 길이를 의미한다고 볼 수 있음.
        self.track_detail_step_check = track_detail_step_check
        
        self._reset()
     
    def _generate(self):
        success = False
        while not success:
            self._reset()
            self._create_checkpoints()
            self._denoise_checkpoints()
            self._validate_closed()
            success = self._validate_glued()
            
        
        
     
    def _reset(self):
        self.checkpoints = []
        self.road = []
        self.track = []

    ##### STEP1: Create the checkpoints that will be connected to generate the track #####
    def _create_checkpoints(self):
        '''
        - start_alpha: 트랙의 중앙 좌표에서 첫번째 checkpoint까지의 X축과 이루는 각도의 크기
        '''
        num_ckpts = self.np_random.choice(np.arange(*self.ckpt_range))
        self.num_ckpts = num_ckpts
        for c in range(num_ckpts):
            noise = self.np_random.uniform(0, 2*math.pi*1/num_ckpts)
            alpha = 2*math.pi*(c / num_ckpts) + noise
            
            # rad = self.np_random.uniform(self.track_radius / 3., self.track_radius * 2.)
            rad = self.np_random.uniform(self.track_radius * self.track_radius_ratio[0], 
                                         self.track_radius * self.track_radius_ratio[1])
            
            
            if c == 0:
                alpha = 0
                # rad = self.np_random.uniform(self.track_radius * (1/4), self.track_radius * (4))
                rad = self.np_random.uniform(self.track_radius * self.track_radius_ratio[0], 
                                             self.track_radius * self.track_radius_ratio[1])
                start_rad = rad
                
                
            if c == num_ckpts-1:
                alpha = 2*math.pi*2 / num_ckpts
                self.start_alpha = 2*math.pi*(-0.5) / num_ckpts
                rad = start_rad

            self.checkpoints.append((alpha,
                                     rad * math.cos(alpha),
                                     rad * math.sin(alpha),
                                     rad))
    
    ##### STEP2: Singularize the noisy track checkpoints #####
    def _denoise_checkpoints(self):
        x, y, beta = self.track_radius * 1.5, 0, 0 # 시작 좌표 (x, y)
        dest_i = 0
        laps = 0
        
        no_freeze = 2500
        visited_other_side = False # 현재 상태 
        
        while True:
            alpha = math.atan2(y, x) 
            if visited_other_side and alpha > 0: # 1, 2 사분면
                laps += 1
                visited_other_side = False
            if alpha < 0: # 3, 4 사분면
                visited_other_side = True
                alpha += 2 * math.pi
            
            while True:
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y, dest_rad = self.checkpoints[dest_i % len(self.checkpoints)]
                    # 다음 target point가 현재 (x, y) 위치보다 이전의, 즉 X축과의 반시계 방향으로의 각도가 더 작으면 안됨
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(self.checkpoints) == 0:
                        break
                
                if not failed:
                    break
                
                alpha -= 2*math.pi
                continue
            
            #checkpoint들 사이를 채우기 위해서 다음 목적지 좌표까지 도달하기 위해서 사이에 찍히는 좌표들을 지정해 주는데, beta각으로 조절한다.
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x
            dest_dy = dest_y - y
                
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            
            prev_beta = beta
            proj *= self.scale_rate

            if proj > self.min_track_turn_rate:
                beta -= min(self.track_turn_rate, abs(0.001 * proj))
            if proj < -self.min_track_turn_rate:
                beta += min(self.track_turn_rate, abs(0.001 * proj))
            
            x += p1x * self.track_detail_step
            y += p1y * self.track_detail_step
            
            self.track.append(
                (alpha, 
                 0.5 * (prev_beta + beta), #prev_beta == beta이면 그냥 beta인 것임.
                 x, y)
            )
            
            if laps > 4:break
            no_freeze -= 1
            if no_freeze == 0:break
    
    
    ##### STEP3: Find closed loop for the track coordinates #####
    def _validate_closed(self):
        i1, i2 = -1, -1
        i = len(self.track)
        while True:
            i -= 1
            if i == 0:
                return False
            pass_through_start = (
                self.track[i][0] > self.start_alpha and self.track[i-1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
                
        assert i1 != -1
        assert i2 != -1
    
        self.track = self.track[i1 : i2-1]

    
    ##### STEP4: Validate track closed #####
    def _validate_glued(self):
        first_beta = self.track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (self.track[0][2] - self.track[-1][2])) + 
            np.square(first_perp_y * (self.track[0][3] - self.track[-1][3]))
        )

        if well_glued_together > self.track_detail_step_check:
            return False
        else:
            return True
    
class Random_TrackGenerator(Base_TrackGenerator):
    def __init__(self, 
                 track_width:float, #전체너비
                 
                 min_num_ckpt:int,
                 max_num_ckpt:int,
                 track_radius:float=200,
                 scale_rate:float=1.,
                 min_track_turn_rate:float=0.31,
                 track_turn_rate:float=0.31,
                 track_detail_step:float=15,
                 track_detail_step_check:float=15,
                 
                 ):
        super().__init__(min_num_ckpt=min_num_ckpt, max_num_ckpt=max_num_ckpt, track_radius=track_radius,
                         scale_rate=scale_rate, min_track_turn_rate=min_track_turn_rate,
                         track_turn_rate=track_turn_rate, track_detail_step=track_detail_step, 
                         track_detail_step_check=track_detail_step_check
                         ) ## 여기서 reset 됨 

        self.track_width = track_width
        
        
    def _get_track_center(self):
        track_t = np.array(self.track).T
        center_x = track_t[2]
        center_y = track_t[3]
        
        return center_x, center_y
    
    def _get_track_beta(self):
        return np.array(self.track).T[1]
    
    def _get_track_sides(self):
        beta_arr = self._get_track_beta()
        cx_arr, cy_arr = self._get_track_center()
        half_track_width = self.track_width / 2
        
        vertice_arr = []
        left_X, left_Y = [], []
        right_X, right_Y = [], []
        
        # for i in range(len(self.track)):
        for i in range(len(cx_arr)):
            beta1, x1, y1 = beta_arr[i], cx_arr[i], cy_arr[i]
            beta2, x2, y2 = beta_arr[i-1], cx_arr[i-1], cy_arr[i-1]
            l1 = (x1 - half_track_width*math.cos(beta1), y1 - half_track_width*math.sin(beta1))
            r1 = (x1 + half_track_width*math.cos(beta1), y1 + half_track_width*math.sin(beta1))
            l2 = (x2 - half_track_width*math.cos(beta2), y2 - half_track_width*math.sin(beta2))
            r2 = (x2 + half_track_width*math.cos(beta2), y2 + half_track_width*math.sin(beta2))
            
            vertices = [l1, r1, r2, l2]
            left_X.append(l1[0])
            left_Y.append(l1[1])
            right_X.append(r1[0])
            right_Y.append(r1[1])
            # left_X.extend([l1[0], l2[0]])
            # left_Y.extend([l1[1], l2[1]])
            # right_X.extend([r1[0], r2[0]])
            # right_Y.extend([r1[1], r2[1]])
            
            vertice_arr.append(vertices)
        
        left_side = np.vstack((np.array(left_X),np.array(left_Y))).T #shape: (length, 2)
        right_side = np.vstack((np.array(right_X), np.array(right_Y))).T #shape: (length, 2)
        
        return vertice_arr, left_side, right_side
    
    def _get_track_kappa(self):
        '''트랙 곡률 계산'''
        X_arr, Y_arr = self._get_track_center()
        
        X_arr = np.array(X_arr)
        Y_arr = np.array(Y_arr)
    
        dx = np.gradient(X_arr)
        dy = np.gradient(Y_arr)
    
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
    
        K = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2)**(3/2)
        # K = np.abs(K)
        
        return K
    
    def _get_track_theta(self):
        '''트랙의 진행 정도 (곡선 거리)'''
        X_arr, Y_arr = self._get_track_center()
        dx = np.gradient(X_arr)
        dy = np.gradient(Y_arr)
        ds = np.sqrt(dx**2 + dy**2)
        
        theta = np.cumsum(ds)
        '''spline이던 1D던 상관 없이 interpolation을 하고자 할 때, 
        theta를 reference로 갖고 phi, kappa, point coords 등의 값을 찾게 되는데,
        theta boundary에 속하지 않는 값일 경우에는 (예를 들어서 처음 시작 theta 값이 0보다 크면 0과 그 초기 값 사이의 빈 공간에서의 연산에 오류가 생김.
        그래서 초반에 dTheta등의 값들이 잘못 계산되어 초반에 후진을 하면 오류가 누적이 되는 양상을 보이는 듯 하다.)'''
        theta = [0] + list(theta[:-1])
        theta = np.array(theta)
        
        return theta
    
    def _get_track_phi(self):
        '''트랙의 각 point에서의 회전각
        X축, 즉 (1, 0)벡터를 기준으로 측정이 되는 각임.
        phi 계산한 값에다가 pi / 2를 빼주면 beta_arr과 동일한 값이 된다.
        beta는 결과적으로 주변 트랙의 side를 계산하기 위해서 실제 두 인접한 track에서의 point들의 방향에 perpendicular한 방향의 각이라고 보면 됨'''
        X_arr, Y_arr = self._get_track_center()
        dx, dy = np.gradient(X_arr), np.gradient(Y_arr)
        phi = np.arctan2(dy, dx)
        phi = np.unwrap(phi) 
        
        return phi
    
    def _calculate_track_dict(self):
        self.cX, self.cY = self._get_track_center()
        self.vertice_arr, self.left_arr, self.right_arr = self._get_track_sides()
        self.kappa_arr = self._get_track_kappa()
        self.theta_arr = self._get_track_theta()
        self.phi_arr = self._get_track_phi()
        self.beta_arr = np.array(self.track).T[1]
        
        track_dict = {
            'theta': self.theta_arr,
            'phi': self.phi_arr,
            'beta': self.beta_arr,
            'kappa': self.kappa_arr,
            'x': self.cX, 'y': self.cY,
            'vertices': self.vertice_arr,
            'left': self.left_arr,
            'right': self.right_arr
        }
        
        return track_dict
        
        
        
class Nam_TrackGenerator(Random_TrackGenerator):
    def __init__(self, 
                 track_width:float,
                 nam_track_path:str,
                 
                 min_num_ckpt:int,
                 max_num_ckpt:int,
                 track_radius:float=200,
                 scale_rate:float=1.,
                 min_track_turn_rate:float=0.31,
                 track_turn_rate:float=0.31,
                 track_detail_step:float=15,
                 track_detail_step_check:float=15,
                 
                 ):
        super().__init__(track_width=track_width,
                         min_num_ckpt=min_num_ckpt, max_num_ckpt=max_num_ckpt, track_radius=track_radius,
                         scale_rate=scale_rate, min_track_turn_rate=min_track_turn_rate,
                         track_turn_rate=track_turn_rate, track_detail_step=track_detail_step, 
                         track_detail_step_check=track_detail_step_check
                         ) ## 여기서 reset 됨 
        self.nam_track = pkl.load(open(nam_track_path, 'rb'))
    
    def _get_track_center(self):
        x, y = np.array(self.nam_track['x']), np.array(self.nam_track['y'])
        return x, y
    
    def _get_track_beta(self):
        return self.beta
    
    def _generate(self):
        self._reset()
        self.cX, self.cY = self._get_track_center()
        # self.phi = np.array(self.nam_track['phi'])
        # self.kappa = np.array(self.nam_track['kappa'])
        # self.theta = np.array(self.nam_track['theta'])
        self.theta = np.array(self._get_track_theta())
        self.kappa = np.array(self._get_track_kappa())
        self.phi = np.array(self._get_track_phi())
        self.beta = self.phi - (np.pi / 2)
        
        
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