import math
import matplotlib.pyplot as plt
from abc import ABC
import pickle
import os, sys
from PIL import Image

class RacerLogger(ABC):
    def __init__(self, env):
        super().__init__()
        self.env = env.unwrapped
        self.car = self.env.car
        self.track_dict = self.env.track_dict

        self._reset()
        
    def _reset(self):
        self.car_traces = [] #[x,y] will be appended
        self.car_status = [] #[vx, vy] will be appended
        self.car_error_c = [] #e_c will be appended
        self.car_error_phi = [] #e_phi will be appended
        

    def _update_logs(self):
        bm = self.car.bicycle_model
        car_x, car_y, car_phi = bm.car_x, bm.car_y, bm.car_phi
        self.car_traces.append([car_x, car_y, car_phi])
        car_vx, car_vy, car_ax, car_ay = bm.Vx, bm.Vy, bm.acc_x, bm.acc_y
        self.car_status.append([car_vx, car_vy, car_ax, car_ay])
        e_c, e_phi = bm.E_c, bm.E_phi
        self.car_error_c.append(e_c)
        self.car_error_phi.append(e_phi)
        
        
    def _log_trajectory_screen(self, log_file_root, log_file_prefix, num_step:int):
        '''the tensorboard logger used during training logs images only on the tensorboard server
        however, the images during simulation for nam-c track should be saved locally for future evaluation
        [TODO]
        1. Faster implementation of plotting (changing the visualization library to dash?)
        2. Real-Time interaction (but no bottleneck should be created)
        3. 
        '''
        render_root = f"{log_file_root}/{log_file_prefix}/render";os.makedirs(render_root, exist_ok=True)
        # global_image = self.env._global_car_on_track(return_image=True)
        local_image = self.env._local_car_on_track(return_image=True)
        
        # Image.fromarray(global_image).save(f"{render_root}/{num_step}_global.png")
        Image.fromarray(local_image).save(f"{render_root}/{num_step}_local.png")
        
    def _dump_sarsa(self,file, log_file_root, log_file_prefix):
        main_root = f"{log_file_root}/{log_file_prefix}";os.makedirs(main_root, exist_ok=True)
        
        sarsa_pkl_fname = f"{main_root}/sarsa.pkl"
        with open(sarsa_pkl_fname, 'wb') as f:
            pickle.dump(file, f)
        
    def _dump_actions(self, log_file_root, log_file_prefix):
        main_root = f"{log_file_root}/{log_file_prefix}";os.makedirs(main_root, exist_ok=True)
        '''simple! just saves the action logs and control values converted from the actions into a pickle'''
        action_dict = self.car.actions
        action_pkl_fname = f"{main_root}/ActionDim{self.car.action_dim}_RawAction.pkl"
        with open(action_pkl_fname, 'wb') as f:
            pickle.dump(action_dict, f)
        
        control_dict = self.car.action2control
        control_pkl_fname = f"{main_root}/ActionDim{self.car.action_dim}_RawControl.pkl"
        with open(control_pkl_fname, 'wb') as f:
            pickle.dump(control_dict, f)
        
    def _dump_trajectory(self, log_file_root, log_file_prefix):
        main_root = f"{log_file_root}/{log_file_prefix}";os.makedirs(main_root, exist_ok=True)
        
        state_pkl_fname = f"{main_root}/car_state.pkl"
        with open(state_pkl_fname, 'wb') as f:
            pickle.dump(self.car_traces, f)
            
        status_pkl_fname = f"{main_root}/car_status.pkl"
        with open(status_pkl_fname, 'wb') as f:
            pickle.dump(self.car_status, f)
            
        ec_pkl_fname = f"{main_root}/e_c.pkl"
        with open(ec_pkl_fname, 'wb') as f:
            pickle.dump(self.car_error_c, f)
            
        ephi_pkl_fname = f"{main_root}/e_phi.pkl"
        with open(ephi_pkl_fname, 'wb') as f:
            pickle.dump(self.car_error_phi, f)
        
        

    
    
        