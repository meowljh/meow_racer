from torch.utils.tensorboard import SummaryWriter
from abc import ABC
from pathlib import Path
import torchvision

class Tensorboard_Plotter_Obj(ABC):
    def __init__(self, log_dir, action_dim):
        super().__init__()
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.action_dim = action_dim
        self.writer = SummaryWriter(
            log_dir=log_dir
        )
        self.rollout_idx = 0
        self.new_rollout = True
        self.global_step = 0
    
    def _log_simulation(self, simulate_arr, tag_name): 
        self.writer.add_image(tag=tag_name, img_tensor=simulate_arr, dataformats='HWC')
        
    def _log_while_train(self, log_dict):
        '''training 중에 loss, auto-tuned alpha(max entropy parameter) 등과 같은 값을 logging 함.'''
        for key, value in log_dict.items():
            self.writer.add_scalar(tag=key, scalar_value=value)
        
    def _update_global_step(self):
        self.global_step += 1
        self.new_rollout = False
        
    def _log_action(self, action_arr, tag_prefix:str=None):
        tag_prefix = '' if tag_prefix is None else tag_prefix + "_"
        if self.action_dim == 2:
            steer_action, torque_action = action_arr
            self.writer.add_scalar(tag=f'{tag_prefix}steer', scalar_value=steer_action, global_step=self.global_step)
            self.writer.add_scalar(tag=f'{tag_prefix}torque', scalar_value=torque_action, global_step=self.global_step)
            
        elif self.action_dim == 3:
            steer_action, throttle_action, brake_action = action_arr
            self.writer.add_scalar(tag=f'{tag_prefix}steer', scalar_value=steer_action, global_step=self.global_step)
            self.writer.add_scalar(tag=f'{tag_prefix}throttle', scalar_value=throttle_action, global_step=self.global_step)
            self.writer.add_scalar(tag=f'{tag_prefix}brake', scalar_value=brake_action, global_step=self.global_step)
        
    def _log_reward(self, reward):
        self.writer.add_scalar(tag='step_reward', scalar_value=reward, global_step=self.global_step)
    
    def _reset_rollout(self):
        '''근데 tensorboard에 logging할 때 이 함수가 필요가 없는게, 어차피 연속적으로 plotting을 하고 episode 구분을 할 필요가 없기 때문이다.'''
        self.global_step = 0
        self.new_rollout = True
        self.rollout_idx += 1
    
            

        
        