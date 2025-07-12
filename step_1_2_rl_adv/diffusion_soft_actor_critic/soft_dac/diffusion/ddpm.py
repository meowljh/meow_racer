import torch
import torch.nn as nn
import torch.nn.functional as F


import os, sys
diffusion_root = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.append(os.path.dirname(diffusion_root).replace('\\', '/'))

from .scheduler import LinearBetaScheduler
from ..networks.module import MLP

class DDPM(nn.Module):
    def __init__(self,
                 device,
                 #diffusion model#
                 input_dim:int,
                 hidden_size:int,
                 output_dim:int,
                 num_layers:int,
                 #time mlp#
                 time_embed_hidden_size:int,
                 time_dim:int,
                 #beta scheduling#
                 num_timesteps:int,
                 beta_1:float,
                 beta_T:float,
                 beta_schedule_mode:str='linear',
                 #reverse sampling dist#
                 reverse_sampling_dist:str='gauss',

                 **kwargs
                 ):