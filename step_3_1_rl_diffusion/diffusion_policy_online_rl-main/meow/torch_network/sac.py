from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from torch_network.common import WithSquashedGaussianPolicy
from torch_network.blocks import QNet, PolicyNet, MultiStyleQNet

 
class SACNet(WithSquashedGaussianPolicy):
    def __init__(self, 
                 device,
                 obs_dim: int,
                 act_dim: int,
                 style_dim: int,
                 hidden_sizes: Sequence[int],
                 activation_fn: str,
                 **kwargs):
        super().__init__()
        self.device = device

        if style_dim == 0:
            self.q1 = QNet(input_size=obs_dim+act_dim, hidden_sizes=hidden_sizes, activation_fn=activation_fn, output_activation_fn='identity').to(device)
            self.q2 = QNet(input_size=obs_dim+act_dim, hidden_sizes=hidden_sizes, activation_fn=activation_fn, output_activation_fn='identity').to(device)
            self.target_q1 = QNet(input_size=obs_dim+act_dim, hidden_sizes=hidden_sizes, activation_fn=activation_fn, output_activation_fn='identity').to(device)
            self.target_q2 = QNet(input_size=obs_dim+act_dim, hidden_sizes=hidden_sizes, activation_fn=activation_fn, output_activation_fn='identity').to(device)

            '''TODO
            (1) implement MultiStyleQNet
                - style conditioned vector 
                - different network for each style 
            '''
        else:
            self.q1 = MultiStyleQNet().to(device)
        
        
        self.policy = PolicyNet(action_dim=act_dim, state_dim=obs_dim, style_dim=style_dim,
                                hidden_sizes=hidden_sizes, activation_fn=activation_fn,
                                output_activation_fn='identity').to(device)
        

        self.target_entropy = -act_dim
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
