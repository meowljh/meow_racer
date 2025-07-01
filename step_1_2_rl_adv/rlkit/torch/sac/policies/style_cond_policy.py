import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import TanhNormal
LOG_SIG_MAX=2
LOG_SIG_MIN=-20

from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.sac.policies.base import TorchStochasticPolicy

"""

"""

class TanhGaussianPolicy_StyleCond(Mlp, TorchStochasticPolicy):
    """
    Actor object for SAC, using Tanh squashing function and Gaussian distribution policy
    - "Style" conditioned
    - style condition factor can be either a scalar, or a vector
    - the "style" factor is necessary for sampling race driving policies of "diverse racing skills"
    """
    def __init__(self,
                 hidden_sizes, 
                 obs_dim:int, 
                 action_dim:int, 
                 style_dim:int,
                 init_w:float=1e-3,
                 layer_norm:bool=False,
                 std=None, 
                 **kwargs):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim + style_dim, 
            output_size=action_dim,
            init_w=init_w,
            layer_norm=layer_norm,
            **kwargs
        )
        self.style_dim=style_dim
        self.obs_dim = obs_dim
        self.std = std
        self.log_std = None
        
        if self.std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(self.std)
            assert LOG_SIG_MIN < self.log_std <= LOG_SIG_MAX
        
    
    def forward(self, obs_with_Z):
        # h = torch.cat((obs, Z_style), dim=1) #(B, obs_dim) (B, style_dim)
        # h = torch.cat(inputs, dim=1) #(B, obs_dim+style_dim)
        assert obs_with_Z.shape[-1] == self.obs_dim + self.style_dim
        h = obs_with_Z
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(
                np.array([self.std]).float().to(ptu.device)
                )

        return TanhNormal(
            normal_mean=mean,
            normal_std=std
        )
            
    