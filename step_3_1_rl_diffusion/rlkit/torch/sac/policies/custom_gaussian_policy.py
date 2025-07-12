import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull
)
LOG_SIG_MAX=2
LOG_SIG_MIN=-20

from rlkit.torch.networks.mlp import Mlp, SplitIntoManyHeads
from rlkit.torch.sac.policies.base import TorchStochasticPolicy




class TanhGaussianPolicy_MultiHead_Mlp(Mlp, TorchStochasticPolicy):
    def __init__(self,
                 hidden_sizes, obs_dim, action_dim,
                 head_conf_dict,
                 init_w:float=1e-3,
                 layer_norm:bool=False,
                 std=None, **kwargs):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim, output_size=action_dim,
            init_w=1e-3, 
            layer_norm=layer_norm,
            **kwargs
        )
        del self.last_fc
        self.init_w = init_w
        self.std = std
        self.hidden_sizes = hidden_sizes
        self.head_conf_dict = head_conf_dict
        self.last_fc_dict, self.last_fc_std_dict = self._build_heads()
    
    def _set_std_learning(self):
        '''policy에서 std 고정을 하다가 어느 정도 안정이 되면 같이 학습 되도록 하기'''
        pass
    
    def _build_heads(self):
        '''head building 시에 module들을 그냥 dict에 넣으면 layer weight들에 device 할당이 안됨.
        그래서 ModuleDict()에 저장을 해 두어야 함.'''
        last_fc_dict = torch.nn.ModuleDict()
        last_fc_std_dict = torch.nn.ModuleDict()
        
        last_hidden_size = self.input_size
        if len(self.hidden_sizes) > 0:
            last_hidden_size = self.hidden_sizes[-1]
            
        for action_name, action_conf in self.head_conf_dict.items():
            fc = nn.Linear(last_hidden_size, action_conf['action_dim'])
            # fc.weight.data.uniform_(-self.init_w, self.init_w)
            # fc.bias.data.uniform_(action_conf['init_w_min'], action_conf['init_w_max'])
            fc.weight.data.uniform_(action_conf['init_w_min'], action_conf['init_w_max'])
            fc.bias.data.fill_(0)
            
            if action_conf['std'] is None:
                fc_log_std = nn.Linear(last_hidden_size, action_conf['action_dim'])
                fc_log_std.weight.data.uniform_(-self.init_w, self.init_w)
                fc_log_std.bias.data.uniform_(-self.init_w, self.init_w)
            else:
                self.log_std = np.log(action_conf['std'])
                fc_log_std = None
                assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
                
            last_fc_dict[action_name] = fc
            last_fc_std_dict[action_name] = fc_log_std
        
        return last_fc_dict, last_fc_std_dict
        
    def forward(self, obs):
        '''Policies returns the distributions based on the parameters estimated 
        with the network layers. 
        For the gaussian distribution, the network estimates mean & log_std value for all action dimensions'''
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        
        mean_arr = [] # torch.zeros((obs.shape[0], self.output_size)).float().to(ptu.device)
        std_arr = [] #  torch.zeros((obs.shape[0], self.output_size)).float().to(ptu.device)
        
        for action_name in self.head_conf_dict.keys():
            fc = self.last_fc_dict[action_name]
            mean = fc(h)
            std_fc = self.last_fc_std_dict[action_name]
            if std_fc is not None:
                log_std = std_fc(h)
                log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
                std = torch.exp(log_std).reshape(-1, 1)
            else:
                std = torch.from_numpy(np.array([self.std])).float().to(ptu.device).reshape(-1, 1)
            
            mean_arr.append(mean)
            std_arr.append(std)
        
        mean = torch.cat(mean_arr, dim=-1)
        std = torch.cat(std_arr, dim=-1)
        
        dist = TanhNormal(normal_mean=mean, normal_std=std)
        
        return dist
    
    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob
            
            
                
            
            