import torch
import torch.nn as nn

from collections.abc import Sequence


import os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #soft_dac/
sys.path.append(root)
from networks.module import init_weights

class Critic(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:type[int], num_layers:type[int]):
        super().__init__()
        inp_dim = state_dim + action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = 1 #critic은 Q-value값을 내뱉는 것이 목적이기 때문에 output dimension의 크기가 1이어야 함.

        self.q1 = self._build_network(inp_dim=inp_dim)
        self.q2 = self._build_network(inp_dim=inp_dim)

        self.apply(init_weights)
    
    def _build_network(self, inp_dim):
        net_list = []
        inp_dim_list = [inp_dim] + [self.hidden_dim for _ in range(self.num_layers)]
        out_dim_list = [self.hidden_dim for _ in range(self.num_layers)] + [self.out_dim]

        for i, (id, od) in enumerate(zip(inp_dim_list, out_dim_list)):
            net_list.append(nn.Linear(id, od))
            if i != len(inp_dim_list)-1:
                net_list.append(nn.Mish())
            
        return nn.Sequential(*net_list)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)

    def q2(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q2(sa)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
        
# class Critic_MultiStyle(Critic):
#     def __init__(self, state_dim, action_dim, hidden_dim:type[Sequence],
#                  style_dim:int):
#         super().__init__(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
#         inp_dim = state_dim + action_dim + style_dim
#         self.inp_dim = inp_dim
        
#         self.q1 = self._build_network(inp_dim=inp_dim)
#         self.q2 = self._build_network(inp_dim=inp_dim)
    
#     def forward(self, style_state, action):
#         ssa = torch.cat([style_state, action], dim=-1)
#         assert ssa.shape[-1] == self.inp_dim

#         return self.q1(ssa), self.q2(ssa)