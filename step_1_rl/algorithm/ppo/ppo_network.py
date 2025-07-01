import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym


class PPO_MLP_Value(nn.Module):
    #critic#
    #State-Value function#
    def __init__(self,
                 state_space,
                 action_space,
                 hidden_dims, 
                 hidden_activation=nn.ReLU()
    ):
        super(PPO_MLP_Value, self).__init__()
        state_dim = state_space.shape[0]
        self.value_head = nn.Linear(state_dim, hidden_dims[0])
        action_dim = action_space.shape[0]
        self.value_layers = []

        for i in range(len(hidden_dims)-1):
            self.value_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.value_layers.append(hidden_activation)
        
        self.value_layers = nn.Sequential(*self.value_layers)

        self.value_fc = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, state):
        state = self.value_head(state)
        state = self.value_layers(state)
        
        state_value = self.value_fc(state)
        
        return state_value
        
class PPO_MLP_Policy(nn.Module):
    #actor#
    def __init__(self, 
                 state_space, 
                 action_space,
                 hidden_dims,
                 hidden_activation=nn.Tanh()
    ):
        super(PPO_MLP_Policy, self).__init__()
        state_dim = state_space.shape[0]
        self.policy_head = nn.Linear(state_dim, hidden_dims[0])
        action_dim = action_space.shape[0]

        self.policy_layers = []
        for i in range(len(hidden_dims)-1):
            self.policy_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.policy_layers.append(hidden_activation)
        self.policy_layers = nn.Sequential(*self.policy_layers)
        
        if isinstance(action_space, gym.spaces.Discrete): # discrete action space #
            self.mu_layer = nn.Linear(hidden_dims[-1], action_dim)
            self.log_var_layer = nn.Sequential(nn.Linear(hidden_dims[-1], action_dim), nn.Softmax(dim=-1))

        else: # continuous action space -> Gaussian distribution policy #
            self.mu_layer = nn.Linear(hidden_dims[-1], action_dim)
            self.log_var_layer = nn.Sequential(nn.Linear(hidden_dims[-1], action_dim), nn.Tanh())

    
    def forward(self, state):
        state = self.policy_head(state)
        for layer in self.policy_layers:
            state = layer(state)
        
        mu = self.mu_layer(state)
        log_var = self.log_var_layer(state)
        std = log_var.exp()

        return mu, std