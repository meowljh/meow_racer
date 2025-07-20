from typing import Callable, NamedTuple, Sequence, Tuple, Union
from dataclasses import dataclass

import math

import torch
import torch.nn as nn

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from torch_network.blocks import DACERPolicyNet, QNet, _get_activation
from torch_utils.diffusion import GaussianDiffusion

class Diffv2Net(nn.Module):
    def __init__(self, 
                 device,
                 num_timesteps: int,
                 time_dim: int, #output embedding dimension size of the time vector for diffusion
                 act_dim: int,
                 obs_dim: int,
                 style_dim: int,
                 hidden_sizes: Sequence[int],
                 num_particles: int,
                 target_entropy: float,
                 noise_scale: float,
                 beta_schedule_scale: float,
                 activation_fn: str,
                 output_activation_fn: str = 'odentity',
                 beta_schedule_type: str='linear'
    ):
        super().__init__()
        self.device = device
        
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.style_dim = style_dim

        self.num_timesteps = num_timesteps
        self.num_particles = num_particles
        self.target_entropy = target_entropy
        self.noise_scale = noise_scale
        self.beta_schedule_scale = beta_schedule_scale
        self.beta_schedule_type = beta_schedule_type
        
        # self.q = QNet(input_size=obs_dim + act_dim, hidden_sizes=hidden_sizes, activation_fn=activation_fn, output_activation_fn=output_activation_fn).to(device)

        self.q1 = QNet(input_size=obs_dim + act_dim, hidden_sizes=hidden_sizes, activation_fn=activation_fn, output_activation_fn=output_activation_fn).to(device)
        self.q2 = QNet(input_size=obs_dim + act_dim, hidden_sizes=hidden_sizes, activation_fn=activation_fn, output_activation_fn=output_activation_fn).to(device)
        self.target_q1 = QNet(input_size=obs_dim + act_dim, hidden_sizes=hidden_sizes, activation_fn=activation_fn, output_activation_fn=output_activation_fn).to(device)
        self.target_q2 = QNet(input_size=obs_dim + act_dim, hidden_sizes=hidden_sizes, activation_fn=activation_fn, output_activation_fn=output_activation_fn).to(device)

        self.policy = DACERPolicyNet(time_dim=time_dim, action_dim=act_dim, obs_dim=obs_dim, style_dim=style_dim,
                                     hidden_sizes=hidden_sizes, activation_fn=activation_fn,
                                     output_activation_fn=output_activation_fn).to(device)

        self.target_policy = DACERPolicyNet(time_dim=time_dim, action_dim=act_dim, obs_dim=obs_dim, style_dim=style_dim,
                                     hidden_sizes=hidden_sizes, activation_fn=activation_fn,
                                     output_activation_fn=output_activation_fn).to(device)
        
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        self.diffusion = GaussianDiffusion(
            device=self.device,
            num_timesteps=self.num_timesteps,
            beta_schedule_scale=self.beta_schedule_scale,
            beta_schedule_type=self.beta_schedule_type
        )
        
    # @property
    # def diffusion(self) -> GaussianDiffusion:
    #     return GaussianDiffusion(
    #         device=self.device,
    #         num_timesteps=self.num_timesteps,
    #         beta_schedule_scale=self.beta_schedule_scale,
    #         beta_schedule_type=self.beta_schedule_type
    #     )

    def get_action(self, obs:torch.Tensor, log_alpha=None) -> torch.Tensor:
        if log_alpha is None:
            log_alpha = self.log_alpha

        def model_fn(t, x):
            return self.policy(obs=obs, act=x, t=t, style=None)
        
        
        def sample() -> Union[torch.Tensor, torch.Tensor]:
            act = self.diffusion.p_sample(model_fn, shape=(*obs.shape[:-1], self.act_dim))
            q1 = self.q1(obs=obs, act=act)
            q2 = self.q2(obs=obs, act=act)
            q = torch.min(q1, q2)
            return act.clip(-1, 1), q
        
        if self.num_particles == 1:
            act, q = sample()
        else:
            acts = None;qs = None
            for n in range(self.num_particles):
                act, q = sample()
                acts = torch.concatenate((acts, act.unsqueeze(0)), 0) if acts is not None else act.unsqueeze(0)
                qs = torch.concatenate((qs, q.unsqueeze(0)), 0) if qs is not None else q.unsqueeze(0)
            q_best_ind = torch.argmax(qs, dim=0, keepdim=True)
            act = torch.take_along_dim(acts, q_best_ind[..., None], dim=0).squeeze(0)
        
        act = act + torch.randn(act.shape, device=self.device) * torch.exp(log_alpha) * self.noise_scale
        return act
    
    def get_batch_actions(self, obs:torch.Tensor, q_func: Callable)->torch.Tensor:
        batch_size = obs.shape[0]
        batch_flatten_obs = obs.repeat(self.num_particles, axis=0) #(repeat_size, B, obs_dim)
        batch_flatten_actions = self.get_action(obs=batch_flatten_obs) #(repeat_size, B. act_dim)
        batch_q = q_func(obs=batch_flatten_obs, act=batch_flatten_actions) #(repeat_size, B)
        max_q_idx = batch_q.argmax(axis=1) #(repeat_size)
        batch_action = batch_flatten_actions.reshape(batch_size, -1, self.act_dim) #(Batch size, repeat_size, act_dim)
        best_action = batch_action[torch.arange(batch_size), max_q_idx]

        return best_action

    def get_deterministic_action(self, obs:torch.Tensor)->torch.Tensor:
        log_alpha = torch.tensor(-torch.inf, dtype=torch.float32, device=self.device)
        return self.get_action(obs, log_alpha)
    

    def q_evaluate(self, obs:torch.Tensor, act:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_mean, q_std = self.q