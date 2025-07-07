import torch
import torch.nn as nn
from tqdm import tqdm

import os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #soft_dac/
sys.path.append(root)
from networks.actor import ActorMLP
from .scheduler import (LinearBetaScheduler, CosineBetaScheduler, NonLinearBetaScheduler)

class ActorDiffusion(nn.Module):
    def __init__(self, 
                 state_dim:int,
                 action_dim:int,
                 #actor#
                 actor_hidden_size:int,
                 actor_hidden_layers:int,
                 #time mlp#
                 time_embed_hidden_size:int,
                 time_dim:int,
                 #beta scheduling#
                 beta_schedule_mode:str,
                 num_timesteps:int,
                 beta_1:float,
                 beta_T:float,
                 #additional params for policy#
                 clip_denoised_action:bool=True,
                 #style conditioning#
                 use_sc:bool=False,

                 **kwargs):
        super(ActorDiffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        #set up the mlp network that will work as the actor policy#
        self.actor_model = ActorMLP(state_dim=state_dim,
                                    action_dim=action_dim,
                                    hidden_size=actor_hidden_size,
                                    num_layers=actor_hidden_layers,
                                    time_embed_hidden_size=time_embed_hidden_size,
                                    time_dim=time_dim)
        #set up the beta scheduler for the diffusion process#
        if beta_schedule_mode == 'linear':
            self.var_scheduler = LinearBetaScheduler(num_train_timesteps=num_timesteps, beta_1=beta_1, beta_T=beta_T)
        elif beta_schedule_mode == 'cosine':
            self.var_scheduler = CosineBetaScheduler(num_train_timesteps=num_timesteps, beta_1=beta_1, beta_T=beta_T, s=kwargs['s']) #parameter s will be in **kwargs..
        elif beta_schedule_mode == 'nonlinear':
            self.var_scheduler = NonLinearBetaScheduler(num_train_timesteps=num_timesteps, beta_1=beta_1, beta_T=beta_T)
        else:
            raise NotImplementedError
        
        
        


        