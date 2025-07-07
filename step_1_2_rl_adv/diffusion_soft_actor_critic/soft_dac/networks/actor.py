import torch
import torch.nn as nn

from collections.abc import Sequence

import os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #soft_dac/
sys.path.append(root)
from networks.module import TimeEmbedding, init_weights

class ActorMLP(nn.Module):
    def __init__(self, 
                 state_dim:type[int], 
                 action_dim:type[int], 
                #  hidden_size:type[Sequence | int],
                 ## for actor layer
                 hidden_size:type[int],
                 num_layers:type[int],
                 ## for time mlp
                 time_embed_hidden_size:type[int],
                 time_dim:type[int],
                 ):
        super().__init__()

        """ActorMLP
        diffusion actor policy that uses the Linear layers 
        default setting:
            hidden_size = 256
            num_layers = 3
            activation_fn = Mish
        """

        self.time_mlp = TimeEmbedding(hidden_size=time_embed_hidden_size, frequency_embedding_size=time_dim) #sinusoidal embedding 이후에 MLP layer 거침.

        input_dim = state_dim + action_dim + time_dim

        actor_layer = []
        inp_dim_arr = [input_dim] + [hidden_size for _ in range(num_layers)]
        out_dim_arr = [hidden_size for _ in range(num_layers)] + [action_dim]
        for i, (id, od) in enumerate(zip(inp_dim_arr, out_dim_arr)):
            actor_layer.append(nn.Linear(id, od))
            if i != len(inp_dim_arr)-1:
                actor_layer.append(nn.Mish()) #same as the Soft-Diffusion Actor Critic paper
        self.actor_layer = nn.Sequential(*actor_layer)

        self.apply(init_weights)


    def forward(self, x:torch.Tensor, state:torch.Tensor, time:torch.Tensor):

        """
        !time embedding vector will be concatenated with the x tensor to denoise, and the state tensor!

        :param x: tensor to denoise, which will be generating the real action for the diffusion policy
        :param state: state tensor of the current time step
        :param time: time step randomly sampled from (1 ~ num_timesteps), which means the network will be predicting the noise level that was added from x_t to x_0 (x means the action for current state)
        """

        time_emb = self.time_mlp(time)
        out = torch.cat([x, time_emb, state], dim=-1)
        out = self.actor_layer(out)

        return out

        
