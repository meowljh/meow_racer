import torch
import torch.nn as nn
import numpy as np
import math

def init_weights(m):
    def truncated_normal_init(t, mean:float=0.0, std:float=0.01):
        nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear:
        input_dim = m.in_features
        truncated_normal_init(t=m.weight, std=1/(2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.)

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size:int, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(), #same activation function as used in the Diffuser official repository
            nn.Linear(hidden_size, frequency_embedding_size, bias=True)
        ) # two linear layers with an activation function betweens
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings
        embedding_i = [cos(w1t), cos(w2t),..., sin(w1t), sin(w2t)...]
        frequency = exp(-log(max_period) * i / half) 
        --> 위와 같이 frequency를 계산할 수 있는 이유는 1 / max_period ^ (2i / dim)과 논리적으로 동치이기 때문이다
        --> a^(-x) = e^(-xloga)

        :param t: a 1-D Tensor of N indices, one per batch element. (NOT just a single number)
        :param dim: the dimension of the output
        :param max_period: the minimum frequency of the embeddings

        :return: an [N x dim] Tensor of positional embeddings
        """

        half = dim // 2 
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)#위의 주석 부분에서 w_i의 역할을 함
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1: #dim이 홀수인 경우에 -> embedding의 마지막 부분에 0을 붙여줌.
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, 0])], dim=-1)
        
        return embedding
    

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(-1) #(1, 1) 이게 아니면 (N, 1)일 것임. (N은 batch size)
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq) #(B, frequency_embedding_size)
 
        return t_emb
    
class MLP(nn.Module):
    def __init__(self, 
                 input_dim:type[int],
                 hidden_size:type[int],
                 output_dim:type[int],
                 num_layers:type[int],
                 #for time mlp
                 time_embed_hidden_size:type[int],
                 time_dim:type[int]
                 ):
        super().__init__()

        """MLP
        diffusion model only for the toy example
        default setting:
            hidden_size = 128
            num_layers = 2
            activation_fn = LeakyReLU
        """

        self.time_mlp = TimeEmbedding(hidden_size=time_embed_hidden_size, 
                                      frequency_embedding_size=time_dim)

        net = []
        inp_dim_arr = [input_dim + time_dim] + [hidden_size for _ in range(num_layers)]
        out_dim_arr = [hidden_size for _ in range(num_layers)] + [output_dim]
        for i, (id, od) in enumerate(zip(inp_dim_arr, out_dim_arr)):
            net.append(nn.Linear(id, od))
            if i != len(inp_dim_arr)-1:
                net.append(nn.LeakyReLU())
        self.net = nn.Sequential(*net)

        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor, time:torch.Tensor):
        
        t = self.time_mlp(time)
        out = torch.cat([x, t], dim=-1)
        out = self.net(out)

        return out



# class TimeLinear(nn.Module):
#     def __init__(self, in_dim:int, out_dim:int, num_timesteps:int,
#                  embed_mode:str):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.num_timesteps = num_timesteps
#         assert embed_mode in ['multiply', 'concat', 'add']

#         self.embed_mode = embed_mode

#         self.time_embedding = TimeEmbedding(hidden_size=self.out_dim)
#         self.fc = nn.Linear(in_dim, out_dim)
    
#     def forward(self, x: torch.Tensor, t: torch.Tensor):
#         x = self.fc(x)
#         time_embed = self.time_embedding(t).view(-1, self.out_dim)

#         if self.embed_mode == 'multiply':
#             return x * time_embed
#         elif self.embed_mode == 'add':
#             return x + time_embed
#         elif self.embed_mode == 'concat':
#             return torch.cat([x, time_embed], dim=-1)
    

