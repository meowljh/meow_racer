import torch
import torch.nn as nn

from einops import einsum

from typing import Sequence, Union

from dataclasses import dataclass

def _get_activation(name: str):
    if name.lower() == 'leaky_relu':
        return nn.LeakyReLU()
    elif name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError(name)

@dataclass
class ValueNet(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_sizes: Sequence[int],
                 activation_fn: str,
                 output_activation_fn: str,
                 **kwargs):
        super().__init__()
        """
        :param input_size: state_size 
        :output: V(s_t)
        """
        
        self.net = MLP(input_size, hidden_sizes, output_size=1, 
                       activation_fn=activation_fn, 
                       output_activation_fn=output_activation_fn, 
                       squeeze_output=True)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # input = torch.concatenate((obs, style), axis=-1)
        input = obs
        out = self.net(input)
        return out


@dataclass
class QNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: Sequence[int],
                 activation_fn: str,
                 output_activation_fn: str,
                 **kwargs):
        super().__init__()
        """
        :param input_size: state_size + action_size
        :output: Q(s_t, a_t)
        """

        self.net = MLP(input_size, hidden_sizes, output_size=1,
                       activation_fn=activation_fn,
                       output_activation_fn=output_activation_fn,
                       squeeze_output=True)
    
    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        input = torch.concatenate((obs, act), axis=-1)
        out = self.net(input)
        return out

@dataclass
class PolicyNet(nn.Module):
    def __init__(self,
                 action_dim: int,
                 state_dim: int, 
                 style_dim: int,
                 hidden_sizes: Sequence[int],
                 activation_fn: str,
                 output_activation_fn: str,
                 min_log_std: float=-20.,
                 max_log_std: float=0.5,
                 log_std_mode: Union[str, float] = 'shared',
                 **kwargs):
        super().__init__()
        
        self.log_std_mode = log_std_mode

        input_dim = state_dim + style_dim
        
        if self.log_std_mode == 'shared':
            output_dim = action_dim * 2
            self.net = MLP(input_dim, hidden_sizes, output_dim, activation_fn, output_activation_fn)
        
        elif self.log_std_mode == 'separate':
            output_dim = action_dim
            self.mean_net = MLP(input_dim, hidden_sizes, output_dim, activation_fn, output_activation_fn)
            self.log_std_net = MLP(input_dim, hidden_sizes, output_dim, activation_fn, output_activation_fn)

        else:
            output_dim = action_dim
            self.initial_log_std = float(self.log_std_mode)
            self.mean_net = MLP(input_dim, hidden_sizes, output_dim, activation_fn, output_activation_fn)
        
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
    def forward(self, obs: torch.Tensor, style: torch.Tensor=None, return_log_std: bool=False)->torch.Tensor:
        input = torch.concatenate((obs, style), dim=-1) if style is not None else obs
        
        if self.log_std_mode == 'shared':
            out = self.net(input)
            mean, log_std = torch.split(out, 2, dim=-1)
        
        elif self.log_std_mode == 'separate':
            mean = self.mean_net(input)
            log_std = self.log_std_net(input)
        
        else: #use constant initial log std
            mean = self.mean_net(input)
            log_std = torch.ones_like(mean) * self.initial_log_std
        
        if not (self.min_log_std is None and self.max_log_std is None):
            log_std = torch.clip(log_std, self.min_log_std, self.max_log_std)
        
        if return_log_std:
            return mean, log_std
        else:
            return mean, torch.exp(log_std)

            
@dataclass
class DiffusionPolicyNet(nn.Module):
    def __init__(self,
                 time_dim: int,
                 action_dim: int,
                 obs_dim: int,
                 style_dim: int,
                 hidden_sizes: Sequence[int],
                 activation_fn: str,
                 output_activation_fn: str,
                 **kwargs):
        super(DiffusionPolicyNet, self).__init__()

        """
        diffusion policy network이기 때문에 "generation"의 관점에서 생각해 볼 때
        (B, action_dim)에 다음 action을 채울 것임.
        단순히 observation state vector만 입력으로 받는게 아님.
        """
        self.input_dim = obs_dim + action_dim + style_dim
        self.diffusion_policy_net = MLP(
            input_size=self.input_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn
        )
        self.time_dim = time_dim
    
    def forward(self, obs: torch.Tensor, act: torch.Tensor, t: torch.Tensor, 
                style: torch.Tensor=None)->torch.Tensor:
        te = scaled_sinusoidal_encoding(t=t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        input = torch.concatenate((obs, act, te), axis=-1) if style is None else \
                    torch.concatenate((style, obs, act, te), axis=-1)
        out = self.diffusion_policy_net(input)

        return out

@dataclass
class DACERPolicyNet(nn.Module):
    def __init__(self,
                 time_dim: int,
                 action_dim: int,
                 obs_dim: int,
                 style_dim: int,
                 hidden_sizes: Sequence[int],
                 activation_fn: str,
                 output_activation_fn: str,
                 **kwargs):
        super(DACERPolicyNet, self).__init__()
        
        self.input_dim = obs_dim + action_dim + style_dim

        self.dacer_policy_net = MLP(
            input_size=self.input_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn
        )

        self.time_emb_fc = nn.Sequential(
            nn.Linear(time_dim, time_dim*2),
            _get_activation(activation_fn),
            nn.Linear(time_dim, time_dim)
        )

        self.time_dim = time_dim

    def forward(self, obs:torch.Tensor, act:torch.Tensor, t:torch.Tensor, style:torch.Tensor=None)->torch.Tensor:
        te = scaled_sinusoidal_encoding(t=t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = self.time_emb_fc(te)
        input = torch.concatenate((obs, act, te), dim=-1) if style is None else \
                    torch.concatenate((style, obs, act, te), dim=-1)
        out = self.dacer_policy_net(input)

        return out
    
class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: Sequence[int],
                 output_size: int,
                 activation_fn: str,
                 output_activation_fn: str,
                 squeeze_output: bool=False,
                 **kwargs):
        super().__init__()
        layers = []

        self.squeeze_output = squeeze_output

        activation = _get_activation(activation_fn)
        output_activation = _get_activation(output_activation_fn)
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                layers += [nn.Linear(input_size, hidden_size), activation]
            elif i == len(hidden_sizes)-1:
                layers += [nn.Linear(hidden_size, output_size), output_activation]
            else:
                layers += [nn.Linear(hidden_sizes[i-1], hidden_size), activation]
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        if self.squeeze_output:
            out = torch.squeeze(out, dim=-1)
        return out

def scaled_sinusoidal_encoding(t: torch.Tensor,
                               dim: int,
                               theta: int = 10000,
                               batch_shape = None) -> torch.Tensor:
    assert dim % 2 == 0

    """
    :param t: (B,)의 크기를 가짐 (각 batch에 대해서 필요로 하는 time step의 "pos" 정보로 구성된 tensor임)
    :param dim: embedding dimension
    """

    device = t.get_device()

    scale = 1 / dim ** 0.5
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim) / half_dim
    inv_freq = theta ** -freq_seq
    inv_freq = inv_freq.to(device)

    emb = einsum(t, inv_freq, '..., j -> ... j')
    # breakpoint()
    emb = torch.concatenate((torch.sin(emb), torch.cos(emb)), axis=-1)
    emb *= scale

    if batch_shape is not None:
        emb = emb.expand(*batch_shape, dim)
    
    return emb

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    obs = torch.rand((1024, 30)).to(device)
    # t = torch.arange(0, 16).to(device) #.repeat(1024, 1)
    t = torch.randint(0, 20, size=(1024, )).to(device)
    print(t.shape)
    dim = 30
    batch_shape = obs.shape[:-1]
    
    emb = scaled_sinusoidal_encoding(t=t, dim=dim, batch_shape=batch_shape)
    print(emb.shape) #(#time, dim)
    # state_dim = 30
    # batch_size = 128
    # action_dim = 3
    # style_dim = 3
    # state = torch.rand((batch_size, state_dim)).to(device)
    # action = torch.rand((batch_size, action_dim)).to(device)
    # style = torch.rand((batch_size, style_dim)).to(device)

    # # print(torch.concatenate((action, style, None)), axis=-1)
    # q_net = QNet(state_dim+action_dim, [128, 128], 'relu', 'sigmoid').to(device)

    # out = q_net(state, action)
    # print(out.shape)