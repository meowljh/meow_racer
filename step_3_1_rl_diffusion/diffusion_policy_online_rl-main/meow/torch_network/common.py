from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

 
class WithSquashedGaussianPolicy(nn.Module):
    # policy: nn.Module
    def __init__(self):
        super().__init__()

    def get_action(self, obs:torch.Tensor, generator: torch.Generator=None)->torch.Tensor:
        mean, std = self.policy(obs)
        noise = torch.randn_like(mean) #generator=generator)
        act = mean + std * noise
        return F.tanh(act)
    
    def get_deterministic_action(self, obs:torch.Tensor)->torch.Tensor:
        mean, _ = self.policy(obs)
        return F.tanh(mean)

    def evaluate(self, obs:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.policy(obs)
        noise = torch.randn_like(mean)
        act = mean + std * noise
        dist = torch.distributions.normal.Normal(loc=mean, scale=std) #mean(mu),  std(sigma)
        logp = dist.log_prob(act)
        return F.tanh(act), logp.sum(axis=-1)

if __name__ == "__main__":
    pass
