from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BetaSchedulerCoefficients(nn.Module):
    def __init__(self, device, timesteps:int, beta_schedule_method: str, **kwargs):
        super().__init__()

        self.device = device

        self.timesteps = timesteps
        self.beta_schedule_method = beta_schedule_method
        
        if self.beta_schedule_method == 'vp':
            self.betas = self._np2tensor(self.vp_beta_schedule(timesteps))
        elif self.beta_schedule_method == 'cosine':
            self.betas = self._np2tensor(self.cosine_beta_schedule(timesteps))
        elif self.beta_schedule_method == 'linear':
            self.betas = self._np2tensor(self.linear_beta_schedule(timesteps, beta_start=kwargs['beta_start'], beta_end=kwargs['beta_end']))
        else:
            raise NotImplementedError(beta_schedule_method)

    def _np2tensor(self, arr):
        arr = torch.from_numpy(arr).to(self.device)
        return arr
    
    def from_beta(self):
        betas = self.betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # q(x_t | x_{t-1}) : forward process (data to noise)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1) #(alphas_cumprod - 1) / alphas_cumprod

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        self.register_buffer('betas', self._np2tensor(betas))
        self.register_buffer('alphas', self._np2tensor(alphas))
        self.register_buffer('alphas_cumprod', self._np2tensor(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', self._np2tensor(alphas_cumprod_prev))

        self.register_buffer('')

        
    @staticmethod
    def vp_beta_schedule(timesteps: int):
        t = np.arange(1, timesteps + 1)
        T = timesteps
        b_max = 10.
        b_min = 0.1
        alphas = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        betas = 1. - alphas
        return betas
    
    @staticmethod
    def linear_beta_schedule(timesteps: int, beta_start:float=1e-4, beta_end: float=0.999):
        betas = np.linspace(beta_start, beta_end, timesteps)
        return betas
    
    @staticmethod
    def cosine_beta_schedule(timesteps: int):
        s = 0.008
        t = np.arange(0, timesteps+1) / timesteps
        alphas_cumprod = np.cos((t+s) / (1+s) * np.pi / 2)**2
        alphas_cumprod /= alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1. - alphas
        betas = np.clip(betas, 0, 0.999)
        return betas