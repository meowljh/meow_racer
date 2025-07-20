from dataclasses import dataclass

from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaSchedulerCoefficients(nn.Module):
    def __init__(self, device, timesteps, beta_schedule_type: str, **kwargs):
        super().__init__()

        self.device = device

        self.timesteps = timesteps
        self.beta_schedule_type = beta_schedule_type
        
        if self.beta_schedule_type == 'vp':
            self.betas_np = self.vp_beta_schedule(timesteps)
        elif self.beta_schedule_type == 'cosine':
            self.betas_np = self.cosine_beta_schedule(timesteps)
        elif self.beta_schedule_type == 'linear':
            self.betas_np = self.linear_beta_schedule(timesteps, beta_start=kwargs['beta_start'], beta_end=kwargs['beta_end'])
        else:
            raise NotImplementedError(beta_schedule_type)
    
        self.from_beta()

    def _np2tensor(self, arr):
        arr = torch.from_numpy(arr).to(self.device)
        return arr
    
    def from_beta(self):
        betas = self.betas_np
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

        """register_buffer을 사용하는 이유는 torch.nn.Module의 parameter에 속하지 않아서 optimizer에 의해서 update되지 않아야 하기 때문이다."""
        self.register_buffer('betas', self._np2tensor(betas))
        self.register_buffer('alphas', self._np2tensor(alphas))
        self.register_buffer('alphas_cumprod', self._np2tensor(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', self._np2tensor(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', self._np2tensor(sqrt_alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', self._np2tensor(sqrt_one_minus_alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', self._np2tensor(log_one_minus_alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', self._np2tensor(sqrt_recip_alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', self._np2tensor(sqrt_recipm1_alphas_cumprod))
        
        self.register_buffer('posterior_variance', self._np2tensor(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', self._np2tensor(posterior_log_variance_clipped))
        self.register_buffer('posterior_mean_coef1', self._np2tensor(posterior_mean_coef1))
        self.register_buffer('posterior_mean_coef2', self._np2tensor(posterior_mean_coef2))

        
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

class GaussianDiffusion:
    def __init__(self, device,
                 num_timesteps: int, beta_schedule_scale:float=0.3,
                 beta_schedule_type: str='linear'):
        
        self.beta_scheduler = BetaSchedulerCoefficients(
            device=device, 
            timesteps=num_timesteps,
            beta_schedule_type=beta_schedule_type
        )
        
        self.beta_schedule_scale = beta_schedule_scale

        self.num_timesteps = num_timesteps

        self.device = device
    
    @staticmethod
    def _extract(arr, t, x):
        if x is not None:
            dtype = x.dtype
            device = x.device
            ndim = x.ndim
        out = torch.as_tensor(arr, dtype=dtype, device=device).gather(0, t)
        return out.reshape((-1, ) + (1,) * (ndim-1))
    
    def p_mean_variance(self, t, x:torch.Tensor, noise_pred:torch.Tensor)->torch.Tensor:
        """
        :param x: diffusion forward process에서의 t번째 x

        output
        x_recon: 1/sqrt(alpha_bar) * x - sqrt(1/sqrt(alphas_bar) - 1) * noise_pred
        model_mean: eq(11) in DDPM paper
        """

        # x_recon = x * self.beta_scheduler.sqrt_recip_alphas_cumprod[t] - \
        #             noise_pred * self.beta_scheduler.sqrt_recipm1_alphas_cumprod[t]
        x_recon = self.get_recon(t=t, x=x, noise=noise_pred)
        x_recon = torch.clip(x_recon, -1, 1) #action 값들이 (-1, 1) 사이의 범위에 속하게 해야 하기 때문이다.
        
        # model_mean = x_recon * self.beta_scheduler.posterior_mean_coef1[t] + \
        #                 x * self.beta_scheduler.posterior_mean_coef2[t] #
        model_mean = self.get_p_mean(x=x_recon, t=t)

        # model_log_variance = self.beta_scheduler.posterior_log_variance_clipped[t] #고정 값 사용
        model_log_variance = self._extract(self.beta_scheduler.posterior_log_variance_clipped, t, x_recon)

        return model_mean, model_log_variance

    def get_p_mean(self, x, t):
        return x * self._extract(self.beta_scheduler.posterior_mean_coef1, t, x) + \
                    x * self._extract(self.beta_scheduler.posterior_mean_coef2, t, x)
    
    def get_recon(self, t, x:torch.Tensor, noise:torch.Tensor)->torch.Tensor:
        """"""
        # x_recon = x * self.beta_scheduler.sqrt_recip_alphas_cumprod[t][:, torch.newaxis] - \
        #             noise * self.beta_scheduler.sqrt_recipm1_alphas_cumprod[t][:, torch.newaxis]
        x_recon = x * self._extract(self.beta_scheduler.sqrt_recip_alphas_cumprod, t, x) - \
                        noise * self._extract(self.beta_scheduler.sqrt_recipm1_alphas_cumprod, t, x)
        
        return x_recon
    
    def q_sample(self, t, x_0: torch.Tensor, noise: torch.Tensor)->torch.Tensor:
        """forward diffusion process
        Eq(4) in DDPM paper (in page 3)
        """
        # x_t = x_0 * self.beta_scheduler.sqrt_alphas_cumprod[t] + \
        #         noise * self.beta_scheduler.sqrt_one_minus_alphas_cumprod[t]
        x_t = self._extract(self.beta_scheduler.sqrt_alphas_cumprod, t, x_0) + \
                    noise * self._extract(self.beta_scheduler.sqrt_one_minus_alphas_cumprod, t, x_0)
        
        return x_t
        

    def p_sample_single_step(self, model_fn, x: torch.Tensor, t:torch.Tensor, noise: torch.Tensor):
        noise_pred = model_fn(t, x)
        # model_mean, model_log_variance = self.p_mean_variance(t=t.item(), x=x, noise_pred=noise_pred)
        model_mean, model_log_variance = self.p_mean_variance(t=t, x=x, noise_pred=noise_pred)
        nonzero_mask = (t > 0).reshape((-1, ) + (1, ) * (x.ndim-1))

        # x = model_mean + (t > 0) * torch.exp(0.5 * model_log_variance) * noise
        sample = model_mean + nonzero_mask * torch.exp(model_log_variance * 0.5) * noise

        # return x, None
        return sample, None
    
    @torch.inference_mode()
    def p_sample(self, model_fn, shape: Tuple[int, ...])->torch.Tensor:
        """reverse diffusion process (denoising)
        repeat p_theta(x_{t-1} | x_t) for num_timesteps
        """
    
        # x = torch.randn(shape).to(self.device) * 0.5
        x = torch.randn(shape, dtype=torch.float32, device=self.device)

        # breakpoint()
        time_steps = torch.arange(self.num_timesteps-1, -1, -1).to(self.device) #T-1, T-2, ... , 0
        
        time_arr = torch.empty(shape[:-1], dtype=torch.int64, device=self.device)
        for t in time_steps:
            time_arr.fill_(t)
            noise = torch.randn(size=shape, dtype=torch.float32, device=self.device)
            x, _ = self.p_sample_single_step(model_fn=model_fn,
                                             x=x,
                                             t=time_arr,
                                             noise=noise)
        x_0_pred = x
        return x_0_pred

    def p_loss(self, model_fn, t:torch.Tensor, x_0: torch.Tensor):
        assert t.ndim == 1 and t.shape[0] == x_0.shape[0]

        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(t=t, x_0=x_0, noise=noise)
        noise_pred = model_fn(t=t, x=x_noisy)
        loss = F.mse_loss(noise_pred, noise, reduction='none') * 0.5
        return loss.mean()
    
    def weighted_p_loss(self, weights:torch.Tensor, model_fn, t:torch.Tensor, 
                        x_0:torch.Tensor, x_t:torch.Tensor):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_t.shape[0] #timestep tensor은 (B,)의 크기여야 함
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(t=t, x_0=x_0, noise=noise)
        noise_pred = model_fn(t, x_noisy)
        loss = weights * F.mse_loss(noise_pred, noise, reduction='none')
        return loss.mean()

    """!!!!!RSSM LOSS!!!!!"""
    def reverse_sampling_weighted_p_loss(self, 
                                         noise:torch.Tensor, 
                                         weights:torch.Tensor,
                                         model_fn, 
                                         x_t:torch.Tensor,
                                         t:torch.Tensor):
        """
        :param noise: random normal noise tensor
        :param weights: will be the q-value predicted from the critic network
        :param x_t: x tensor from the randomly selected timestep t
        """
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        
        assert t.ndim == 1 and t.shape[0] == x_t.shape[0]
        noise_pred = model_fn(t, x_t)
        loss = weights * F.mse_loss(noise_pred, noise, reduction='none')
        return loss.mean()
    

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torch_network.diffv2 import DACERPolicyNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_dim=32
    action_dim=3
    obs_dim=30
    style_dim=0 #3
    hidden_sizes=[128, 128]

    diffusion_model = DACERPolicyNet(time_dim=time_dim,
                                     action_dim=action_dim,
                                     obs_dim=obs_dim,
                                     style_dim=style_dim,
                                     hidden_sizes=hidden_sizes,
                                     activation_fn='relu', 
                                     output_activation_fn='relu').to(device)
    
    gauss = GaussianDiffusion(device = device,
                              num_timesteps=20,
                              beta_schedule_type='cosine')

    t = 3
    B = 1024
    obs = torch.rand((B, obs_dim)).to(device)
    x = torch.rand((B, 3)).to(device)
    noise_pred = torch.randn((B, 3)).to(device)

    def model_fn(t, x):
        return diffusion_model(obs=obs, act=x, t=t, style=None)
    

    model_mean, model_log_var = gauss.p_mean_variance(t=t, x=x, noise_pred=noise_pred)
    print(f"Mean: {model_mean.shape}")
    print(f"LogVar: {model_log_var}")

    x_0_pred = gauss.p_sample(model_fn=model_fn, 
                   shape=(B, action_dim))
    
    print(f"X0: {x_0_pred.shape}")