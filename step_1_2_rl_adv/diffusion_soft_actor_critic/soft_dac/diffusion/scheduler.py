import torch
import torch.nn as nn

import numpy as np

class BaseBetaScheduler(nn.Module):
    def __init__(
            self, num_train_timesteps:int, beta_1:float, beta_T:float
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        ) #T-1, T-2,..., 0
    
    def _register_other_buffers(self):
        #needed for q(x_t | x_{t-1}) -> but actually will not be needed for sdac because no forward process is necessary
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1. - self.alphas_cumprod))
        
        # needed for reverse diffusion process q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)


# Linear Beta Scheduler #
class LinearBetaScheduler(BaseBetaScheduler):
    def __init__(
            self, num_train_timesteps:int, beta_1:float, beta_T:float
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T)
        betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.ones(1), alphas_cumprod[:-1]]))

        self._register_other_buffers()

# Cosine Beta Scheduler #
class CosineBetaScheduler(BaseBetaScheduler):
    def __init__(
            self, num_train_timesteps:int, beta_1:float, beta_T:float,
            s:float=0.008
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T)

        T = self.num_train_timesteps + 1
        t = np.linspace(0, T, T)
        f_t = np.cos(((t / T) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=beta_1, a_max=beta_T)

        self.register_buffer("betas", torch.tensor(betas_clipped, dtype=torch.float32))
        self.register_buffer("alphas_cumprod", torch.tensor(alphas_cumprod[1:], dtype=torch.float32))
        # self.register_buffer("alphas_cumprod", torch.tensor(alphas_cumprod[:-1], dtype=torch.float32))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.ones(1), alphas_cumprod[:-1]]))

        self._register_other_buffers()

# Non linear Beta Scheduler #
class NonLinearBetaScheduler(BaseBetaScheduler):
    """Non Linear Beta Scheduler
    - Variance preserving beta scheduler은 DDPM의 forward process를 SDE 형태로 일반화 한 것이다.
    """
    def __init__(
            self, num_train_timesteps:int, beta_1:float, beta_T:float
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T)
    
        t = np.arange(1, self.num_train_timesteps + 1)
        T = self.num_train_timesteps
        alpha = np.exp(-self.beta_1 / T - 0.5 * (self.beta_T - self.beta_1) * (2 * t - 1) / T ** 2)
        betas = 1 - alpha

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas))
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.ones(1), self.alphas_cumprod[:-1]]))

        self._register_other_buffers()