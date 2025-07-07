"""rssm_diffusion.py
Implementation of the diffusion model, not the diffusion policy, but with the usage of RSSM(Reverse Sampling Score Matching) loss
-> will be useful for debugging whether the implementation of the RSSM loss is correct
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
sys.path.append(f"{root}/networks")
from networks.module import MLP
from .scheduler import LinearBetaScheduler
from .tools import extract

class ToyDiffusion(nn.Module):
    def __init__(self,
                 device,
                 #diffusion model#
                 input_dim:int,
                 hidden_size:int,
                 output_dim:int,
                 num_layers:int,
                 #time mlp#
                 time_embed_hidden_size:int,
                 time_dim:int,
                 #beta scheduling#
                 num_timesteps:int,
                 beta_1:float,
                 beta_T:float,
                 beta_schedule_mode:str='linear',
                 #reverse sampling dist#
                 reverse_sampling_dist:str='gauss',

                 **kwargs
                 ):
        super().__init__()

        assert beta_schedule_mode == 'linear'
        assert reverse_sampling_dist in ['gauss', 'uniform']
        
        self.device = device

        self.model = MLP(input_dim=input_dim,
                         hidden_size=hidden_size,
                         output_dim=output_dim,
                         num_layers=num_layers,
                         time_embed_hidden_size=time_embed_hidden_size,
                         time_dim=time_dim).to(self.device)
        
        self.var_scheduler = LinearBetaScheduler(num_train_timesteps=num_timesteps, beta_1=beta_1, beta_T=beta_T)

        self.reverse_sampling_dist = reverse_sampling_dist
    
    def gmm_density_torch(self, x0: torch.Tensor, do_log:bool=False):
        mean1 = torch.tensor([3., 3.], device=self.device)
        mean2 = torch.tensor([-3., -3.], device=self.device)
        diff1 = x0 - mean1
        diff2 = x0 - mean2
        exp1 = torch.exp(-0.5 * torch.sum(diff1**2, dim=1))
        exp2 = torch.exp(-0.5 * torch.sum(diff2**2, dim=1))

        weight1, weight2 = 0.8, 0.2

        if do_log:
            log_coef = -1 * torch.log(torch.tensor(2 * torch.pi)) #분수니까 log 붙이면 -1 곱해줘야 함.
            log_density = torch.log(
                weight1 * torch.exp(torch.log(exp1) * log_coef) + \
                weight2 * torch.exp(torch.log(exp2) * log_coef)
            )
            return log_density
        else:
            coef = 1 / (2 * torch.pi)
            density = coef * (weight1 * exp1 + weight2 * exp2)
            return density
    
    def gmm_density(self, x:torch.Tensor, y:torch.Tensor):
        """
        returns the probability density of the predicted x_0
        implementation of the gaussian mixture model for the toy example
        :param x: 1-D tensor
        :param y: 1-D tensor
        """
        # x = x[:, None] if len(x.shape) == 1 else x
        # y = y[:, None] if len(y.shape) == 1 else y
        X, Y = torch.meshgrid(x, y)
        pos = torch.stack([X, Y], dim=-1)

        #Gaussian 1#
        mu1 = torch.Tensor([3, 3]).to(self.device)
        diff1 = pos - mu1
        gauss1 = 0.8 * (1 / (2 * torch.pi)) * torch.exp(-0.5 * torch.sum(diff1 ** 2, dim=-1))

        #Gaussian 2#
        mu2 = torch.Tensor([-3, -3]).to(self.device)
        diff2 = pos - mu2 
        gauss2 = 0.2 * (1 / (2 * torch.pi)) * torch.exp(-0.5 * torch.sum(diff2 ** 2, dim=-1))

        return gauss1 + gauss2

    def compute_rssm_loss(self, batch_size:int, x_shape:tuple, do_log:bool=False):
        #step1: randomly sample t#
        t = (
            torch.randint(0, self.var_scheduler.num_train_timesteps, size=(batch_size,))
            .to(self.device)
            .long()
        )
        #step2: randomly sample x_t#
        if self.reverse_sampling_dist == 'gauss': #N(0,4I)
            x_t = torch.randn(x_shape) * 2
        elif self.reverse_sampling_dist == 'uniform':
            high, low = 6, -6
            x_t = (high - low) * torch.rand(x_shape) + low #Unif(-6, 6)
        else:
            raise NotImplementedError(f"{self.reverse_sampling_dist} is not supported")
        #step3: randomly sample target noise#
        target_noise = torch.randn(x_shape).to(self.device)
        x_t = x_t.to(self.device)
        #step4: compute ~{x_0}#
        x_0_tilda = (1 / extract(self.var_scheduler.sqrt_alphas_cumprod, t, x_t).to(self.device)) * x_t - \
                        target_noise * (extract(self.var_scheduler.sqrt_one_minus_alphas_cumprod, t, x_t).to(self.device) / extract(self.var_scheduler.sqrt_alphas_cumprod, t, x_t).to(self.device))
        #step5: compute RSSM loss#
        score_prediction = self.model(x=x_t, time=t)
        with torch.no_grad():
            # p0_loss_weight = self.gmm_density(x=x_0_tilda[:, 0], y=x_0_tilda[:, 1])
            p0_loss_weight = self.gmm_density_torch(x0 = x_0_tilda, do_log=do_log)
            p0_loss_weight = p0_loss_weight.unsqueeze(1)
        target_score = -target_noise / extract(self.var_scheduler.sqrt_one_minus_alphas_cumprod, t, x_t).to(self.device)
        rssm_loss = p0_loss_weight * F.mse_loss(score_prediction, target_score, reduction='none')
        # rssm_loss = p0_loss_weight * F.mse_loss(score_prediction, target_noise)
        rssm_loss = rssm_loss.mean()

        return rssm_loss
        

        
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """p_sample p(x_{t-1} | x_{t})
        :param x_t: 
        :param t:
        """
        score_theta = self.model(x=x_t, time=t)
        a = 1 / extract(self.var_scheduler.sqrt_alphas_cumprod, t, x_t).to(self.device)
        beta_t = extract(self.var_scheduler.betas, t, x_t).to(self.device)
        c = ((1 - extract(self.var_scheduler.alphas_cumprod_prev, t, x_t).to(self.device)) / (1 - extract(self.var_scheduler.alphas_cumprod, t, x_t).to(self.device)))
        z_t = torch.randn(x_t.shape).to(self.device)

        x_t_prev = (a * (beta_t*score_theta + x_t)) + (beta_t * c * z_t)

        return x_t_prev
    

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """p_sample_loop
        
        """
        if self.reverse_sampling_dist == 'gauss':
            x_T = torch.randn(shape) * 2
        elif self.reverse_sampling_dist == 'uniform':
            high, low = 6, -6
            x_T = (high - low) * torch.rand(shape) + low #Unif(-6, 6)
        else:
            raise NotImplementedError(f"{self.reverse_sampling_dist} is not supported")
        x_0_pred = None
        # x_T = torch.randn(shape) 
        x_T = x_T.to(self.device)

        # for t in self.var_scheduler.timesteps:
        for t in range(self.num_timesteps+1, 0, -1): #T, T-1,.., 1
            # if isinstance(t, int):
            #     t = torch.tensor([t])
            # t = t.to(self.device)
            timesteps = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            x_T = self.p_sample(x_t=x_T, t=timesteps)
        
        x_0_pred = x_T
        return x_0_pred
    

