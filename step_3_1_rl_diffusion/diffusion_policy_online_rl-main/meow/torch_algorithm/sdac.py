from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import math
import pickle
import os
import numpy as np
from pathlib import Path

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from torch_algorithm.base import Algorithm
from torch_network.diffv2 import Diffv2Net
from torch_utils.experience import Experience
from torch_utils.tensor_utils import _add_batch_dim

def polyak_update_from_to(source, target, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1. - tau) + source_param.data * tau
        )


class LinearLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, 
                 init_lr_value:float, end_lr_value:float, 
                 transition_steps:int, transition_begin:int):
        self.optimizer = optimizer
        self.init_lr_value = init_lr_value
        self.end_lr_value = end_lr_value
        self.transition_steps = transition_steps
        self.transition_begin = transition_begin
        self.transition_end = transition_begin + transition_steps
        self.power = 1
        self.step_counter = 0

        super().__init__(optimizer=optimizer, last_epoch=-1)
    
    def get_lr(self):
        if (self.transition_begin <= self.step_counter <= self.transition_end):
            count = np.clip(self.step_counter - self.transition_begin, 0, self.transition_steps)
            frac = 1 - count / self.transition_steps
            ret_lr = [(self.init_lr_value - self.end_lr_value) * (frac**self.power) + self.end_lr_value for _ in range(len(self.optimizer.param_groups))]
        else:
            ret_lr = [g['lr'] for g in self.optimizer.param_groups]

        self.step_counter += 1
        return ret_lr
    
class SDAC(Algorithm):
    def __init__(self,
                 device,
                 agent: Diffv2Net,
                 gamma: float=0.99,
                 lr: float=1e-4,
                 alpha_lr: float=3e-2,
                 lr_schedule_end: float=5e-5,
                 tau: float=0.005,
                 delay_alpha_update: int=250,
                 delay_update: int=2,
                 reward_scale: float=0.2,
                 num_samples: int=200,
                 reverse_mc_num: int=64,
    ):
        super().__init__()
        self.device = device

        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.reverse_mc_num = reverse_mc_num

        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.q1_optim = torch.optim.Adam(params=self.agent.q1.parameters(), lr=lr) #critic optimizer (q1)
        self.q2_optim = torch.optim.Adam(params=self.agent.q2.parameters(), lr=lr) #critic optimizer (q2)
        self.policy_optim = torch.optim.Adam(params=self.agent.policy.parameters(), lr=lr) #policy optimizer
        self.policy_lr_scheduler = LinearLRScheduler(
            optimizer=self.policy_optim, init_lr_value=lr,
            end_lr_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4)
        )
        self.log_alpha_optim = torch.optim.Adam([self.agent.log_alpha], lr=alpha_lr, betas=(0.9, 0.999))
        self.entropy = 0.

        self.step = 0
        self.running_mean = 0.
        self.running_std = 0.
    
    def stateless_update(self,  data: Experience):
        try:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_ois, data.done
        except:
            obs, action, reward, next_obs, done = data

        reward *= self.reward_scale

        ## Sampling and experience replay ##
        # Cast the array data to tensor and cuda device
        obs = _add_batch_dim(torch.tensor(obs, dtype=torch.float32, device=self.device))
        action = _add_batch_dim(torch.tensor(action, dtype=torch.float32, device=self.device))
        reward = _add_batch_dim(torch.tensor(reward, dtype=torch.float32, device=self.device))
        next_obs = _add_batch_dim(torch.tensor(next_obs, dtype=torch.float32, device=self.device))
        done = _add_batch_dim(torch.tensor(done, dtype=torch.float32, device=self.device))

        def get_min_q(s, a):
            q1 = self.agent.q1(s, a)
            q2 = self.agent.q2(s, a)
            min_q = torch.minimum(q1, q2)
            return min_q
        
        def get_min_target_q(s, a):
            q1 = self.agent.target_q1(s, a)
            q2 = self.agent.target_q2(s, a)
            min_target_q = torch.minimum(q1, q2)
            return min_target_q
        
        next_action = self.agent.get_action(obs=next_obs, log_alpha=self.agent.log_alpha)
        q1_target = self.agent.target_q1(obs=next_obs, act=next_action)
        q2_target = self.agent.target_q2(obs=next_obs, act=next_action)
        q_target = torch.minimum(q1_target, q2_target)
        q_backup = reward + (1. - done) * self.gamma * q_target

        ## Policy Evaluation ##
        q1 = self.agent.q1(obs=obs, act=action)
        q2 = self.agent.q2(obs=obs, act=action)
        q1_loss = torch.mean((q1 - q_backup.detach()) ** 2)
        q2_loss = torch.mean((q2 - q_backup.detach()) ** 2)

        new_action = self.agent.get_action(obs=obs, log_alpha=self.agent.log_alpha)
        timesteps = torch.randint(size=(next_obs.shape[0], ), low=0, high=self.agent.num_timesteps, device=self.device) # random timestep sampling (a_t -> a_0 from random t training)
        # timesteps = torch.randint(size=(next_obs.shape[0], ), low=0, high=self.agent.num_timesteps, device=self.device, dtype=torch.float32) # random timestep sampling (a_t -> a_0 from random t training)

        noise1 = torch.randn(size=action.shape, device=self.device, dtype=torch.float32)

        """RSSM 계산할 때의 reverse process에서의 target distribution을 모르기 때문에 q_sample을 할 때에 
        policy의 action을 x_0, 즉 원래 데이터로 보고 거기에 noise를 더해서 Q-value를 더 높일 수 있는 방향으로 찾고자 한다."""
        tilde_at = self.agent.diffusion.q_sample(t=timesteps, x_0=new_action, noise=noise1)
        # tilde_at = torch.stack([
        #     self.agent.diffusion.q_sample(timesteps[i], x_0=new_action[i], noise=noise1[i]) for i in range(timesteps.shape[0])
        # ]) #a_0_tilde at line 10 of the SDAC algorithm
        
        # Try multiple samples to fit loss
        """as stated in pg.6 of the paper,
        they sample multiple actions for every s, a_t and use the logsumexp trick to avoid the explosion of the weights"""
        reverse_mc_num = self.reverse_mc_num
        tilde_at = tilde_at.repeat_interleave(repeats=reverse_mc_num, dim=0)
        timesteps = timesteps.repeat_interleave(repeats=reverse_mc_num, dim=0)
        wide_obs = obs.repeat_interleave(repeats=reverse_mc_num, dim=0)

        ## RSSM Loss ##
        noise2 = torch.randn(size=(action.shape[0] * reverse_mc_num, action.shape[1]), device=self.device, dtype=torch.float32)
        recon = self.agent.diffusion.get_recon(t=timesteps, x=tilde_at, noise=noise2).clip(-1., 1.).float()
        """alpha: regularization coefficient for the entropy"""
        q_min = get_min_q(s=wide_obs, a=recon) * 5. / torch.exp(self.agent.log_alpha) # initial alpha value: 5
        q_mean, q_std = q_min.mean(), q_min.std()
        q_reshape = q_min.reshape((-1, reverse_mc_num)) # [batch size, mc_num]
        Z = torch.logsumexp(input=q_reshape, dim=1, keepdim=True) # [batch size, 1]
        q_weights = torch.exp(q_reshape - Z).flatten() # [batch size, mc_num]
        
        # breakpoint()


        def denoiser(t, x):
            return self.agent.policy(obs=wide_obs, act=x, t=t)
        
        rssm_loss = self.agent.diffusion.reverse_sampling_weighted_p_loss(
            noise=noise2,
            weights=q_weights.detach(),
            model_fn=denoiser,
            x_t=tilde_at,
            t=timesteps 
        )

        ## log alpha loss (for Entropy Maximization RL) ##
        approx_entropy = 0.5 * self.agent.act_dim * torch.log(2 * torch.pi * math.exp(1) * (0.1 * torch.exp(self.agent.log_alpha)) ** 2) #log_alpha is on CUDA device
        log_alpha_loss = -1 * self.agent.log_alpha * (-1 * approx_entropy.detach() + self.agent.target_entropy)

        ## update critic (q1, q2)
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        ## update policy
        if self.step % self.delay_update == 0:
            self.policy_optim.zero_grad()
            rssm_loss.backward()
            self.policy_optim.step()

        ## update log alpha
        if self.step % self.delay_alpha_update == 0:
            self.log_alpha_optim.zero_grad()
            log_alpha_loss.backward()
            self.log_alpha_optim.step()
        
        ## update target critic (target_q1, target_q2) & target policy
        if self.step % self.delay_update == 0:
            polyak_update_from_to(source=self.agent.q1, target=self.agent.target_q1, tau=self.tau)
            polyak_update_from_to(source=self.agent.q2, target=self.agent.target_q2, tau=self.tau)
            polyak_update_from_to(source=self.agent.policy, target=self.agent.target_policy, tau=self.tau)
        
        self.running_mean += 0.001 * (q_mean.detach().cpu().numpy() - self.running_mean)
        self.running_std += 0.001 * (q_std.detach().cpu().numpy() - self.running_std)
        self.step += 1

        info = {
            "q1_loss": q1_loss.item(),
            "q1_mean": torch.mean(q1.detach()),
            "q1_max": torch.max(q1.detach()), 
            "q1_min": torch.min(q1.detach()),
            "q2_loss": q2_loss.item(),
            "q2_mean": torch.mean(q2.detach()),
            "q2_max": torch.max(q2.detach()),
            "q2_min": torch.min(q2.detach()),
            "q_weights_std": torch.std(q_weights.detach()),
            "q_weights_mean": torch.mean(q_weights.detach()),
            "q_weights_min": torch.min(q_weights.detach()),
            "q_weights_max": torch.max(q_weights.detach()),
            "hist_q_weights": q_weights.detach(),
            "hist_t": timesteps.detach(),
            "scale_q_mean": torch.mean(q_min.detach()),
            "scale_q_std": torch.std(q_min.detach()),
            "entropy_approx": (0.5 * self.agent.act_dim * torch.log(2 * torch.pi * math.exp(1) * (0.1 * torch.exp(self.agent.log_alpha)) ** 2)).detach(),
            "running_q_mean": self.running_mean,
            "running_q_std": self.running_std,
            "step": self.step
        }

        #### CUDA Memory Save ####
        obs=obs.detach().cpu().numpy()
        action=action.detach().cpu().numpy()
        reward=reward.detach().cpu().numpy()
        next_obs=next_obs.detach().cpu().numpy()
        done=done.detach().cpu().numpy()
        tilde_at = tilde_at.detach().cpu().numpy()
        wide_obs = wide_obs.detach().cpu().numpy()
        q1 = q1.detach().cpu().numpy()
        q2 = q2.detach().cpu().numpy()
        q_weights = q_weights.detach().cpu().numpy()
        q_mean = q_mean.detach().cpu().numpy()
        q_std = q_std.detach().cpu().numpy()
        Z = Z.detach().cpu().numpy()
        q_reshape = q_reshape.detach().cpu().numpy()
        timesteps = timesteps.detach().cpu().numpy()
        q_min = q_min.detach().cpu().numpy() 
        noise1 = noise1.detach().cpu().numpy()
        noise2 = noise2.detach().cpu().numpy()
        
        q1_loss = q1_loss.item();q2_loss = q2_loss.item();rssm_loss = rssm_loss.item();log_alpha_loss = log_alpha_loss.item()
        del obs;del action;del reward;del next_obs;del done
        del noise1;del noise2
        del tilde_at;del wide_obs
        del Z;del q_mean;del q_std;del q_reshape
        del q1;del q2;del q_weights;del timesteps;del q_min

        torch.cuda.empty_cache()

        return info

    def get_policy_params(self):
        return (self.agent.policy.state_dict(), self.agent.log_alpha, self.agent.q1.state_dict(), self.agent.q2.state_dict())

    def get_policy_params_to_save(self):
        return (self.agent.target_policy.state_dict(), self.agent.log_alpha, self.agent.target_q1.state_dict(), self.agent.target_q2.state_dict())
    
    def get_value_params(self):
        return (self.agent.q1.state_dict(), self.agent.q2.state_dict())

    def get_action(self, obs:np.ndarray)->np.ndarray:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.float().to(self.device)
        # tensor_obs = _add_batch_dim(torch.tensor(obs, dtype=torch.float32, device=self.device))
        tensor_obs = _add_batch_dim(obs)

        action = self.agent.get_action(obs=tensor_obs, log_alpha=self.agent.log_alpha)

        return np.asarray(action.detach().cpu())
    
    def get_state_dicts(self) -> dict:
        return {
            'policy': self.agent.policy.state_dict(),
            'q1': self.agent.q1.state_dict(),
            'q2': self.agent.q2.state_dict(),
            'target_q1': self.agent.target_q1.state_dict(),
            'target_q2': self.agent.target_q2.state_dict(),
            'target_policy': self.agent.target_policy.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'q1_optim': self.q1_optim.state_dict(),
            'q2_optim': self.q2_optim.state_dict(),
            'log_alpha': self.agent.log_alpha,
            'log_alpha_optim': self.log_alpha_optim.state_dict()
        }
    
    def save_network(self, path:str) -> None:
        torch.save(self.agent.policy.state_dict(), f"{path}/policy.pth")
        torch.save(self.agent.q1.state_dict(), f"{path}/q1.pth")
        torch.save(self.agent.q2.state_dict(), f"{path}/q2.pth")
        torch.save(self.agent.target_q1.state_dict(), f"{path}/target_q1.pth")
        torch.save(self.agent.target_q2.state_dict(), f"{path}/target_q2.pth")
        torch.save(self.agent.target_policy.state_dict(), f"{path}/target_policy.pth")
    
    def load_network(self, path:str) -> None:
        self.agent.policy.load_state_dict(f"{path}/policy.pth")
        self.agent.q1.load_state_dict(f"{path}/q1.pth")
        self.agent.q2.load_state_dict(f"{path}/q2.pth")
        self.agent.target_q1.load_state_dict(f"{path}/target_q1.pth")
        self.agent.target_q2.load_state_dict(f"{path}/target_q2.pth")
        self.agent.target_policy.load_state_dict(f"{path}/target_policy.pth")

    def save_optimizer(self, path:str) -> None:
        torch.save(self.policy_optim.state_dict(), f"{path}/policy_optim.pth")
        torch.save(self.q1_optim.state_dict(), f"{path}/q1_optim.pth")
        torch.save(self.q2_optim.state_dict(), f"{path}/q2_optim.pth")
        torch.save(self.log_alpha_optim.state_dict(), f"{path}/log_alpha_optim.pth")
    
    def load_optimizer(self, path:str) -> None:
        self.policy_optim.load_state_dict(f"{path}/policy_optim.pth")
        self.q1_optim.load_state_dict(f"{path}/q1_optim.pth")
        self.q2_optim.load_state_dict(f"{path}/q2_optim.pth")
        self.log_alpha_optim.load_state_dict(f"{path}/log_alpha_optim.pth")