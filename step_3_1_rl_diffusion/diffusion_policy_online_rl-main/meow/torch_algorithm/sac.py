from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from torch_algorithm.base import Algorithm
from torch_network.sac import SACNet
from torch_utils.experience import Experience
from torch_utils.tensor_utils import _add_batch_dim
# from torch_utils.typing_utils import Metric

def polyak_update_from_to(source, target, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1. - tau) + source_param.data * tau
        )

class SAC(Algorithm):
    def __init__(self, 
                 device,
                 agent: SACNet,
                 gamma: float=0.99,
                 lr: float=1e-4,
                 alpha_lr: float=3e-4,
                 tau: float=0.005,
                 reward_scale: float=0.2,
                 **kwargs):
        super().__init__()
        self.device = device

        self.agent = agent
        self.gamma = gamma #reward discount factor
        self.tau = tau #for target q1, q2 update factor
        self.reward_scale = reward_scale
        
    
        #load optimizers (policy, q1, q2, log_alpha) - target_q1, target_q2 will be 'soft updated'
        self.policy_optim = Adam(self.agent.policy.parameters(), lr=lr, betas=(0.9, 0.999))
        self.q1_optim = Adam(self.agent.q1.parameters(), lr=lr, betas=(0.9, 0.999))
        self.q2_optim = Adam(self.agent.q2.parameters(), lr=lr, betas=(0.9, 0.999))
        self.log_alpha_optim = Adam([self.agent.log_alpha], lr=alpha_lr, betas=(0.9, 0.999))

    def stateless_update(self, data: Experience):
        """called as self.update from the base object  Algorithm
        """
        try:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
        except:
            obs, action, reward, next_obs, done = data # data.obs, data.action, data.reward, data.next_obs, data.done

        # load to tensor and gpu device
        obs = _add_batch_dim(torch.tensor(obs, dtype=torch.float32, device=self.device))
        action = _add_batch_dim(torch.tensor(action, dtype=torch.float32, device=self.device))
        reward = _add_batch_dim(torch.tensor(reward, dtype=torch.float32, device=self.device))
        next_obs = _add_batch_dim(torch.tensor(next_obs, dtype=torch.float32, device=self.device))
        done = _add_batch_dim(torch.tensor(done, dtype=torch.float32, device=self.device))

        reward *= self.reward_scale
        alpha = self.agent.log_alpha.exp()

        # compute target q
        next_action, next_logp = self.agent.evaluate(obs=next_obs) #P(s_{t+1})
        q1_target = self.agent.target_q1(obs=next_obs, act=next_action)
        q2_target = self.agent.target_q2(obs=next_obs, act=next_action)
        # q_target = torch.min(q1_target, q2_target) - torch.exp(self.agent.log_alpha) * next_logp
        q_target = torch.min(q1_target, q2_target) - alpha * next_logp
        q_backup = reward + (1. - done) * self.gamma * q_target 

        # update q
        q1_pred = self.agent.q1(obs=obs, act=action)
        qf1_loss = torch.mean((q1_pred - q_backup.detach()) ** 2)
        q2_pred = self.agent.q2(obs=obs, act=action)
        qf2_loss = torch.mean((q2_pred - q_backup.detach()) ** 2)

        # update policy
        new_action, new_logp = self.agent.evaluate(obs=obs)
        q1 = self.agent.q1(obs=obs, act=new_action)
        q2 = self.agent.q2(obs=obs, act=new_action)
        q = torch.min(q1, q2)

        # policy_loss = torch.mean(torch.exp(self.agent.log_alpha) * new_logp - q)
        policy_loss = torch.mean(alpha * new_logp - q)

        # update alpha
        log_alpha_loss = -torch.mean(self.agent.log_alpha * (new_logp + self.agent.target_entropy).detach())
        
        # update all parameters
        self.log_alpha_optim.zero_grad()
        log_alpha_loss.backward()
        self.log_alpha_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.q1_optim.zero_grad()
        qf1_loss.backward()
        self.q1_optim.step()
        
        self.q2_optim.zero_grad()
        qf2_loss.backward()
        self.q2_optim.step()

        # polyak average update of the target_q1, q2
        polyak_update_from_to(source=self.agent.q1, target=self.agent.target_q1, tau=self.tau)
        polyak_update_from_to(source=self.agent.q2, target=self.agent.target_q2, tau=self.tau)

        info = {
            'q1_loss': qf1_loss,
            'q2_loss': qf2_loss,
            'q1': torch.mean(q1),
            'q2': torch.mean(q2),
            'policy_loss': policy_loss,
            'entropy': -torch.mean(new_logp),
            'alpha': torch.exp(self.agent.log_alpha)
        }

        return info
    

    def get_state_dicts(self) -> dict:
        return {
            'policy': self.agent.policy.state_dict(),
            'q1': self.agent.q1.state_dict(),
            'q2': self.agent.q2.state_dict(),
            'target_q1': self.agent.target_q1.state_dict(),
            'target_q2': self.agent.target_q2.state_dict(),
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
    
    def load_network(self, path:str) -> None:
        self.agent.policy.load_state_dict(f"{path}/policy.pth")
        self.agent.q1.load_state_dict(f"{path}/q1.pth")
        self.agent.q2.load_state_dict(f"{path}/q2.pth")
        self.agent.target_q1.load_state_dict(f"{path}/target_q1.pth")
        self.agent.target_q2.load_state_dict(f"{path}/target_q2.pth")

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

