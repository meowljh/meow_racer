from typing import NamedTuple, Optional
import numpy as np

import torch

def probe_batch_size(reward: torch.Tensor) -> Optional[int]:
    try:
        if reward.ndim > 0:
            return reward.shape[0]
        else:
            return None
    except AttributeError:
        return None
    

class Experience(NamedTuple):
    obs: np.ndarray #torch.Tensor
    action: np.ndarray #torch.Tensor
    reward: np.ndarray #torch.Tensor
    next_obs: np.ndarray #torch.Tensor
    done: np.ndarray #torch.Tensor

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def as_dict(self):
        return {
            'obs': self.obs,
            'action': self.action,
            'reward': self.reward,
            'done': self.done,
            'next_obs': self.next_obs
        }
    
    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int]=None, horizon_size: Optional[int]=None):
        if batch_size is not None and horizon_size is not None:
            leading_dims = (batch_size, horizon_size)
        elif batch_size is not None:
            leading_dims = (batch_size,)
        elif horizon_size is not None:
            leading_dims = (horizon_size, )
        else:
            leading_dims = ()
        # leading_dims = (batch_size,) if batch_size is not None else ()
        return Experience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32)
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info=None):
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().numpy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.detach().cpu().numpy()
        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.detach().cpu().numpy()
        return Experience(obs=obs, action=action, reward=reward, done=terminated, next_obs=next_obs)
    
if __name__ == "__main__":
    exp_sample = Experience.create_example(obs_dim=30, action_dim=3, batch_size=1000)
    # breakpoint()
    for key, value in exp_sample.as_dict().items():
    # for name, data in zip(keys(exp_sample), exp_sample):
        print(f"{key}: {value.shape}")
        # print(name, data.shape)
