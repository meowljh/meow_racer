from pathlib import Path
from typing import Callable, Tuple, TypeVar, Optional

import pickle
import numpy as np

import torch
import torch.nn as nn

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from torch_buffer.base import Buffer
from torch_utils.experience import Experience

T = TypeVar("T")

class ReplayBuffer(Buffer[T]):
    """
    - Vanilla Replay Buffer
        -> Transition (s, a, r, s', d)
    - Diffusion Replay Buffer
        -> Fixed length trajectory
    """
    def __init__(self, 
                 max_len:int,
                 obs_dim: int,
                 action_dim: int,
                 horizon_len: Optional[int] = None,
                 ):
        super().__init__()
        self.max_len = max_len
        self.horizon_len = horizon_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.cur_len = 0 #current length of the buffer
        self.cur_ptr = 0 #index pointer

        leading_shape = (max_len, horizon_len) if horizon_len is not None else (max_len,)
        # save as numpy -> change to tensor when sampling from the buffer
        self.action_buffer = np.zeros((*leading_shape, self.action_dim), dtype=np.float32)
        self.obs_buffer = np.zeros((*leading_shape, self.obs_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((*leading_shape,), dtype=np.float32)
        self.next_obs_buffer = np.zeros((*leading_shape, self.obs_dim), dtype=np.float32)
        self.done_buffer = np.zeros((*leading_shape, ), dtype=np.bool_)

    def __len__(self):
        return self.cur_len
    
    def _buffers(self):
        # return (self.action_buffer, self.obs_buffer, self.reward_buffer, self.next_obs_buffer, self.done_buffer)
        """Algorithm의 stateless_update하는 부분에서 입력으로 들어가는 data의 순서가
        obs / action / reward / next_obs / done의 순서를 만족해야 함. """
        return (self.obs_buffer, self.action_buffer, self.reward_buffer, self.next_obs_buffer, self.done_buffer)


    def add(self, sample: T) -> None:
        self.cur_len += 1
        self.cur_len = min(self.cur_len, self.max_len)
        self.obs_buffer[self.cur_ptr] = sample.obs
        self.action_buffer[self.cur_ptr] = sample.action
        self.reward_buffer[self.cur_ptr] = sample.reward
        self.done_buffer[self.cur_ptr] = sample.done
        self.next_obs_buffer[self.cur_ptr] = sample.next_obs

        self.cur_ptr += 1
        self.cur_ptr %= self.max_len

    def add_batch(self, samples: T) -> None:
        B = samples.batch_size()

        for b in range(B):
            self.add(Experience.create(obs=samples.obs[b], action=samples.action[b], reward=samples.reward[b], terminated=samples.done[b], truncated=samples.done[b], next_obs=samples.next_obs[b]))
            # print(f"Added number {b}")
    
    def sample(self, size: int) -> T:
        return self.sample_with_indices(size=size)[0]
    
    def sample_with_indices(self, size:int):
        indices = np.random.choice(np.arange(0, self.cur_len), size=size)
        samples = tuple(np.take(buf, indices, axis=0) for buf in self._buffers())
        return samples, indices

    def replace(self, indices: np.ndarray, samples: T) -> None:
        for cnt, idx in enumerate(indices):
            self.obs_buffer[idx] = samples.obs[cnt]
            self.action_buffer[idx] = samples.action[cnt]
            self.reward_buffer[idx] = samples.reward[cnt]
            self.done_buffer[idx] = samples.done[cnt]
            self.next_obs_buffer[idx] = samples.next_obs[cnt]


    def save(self, path: Path) -> None:
        if self.cur_len < self.max_len:
            data = tuple(buf[:self.cur_len] for buf in self._buffers())
        else:
            data = self._buffers()

        with path.open('wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    obs_dim = 30
    action_dim = 3
    batch_size=256
    horizon_len = 10

    rb = ReplayBuffer(max_len=1000, horizon_len=horizon_len, obs_dim=obs_dim, action_dim=action_dim)

    sample = Experience.create_example(obs_dim=obs_dim, action_dim=action_dim, batch_size=None, horizon_size=horizon_len)
    batch_sample = Experience.create_example(obs_dim=obs_dim, action_dim=action_dim, batch_size=batch_size, horizon_size=horizon_len)


    rb.add(sample)

    rb.add_batch(batch_sample)

    sample, indices = rb.sample_with_indices(size=1)

    rb.replace(indices=np.array([3, 5, 7]), samples=Experience.create_example(obs_dim, action_dim, 3, horizon_size=horizon_len))

    print(f"Length of buffer: {len(rb)}")