from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

import numpy as np

from noise import (
    NormalActionNoise, OrnsteinUhlenbeckActionNoise, 
    Linear_NormalActionNoise,
    VectorizedActionNoise
)
from vec_env import DummyVecEnv

import os, sys
RL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(RL_ROOT))
from load_rl_model import get_action_noise

"""Action Noise Scheduler
- 중요한건, action noise는 off-policy algorithm에서만 필요로 한다는 것이다.

"""
class NoiseScheduler(ABC):
    """Action Noise Scheduler base class"""
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def __call__(self, env) -> np.ndarray:
        raise NotImplementedError()
    
    def reset(self) -> None:
        """call end of episode reset for the noise"""
        pass

def load_normal_action_noise(args, env):
    if args.linear_schedule:
        return Linear_NormalActionNoise(mean=args.mean, sigma=args.sigma)
    else:
        return NormalActionNoise(**args)
class Step_NoiseScheduler(NoiseScheduler):
    def __init__(self, 
                 env,
                 n_envs:int,
                 normal_noise_args,
                 brownian_noise_args,
                 noise_change_interval:str="step_1000"
    ):
        assert "_" in noise_change_interval
        interval_type, interval_step = noise_change_interval.split("_")
        self.interval_type = str(interval_type)
        self.interval_step = float(interval_step)
        
        self.n_envs = n_envs
        if self.n_envs > 1:
            if normal_noise_args.is_linear:
                self.gaussian_noise = VectorizedActionNoise(get_action_noise(normal_noise_args, env, "linear_gaussian"), self.n_envs)
            else:
                self.gaussian_noise = VectorizedActionNoise(get_action_noise(normal_noise_args, env, "gaussian"), self.n_envs)
            self.brownian_noise = VectorizedActionNoise(get_action_noise(brownian_noise_args, env, "brownian"), self.n_evns)
        else:
            if normal_noise_args.is_linear:
                self.gaussian_noise = get_action_noise(normal_noise_args, env, "linear_gaussian")
            else:
                self.gaussian_noise = get_action_noise(normal_noise_args, env, "gaussian")
            self.brownian_noise = get_action_noise(brownian_noise_args, env, "brownian")
        
        super().__init__()
        
        self.step_count = 0
        self.episode = 0
        
    def reset(self) -> None:
        self.brownian_noise.reset()
        self.gaussian_noise.reset()
        
        
    def __call__(self, env) -> np.ndarray:
        if isinstance(env, DummyVecEnv):
            num_timestep = env.get_attr('num_timesteps')
            num_episode = env.get_attr('num_episode')
            num_timestep /= len(env)
        else:
            num_timestep = env.num_timestep
            num_episode = env.num_episode
        
        self.step_count += num_timestep
        self.episode = num_episode
        
        '''vectorized environment인 경우에는 각 environment에서는 single step의 환경 update만 있지만 동시에 n_envs의 개수만큼 timestep이 증가함.'''
            
        if self.interval_type == "episode":
            if self.episode <= self.interval_step:
                return self.gaussian_noise()
            else:
                return self.brownian_noise()
        elif self.interval_type == "step":
            if self.step_count <= self.interval_step:
                return self.gaussian_noise()
            else:
                return self.brownian_noise()
        