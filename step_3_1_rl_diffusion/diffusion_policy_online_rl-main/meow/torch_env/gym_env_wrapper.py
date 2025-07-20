import numpy as np
import gymnasium as gym
from gymnasium import Env, Wrapper, make
from gymnasium.spaces import Box

class MeowGymWrapper(Wrapper):
    def __init__(self, env:Env):
        super().__init__(env)
        self.env = env

        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Box) and env.action_space.is_bounded()
    
        self.obs_dim, = env.observation_space.shape
        self.act_dim, = env.action_space.shape
        single_action_space = env.action_space

        if np.any(single_action_space.low != -1.) or np.any(single_action_space.high != 1.):
            print(f"Action Space is not normalized, but {single_action_space.low} to {single_action_space.high} will be rescaled...")
            self.needs_rescale = True
            self.original_action_center = (single_action_space.low + single_action_space.high) * 0.5
            self.original_action_half_range = (single_action_space.high - single_action_space.low) * 0.5
        else:
            self.needs_rescale = False

        self.original_action_dtype = env.action_space.dtype

        self._action_space = Box(
            low=-1.,
            high=1.,
            shape=env.action_space.shape,
            dtype=np.float32,
        )
    
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.astype(np.float32, copy=False), info
    
    def step(self, action:np.ndarray):
        action = action.astype(self.original_action_dtype)
        if self.needs_rescale:
            action *= self.original_action_half_range
            action += self.original_action_center

        obs, reward, terminated, truncated, info = self.env.step(action)
        
        return obs.astype(np.float32, copy=False), reward, terminated, truncated, info



        