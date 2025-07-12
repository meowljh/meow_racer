import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper, RewardWrapper, ObservationWrapper

from typing import Any
class NormalizedObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = env.observation_space
        self.environment_config = env.environment_config
        self._get_obs_mean_std_uniform()

    def step(
        self, 
        action,
        style_setting=None
    ):
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action, style_setting=style_setting)
        return self.observation(observation), reward, terminated, truncated, info


    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
        style=None,
    ):
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options, style=style)
        return self.observation(obs), info
    
    def _get_obs_mean_std_uniform(self):
        '''uniform distribution이라고 가정하고 구현'''
        stats_dict = {}
        
        for obs_name, box_state in self.observation_space.items():
            low_arr = box_state.low
            high_arr = box_state.high
            
            if 'lookahead' in obs_name:
                scale_method = getattr(self.environment_config['observation']['lookahead'], 'scale_method', None)
            else:
                scale_method = getattr(self.environment_config['observation'][obs_name], 'scale_method', None)
            
            if scale_method is None or scale_method == 'standard':
                mean = (low_arr + high_arr) / 2
                var = ((high_arr - low_arr)**2) / 12
                stats_dict[obs_name] = {'mean': mean, 'std': np.sqrt(var)}
                
            elif scale_method in ['minmax', 'none']:
                stats_dict[obs_name] = {'min': low_arr, 'max': high_arr}
        
        self.stats_dict = stats_dict
        
    def observation(self, obs):
        def safe_divide(a, b, fill=0):
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
                c[~np.isfinite(c)] = fill
            return c
        new_obs = np.zeros_like(obs)
        ptr = 0
        for obs_name, box_state in self.observation_space.items():
            stats = self.stats_dict[obs_name]
            temp_obs = obs[ptr:ptr + box_state.shape[0]]
            if 'mean' in stats:
                temp_obs = np.true_divide(temp_obs - stats['mean'], stats['std'])
            elif 'min' in stats:
                temp_obs = np.true_divide(temp_obs - stats['min'], stats['max'] - stats['min'])
                
            temp_obs = np.nan_to_num(temp_obs, nan=0., posinf=0., neginf=0.)
            new_obs[ptr:ptr+box_state.shape[0]] = temp_obs
            ptr += box_state.shape[0]
            
        assert ptr == obs.shape[0]
        # if len(obs.shape)> 1:
        #     breakpoint()
        # breakpoint()
        
        return new_obs
        
