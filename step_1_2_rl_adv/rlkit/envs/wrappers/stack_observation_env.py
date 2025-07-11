import numpy as np
from gymnasium.spaces import Box

from rlkit.envs.proxy_env import ProxyEnv


class StackObservationEnv(ProxyEnv):
    """
    Env wrapper for passing history of observations as the new observation
    """

    def __init__(
            self,
            env,
            stack_obs=1,
    ):
        ProxyEnv.__init__(self, env)
        self.stack_obs = stack_obs
        low = env.observation_space.low
        high = env.observation_space.high
        self.obs_dim = low.size
        self._last_obs = np.zeros((self.stack_obs, self.obs_dim))
        self.observation_space = Box(
            low=np.repeat(low, stack_obs),
            high=np.repeat(high, stack_obs),
        )

    def reset(self):
        self._last_obs = np.zeros((self.stack_obs, self.obs_dim))
        next_obs = self._wrapped_env.reset()
        self._last_obs[-1, :] = next_obs
        return self._last_obs.copy().flatten()

    def step(self, action): 
        env_step_ret = self._wrapped_env.step(action)
        if len(env_step_ret) == 4:
            next_obs, reward, done, info = env_step_ret
        elif len(env_step_ret) == 5:
            next_obs, reward, terminated, truncated, info = env_step_ret
            done = terminated | truncated

        self._last_obs = np.vstack((
            self._last_obs[1:, :],
            next_obs
        ))
        return self._last_obs.copy().flatten(), reward, done, info


