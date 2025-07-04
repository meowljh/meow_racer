import numpy as np
from gymnasium.spaces import Box

from rlkit.envs.proxy_env import ProxyEnv


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        ##만약에 observation state의 mean, std가 특정이 되어 있는 상황이 아니라면 임의로 평균 0, 표준편차 1로 설정한 정규 분포로 가정한다.
        ##그렇기 떄문에 statistics가 특정이 안되어 있으면 그냥 training batch의 평균과 표준편차를 계산해서 사용하게 된다.
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action) 
        if len(wrapped_step) == 4:
            next_obs, reward, done, info = wrapped_step
        elif len(wrapped_step) == 5:
            next_obs, reward, terminated, truncated, info = wrapped_step
            done = terminated | truncated

        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env


