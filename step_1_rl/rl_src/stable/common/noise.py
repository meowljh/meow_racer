import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike

class ActionNoise(ABC):
    """
    The action noise base class
    """

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """
        Call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class NormalActionNoise(ActionNoise):
    """
    A Gaussian action noise.

    :param mean: Mean value of the noise
    :param sigma: Scale of the noise (std here)
    :param dtype: Type of the output noise
    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray, dtype: DTypeLike = np.float32) -> None:
        self._mu = mean
        self._sigma = sigma
        self._dtype = dtype
        super().__init__()

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma).astype(self._dtype)

    def __repr__(self) -> str:
        return f"NormalActionNoise(mu={self._mu}, sigma={self._sigma})"

class Linear_NormalActionNoise(NormalActionNoise):
    def __init__(self, mean: np.ndarray, sigma: np.ndarray, 
                 max_steps: int, 
                 final_sigma: np.ndarray=None, 
                 dtype=np.float32) -> None:
        super().__init__(mean=mean, sigma=sigma, dtype=dtype)
        self._step = 0
        self._max_steps = max_steps
        if final_sigma is None:
            final_sigma = np.zeros_like(sigma)
        self._final_sigma = final_sigma
    
    def __call__(self):
        t = min(1., self._step / self._max_steps)
        sigma = (1. - t) * self._sigma + t * self._final_sigma
        self._step += 1
        return np.random.normal(self._mu, sigma).astype(self._dtype)

    def __repr__(self) -> str:
        return f"LinearScheduled_NormalActionNoise(mu={self._mu}, sigma={self._sigma})"

class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: Mean of the noise
    :param sigma: Scale of the noise
    :param theta: Rate of mean reversion
    :param dt: Timestep for the noise
    :param initial_noise: Initial value for the noise output, (if None: 0)
    :param dtype: Type of the output noise
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[np.ndarray] = None,
        dtype: DTypeLike = np.float32,
    ) -> None:
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self._dtype = dtype
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()
        super().__init__()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise.astype(self._dtype)

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"


class VectorizedActionNoise(ActionNoise):
    """
    A Vectorized action noise for parallel environments.

    :param base_noise: Noise generator to use
    :param n_envs: Number of parallel environments
    """

    def __init__(self, base_noise: ActionNoise, n_envs: int) -> None:
        try:
            self.n_envs = int(n_envs)
            assert self.n_envs > 0
        except (TypeError, AssertionError) as e:
            raise ValueError(f"Expected n_envs={n_envs} to be positive integer greater than 0") from e

        self.base_noise = base_noise
        self.noises = [copy.deepcopy(self.base_noise) for _ in range(n_envs)]

    def reset(self, indices: Optional[Iterable[int]] = None) -> None:
        """
        Reset all the noise processes, or those listed in indices.

        :param indices: The indices to reset. Default: None.
            If the parameter is None, then all processes are reset to their initial position.
        """
        if indices is None:
            indices = range(len(self.noises))

        for index in indices:
            self.noises[index].reset()

    def __repr__(self) -> str:
        return f"VecNoise(BaseNoise={self.base_noise!r}), n_envs={len(self.noises)})"

    def __call__(self) -> np.ndarray:
        """
        Generate and stack the action noise from each noise object.
        """
        noise = np.stack([noise() for noise in self.noises])
        return noise

    @property
    def base_noise(self) -> ActionNoise:
        return self._base_noise

    @base_noise.setter
    def base_noise(self, base_noise: ActionNoise) -> None:
        if base_noise is None:
            raise ValueError("Expected base_noise to be an instance of ActionNoise, not None", ActionNoise)
        if not isinstance(base_noise, ActionNoise):
            raise TypeError("Expected base_noise to be an instance of type ActionNoise", ActionNoise)
        self._base_noise = base_noise

    @property
    def noises(self) -> list[ActionNoise]:
        return self._noises

    @noises.setter
    def noises(self, noises: list[ActionNoise]) -> None:
        noises = list(noises)  # raises TypeError if not iterable
        assert len(noises) == self.n_envs, f"Expected a list of {self.n_envs} ActionNoises, found {len(noises)}."

        different_types = [i for i, noise in enumerate(noises) if not isinstance(noise, type(self.base_noise))]

        if len(different_types):
            raise ValueError(
                f"Noise instances at indices {different_types} don't match the type of base_noise", type(self.base_noise)
            )

        self._noises = noises
        for noise in noises:
            noise.reset()
