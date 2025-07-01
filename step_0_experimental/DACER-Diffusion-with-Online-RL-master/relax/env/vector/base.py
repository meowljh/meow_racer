from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class VectorEnv(ABC):
    """Base class for vectorized environments in relax."""

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        ...

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self

    num_envs: int
