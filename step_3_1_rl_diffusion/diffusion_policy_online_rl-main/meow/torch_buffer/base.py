from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Generic, Tuple, TypeVar


T = TypeVar("T")


class Buffer(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    def add(self, sample: T) -> None:
        ...
    
    @abstractmethod
    def add_batch(self, samples: T) -> None:
        ...
    
    @abstractmethod
    def sample(self, size: int) -> T:
        ...

    @abstractmethod
    def sample_with_indices(self, size:int):
        ...

    @abstractmethod
    def replace(self, indices, samples: T) -> None:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...