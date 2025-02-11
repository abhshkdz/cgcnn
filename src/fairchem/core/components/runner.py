from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from fairchem.core._cli_hydra import JobConfig


class Runner(metaclass=ABCMeta):
    """
    Represents an abstraction over things that run in a loop and can save/load state.
    ie: Trainers, Validators, Relaxation all fall in this category.
    This allows us to decouple away from a monolithic trainer class
    """

    @property
    def job_config(self) -> JobConfig:
        return self._job_config

    @job_config.setter
    def job_config(self, cfg: DictConfig):
        self._job_config = cfg

    @abstractmethod
    def run(self, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_state(self) -> None:
        raise NotImplementedError


# Used for testing
class MockRunner(Runner):
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def run(self) -> Any:
        if self.x * self.y > 1000:
            raise ValueError("sum is greater than 1000!")
        return self.x + self.y

    def save_state(self) -> None:
        pass

    def load_state(self) -> None:
        pass
