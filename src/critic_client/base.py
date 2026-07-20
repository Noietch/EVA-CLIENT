"""Critic client interface, errors, and the server-free mock backend."""

from __future__ import annotations

import abc
import dataclasses

import numpy as np

from core.config import ConfigDict
from core.registry import CRITIC_REGISTRY


class CriticConnectionError(RuntimeError):
    """Raised when the initial critic connection fails."""


class CriticRequestError(RuntimeError):
    """Raised when a critic evaluation request fails."""


@dataclasses.dataclass
class CriticBuildContext:
    """Runtime-only critic construction options."""

    retry_until_connected: bool = False
    max_retries: int = 0


class CriticClient(abc.ABC):
    """Evaluate an observation and candidate action sequence to one scalar value."""

    @abc.abstractmethod
    def evaluate(self, observation: dict, actions: np.ndarray) -> float:
        """Return one scalar critic value for observation + actions."""
        ...

    @property
    @abc.abstractmethod
    def metadata(self) -> dict:
        """Return critic server capability metadata."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        return None


@CRITIC_REGISTRY.register_client("mock")
class MockCriticClient(CriticClient):
    """Deterministic scalar critic used by unit tests and offline demos."""

    @classmethod
    def from_config(
        cls, config: ConfigDict, ctx: CriticBuildContext
    ) -> MockCriticClient:
        _ = config, ctx
        return cls()

    def evaluate(self, observation: dict, actions: np.ndarray) -> float:
        action_array = np.asarray(actions, dtype=np.float32)
        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        target = action_array.reshape(-1, action_array.shape[-1]).mean(axis=0)
        distance = float(np.mean(np.square(target - state[: target.shape[0]])))
        return 1.0 / (1.0 + distance)

    @property
    def metadata(self) -> dict:
        return {"server_name": "eva-mock-critic"}
