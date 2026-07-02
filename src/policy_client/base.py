"""Policy client core — the abstract interface, error types, and the two built-in
clients that need no policy server (a synthetic random client and a dataset
trajectory replay client).

Server-backed clients live in the ``policy_client.openpi`` subpackage.
"""

from __future__ import annotations

import abc
import dataclasses
import time

import numpy as np

from core.config import ConfigDict
from core.registry import POLICY_REGISTRY


class PolicyConnectionError(RuntimeError):
    """Raised when the initial WebSocket connection to the policy server fails."""


class PolicyRequestError(RuntimeError):
    """Raised when an inference request to a connected policy server fails."""


@dataclasses.dataclass
class PolicyBuildContext:
    """Cross-domain inputs a policy backend needs at construction time.

    action_dim / action_trajectory feed the server-free mock and replay clients;
    the retry knobs control server connection startup behavior. Static client
    parameters (host, port, RTC variant, ...) live in the policy config, not here.
    """

    action_dim: int = 0
    action_trajectory: np.ndarray | None = None
    retry_until_connected: bool = False
    max_retries: int = 0


class PolicyClient(abc.ABC):
    """Abstract base class for policy clients.

    Concrete implementations must provide infer() to send an observation and
    receive an action chunk, reset() to clear per-episode state, and a metadata
    property for capability discovery (action_mode, chunk_size, etc.).
    """

    @abc.abstractmethod
    def infer(
        self,
        observation: dict,
        prev_action: np.ndarray | None = None,
        rtc_params: dict | None = None,
    ) -> dict:
        """Run one inference and return the next action chunk.

        Args:
            observation: Current observation. Keys are backend-specific but the
                common EVA shape is ``{"state": ndarray[action_dim] float32,
                "images": {cam_name: ndarray[H, W, C] uint8}, "prompt": str}``.
            prev_action: Previously executed actions fed back for conditioning,
                ``[T, action_dim] float32`` or None. Only meaningful for RTC
                clients; ignored otherwise.
            rtc_params: RTC conditioning parameters (e.g. latency shift), or None
                for non-RTC inference.

        Returns:
            ``{"actions": ndarray[T, action_dim] float32}`` — the predicted
            action chunk, ``T`` time steps of ``action_dim`` per step.
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear per-episode state so the next infer() starts a fresh episode."""
        ...

    @property
    @abc.abstractmethod
    def metadata(self) -> dict:
        """Capability descriptor: ``server_name``, ``action_mode``, ``chunk_size``, etc."""
        ...


@POLICY_REGISTRY.register_client("mock")
class RandomPolicyClient(PolicyClient):
    """Returns smooth mock actions that simulate real model stochasticity.

    Base trajectory: deterministic sum-of-sinusoids (same at a given step
    regardless of which chunk requested it).
    Per-chunk perturbation: random low-degree polynomial unique to each
    ``infer()`` call — models the prediction variance a real policy exhibits.

    When the inference strategy ensembles overlapping chunks the base aligns
    perfectly while the per-chunk polynomials average out, producing the
    characteristic "smooth core + noisy edges" pattern visible in real runs.
    """

    def __init__(
        self, action_dim: int, chunk_size: int = 50, step_advance: int | None = None
    ) -> None:
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        # step_advance: how many steps the mock advances per inference call.
        # In async mode, only a fraction of each chunk is consumed before the
        # next inference, so this should be << chunk_size (default: chunk_size // 3).
        self._step_advance = step_advance if step_advance is not None else max(1, chunk_size // 3)
        self._chunk_idx = 0
        # Per-dimension sinusoidal parameters (deterministic base trajectory)
        rng = np.random.default_rng(42)
        self._freqs = rng.uniform(0.3, 1.2, size=action_dim).astype(np.float32)
        self._amps = rng.uniform(0.008, 0.025, size=action_dim).astype(np.float32)
        self._phases = rng.uniform(0.0, 2 * np.pi, size=action_dim).astype(np.float32)

    @classmethod
    def from_config(cls, config: ConfigDict, ctx: PolicyBuildContext) -> RandomPolicyClient:
        """Build from the policy config; chunk_size is an optional backend_option."""
        opts = config.backend_options
        if "chunk_size" in opts:
            return cls(action_dim=ctx.action_dim, chunk_size=opts["chunk_size"])
        return cls(action_dim=ctx.action_dim)

    def infer(
        self,
        observation: dict,
        prev_action: np.ndarray | None = None,
        rtc_params: dict | None = None,
    ) -> dict:
        """Return a synthetic action chunk (deterministic base + per-chunk noise).

        Args:
            observation: Ignored; present for interface compatibility.
            prev_action: Ignored.
            rtc_params: Ignored.

        Returns:
            ``{"actions": ndarray[chunk_size, action_dim] float32}``.
        """
        # Simulate inference latency (150–250 ms)
        time.sleep(np.random.uniform(0.15, 0.25))

        # Base step determined by chunk index and step_advance so that
        # consecutive inferences overlap correctly (like a real async strategy).
        base_step = self._chunk_idx * self._step_advance
        t = np.arange(base_step, base_step + self._chunk_size, dtype=np.float32) / 30.0
        actions = np.zeros((self._chunk_size, self._action_dim), dtype=np.float32)

        # Deterministic base: same output for same global step
        for d in range(self._action_dim):
            actions[:, d] = self._amps[d] * np.sin(2 * np.pi * self._freqs[d] * t + self._phases[d])

        # Per-chunk smooth perturbation: random cubic polynomial
        chunk_rng = np.random.default_rng(self._chunk_idx * 31 + 7)
        t_local = np.linspace(0.0, 1.0, self._chunk_size, dtype=np.float32)
        for d in range(self._action_dim):
            coeffs = chunk_rng.normal(0, 0.004, size=4).astype(np.float32)
            actions[:, d] += np.polyval(coeffs, t_local)

        self._chunk_idx += 1
        return {"actions": actions}

    def reset(self) -> None:
        """Rewind the chunk counter to the start of an episode."""
        self._chunk_idx = 0

    @property
    def metadata(self) -> dict:
        """Capability descriptor for the random mock client."""
        return {
            "server_name": "eva-random-mock",
            "action_mode": "qpos",
            "chunk_size": self._chunk_size,
            "action_dim": self._action_dim,
        }


@POLICY_REGISTRY.register_client("replay")
class DatasetReplayPolicyClient(PolicyClient):
    """Replays the dataset's recorded action trajectory as successive chunks.

    Unlike RandomPolicyClient (smooth-but-meaningless sinusoids that make the
    arm jitter in place), this returns slices of the real episode actions so
    the 3D canvas plays the actual demonstrated motion. Each infer() advances a
    cursor by chunk_size; once the trajectory is exhausted the last pose is held.
    """

    # actions: (n_steps, action_dim) ground-truth trajectory. chunk_size: actions per infer.
    def __init__(self, actions: np.ndarray, chunk_size: int = 50) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)
        self._n_steps = self._actions.shape[0]
        self._action_dim = self._actions.shape[1]
        self._chunk_size = chunk_size
        self._cursor = 0

    @classmethod
    def from_config(cls, config: ConfigDict, ctx: PolicyBuildContext) -> DatasetReplayPolicyClient:
        """Build from the replay trajectory on ctx; chunk_size is an optional backend_option."""
        if ctx.action_trajectory is None:
            raise PolicyConnectionError("replay policy requires a dataset transport")
        opts = config.backend_options
        if "chunk_size" in opts:
            return cls(ctx.action_trajectory, chunk_size=opts["chunk_size"])
        return cls(ctx.action_trajectory)

    def infer(
        self,
        observation: dict,
        prev_action: np.ndarray | None = None,
        rtc_params: dict | None = None,
    ) -> dict:
        """Return the next ``chunk_size`` recorded actions, holding the last pose at the end.

        Args:
            observation: Ignored; present for interface compatibility.
            prev_action: Ignored.
            rtc_params: Ignored.

        Returns:
            ``{"actions": ndarray[chunk_size, action_dim] float32}`` — the
            trajectory slice starting at the internal cursor, tail-padded with
            the final recorded pose once the trajectory is exhausted.
        """
        start = min(self._cursor, max(self._n_steps - 1, 0))
        end = min(start + self._chunk_size, self._n_steps)
        chunk = self._actions[start:end]
        if chunk.shape[0] < self._chunk_size and self._n_steps > 0:
            # Hold the final recorded pose to pad a short tail chunk to chunk_size.
            pad = np.repeat(self._actions[-1:], self._chunk_size - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        self._cursor = end
        return {"actions": chunk.astype(np.float32)}

    def reset(self) -> None:
        """Rewind the replay cursor to the start of the trajectory."""
        self._cursor = 0

    @property
    def first_action(self) -> np.ndarray | None:
        """First recorded pose ``[action_dim] float32``, or None if the trajectory is empty."""
        # Recorded trajectory's first pose — used to smoothly transition the real
        # robot to the trajectory start before replay begins.
        return self._actions[0].copy() if self._n_steps > 0 else None

    @property
    def metadata(self) -> dict:
        """Capability descriptor for the dataset replay client."""
        return {
            "server_name": "eva-dataset-replay",
            "action_mode": "qpos",
            "chunk_size": self._chunk_size,
            "action_dim": self._action_dim,
            "n_steps": self._n_steps,
        }
