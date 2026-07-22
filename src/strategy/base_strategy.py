"""Base inference strategy — defines BaseInferStrategy (sync) and the shared
BackgroundLoopInferStrategy base for fixed-Hz background-loop variants.
Strategy classes ARE the config: their dataclass fields become both YAML config
keys and runtime state. STRATEGY_REGISTRY dispatches by class name; the registered
"builder" is the class itself, called with kwargs from the YAML spec.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from core.registry import STRATEGY_REGISTRY
from strategy.action_buffer import StreamActionBuffer

logger = logging.getLogger(__name__)

FetchChunkFn = Callable[[str, dict | None], np.ndarray]
ObservationFn = Callable[[], dict | None]


@STRATEGY_REGISTRY.register("BaseInferStrategy")
@dataclasses.dataclass
class BaseInferStrategy:
    """Synchronous (blocking) inference strategy — the simplest execution mode.

    Implements step-by-step inference: take_or_fetch_chunk() calls the policy,
    crops the result to execute_horizon, and returns the full chunk for the
    caller to execute sequentially. No background threads, no smoothing.

    Also serves as the base class for the fixed-Hz background-loop variants,
    providing shared chunk-cropping logic, step tracking, and the
    start_loop/stop_loop/pop_next_action interface (all no-ops here).

    Fields are the YAML config keys: execute_horizon (chunk crop length, >=1),
    sync_wait_ignore_gripper (sync-mode gripper wait toggle),
    sync_wait_threshold (arrival L2-error threshold) and sync_wait_max_ticks
    (sync-mode arrival polling cap).

    tune_fields declares the online-tunable params the console renders as inputs;
    each entry is {"key", "label", "min": int|None, "step": int|float|None} and a
    null step renders a text input. Subclasses override to expose their own params.
    """

    tune_fields: ClassVar[list[dict]] = [
        {"key": "execute_horizon", "label": "exec steps", "min": 1, "step": 1},
    ]

    execute_horizon: int | None = None
    sync_wait_ignore_gripper: bool = False
    sync_wait_threshold: float = 0.05
    sync_wait_max_ticks: int = 100
    runs_background_loop: bool = dataclasses.field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.execute_horizon is not None:
            self.execute_horizon = max(1, int(self.execute_horizon))
        self.reset()

    def take_or_fetch_chunk(self, prompt: str, fetch_chunk: FetchChunkFn) -> np.ndarray:
        """Fetch a fresh action chunk from the policy and return it for sequential execution.

        Args:
            prompt: Task instruction passed to the policy.
            fetch_chunk: Callable invoking the policy; returns a raw action chunk.

        Returns:
            Action chunk cropped to execute_horizon, shape [T, action_dim] float32.
        """
        start_step = self._current_chunk_end_step()
        chunk = self._crop_chunk(fetch_chunk(prompt, None))
        self._activate_chunk(chunk, start_step)
        return chunk

    def prepare_warmup_chunk(self, prompt: str, fetch_chunk: FetchChunkFn) -> np.ndarray:
        """Fetch one warmup chunk before execution starts.

        Args:
            prompt: Task instruction passed to the policy.
            fetch_chunk: Callable invoking the policy; returns [T, action_dim].

        Returns:
            Prepared action chunk with shape [T, action_dim].
        """
        return self.take_or_fetch_chunk(prompt, fetch_chunk)

    def start_loop(
        self,
        prompt: str,
        fetch_chunk: FetchChunkFn,
        get_observation: ObservationFn,
    ) -> None:
        """Start continuous inference loop. No-op for sync strategies."""
        _ = prompt, fetch_chunk, get_observation

    def stop_loop(self) -> None:
        """Stop continuous inference loop. No-op for sync strategies."""

    def pop_next_action(self) -> np.ndarray | None:
        """Pop next action from the continuous buffer. Returns None if unavailable."""
        return None

    def is_loop_running(self) -> bool:
        """Check if the continuous inference loop is active."""
        return False

    def seed_buffer(self, action: np.ndarray) -> None:
        """Anchor the action buffer to a known position. No-op for sync."""
        _ = action

    def reset(self) -> None:
        """Reset chunk step tracking to the start state."""
        self._current_chunk_start_step = 0
        self._current_chunk_length = 0

    def close(self) -> None:
        """Stop any running loop and reset the strategy state."""
        self.stop_loop()
        self.reset()

    def _crop_chunk(self, action_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with 2 dims, got {chunk.shape}")
        if self.execute_horizon is not None and self.execute_horizon < len(chunk):
            chunk = chunk[: self.execute_horizon]
        return chunk.copy()

    def _current_chunk_end_step(self) -> int:
        return self._current_chunk_start_step + self._current_chunk_length

    def _activate_chunk(self, action_chunk: np.ndarray, start_step: int) -> None:
        self._current_chunk_start_step = start_step
        self._current_chunk_length = len(action_chunk)


@dataclasses.dataclass
class BackgroundLoopInferStrategy(BaseInferStrategy):  # pyright: ignore[reportGeneralTypeIssues]
    """Shared fixed-Hz background-loop inference strategy backed by an action buffer.

    A background thread runs inference at inference_rate Hz, integrating each
    cropped chunk into the buffer (subclasses choose the buffer kind via
    _make_buffer). The main loop pops individual actions via pop_next_action().
    Subclasses set _thread_name and _log_label.

    Adds two YAML config keys on top of BaseInferStrategy: inference_rate (Hz,
    clamped to >= 0) and latency_k (front-trim of new chunks, clamped to >= 0).
    """

    inference_rate: float = 3.0
    latency_k: int | None = None
    runs_background_loop: bool = dataclasses.field(default=True, init=False)
    _thread_name: str = dataclasses.field(default="eva-loop", init=False)
    _log_label: str = dataclasses.field(default="continuous", init=False)
    _generation: int = dataclasses.field(default=0, init=False)
    _stop_event: threading.Event = dataclasses.field(default_factory=threading.Event, init=False)
    _loop_thread: threading.Thread | None = dataclasses.field(default=None, init=False)
    _loop_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, init=False)
    _empty_buffer_started: float | None = dataclasses.field(default=None, init=False)
    _empty_buffer_polls: int = dataclasses.field(default=0, init=False)
    _missing_observation_started: float | None = dataclasses.field(default=None, init=False)
    _missing_observation_polls: int = dataclasses.field(default=0, init=False)

    def __post_init__(self) -> None:
        self.inference_rate = max(0.0, float(self.inference_rate))
        if self.latency_k is not None:
            self.latency_k = max(0, int(self.latency_k))
        self._smooth_buffer = self._make_buffer()
        super().__post_init__()

    def _make_buffer(self) -> StreamActionBuffer:
        """Create the action buffer backing this strategy. Subclasses override for variants."""
        return StreamActionBuffer()

    def start_loop(
        self,
        prompt: str,
        fetch_chunk: FetchChunkFn,
        get_observation: ObservationFn,
    ) -> None:
        """Start the background inference thread that prefetches chunks into the buffer.

        No-op if a loop thread is already alive.

        Args:
            prompt: Task instruction passed to the policy each iteration.
            fetch_chunk: Callable invoking the policy with (prompt, observation).
            get_observation: Callable returning the latest observation, or None to skip.
        """
        with self._loop_lock:
            if self._loop_thread is not None and self._loop_thread.is_alive():
                return
            self._stop_event.clear()
            self._loop_thread = threading.Thread(
                target=self._inference_loop,
                args=(self._generation, prompt, fetch_chunk, get_observation),
                name=self._thread_name,
                daemon=True,
            )
            self._loop_thread.start()
            logger.info(
                "Started %s inference loop at %.1f Hz",
                self._log_label,
                self.inference_rate,
            )

    def stop_loop(self) -> None:
        """Signal the background inference thread to stop and join it (5s timeout)."""
        with self._loop_lock:
            self._stop_event.set()
            thread = self._loop_thread
            self._loop_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def is_loop_running(self) -> bool:
        """Return True if the background inference thread is alive."""
        with self._loop_lock:
            return self._loop_thread is not None and self._loop_thread.is_alive()

    def pop_next_action(self) -> np.ndarray | None:
        """Pop the next action [action_dim] float32 from the action buffer, else None."""
        action = self._smooth_buffer.pop_next_action()
        now = time.monotonic()
        if action is None:
            self._empty_buffer_polls += 1
            if self._empty_buffer_started is None:
                self._empty_buffer_started = now
                logger.warning("[ACTION_BUFFER_GAP] state=start")
            return None
        if self._empty_buffer_started is not None:
            logger.warning(
                "[ACTION_BUFFER_GAP] state=end duration_ms=%.1f empty_polls=%d",
                (now - self._empty_buffer_started) * 1000.0,
                self._empty_buffer_polls,
            )
            self._empty_buffer_started = None
            self._empty_buffer_polls = 0
        return action

    def _inference_loop(
        self,
        generation: int,
        prompt: str,
        fetch_chunk: FetchChunkFn,
        get_observation: ObservationFn,
    ) -> None:
        interval = 1.0 / max(self.inference_rate, 0.1)
        while not self._stop_event.is_set():
            if generation != self._generation:
                return
            observation_started = time.monotonic()
            obs = get_observation()
            observation_done = time.monotonic()
            observation_ms = (observation_done - observation_started) * 1000.0
            if observation_ms >= 200.0:
                logger.warning(
                    "[INFERENCE_TIMING] stage=observation duration_ms=%.1f",
                    observation_ms,
                )
            if obs is None:
                self._missing_observation_polls += 1
                if self._missing_observation_started is None:
                    self._missing_observation_started = observation_done
                    logger.warning("[INFERENCE_OBSERVATION_GAP] state=start")
                self._stop_event.wait(0.02)
                continue
            if self._missing_observation_started is not None:
                logger.warning(
                    "[INFERENCE_OBSERVATION_GAP] state=end duration_ms=%.1f polls=%d",
                    (observation_done - self._missing_observation_started) * 1000.0,
                    self._missing_observation_polls,
                )
                self._missing_observation_started = None
                self._missing_observation_polls = 0
            loop_start = time.monotonic()
            try:
                raw_chunk = self._crop_chunk(fetch_chunk(prompt, obs))
                inference_done = time.monotonic()
                if generation != self._generation:
                    return
                self._smooth_buffer.integrate_new_chunk(
                    raw_chunk,
                    max_k=self.latency_k or 0,
                )
                integrate_done = time.monotonic()
                inference_ms = (inference_done - loop_start) * 1000.0
                integrate_ms = (integrate_done - inference_done) * 1000.0
                if inference_ms >= 250.0 or integrate_ms >= 20.0:
                    logger.warning(
                        "[INFERENCE_TIMING] stage=chunk inference_ms=%.1f "
                        "integrate_ms=%.1f chunk_steps=%d",
                        inference_ms,
                        integrate_ms,
                        len(raw_chunk),
                    )
                logger.debug(
                    "%s loop: integrated %d-step chunk",
                    self._log_label,
                    len(raw_chunk),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("%s inference loop error: %s", self._log_label, e)
            elapsed = time.monotonic() - loop_start
            remaining = max(0.0, interval - elapsed)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def take_or_fetch_chunk(self, prompt: str, fetch_chunk: FetchChunkFn) -> np.ndarray:
        """Synchronous fetch used for the first chunk during setup."""
        start_step = self._current_chunk_end_step()
        raw_chunk = self._crop_chunk(fetch_chunk(prompt, None))
        self._smooth_buffer.integrate_new_chunk(
            raw_chunk,
            max_k=self.latency_k or 0,
        )
        actions: list[np.ndarray] = []
        while self._smooth_buffer.has_actions():
            act = self._smooth_buffer.pop_next_action()
            if act is not None:
                actions.append(act)
        chunk = np.array(actions, dtype=np.float32) if actions else raw_chunk.copy()
        self._activate_chunk(chunk, start_step)
        return chunk

    def prepare_warmup_chunk(self, prompt: str, fetch_chunk: FetchChunkFn) -> np.ndarray:
        """Fetch and retain one warmup chunk in the continuous action buffer.

        Args:
            prompt: Task instruction passed to the policy.
            fetch_chunk: Callable invoking the policy; returns [T, action_dim].

        Returns:
            Retained action chunk with shape [T, action_dim].
        """
        chunk = self._crop_chunk(fetch_chunk(prompt, None))
        self._smooth_buffer.integrate_new_chunk(
            chunk,
            max_k=self.latency_k or 0,
        )
        return chunk

    def seed_buffer(self, action: np.ndarray) -> None:
        """Anchor the action buffer to a known action [action_dim] float32, clearing state."""
        self._smooth_buffer.seed(action)

    def reset(self) -> None:
        """Stop the loop, invalidate the running generation, and reset the buffer and base state."""
        self.stop_loop()
        self._generation += 1
        if hasattr(self, "_smooth_buffer"):
            self._smooth_buffer.reset()
        self._empty_buffer_started = None
        self._empty_buffer_polls = 0
        self._missing_observation_started = None
        self._missing_observation_polls = 0
        super().reset()
