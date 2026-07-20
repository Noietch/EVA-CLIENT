"""Non-blocking critic request runner with explicit latest-pending coalescing."""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable

import numpy as np

from critic_client.base import CriticClient


@dataclasses.dataclass
class _CriticTask:
    generation: int
    timestamp: float
    source: str
    observation: dict | None = None
    actions: np.ndarray | None = None
    builder: Callable[[], tuple[dict, np.ndarray]] | None = None


class CriticRunner:
    """Evaluate critic requests off the control loop and retain their scalar series.

    There can be one active network request and one pending request. A newer sample
    replaces an older pending sample when the critic is slower than the producer;
    ``coalesced_requests`` exposes every replacement to the UI and tests.
    """

    def __init__(self, client: CriticClient) -> None:
        self._client = client
        self._condition = threading.Condition()
        self._pending: _CriticTask | None = None
        self._closed = False
        self._generation = 0
        self._samples: list[tuple[float, float, str]] = []
        self._last_request_ms = 0.0
        self._last_error = ""
        self._coalesced_requests = 0
        self._thread = threading.Thread(
            target=self._run,
            name="eva-critic-runner",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        observation: dict,
        actions: np.ndarray,
        timestamp: float,
        source: str,
    ) -> None:
        """Queue the latest observation/action request without blocking its producer."""
        with self._condition:
            task = _CriticTask(
                generation=self._generation,
                timestamp=float(timestamp),
                source=str(source),
                observation=observation,
                actions=np.asarray(actions, dtype=np.float32).copy(),
            )
            if self._pending is not None:
                self._coalesced_requests += 1
            self._pending = task
            self._condition.notify()

    def submit_builder(
        self,
        builder: Callable[[], tuple[dict, np.ndarray]],
        timestamp: float,
        source: str,
    ) -> None:
        """Queue deferred observation construction and critic evaluation."""
        with self._condition:
            task = _CriticTask(
                generation=self._generation,
                timestamp=float(timestamp),
                source=str(source),
                builder=builder,
            )
            if self._pending is not None:
                self._coalesced_requests += 1
            self._pending = task
            self._condition.notify()

    def reset_series(self) -> None:
        """Start a new rollout/replay series and invalidate outstanding results."""
        with self._condition:
            self._generation += 1
            self._pending = None
            self._samples = []
            self._last_request_ms = 0.0
            self._last_error = ""

    def series(self, since: int = 0) -> dict:
        """Return scalar samples at indices greater than or equal to ``since``."""
        with self._condition:
            start = max(0, int(since))
            samples = self._samples[start:]
            return {
                "n": len(self._samples),
                "timestamp": [sample[0] for sample in samples],
                "value": [sample[1] for sample in samples],
                "source": [sample[2] for sample in samples],
            }

    def status(self) -> dict:
        """Return connection, latency, error, and coalescing telemetry."""
        with self._condition:
            return {
                "connected": not self._closed,
                "metadata": dict(self._client.metadata),
                "last_request_ms": round(self._last_request_ms, 1),
                "last_error": self._last_error,
                "coalesced_requests": self._coalesced_requests,
                "samples": len(self._samples),
            }

    def close(self) -> None:
        """Stop the worker after its active request and release the client."""
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()
        self._client.close()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("Critic worker did not stop after its client was closed")

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                task = self._pending
                self._pending = None
            assert task is not None
            started = time.monotonic()
            try:
                if task.builder is not None:
                    observation, actions = task.builder()
                else:
                    assert task.observation is not None and task.actions is not None
                    observation, actions = task.observation, task.actions
                value = self._client.evaluate(observation, actions)
            except Exception as error:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                with self._condition:
                    self._last_request_ms = elapsed_ms
                    self._last_error = str(error.__cause__ or error)
                continue
            elapsed_ms = (time.monotonic() - started) * 1000.0
            with self._condition:
                self._last_request_ms = elapsed_ms
                self._last_error = ""
                if task.generation == self._generation:
                    self._samples.append((task.timestamp, float(value), task.source))
