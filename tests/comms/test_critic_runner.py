from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from critic_client.base import CriticClient
from critic_client.runner import CriticRunner


class _RecordingCritic(CriticClient):
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def evaluate(self, observation: dict, actions: np.ndarray) -> float:
        _ = observation
        self.actions.append(np.asarray(actions).copy())
        return float(np.asarray(actions).mean())

    @property
    def metadata(self) -> dict:
        return {"server_name": "recording"}


class _BlockingCritic(_RecordingCritic):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def evaluate(self, observation: dict, actions: np.ndarray) -> float:
        if not self.actions:
            self.started.set()
            self.release.wait(timeout=2.0)
        return super().evaluate(observation, actions)


def _wait_for_samples(runner: CriticRunner, expected: int) -> dict:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        series = runner.series()
        if series["n"] >= expected:
            return series
        time.sleep(0.01)
    raise TimeoutError(f"critic samples did not reach {expected}: {runner.status()}")


def test_critic_runner_evaluates_off_thread_and_exposes_incremental_series():
    runner = CriticRunner(_RecordingCritic())
    try:
        runner.submit(
            {"state": np.array([0.0], dtype=np.float32)},
            np.array([[0.4]], dtype=np.float32),
            timestamp=1.25,
            source="policy",
        )

        series = _wait_for_samples(runner, 1)

        assert series == {
            "n": 1,
            "timestamp": [1.25],
            "value": [pytest.approx(0.4)],
            "source": ["policy"],
        }
        assert runner.series(since=1)["value"] == []
        assert runner.status()["last_request_ms"] >= 0.0
    finally:
        runner.close()


def test_critic_runner_coalesces_only_pending_requests_and_reports_drop_count():
    critic = _BlockingCritic()
    runner = CriticRunner(critic)
    try:
        runner.submit({"state": np.zeros(1)}, np.array([[1.0]]), 1.0, "policy")
        assert critic.started.wait(timeout=1.0)
        runner.submit({"state": np.zeros(1)}, np.array([[2.0]]), 2.0, "policy")
        runner.submit({"state": np.zeros(1)}, np.array([[3.0]]), 3.0, "intervention")
        critic.release.set()

        series = _wait_for_samples(runner, 2)

        assert series["value"] == [1.0, 3.0]
        assert series["source"] == ["policy", "intervention"]
        assert runner.status()["coalesced_requests"] == 1
    finally:
        runner.close()
