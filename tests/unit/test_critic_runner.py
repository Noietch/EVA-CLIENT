from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from critic_client.base import CriticClient
from critic_client.runner import CriticRunner

pytestmark = pytest.mark.unit


class _BlockingCritic(CriticClient):
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def evaluate(self, observation: dict, actions: np.ndarray) -> float:
        _ = observation
        if not self.actions:
            self.started.set()
            self.release.wait(timeout=2.0)
        self.actions.append(np.asarray(actions).copy())
        return float(np.asarray(actions).mean())

    @property
    def metadata(self) -> dict:
        return {"server_name": "recording"}


def test_critic_runner_coalesces_only_pending_requests_and_reports_drop_count():
    critic = _BlockingCritic()
    runner = CriticRunner(critic)
    try:
        runner.submit({"state": np.zeros(1)}, np.array([[1.0]]), 1.0, "policy")
        assert critic.started.wait(timeout=1.0)
        runner.submit({"state": np.zeros(1)}, np.array([[2.0]]), 2.0, "policy")
        runner.submit({"state": np.zeros(1)}, np.array([[3.0]]), 3.0, "intervention")
        critic.release.set()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            series = runner.series()
            if series["n"] >= 2:
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"critic samples did not reach 2: {runner.status()}")

        assert series["value"] == [1.0, 3.0]
        assert series["source"] == ["policy", "intervention"]
        assert runner.status()["coalesced_requests"] == 1
    finally:
        runner.close()
