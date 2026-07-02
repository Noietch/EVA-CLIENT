"""Console live-series API contract (/api/live_series).

The DEBUG/REPLAY/MANUAL stage charts plot the in-flight rollout's action + state
straight from the recorder's in-memory buffer, so an operator sees the numbers
scroll during RUN and can scrub back through them after STOP — without ever
saving. These tests pin that contract:

- ``/api/live_series`` is empty (active=false, n=0) when nothing is recording.
- When a rollout logger is present it returns ``{active, n, timestamp, state,
  action}`` and honors ``?since=`` for incremental polling.

The buffer-shape semantics themselves live in tests/recording/test_episode_logger.py
(test_live_series_streams_in_flight_buffer); here we only assert the HTTP wiring.
"""

from __future__ import annotations

from typing import cast

from _harness import WebHarness

from core.app import handlers
from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger


def test_live_series_empty_without_logger(console: WebHarness):
    body = console.get("/api/live_series").json
    assert body == {"active": False, "n": 0, "timestamp": [], "state": [], "action": []}


class _StubLogger:
    """Minimal stand-in exposing the live_series accessor the endpoint calls."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    @property
    def is_collection_enabled(self) -> bool:
        return False

    def live_series(self, since: int = 0) -> dict:
        self.calls.append(since)
        frames = [
            {"timestamp": 0.0, "state": [0.0, 1.0], "action": [10.0, 11.0]},
            {"timestamp": 1.0, "state": [2.0, 3.0], "action": [12.0, 13.0]},
            {"timestamp": 2.0, "state": [4.0, 5.0], "action": [14.0, 15.0]},
        ][since:]
        return {
            "active": True,
            "n": 3,
            "timestamp": [f["timestamp"] for f in frames],
            "state": [f["state"] for f in frames],
            "action": [f["action"] for f in frames],
        }


def test_live_series_reads_default_episode_logger(console: WebHarness):
    stub = _StubLogger()
    console.runtime.episode_logger = cast(EpisodeLogger, stub)

    body = console.get("/api/live_series").json
    assert body["active"] is True
    assert body["n"] == 3
    assert body["state"][0] == [0.0, 1.0]
    assert body["action"][-1] == [14.0, 15.0]

    tail = console.get("/api/live_series?since=2").json
    assert tail["n"] == 3  # total is unchanged; only the returned slice shrinks
    assert tail["timestamp"] == [2.0]
    assert tail["state"] == [[4.0, 5.0]]
    assert stub.calls == [0, 2]  # ?since= flows through to the accessor


def test_live_series_streams_debug_run_episode_logger(console: WebHarness, tmp_path):
    console.runtime.episode_logger = EpisodeLogger(
        tmp_path / "episodes",
        console.runtime.robot,
        fps=30,
        dataset_keys=ConfigDict(),
        async_save=False,
    )
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    console.do("/api/setup")
    console.do("/api/run")

    for _ in range(3):
        handlers.publish_next_action(console.config, console.runtime, console.session)

    body = console.get("/api/live_series").json
    assert body["active"] is True
    assert body["n"] == 3
    assert len(body["state"]) == 3
    assert len(body["action"]) == 3
    assert len(body["state"][0]) == console.runtime.robot.total_action_dim
    assert len(body["action"][0]) == console.runtime.robot.total_action_dim
