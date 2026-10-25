from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.app.collection_capture import CollectionCaptureRunner

pytestmark = pytest.mark.unit


def _wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)


def test_collection_capture_stop_does_not_drain_forever_when_source_keeps_publishing():
    ingested = []
    calls = 0

    class _Transport:
        def acquire_collection_raw(self):
            nonlocal calls
            calls += 1
            if calls > 6:
                raise AssertionError("stop drain must be bounded")
            return SimpleNamespace(timestamp=float(calls))

    class _Logger:
        is_collection_enabled = True
        has_active_episode = True

        def ingest_collection_snapshot(self, snapshot):
            ingested.append(snapshot)

    runtime = SimpleNamespace(episode_logger=_Logger(), transport=_Transport())
    runner = CollectionCaptureRunner(runtime, fps=0.1, max_raw_snapshots_per_tick=3)
    runner.start()
    _wait_until(lambda: len(ingested) >= 1)

    runner.stop()

    assert len(ingested) <= 6


def test_policy_recording_uses_video_rate_and_latest_sampling_without_changing_collection(
    monkeypatch,
):
    import core.app.collection_capture as capture

    events = []

    class Runner:
        def __init__(self, runtime, fps, max_raw_snapshots_per_tick):
            events.append(("fps", fps))

        def start(self):
            pass

        def stop(self):
            pass

    transport = SimpleNamespace(
        prepare_collection_capture=lambda _: None,
        finish_collection_capture=lambda: None,
        set_collection_capture_latest_only=lambda enabled: events.append(("latest", enabled)),
    )
    rollout = SimpleNamespace(has_active_episode=True, is_collection_enabled=False, _fps=30)
    runtime = SimpleNamespace(
        episode_logger=None, rollout_episode_logger=rollout, transport=transport
    )
    monkeypatch.setattr(capture, "CollectionCaptureRunner", Runner)
    monkeypatch.setattr(capture, "_suspend_cyclic_gc", lambda: None)
    monkeypatch.setattr(capture, "_resume_cyclic_gc", lambda: None)
    capture.start_collection_capture(runtime, fps=10, max_raw_snapshots_per_tick=1)
    assert ("fps", 30) in events
    assert ("latest", True) in events
    capture.stop_collection_capture(runtime)
    events.clear()
    rollout.has_active_episode = False
    runtime.episode_logger = SimpleNamespace(
        has_active_episode=True, is_collection_enabled=True, _fps=30
    )
    capture.start_collection_capture(runtime, fps=10, max_raw_snapshots_per_tick=16)
    assert ("fps", 10) in events
    assert ("latest", False) in events
    capture.stop_collection_capture(runtime)
