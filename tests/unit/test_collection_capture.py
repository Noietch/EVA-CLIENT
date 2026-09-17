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
