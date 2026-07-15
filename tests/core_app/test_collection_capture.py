from __future__ import annotations

import time
from types import SimpleNamespace

from core.app.collection_capture import CollectionCaptureRunner


def test_collection_capture_stop_drains_available_raw_snapshots():
    snapshots = [object(), object(), object()]
    ingested = []

    class _Transport:
        def acquire_collection_raw(self):
            if snapshots:
                return snapshots.pop(0)
            return None

    class _Logger:
        is_collection_enabled = True

        def ingest_collection_snapshot(self, snapshot):
            ingested.append(snapshot)

    runtime = SimpleNamespace(episode_logger=_Logger(), transport=_Transport())
    runner = CollectionCaptureRunner(runtime, fps=0.1, max_raw_snapshots_per_tick=16)
    runner.start()
    deadline = time.monotonic() + 1.0
    while len(ingested) < 1 and time.monotonic() < deadline:
        time.sleep(0.005)

    runner.stop()

    assert len(ingested) == 3


def test_collection_capture_stop_does_not_drain_forever_when_source_keeps_publishing():
    ingested = []
    calls = 0

    class _Transport:
        def acquire_collection_raw(self):
            nonlocal calls
            calls += 1
            if calls > 6:
                raise AssertionError("stop drain must be bounded")
            return object()

    class _Logger:
        is_collection_enabled = True

        def ingest_collection_snapshot(self, snapshot):
            ingested.append(snapshot)

    runtime = SimpleNamespace(episode_logger=_Logger(), transport=_Transport())
    runner = CollectionCaptureRunner(runtime, fps=0.1, max_raw_snapshots_per_tick=3)
    runner.start()
    deadline = time.monotonic() + 1.0
    while len(ingested) < 1 and time.monotonic() < deadline:
        time.sleep(0.005)

    runner.stop()

    assert len(ingested) <= 6
