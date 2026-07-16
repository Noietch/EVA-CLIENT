from __future__ import annotations

import queue
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
        has_active_episode = True

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
        has_active_episode = True

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


def test_rollout_capture_buffers_raw_snapshots_for_action_pairing():
    snapshots = [object(), object()]

    class _Transport:
        def acquire_collection_raw(self):
            return snapshots.pop(0) if snapshots else None

    class _RolloutLogger:
        has_active_episode = True

    runtime = SimpleNamespace(
        episode_logger=None,
        rollout_episode_logger=_RolloutLogger(),
        rollout_raw_snapshots=queue.Queue(),
        transport=_Transport(),
    )
    runner = CollectionCaptureRunner(runtime, fps=100.0, max_raw_snapshots_per_tick=2)
    runner.start()
    deadline = time.monotonic() + 1.0
    while runtime.rollout_raw_snapshots.qsize() < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    runner.stop()

    assert runtime.rollout_raw_snapshots.qsize() == 2


def test_active_rollout_capture_is_not_routed_to_collection_logger():
    snapshot = object()
    ingested = []

    class _Transport:
        def acquire_collection_raw(self):
            nonlocal snapshot
            value, snapshot = snapshot, None
            return value

    class _CollectionLogger:
        is_collection_enabled = True
        has_active_episode = True

        def ingest_collection_snapshot(self, value):
            ingested.append(value)

    class _RolloutLogger:
        has_active_episode = True

    runtime = SimpleNamespace(
        episode_logger=_CollectionLogger(),
        rollout_episode_logger=_RolloutLogger(),
        rollout_raw_snapshots=queue.Queue(),
        transport=_Transport(),
    )
    runner = CollectionCaptureRunner(runtime, fps=100.0, max_raw_snapshots_per_tick=1)
    runner.start()
    deadline = time.monotonic() + 1.0
    while runtime.rollout_raw_snapshots.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    runner.stop()

    assert runtime.rollout_raw_snapshots.qsize() == 1
    assert ingested == []
