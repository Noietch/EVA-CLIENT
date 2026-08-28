from __future__ import annotations

import queue
import time
from types import SimpleNamespace

import pytest

from core.app import collection_capture
from core.app.collection_capture import CollectionCaptureRunner


def _wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)


def test_collection_capture_stop_drains_available_raw_snapshots():
    snapshots = [SimpleNamespace(timestamp=float(index)) for index in range(3)]
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
    _wait_until(lambda: len(ingested) >= 1)

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


def test_rollout_capture_buffers_raw_snapshots_for_action_pairing():
    snapshots = [SimpleNamespace(timestamp=float(index)) for index in range(2)]

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
    _wait_until(lambda: runtime.rollout_raw_snapshots.qsize() >= 2)
    runner.stop()

    assert runtime.rollout_raw_snapshots.qsize() == 2


def test_active_rollout_capture_is_not_routed_to_collection_logger():
    snapshot = SimpleNamespace(timestamp=1.0)
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
    _wait_until(lambda: not runtime.rollout_raw_snapshots.empty())
    runner.stop()

    assert runtime.rollout_raw_snapshots.qsize() == 1
    assert ingested == []


def test_capture_loop_waits_only_for_remaining_fixed_clock_interval(monkeypatch):
    runtime = SimpleNamespace(episode_logger=None, rollout_episode_logger=None)
    runner = CollectionCaptureRunner(runtime, fps=10.0, max_raw_snapshots_per_tick=1)
    now = [0.0]
    waits = []

    class _StopEvent:
        def is_set(self):
            return len(waits) >= 3

        def wait(self, duration):
            waits.append(duration)
            now[0] += duration

    def capture_tick():
        now[0] += 0.03
        return True

    runner._stop_event = _StopEvent()
    runner._capture_tick = capture_tick
    monkeypatch.setattr(collection_capture.time, "monotonic", lambda: now[0])

    runner._run()

    assert waits == pytest.approx([0.07, 0.07, 0.07])


def test_capture_lifecycle_suspends_gc_until_capture_stops(monkeypatch):
    events = []

    class _Runner:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            events.append("runner_start")

        def stop(self):
            events.append("runner_stop")

    class _Thread:
        def __init__(self, *, target, name, daemon):
            _ = name, daemon
            self._target = target

        def start(self):
            self._target()

        def join(self):
            pass

    runtime = SimpleNamespace(
        collection_capture_runner=None,
        episode_logger=SimpleNamespace(_log_dir="/data/collect"),
        rollout_episode_logger=None,
        transport=SimpleNamespace(
            prepare_collection_capture=lambda directory: events.append(
                ("capture_prepare", directory)
            ),
            finish_collection_capture=lambda: events.append("capture_finish"),
        ),
    )
    monkeypatch.setattr(collection_capture, "CollectionCaptureRunner", _Runner)
    monkeypatch.setattr(collection_capture.threading, "Thread", _Thread)
    monkeypatch.setattr(collection_capture.gc, "collect", lambda: events.append("collect") or 0)
    monkeypatch.setattr(collection_capture.gc, "disable", lambda: events.append("disable"))
    monkeypatch.setattr(collection_capture.gc, "enable", lambda: events.append("enable"))
    monkeypatch.setattr(collection_capture, "_gc_collect_thread", None)

    collection_capture.start_collection_capture(runtime, fps=20, max_raw_snapshots_per_tick=2)
    collection_capture.stop_collection_capture(runtime)

    assert events == [
        ("capture_prepare", "/data/collect"),
        "collect",
        "disable",
        "runner_start",
        "runner_stop",
        "capture_finish",
        "enable",
        "collect",
    ]


def test_capture_constructor_failure_restores_gc(monkeypatch):
    events = []
    runtime = SimpleNamespace(
        collection_capture_runner=None,
        episode_logger=SimpleNamespace(_log_dir="/data/collect"),
        rollout_episode_logger=None,
        transport=SimpleNamespace(prepare_collection_capture=lambda _directory: None),
    )
    monkeypatch.setattr(collection_capture, "_suspend_cyclic_gc", lambda: events.append("suspend"))
    monkeypatch.setattr(collection_capture, "_resume_cyclic_gc", lambda: events.append("resume"))
    monkeypatch.setattr(
        collection_capture,
        "CollectionCaptureRunner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid capture rate")),
    )

    with pytest.raises(ValueError, match="invalid capture rate"):
        collection_capture.start_collection_capture(
            runtime,
            fps=0,
            max_raw_snapshots_per_tick=1,
        )

    assert runtime.collection_capture_runner is None
    assert events == ["suspend", "resume"]


def test_capture_stop_restores_gc_when_transport_finish_fails(monkeypatch):
    events = []

    class _Runner:
        def stop(self):
            events.append("runner_stop")

    def finish_capture():
        events.append("capture_finish")
        raise OSError("journal release failed")

    runtime = SimpleNamespace(
        collection_capture_runner=_Runner(),
        transport=SimpleNamespace(finish_collection_capture=finish_capture),
    )
    monkeypatch.setattr(collection_capture, "_resume_cyclic_gc", lambda: events.append("gc_resume"))

    with pytest.raises(OSError, match="journal release failed"):
        collection_capture.stop_collection_capture(runtime)

    assert runtime.collection_capture_runner is None
    assert events == ["runner_stop", "capture_finish", "gc_resume"]
