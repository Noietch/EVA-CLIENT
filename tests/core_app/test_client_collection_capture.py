from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.app import handlers as recording_handlers
from core.app.handlers.recording import ingest_client_teleop_action
from core.app.handlers.teleop import PublishedTeleopAction
from core.app.state import SessionState
from core.config import ConfigDict


class _Logger:
    is_collection_enabled = True

    def __init__(self, active: bool) -> None:
        self.has_active_episode = active
        self.actions: list[tuple[float, np.ndarray]] = []
        self.snapshots: list[object] = []

    def ingest_collection_action(self, timestamp, action_qpos) -> None:
        self.actions.append((float(timestamp), np.asarray(action_qpos).copy()))

    def ingest_collection_client_snapshot(self, snapshot) -> None:
        self.snapshots.append(snapshot)


class _Transport:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def acquire_collection_raw(self):
        self.calls += 1
        return self.snapshot

    def supports_collection(self):
        return True

    def clear_collection_backlog(self):
        return 0.0


def _published() -> PublishedTeleopAction:
    return PublishedTeleopAction(np.asarray([1.0, 2.0], dtype=np.float32), timestamp=12.5)


def _runtime(logger: object, transport: object) -> SimpleNamespace:
    return SimpleNamespace(episode_logger=logger, transport=transport)


def test_client_collection_does_not_read_snapshot_outside_recording() -> None:
    logger = _Logger(active=False)
    transport = _Transport(object())
    runtime = _runtime(logger, transport)

    assert not ingest_client_teleop_action(None, runtime, _published())
    assert transport.calls == 0


def test_client_collection_reads_at_most_one_snapshot_and_pairs_published_qpos() -> None:
    snapshot = object()
    logger = _Logger(active=True)
    transport = _Transport(snapshot)
    runtime = _runtime(logger, transport)

    published = _published()
    assert ingest_client_teleop_action(None, runtime, published)
    assert transport.calls == 1
    assert logger.snapshots == [snapshot]
    assert logger.actions[0][0] == 12.5
    np.testing.assert_allclose(logger.actions[0][1], published.qpos)


def test_client_collection_keeps_action_when_snapshot_is_not_ready() -> None:
    logger = _Logger(active=True)
    transport = _Transport(None)
    runtime = _runtime(logger, transport)

    assert not ingest_client_teleop_action(None, runtime, _published())
    assert transport.calls == 1
    assert logger.snapshots == []
    assert logger.actions[0][0] == 12.5
    np.testing.assert_allclose(logger.actions[0][1], [1.0, 2.0])


def test_client_collection_start_does_not_start_background_capture(monkeypatch) -> None:
    calls: list[str] = []

    class _EpisodeLogger:
        is_collection_enabled = True
        has_active_episode = False

        def is_queue_full(self):
            return False

        def start_episode(self, **_kwargs):
            calls.append("episode")

    class _Transport:
        def clear_collection_backlog(self):
            calls.append("clear")
            return 1.0

    runtime = type("Runtime", (), {})()
    runtime.episode_logger = _EpisodeLogger()
    runtime.transport = _Transport()
    runtime.collection_replay_qpos = None
    runtime.collection_replay_episode = None
    runtime.collection_teleop_armed = True
    runtime.collection_teleop_active = True
    session = SessionState(selected_collect_task="task")
    config = ConfigDict(
        collection=ConfigDict(teleop=ConfigDict(control_source="client")),
    )

    monkeypatch.setattr(recording_handlers, "activate_teleop", lambda *_args: True)
    monkeypatch.setattr(
        recording_handlers,
        "start_collection_capture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("client collection must not start CollectionCaptureRunner")
        ),
    )

    assert recording_handlers.collect_start(config, runtime, session)
    assert calls == ["clear", "episode"]
