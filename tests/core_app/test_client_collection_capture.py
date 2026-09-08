from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.app.handlers.recording import (
    collect_start,
    ingest_client_teleop_action,
    start_episode,
)
from core.app.handlers.teleop import PublishedTeleopAction
from core.app.state import SessionState
from core.types import RawCollectionSnapshot

pytestmark = pytest.mark.unit


class _Logger:
    is_collection_enabled = True

    def __init__(self, active: bool) -> None:
        self.has_active_episode = active
        self.paired: list[tuple[RawCollectionSnapshot, np.ndarray]] = []

    def ingest_collection_action_snapshot(
        self,
        snapshot: RawCollectionSnapshot,
        action_qpos: np.ndarray,
    ) -> None:
        self.paired.append((snapshot, np.asarray(action_qpos).copy()))


class _Transport:
    def __init__(self, snapshot: RawCollectionSnapshot | None) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
        self.calls += 1
        return self.snapshot


def _published(timestamp: float = 63_648.0) -> PublishedTeleopAction:
    return PublishedTeleopAction(
        np.asarray([1.0, 2.0], dtype=np.float32),
        timestamp=timestamp,
    )


def test_client_collection_uses_snapshot_source_clock() -> None:
    snapshot = RawCollectionSnapshot(timestamp=63_608.0, decode_raw=lambda: None)
    logger = _Logger(active=True)
    transport = _Transport(snapshot)
    runtime = SimpleNamespace(
        episode_logger=logger,
        transport=transport,
        last_collection_timestamp=None,
    )

    published = _published()
    assert ingest_client_teleop_action(None, runtime, published)

    assert transport.calls == 1
    assert runtime.last_collection_timestamp == snapshot.timestamp
    assert logger.paired[0][0] is snapshot
    np.testing.assert_allclose(logger.paired[0][1], published.qpos)


def test_client_collection_drops_action_without_source_timestamp() -> None:
    logger = _Logger(active=True)
    transport = _Transport(None)
    runtime = SimpleNamespace(
        episode_logger=logger,
        transport=transport,
        last_collection_timestamp=None,
    )

    assert not ingest_client_teleop_action(None, runtime, _published())
    assert transport.calls == 1
    assert logger.paired == []


def test_client_collection_does_not_read_outside_active_episode() -> None:
    snapshot = RawCollectionSnapshot(timestamp=1.0, decode_raw=lambda: None)
    logger = _Logger(active=False)
    transport = _Transport(snapshot)
    runtime = SimpleNamespace(
        episode_logger=logger,
        transport=transport,
        last_collection_timestamp=None,
    )

    assert not ingest_client_teleop_action(None, runtime, _published())
    assert transport.calls == 0


def test_eval_episode_clears_backlog_before_capture_starts() -> None:
    calls: list[object] = []

    class _EpisodeLogger:
        _log_dir = "/tmp/eval"

        def start_episode(self, *, task: str) -> None:
            calls.append(("start_episode", task))

    class _Transport:
        def clear_collection_backlog(self) -> None:
            calls.append("clear_collection_backlog")

        def prepare_collection_capture(self, directory: str | None) -> None:
            calls.append(("prepare_collection_capture", directory))

    runtime = SimpleNamespace(
        episode_logger=_EpisodeLogger(),
        transport=_Transport(),
    )

    start_episode(runtime, SessionState(selected_task="pick"))

    assert calls == [
        "clear_collection_backlog",
        ("prepare_collection_capture", "/tmp/eval"),
        ("start_episode", "pick"),
    ]


def test_collection_start_passes_sim_metadata_to_episode_logger() -> None:
    calls: list[object] = []

    class _EpisodeLogger:
        is_collection_enabled = False
        has_active_episode = False

        def is_queue_full(self) -> bool:
            return False

        def start_episode(self, *, task: str) -> None:
            calls.append(("start_episode", task))

        def set_episode_meta(self, **fields: object) -> None:
            calls.append(("set_episode_meta", fields))

    sim_metadata = {
        "schema_version": 4,
        "scene": {"scene_id": "scene-01"},
        "task_item": {"task_item_id": "task-01"},
    }
    session = SessionState(
        selected_collect_task="pick up the block",
        collection_sim_metadata=sim_metadata,
    )
    runtime = SimpleNamespace(episode_logger=_EpisodeLogger())

    assert collect_start(None, runtime, session)

    assert calls == [
        ("start_episode", "pick up the block"),
        ("set_episode_meta", {"sim_metadata": sim_metadata}),
    ]
    assert session.collection_sim_metadata is None
