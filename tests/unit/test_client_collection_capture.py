from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import core.app.handlers.recording as recording
from core.app.handlers.recording import (
    COLLECT_STEP_MAX_RAW_SNAPSHOTS,
    collect_start,
    ingest_client_teleop_action,
)
from core.app.handlers.teleop import PublishedTeleopAction
from core.app.state import SessionState
from core.config import ConfigDict
from core.types import RawCollectionSnapshot

pytestmark = pytest.mark.unit


class _Logger:
    is_collection_enabled = True

    def __init__(self, active: bool) -> None:
        self.has_active_episode = active
        self.paired: list[tuple[RawCollectionSnapshot, np.ndarray]] = []
        self.unpaired: list[RawCollectionSnapshot] = []

    def ingest_collection_client_snapshot(self, snapshot: RawCollectionSnapshot) -> None:
        self.unpaired.append(snapshot)

    def ingest_collection_action_snapshot(
        self,
        snapshot: RawCollectionSnapshot,
        action_qpos: np.ndarray,
    ) -> None:
        self.paired.append((snapshot, np.asarray(action_qpos).copy()))


class _Transport:
    def __init__(self, snapshot: RawCollectionSnapshot | None) -> None:
        self.snapshots = deque([] if snapshot is None else [snapshot])
        self.calls = 0

    def acquire_collection_raw(self) -> RawCollectionSnapshot | None:
        self.calls += 1
        return self.snapshots.popleft() if self.snapshots else None


def _published(timestamp: float = 63_648.0) -> PublishedTeleopAction:
    return PublishedTeleopAction(
        np.asarray([1.0, 2.0], dtype=np.float32),
        timestamp=timestamp,
    )


def test_client_collection_preserves_batch_and_pairs_only_latest_observation():
    transport = _Transport(None)
    snapshots = [RawCollectionSnapshot(timestamp=i / 60, decode_raw=lambda: None) for i in (1, 2)]
    transport.snapshots.extend(snapshots)
    logger = _Logger(active=True)
    runtime = SimpleNamespace(
        episode_logger=logger, transport=transport, last_collection_timestamp=None
    )

    config = ConfigDict(collection=ConfigDict(sim_recording_url=""))
    assert ingest_client_teleop_action(config, runtime, _published())
    assert logger.unpaired == snapshots[:1]
    assert [snapshot for snapshot, _ in logger.paired] == snapshots[1:]
    assert runtime.last_collection_timestamp == 2 / 60
    assert not transport.snapshots


def test_client_collection_bounds_work_per_control_tick():
    transport = _Transport(None)
    transport.snapshots.extend(
        RawCollectionSnapshot(timestamp=i, decode_raw=lambda: None)
        for i in range(COLLECT_STEP_MAX_RAW_SNAPSHOTS + 1)
    )
    logger = _Logger(active=True)
    runtime = SimpleNamespace(
        episode_logger=logger, transport=transport, last_collection_timestamp=None
    )

    config = ConfigDict(collection=ConfigDict(sim_recording_url=""))
    assert ingest_client_teleop_action(config, runtime, _published())
    assert transport.calls == COLLECT_STEP_MAX_RAW_SNAPSHOTS
    assert len(logger.unpaired) == COLLECT_STEP_MAX_RAW_SNAPSHOTS - 1
    assert len(transport.snapshots) == 1


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


def test_collection_start_records_metadata_from_automatic_slot_sync(monkeypatch):
    session = SessionState(selected_collect_task="pick up the block", selected_collect_set="set")
    sim_metadata = {"schema_version": 4, "scene": {"scene_id": "SC-1"}}

    def sync_slot():
        session.collection_slot_id = "TASK-1:SC-1:0"
        session.collection_scene_id = "SC-1"
        session.collection_scene_round = 0
        session.collection_task_id = "TASK-1"
        session.collection_sim_metadata = sim_metadata

    logger = SimpleNamespace(
        is_collection_enabled=True,
        is_queue_full=lambda: False,
        start_episode=Mock(),
        set_episode_meta=Mock(),
    )
    runtime = SimpleNamespace(
        episode_logger=logger,
        console_ctx=SimpleNamespace(sync_collection_slot=sync_slot),
        transport=SimpleNamespace(clear_collection_backlog=lambda: None),
    )
    config = ConfigDict(
        collection=ConfigDict(task_set_dir="task_sets", teleop=ConfigDict(control_source="client"))
    )
    monkeypatch.setattr(recording, "activate_teleop", lambda *_args: True)

    assert session.collection_slot_id is None
    assert collect_start(config, runtime, session)

    logger.set_episode_meta.assert_called_once_with(
        slot_id="TASK-1:SC-1:0",
        scene_id="SC-1",
        scene_round=0,
        task_id="TASK-1",
        sim_metadata=sim_metadata,
    )
    assert session.collection_sim_metadata is None
