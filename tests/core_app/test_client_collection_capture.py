from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from core.app.handlers.recording import COLLECT_STEP_MAX_RAW_SNAPSHOTS, ingest_client_teleop_action
from core.app.handlers.teleop import PublishedTeleopAction
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

    assert transport.calls == 2
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


def test_client_collection_preserves_batch_and_pairs_only_latest_observation():
    transport = _Transport(None)
    snapshots = [RawCollectionSnapshot(timestamp=i / 60, decode_raw=lambda: None) for i in (1, 2)]
    transport.snapshots.extend(snapshots)
    logger = _Logger(active=True)
    runtime = SimpleNamespace(
        episode_logger=logger, transport=transport, last_collection_timestamp=None
    )

    assert ingest_client_teleop_action(None, runtime, _published())
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

    assert ingest_client_teleop_action(None, runtime, _published())
    assert transport.calls == COLLECT_STEP_MAX_RAW_SNAPSHOTS
    assert len(logger.unpaired) == COLLECT_STEP_MAX_RAW_SNAPSHOTS - 1
    assert len(transport.snapshots) == 1
