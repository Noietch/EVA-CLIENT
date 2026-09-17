from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from core.app.handlers.recording import ingest_client_teleop_action
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
