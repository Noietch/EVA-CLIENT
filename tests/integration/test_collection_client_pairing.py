from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.app.handlers.recording import ingest_client_teleop_action
from core.app.handlers.teleop import PublishedTeleopAction
from core.config import ConfigDict
from core.recorder import episode as episode_module
from core.recorder.episode import EpisodeLogger
from core.types import CollectionRawBatch, CollectionRawSample, RawCollectionSnapshot
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot

pytestmark = pytest.mark.integration

_STATE_QPOS = np.asarray([1.0, 2.0, 3.0, 0.1], dtype=np.float32)
_REMOTE_ACTION_QPOS = np.asarray([9.0, 9.0, 9.0, 0.9], dtype=np.float32)
_CLIENT_ACTION_QPOS = np.asarray([4.0, 5.0, 6.0, 0.2], dtype=np.float32)


class _FakeFk:
    def fk_chunk(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32)
        return np.stack(
            [
                np.asarray(
                    [row[0], row[1], row[2], 1.0, 0.0, 0.0, 0.0, row[3]],
                    dtype=np.float32,
                )
                for row in qpos
            ]
        )


class _FakeRobot(Robot):
    def __init__(self) -> None:
        super().__init__(
            name="fake",
            actuator_groups=(ActuatorGroup("arm", 4, ("x", "y", "z", "gripper"), gripper_index=3),),
            initial_qpos=np.zeros(4, dtype=np.float32),
            observation_schema=ObservationSchema(
                cameras=(CameraSpec("front", "cam_high"),),
                state_composition=("arm",),
            ),
        )

    def build_kinematics(self, **kwargs: object) -> _FakeFk:
        return _FakeFk()


def _collection_task_dir(tmp_path, task: str = "task"):
    """The dataset's one directory under the configured data root."""
    return tmp_path / "data" / "datasets" / "real_robot" / "fake" / task


def _collection_config(control_source: str = "client", data_root=None) -> ConfigDict:
    return ConfigDict(
        enabled=True,
        storage=ConfigDict(data_root=str(data_root) if data_root else ""),
        teleop=ConfigDict(control_source=control_source),
        schema=ConfigDict(
            robot_type="fake",
            min_episode_frames=1,
            arms={"arm": "arm"},
            cameras={"cam_high": "observation.images.cam_high"},
            columns={
                "qpos": "observation.qpos",
                "eef": "observation.eef",
                "action_qpos": "action.qpos",
                "action_eef": "action.eef",
            },
        ),
    )


def _logger(tmp_path, *, control_source: str = "client", fps: float = 10):
    return EpisodeLogger(
        tmp_path,
        _FakeRobot(),
        fps=fps,
        dataset_keys=ConfigDict(
            state_key="observations.state.qpos",
            eef_key="observations.state.eef",
            action_key="action",
            video_keys={},
        ),
        collection=_collection_config(control_source, data_root=tmp_path / "data"),
        async_save=False,
    )


def test_60hz_observations_with_30hz_actions_preserve_recording_duration(tmp_path):
    logger = _logger(tmp_path, fps=30)
    logger.start_episode("task")
    pending = deque()
    runtime = SimpleNamespace(
        episode_logger=logger,
        transport=SimpleNamespace(
            acquire_collection_raw=lambda: pending.popleft() if pending else None
        ),
        last_collection_timestamp=None,
    )

    def snapshot(index):
        timestamp = 1 + index / 60
        vector = np.asarray([timestamp, timestamp, timestamp, 0.2], dtype=np.float32)
        batch = CollectionRawBatch(
            images={
                "cam_high": [
                    CollectionRawSample(timestamp, np.full((2, 2, 3), index % 255, dtype=np.uint8))
                ]
            },
            vectors={"state_qpos": [CollectionRawSample(timestamp, vector)]},
        )
        return RawCollectionSnapshot(timestamp=timestamp, decode_raw=lambda: batch)

    for tick in range(300):
        # Two new observations arrive during each 30 Hz control tick.
        pending.extend((snapshot(2 * tick), snapshot(2 * tick + 1)))
        latest_time = 1 + (2 * tick + 1) / 60
        published = PublishedTeleopAction(
            np.asarray([latest_time, latest_time, latest_time, 0.2], dtype=np.float32),
            timestamp=10000 + tick / 30,
        )
        assert ingest_client_teleop_action(None, runtime, published)
        assert not pending

    assert logger.end_episode()
    table = pq.read_table(_collection_task_dir(tmp_path) / "data/chunk-000/episode_000000.parquet")
    times = np.asarray(table["timestamp"].to_pylist())
    assert len(times) == 300
    assert times[-1] == pytest.approx(299 / 30)
    assert np.ptp(table["capture_time"].to_numpy()) == pytest.approx(299 / 30)
    np.testing.assert_allclose(
        table["observation.qpos"].to_pylist(),
        table["action.qpos"].to_pylist(),
        atol=1e-5,
    )


def _batch(*, include_images: bool = True) -> CollectionRawBatch:
    remote_eef = np.full(8, 99.0, dtype=np.float32)
    return CollectionRawBatch(
        images=(
            {
                "cam_high": [
                    CollectionRawSample(1.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                    CollectionRawSample(1.1, np.zeros((2, 2, 3), dtype=np.uint8)),
                ]
            }
            if include_images
            else {}
        ),
        vectors={
            "state_qpos": [
                CollectionRawSample(1.0, _STATE_QPOS),
                CollectionRawSample(1.1, _STATE_QPOS),
            ],
            "action_qpos": [
                CollectionRawSample(1.0, _REMOTE_ACTION_QPOS),
                CollectionRawSample(1.1, _REMOTE_ACTION_QPOS),
            ],
            "state_eef": [CollectionRawSample(1.0, remote_eef)],
            "action_eef": [CollectionRawSample(1.0, remote_eef)],
            "action_qpos:arm": [CollectionRawSample(1.0, _REMOTE_ACTION_QPOS)],
            "action_eef:arm": [CollectionRawSample(1.0, remote_eef)],
        },
    )


def test_client_pairing_replaces_remote_action_stream(tmp_path) -> None:
    logger = _logger(tmp_path)
    logger.start_episode("task")
    snapshot = RawCollectionSnapshot(timestamp=1.0, decode_raw=_batch)

    logger.ingest_collection_action_snapshot(snapshot, _CLIENT_ACTION_QPOS)

    paired = logger._collection_writer._raw_snapshots[0].decode_raw()
    assert set(paired.vectors) == {"state_qpos", "action_qpos"}
    np.testing.assert_allclose(paired.vectors["state_qpos"][0].value, [1, 2, 3, 0.1])
    np.testing.assert_allclose(paired.vectors["action_qpos"][0].value, _CLIENT_ACTION_QPOS)
    assert paired.vectors["action_qpos"][0].timestamp == 1.0


def test_high_rate_client_actions_are_interpolated_to_collection_fps(tmp_path) -> None:
    logger = _logger(tmp_path)
    logger.start_episode("task")

    def raw_observations() -> CollectionRawBatch:
        return CollectionRawBatch(
            images={
                "cam_high": [
                    CollectionRawSample(t, np.zeros((2, 2, 3), dtype=np.uint8))
                    for t in (1.0, 1.1, 1.2)
                ]
            },
            vectors={"state_qpos": [CollectionRawSample(t, _STATE_QPOS) for t in (1.0, 1.1, 1.2)]},
            start_time=1.0,
            end_time=1.2,
        )

    logger.ingest_collection_client_snapshot(
        RawCollectionSnapshot(timestamp=1.0, decode_raw=raw_observations)
    )
    for timestamp, value in ((1.0, 0.0), (1.07, 7.0), (1.14, 14.0), (1.2, 20.0)):
        logger.ingest_collection_action(
            timestamp,
            np.asarray([value, value, value, 0.2], dtype=np.float32),
        )
    assert logger._collection_writer.frame_counts()["received"] == 1

    assert logger.end_episode()
    logger.finalize()

    table = pq.read_table(
        _collection_task_dir(tmp_path) / "data" / "chunk-000" / "episode_000000.parquet"
    )
    actions = np.asarray(table.column("action.qpos").to_pylist(), dtype=np.float32)
    np.testing.assert_allclose(actions[:, 0], [0.0, 10.0, 20.0], atol=1e-5)
    np.testing.assert_allclose(actions[:, 3], [0.2, 0.2, 0.2], atol=1e-5)


@pytest.mark.parametrize(
    ("control_source", "include_images"),
    [("transport", False), ("client", True)],
)
def test_collection_image_mode_writes_vectors_and_actual_video_metadata(
    tmp_path,
    monkeypatch,
    control_source: str,
    include_images: bool,
) -> None:
    """Both action sources share qpos/FK output, while video follows raw images."""

    class _Writer:
        def append_data(self, frame: np.ndarray) -> None:
            assert frame.shape == (2, 2, 3)

        def close(self) -> None:
            pass

    monkeypatch.setattr(episode_module.imageio, "get_writer", lambda *args, **kwargs: _Writer())
    logger = _logger(tmp_path, control_source=control_source)
    logger.start_episode("task")
    snapshot = RawCollectionSnapshot(
        timestamp=1.0,
        decode_raw=lambda: _batch(include_images=include_images),
    )
    if control_source == "client":
        logger.ingest_collection_action_snapshot(snapshot, _CLIENT_ACTION_QPOS)
    else:
        logger.ingest_collection_snapshot(snapshot)

    assert logger.end_episode()
    logger.finalize()

    task_dir = _collection_task_dir(tmp_path)
    table = pq.read_table(task_dir / "data" / "chunk-000" / "episode_000000.parquet")
    assert {
        "observation.qpos",
        "observation.eef",
        "action.qpos",
        "action.eef",
    }.issubset(table.column_names)
    np.testing.assert_allclose(
        table.column("observation.eef").to_pylist()[0],
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.1],
    )
    expected_action_eef = (
        [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.2]
        if control_source == "client"
        else [9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.9]
    )
    np.testing.assert_allclose(table.column("action.eef").to_pylist()[0], expected_action_eef)

    episode = json.loads((task_dir / "meta" / "episodes.jsonl").read_text().splitlines()[0])
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    if include_images:
        assert episode["total_videos"] == 1
        assert episode["video_keys"] == ["observation.images.cam_high"]
        assert info["total_videos"] == 1
        assert "observation.images.cam_high" in info["features"]
    else:
        assert episode["total_videos"] == 0
        assert episode["video_keys"] == []
        assert info["total_videos"] == 0
        assert "observation.images.cam_high" not in info["features"]
