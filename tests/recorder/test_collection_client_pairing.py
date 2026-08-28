from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.config import ConfigDict
from core.recorder import episode as episode_module
from core.recorder.episode import EpisodeLogger
from core.types import CollectionRawBatch, CollectionRawSample, RawCollectionSnapshot
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot


class _FakeFk:
    def __init__(
        self,
        nonfinite: float | None = None,
        nonfinite_on_call: int | None = None,
    ) -> None:
        self.nonfinite = nonfinite
        self.nonfinite_on_call = nonfinite_on_call
        self.calls = 0

    def fk_chunk(self, qpos: np.ndarray) -> np.ndarray:
        self.calls += 1
        qpos = np.asarray(qpos, dtype=np.float32)
        result = np.stack(
            [
                np.asarray(
                    [row[0], row[1], row[2], 1.0, 0.0, 0.0, 0.0, row[3]],
                    dtype=np.float32,
                )
                for row in qpos
            ]
        )
        if self.nonfinite is not None and (
            self.nonfinite_on_call is None or self.calls == self.nonfinite_on_call
        ):
            result[:, 0] = self.nonfinite
        return result


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
        self.fk_builds: list[dict[str, object]] = []
        self.fk_nonfinite: float | None = None
        self.fk_nonfinite_on_call: int | None = None

    def build_kinematics(self, **kwargs: object) -> _FakeFk:
        self.fk_builds.append(dict(kwargs))
        return _FakeFk(self.fk_nonfinite, self.fk_nonfinite_on_call)


def _collection_config(control_source: str = "client") -> ConfigDict:
    return ConfigDict(
        enabled=True,
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


def _logger(tmp_path, *, control_source: str = "client"):
    return EpisodeLogger(
        tmp_path,
        _FakeRobot(),
        fps=10,
        dataset_keys=ConfigDict(
            state_key="observations.state.qpos",
            eef_key="observations.state.eef",
            action_key="action",
            video_keys={},
        ),
        collection=_collection_config(control_source),
        async_save=False,
    )


def _batch(*, include_images: bool = True) -> CollectionRawBatch:
    state = np.asarray([1.0, 2.0, 3.0, 0.1], dtype=np.float32)
    remote_action = np.asarray([9.0, 9.0, 9.0, 0.9], dtype=np.float32)
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
                CollectionRawSample(1.0, state),
                CollectionRawSample(1.1, state),
            ],
            "action_qpos": [
                CollectionRawSample(1.0, remote_action),
                CollectionRawSample(1.1, remote_action),
            ],
            "state_eef": [CollectionRawSample(1.0, remote_eef)],
            "action_eef": [CollectionRawSample(1.0, remote_eef)],
            "action_qpos:arm": [CollectionRawSample(1.0, remote_action)],
            "action_eef:arm": [CollectionRawSample(1.0, remote_eef)],
        },
    )


def test_client_pairing_replaces_remote_action_stream(tmp_path) -> None:
    logger = _logger(tmp_path)
    logger.start_episode("task")
    client_action = np.asarray([4.0, 5.0, 6.0, 0.2], dtype=np.float32)
    snapshot = RawCollectionSnapshot(timestamp=1.0, decode_raw=_batch)

    logger.ingest_collection_action_snapshot(snapshot, client_action)

    paired = logger._collection_writer._raw_snapshots[0].decode_raw()
    assert set(paired.vectors) == {"state_qpos", "action_qpos"}
    np.testing.assert_allclose(paired.vectors["state_qpos"][0].value, [1, 2, 3, 0.1])
    np.testing.assert_allclose(paired.vectors["action_qpos"][0].value, client_action)
    assert paired.vectors["action_qpos"][0].timestamp == 1.0


def test_client_alignment_derives_both_eef_columns_from_aligned_qpos(tmp_path) -> None:
    logger = _logger(tmp_path)
    writer = logger._collection_writer
    writer._raw_batch = _batch()

    writer._align_raw_records()

    assert len(writer._records) == 2
    for frame in writer._records:
        np.testing.assert_allclose(frame.state_eef, [1, 2, 3, 1, 0, 0, 0, 0.1])
        np.testing.assert_allclose(frame.action_eef, [9, 9, 9, 1, 0, 0, 0, 0.9])
    assert len(logger._robot.fk_builds) == 1


@pytest.mark.parametrize(
    ("nonfinite", "call", "field"),
    [
        (np.nan, 1, "state_eef"),
        (np.inf, 1, "state_eef"),
        (np.nan, 2, "action_eef"),
        (np.inf, 2, "action_eef"),
    ],
)
def test_client_alignment_rejects_nonfinite_fk_output(
    tmp_path, nonfinite: float, call: int, field: str
) -> None:
    logger = _logger(tmp_path)
    logger._robot.fk_nonfinite = nonfinite
    logger._robot.fk_nonfinite_on_call = call
    writer = logger._collection_writer
    writer._raw_batch = _batch()

    with pytest.raises(ValueError, match=f"non-finite {field}"):
        writer._align_raw_records()


@pytest.mark.parametrize("control_source", ["transport", "client"])
@pytest.mark.parametrize("include_images", [False, True])
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
    client_action = np.asarray([4.0, 5.0, 6.0, 0.2], dtype=np.float32)
    if control_source == "client":
        logger.ingest_collection_action_snapshot(snapshot, client_action)
    else:
        logger.ingest_collection_snapshot(snapshot)

    assert logger.end_episode()
    logger.finalize()

    task_dir = tmp_path / "task" / "raw"
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


def test_collection_image_key_mismatch_is_reported_without_blocking_vector_save(tmp_path) -> None:
    logger = _logger(tmp_path, control_source="transport")
    writer = logger._collection_writer
    batch = _batch(include_images=False)
    batch.images = {
        "unexpected_camera": [
            CollectionRawSample(1.0, np.zeros((2, 2, 3), dtype=np.uint8)),
            CollectionRawSample(1.1, np.zeros((2, 2, 3), dtype=np.uint8)),
        ]
    }
    writer._raw_batch = batch

    writer._align_raw_records()

    assert len(writer._records) == 2
    assert {issue.code for issue in writer._quality_issues} >= {
        "missing_camera_stream",
        "unexpected_image_stream",
    }
