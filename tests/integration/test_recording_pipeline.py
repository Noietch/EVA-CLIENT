"""Record, replay and upload an actual dataset with encoded video."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.config import ConfigDict, load_config
from core.recorder.episode import EpisodeLogger
from core.types import CollectionRawBatch, CollectionRawSample, RawCollectionSnapshot
from core.utils import dataset_upload
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot
from transport.dataset import DatasetTransport

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("async_save", [False, True], ids=["sync", "async"])
def test_record_replay_export_upload_round_trip(tmp_path, async_save):
    robot = Robot(
        name="pipeline_arm",
        actuator_groups=(ActuatorGroup("arm", 2, ("j0", "j1")),),
        initial_qpos=np.zeros(2, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),), state_composition=("arm",)
        ),
    )
    keys = ConfigDict(
        state_key="observation.qpos",
        action_key="action",
        video_keys={"cam_high": "observation.images.cam_high"},
    )
    logger = EpisodeLogger(
        tmp_path / "recorded",
        robot,
        fps=10,
        dataset_keys=keys,
        collection=ConfigDict(
            enabled=True,
            storage=ConfigDict(data_root=str(tmp_path / "data")),
            schema=ConfigDict(
                robot_type=robot.name,
                min_episode_frames=1,
                arms={"arm": "arm"},
                cameras=dict(keys.video_keys),
                columns={"qpos": keys.state_key, "action_qpos": keys.action_key},
            ),
        ),
        async_save=async_save,
    )
    states = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.float32)
    actions = states + 10
    try:
        for episode in range(2):
            times = [1.0 + episode + i / 10 for i in range(3)]
            batch = CollectionRawBatch(
                images={
                    "cam_high": [
                        CollectionRawSample(t, np.full((16, 16, 3), 40 + i * 60, dtype=np.uint8))
                        for i, t in enumerate(times)
                    ]
                },
                vectors={
                    "state_qpos": [
                        CollectionRawSample(t, q) for t, q in zip(times, states, strict=True)
                    ],
                    "action_qpos": [
                        CollectionRawSample(t, q) for t, q in zip(times, actions, strict=True)
                    ],
                },
            )
            logger.start_episode("task")
            logger.ingest_collection_snapshot(
                RawCollectionSnapshot(timestamp=times[-1], decode_raw=lambda batch=batch: batch)
            )
            assert logger.end_episode()
        assert logger.wait_for_saves()
        assert logger.mark_collection_qc("task", 0, "pass", "accepted")
        assert logger.mark_collection_qc("task", 1, "fail", "retake")
    finally:
        logger.finalize()

    # The dataset is the one copy: QC verdicts live in its ledger, and it is
    # what gets delivered.
    dataset = tmp_path / "data" / "datasets" / "real_robot" / robot.name / "task"
    ledger = [
        json.loads(line)
        for line in (dataset / "meta" / "qc.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [(row["episode_index"], row["qc_verdict"]) for row in ledger] == [
        (0, "pass"),
        (1, "fail"),
    ]
    specs = dataset_upload.resolve_dataset_uploads(
        {"loopback": {"remote_dir": str(tmp_path / "uploaded")}}, "task"
    )
    plan = dataset_upload.scan_dataset_directory(dataset, specs)
    result = dataset_upload.upload_dataset_directory(dataset, specs, plan=plan)
    assert result.files == plan.new_files
    assert result.files > 0
    assert dataset_upload.scan_dataset_directory(dataset, specs).files_to_upload == 0

    # Replay both the recorder output and the delivered copy through the real reader
    config = load_config(Path(__file__).resolve().parents[2] / "configs/00_base/defaults.py")
    config.transport.dataset_keys = keys
    for source in (dataset, tmp_path / "uploaded/task"):
        replay = DatasetTransport(config, robot, source)
        try:
            assert replay.n_steps == 3
            assert replay.current_task == "task"
            np.testing.assert_allclose(replay.get_action_trajectory(), actions)
            for index in range(3):
                replay.seek(index)
                np.testing.assert_allclose(replay.get_obs_state(index), states[index])
                frame = replay.get_camera_frame("cam_high")
                assert frame.shape == (16, 16, 3)
                assert frame.mean() == pytest.approx(40 + index * 60, abs=5)
            assert not replay.advance()
        finally:
            replay.close()
    # The delivered copy carries the same recordings and the same ledger.
    published = tmp_path / "uploaded/task"
    rows = [
        json.loads(line)
        for line in (published / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all("qc_verdict" not in row for row in rows)
    delivered = [
        json.loads(line)
        for line in (published / "meta" / "qc.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [(row["episode_index"], row["qc_verdict"]) for row in delivered] == [
        (0, "pass"),
        (1, "fail"),
    ]
