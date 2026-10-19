from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

import core.recorder.sim_collection as sim_collection
from core.cfg import ConfigDict
from core.recorder.episode import EpisodeLogger
from robots.base import ActuatorGroup, ObservationSchema, Robot

pytestmark = pytest.mark.integration


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


def _robot() -> Robot:
    return Robot(
        name="fake_arm",
        actuator_groups=(ActuatorGroup("arm", 2, ("j0", "j1")),),
        initial_qpos=np.zeros(2, dtype=np.float32),
        observation_schema=ObservationSchema(cameras=(), state_composition=("arm",)),
    )


def _state(qpos: list[float]) -> dict[str, object]:
    return {"qpos": qpos, "scene_state": {}, "objects": {}, "liquids": {}}


def test_episode_logger_imports_sim_steps_without_sidecars(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    recording_path: str | None = None
    recording_episode: str | None = None
    control_dt = 0.1

    def fake_urlopen(request, timeout: float):
        nonlocal recording_episode, recording_path
        del timeout
        calls.append(request.get_method())
        if request.get_method() == "GET":
            return _Response(
                {"ok": True, "data": {"scene_revision": 3, "metadata": {"layout": "test"}}}
            )
        body = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/recording/start"):
            path = Path(body["path"])
            recording_episode = body["episode_id"]
            recording_path = body["path"]
            records = [
                {
                    "type": "header",
                    "schema_version": 1,
                    "episode_id": body["episode_id"],
                    "sim_metadata": {"layout": "test"},
                    "control_dt": control_dt,
                    "physics_dt": 0.01,
                    "initial_state": _state([1.0, 2.0]),
                },
                {
                    "type": "step",
                    "step_id": 0,
                    "sim_time": control_dt,
                    "source_frame_id": 10,
                    "action": [5.0, 6.0],
                    "state": _state([3.0, 4.0]),
                },
                {
                    "type": "step",
                    "step_id": 1,
                    "sim_time": 2 * control_dt,
                    "source_frame_id": 11,
                    "action": [7.0, 8.0],
                    "state": _state([9.0, 10.0]),
                },
                {"type": "footer", "steps": 2, "complete": True},
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            return _Response(
                {
                    "ok": True,
                    "data": {
                        "episode_id": recording_episode,
                        "path": recording_path,
                        "status": "recording",
                        "steps": 0,
                    },
                }
            )
        return _Response(
            {
                "ok": True,
                "data": {
                    "episode_id": recording_episode,
                    "path": recording_path,
                    "status": "complete",
                    "steps": 2,
                },
            }
        )

    monkeypatch.setattr(sim_collection._SIM_OPENER, "open", fake_urlopen)
    collection = ConfigDict(
        sim_recording_url="http://sim",
        storage=ConfigDict(
            data_root=str(tmp_path), log_dir=str(tmp_path), fps=30, save_queue_max=1
        ),
        schema=ConfigDict(
            robot_type="fake_arm",
            min_episode_frames=1,
            max_frame_dt_factor=3.0,
            arms={},
            cameras={},
            columns={
                "qpos": "observations.state.qpos",
                "action_qpos": "action.qpos",
            },
        ),
        tasks={"dataset": [("task", 1)]},
        task_set_dir="",
    )
    logger = EpisodeLogger(
        tmp_path / "runtime",
        _robot(),
        fps=30,
        dataset_keys=ConfigDict(video_keys={}),
        collection=collection,
        async_save=False,
    )

    logger.start_episode("task", collection_dataset="dataset")
    logger.set_episode_meta(scene_id="test")
    logger.start_sim_recording({"scene_id": "test"})
    logger.stop_sim_recording()
    assert logger.end_episode()

    job = logger._last_save_job
    assert job is not None and job.collection_episode_row is not None
    dataset_dir = job.dataset_dir
    assert dataset_dir is not None
    row = job.collection_episode_row
    assert row["sim_execution_schema_version"] == 1
    assert row["sim_metadata"] == {"layout": "test"}
    raw_path = dataset_dir / row["sim_execution_path"]
    assert raw_path.is_file()
    assert not (dataset_dir / "meta" / "sim_state").exists()

    table = pq.read_table(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.column("observations.state.qpos").to_pylist()[0] == [1.0, 2.0]
    assert table.column("action.qpos").to_pylist()[0] == [5.0, 6.0]
    assert table.column("capture_time").to_pylist() == [0.0, 0.1]
    assert table.column("timestamp").to_pylist() == [0.0, 0.1]
    assert json.loads((dataset_dir / "meta/info.json").read_text())["fps"] == 10.0
    assert calls == ["GET", "GET", "POST", "POST"]

    metadata_before = {
        name: (dataset_dir / "meta" / name).read_bytes()
        for name in ("info.json", "episodes.jsonl")
    }
    control_dt = 0.05
    logger.start_episode("task", collection_dataset="dataset")
    logger.start_sim_recording()
    logger.stop_sim_recording()
    with pytest.raises(ValueError, match="same control FPS"):
        logger.end_episode()
    for name, contents in metadata_before.items():
        assert (dataset_dir / "meta" / name).read_bytes() == contents
    assert not (dataset_dir / "data/chunk-000/episode_000001.parquet").exists()
