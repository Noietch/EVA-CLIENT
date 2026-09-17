"""Automatic QC must flag a stalled camera without touching human verdicts."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.datasets.auto_qc import (
    ARM_TRAVEL,
    CAMERA_OFFLINE_REASON,
    CameraMotion,
    DatasetAutoQc,
    offline_cameras,
)
from tools.datasets.collection import STATIC_FRAMES_REASON, PlanCatalog

pytestmark = pytest.mark.integration

BATCH = "bench_batch"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
QPOS_NAMES = (
    [f"left.j{index}" for index in range(6)]
    + ["left.gripper"]
    + [f"right.j{index}" for index in range(6)]
    + ["right.gripper"]
)
FRAMES = 40


def _write_video(path: Path, frozen: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    still = np.full((240, 320, 3), 120, np.uint8)
    for index in range(FRAMES):
        frame = still.copy()
        if not frozen:
            frame[100:140, index * 6 : index * 6 + 40] = 255
        writer.write(frame)
    writer.release()


def _dataset(collection: Path, name: str, frozen: tuple[str, ...], frozen_episode: int = 1) -> Path:
    """Write one two-episode dataset; episode 0 is reviewed, episode 1 is not."""
    root = collection / name / "raw"
    (root / "meta").mkdir(parents=True, exist_ok=True)
    features = {
        "observations.state.qpos": {"dtype": "float32", "shape": [14], "names": QPOS_NAMES},
        "action.qpos": {"dtype": "float32", "shape": [14], "names": QPOS_NAMES},
        **{
            f"observation.images.{camera}": {"dtype": "video", "shape": [240, 320, 3]}
            for camera in CAMERAS
        },
    }
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "robot_type": "dual_yam",
                "total_episodes": 2,
                "fps": 30.0,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
                ),
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    (root / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {"episode_index": index, "slot_id": f"TASK-1:SC-1:{index}", "length": FRAMES}
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    (root / "meta/qc.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "qc_verdict": "pass",
                "qc_note": "",
                "qc_reason": "",
                "qc_updated_at": "2026-09-15T00:00:00+08:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for episode in range(2):
        qpos = np.zeros((FRAMES, 14), dtype=np.float32)
        # Only the left arm travels, so a stalled wrist camera must answer.
        qpos[:, 0] = np.linspace(0.0, 0.5, FRAMES)
        block = [row.tolist() for row in qpos]
        path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"observations.state.qpos": block, "action.qpos": block}), path)
        for camera in CAMERAS:
            _write_video(
                root / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4",
                frozen=episode == frozen_episode and camera in frozen,
            )
    return root


def _catalog(tmp_path: Path, names: tuple[str, ...]) -> PlanCatalog:
    plans = tmp_path / "task_sets"
    for name in names:
        root = plans / name
        root.mkdir(parents=True)
        (root / "info.yaml").write_text(
            f"dataset_name: {name}\nrobot_type: dual_yam\n"
            f"collection_dir: {name}/raw\ntarget_episodes: 2\n",
            encoding="utf-8",
        )
        (root / "tasks.csv").write_text(
            "﻿task_id,prompt_en,total_epsiodes_count,scene_ids,scene_epsiodes_count\n"
            "TASK-1,pick up cup,2,SC-1,2\n",
            encoding="utf-8",
        )
        (root / "scene.csv").write_text("﻿scene_id,placements\nSC-1,\n", encoding="utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "objects.csv").write_text("﻿object_id,object_name\nOBJ-1,cup\n", encoding="utf-8")
    return PlanCatalog(plans, assets, tmp_path / "collection")


def _qc_rows(root: Path) -> dict[int, dict]:
    return {
        int(row["episode_index"]): row
        for row in (json.loads(line) for line in (root / "meta/qc.jsonl").read_text().splitlines())
    }


def test_scan_records_a_stalled_wrist_camera_without_touching_reviews(tmp_path):
    dataset = _dataset(tmp_path / "collection", BATCH, frozen=("cam_left_wrist",))
    catalog = _catalog(tmp_path, (BATCH,))
    result = DatasetAutoQc(catalog, dataset, "dual_yam", threading.Event()).run(lambda *_: None)

    assert (result["checked"], result["flagged"], result["reviewed"]) == (2, 1, 0)
    assert result[CAMERA_OFFLINE_REASON] == 1
    assert result[STATIC_FRAMES_REASON] == 0
    rows = _qc_rows(dataset)
    assert rows[0]["qc_verdict"] == "pass"
    assert rows[1]["qc_verdict"] == "fail"
    assert rows[1]["qc_reason"] == CAMERA_OFFLINE_REASON
    assert rows[1]["qc_auto"] is True
    assert "cam_left_wrist" in rows[1]["qc_note"]

    state = catalog.state(BATCH)
    slots = {slot["slot_id"]: slot for task in state["tasks"] for slot in task["slots"]}
    assert slots["TASK-1:SC-1:0"]["qc_state"] == "passed"
    assert slots["TASK-1:SC-1:1"]["qc_state"] == "failed"
    assert slots["TASK-1:SC-1:1"]["episode"]["qc_auto"] is True


def test_scan_keeps_a_human_verdict_and_appends_the_machine_finding(tmp_path):
    dataset = _dataset(tmp_path / "collection", BATCH, frozen=("cam_left_wrist",), frozen_episode=0)
    catalog = _catalog(tmp_path, (BATCH,))
    result = DatasetAutoQc(catalog, dataset, "dual_yam", threading.Event()).run(lambda *_: None)

    assert (result["flagged"], result["reviewed"]) == (0, 1)
    row = _qc_rows(dataset)[0]
    assert row["qc_verdict"] == "pass"
    assert row.get("qc_auto") is not True
    assert row["qc_auto_verdict"] == "fail"
    assert row["qc_auto_reason"] == CAMERA_OFFLINE_REASON
    assert "cam_left_wrist" in row["qc_auto_note"]

    state = catalog.state(BATCH)
    slots = {slot["slot_id"]: slot for task in state["tasks"] for slot in task["slots"]}
    assert slots["TASK-1:SC-1:0"]["qc_state"] == "passed"
    assert slots["TASK-1:SC-1:0"]["episode"]["qc_auto_note"]


def test_offline_rule_needs_arm_travel_and_repeated_frames():
    stalled = CameraMotion(pairs=100, repeated=80, median_diff=0.0)
    moving = CameraMotion(pairs=100, repeated=0, median_diff=2.0)
    attached = {"cam_left_wrist": "left_arm"}

    assert offline_cameras(attached, {"cam_left_wrist": moving}, {"left_arm": 1.0}) == []
    assert offline_cameras(attached, {"cam_left_wrist": stalled}, {"left_arm": 0.0}) == []
    issues = offline_cameras(attached, {"cam_left_wrist": stalled}, {"left_arm": ARM_TRAVEL})
    assert [issue["camera"] for issue in issues] == ["cam_left_wrist"]
