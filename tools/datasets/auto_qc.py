"""Automatic dataset QC: middle static frames and cameras that stop updating.

A camera whose stream drops keeps publishing its last frame, so the recording
repeats one image while the joints keep travelling. Live cameras never repeat a
frame exactly: sensor and H.264 noise leave a nonzero mean absolute difference
between neighbouring frames, which is what separates a stalled stream from a
merely static scene.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from core.utils.lerobot import LeRobotDatasetIO
from tools.datasets.collection import (
    CAMERA_OFFLINE_REASON,
    MANUAL_REVIEW_REASON,
    SHORT_EPISODE_REASON,
    STATIC_FRAMES_REASON,
    PlanCatalog,
)

# A repeated frame decodes identically, so its mean absolute grey difference is
# zero; even a still scene on a live camera stays above 0.1.
STALL_DIFF = 0.02
STALL_FRACTION = 0.5
# Radians a joint must travel before its camera is required to show changes.
ARM_TRAVEL = 0.05
SAMPLE_STRIDE = 2
MIN_PAIRS = 8
# Durations differ per task, so one fixed floor fits no dataset; a broken take
# is one far below the rest of its own. Tukey's far-out fence (Q1 - 3 IQR) marks
# that extreme and leaves ordinary takes and uniformly short datasets alone.
FAR_OUT_IQR = 3.0
QPOS_KEYS = ("observations.state.qpos", "action.qpos")


@dataclass(frozen=True)
class CameraMotion:
    """Frame-pair statistics of one recorded camera stream."""

    pairs: int
    repeated: int
    median_diff: float

    @property
    def repeated_fraction(self) -> float:
        return self.repeated / self.pairs if self.pairs else 0.0


def episode_seconds(row: dict[str, Any], fps: float) -> float:
    """Recorded duration of one episode row."""
    return int(row.get("length", 0) or 0) / fps


def short_episode_fence(seconds: list[float]) -> float:
    """Duration floor of one dataset: the far-out bound of its own episodes."""
    q1, q3 = np.percentile(seconds, [25, 75])
    return float(q1 - FAR_OUT_IQR * (q3 - q1))


def camera_motion(path: Path) -> CameraMotion:
    """Measure how often a camera video repeats the previous sampled frame."""
    capture = cv2.VideoCapture(str(path))
    previous = None
    diffs: list[float] = []
    index = 0
    while capture.grab():
        if index % SAMPLE_STRIDE == 0:
            ok, frame = capture.retrieve()
            if not ok:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if previous is not None:
                diffs.append(float(np.abs(image - previous).mean()))
            previous = image
        index += 1
    capture.release()
    return CameraMotion(
        pairs=len(diffs),
        repeated=sum(diff <= STALL_DIFF for diff in diffs),
        median_diff=float(np.median(diffs)) if diffs else 0.0,
    )


def group_travel(qpos: np.ndarray, groups: tuple) -> dict[str, float]:
    """Largest joint travel in radians per actuator group, grippers aside."""
    travel = {}
    offset = 0
    for group in groups:
        block = qpos[:, offset : offset + group.dof]
        joints = [index for index in range(block.shape[1]) if index != group.gripper_index]
        travel[group.name] = float(np.ptp(block[:, joints], axis=0).max()) if joints else 0.0
        offset += group.dof
    return travel


def offline_cameras(
    attached_to: dict[str, str | None],
    motions: dict[str, CameraMotion],
    travel: dict[str, float],
) -> list[dict[str, Any]]:
    """Cameras that kept repeating frames while their arm was travelling.

    A wrist camera answers to its own arm. A fixed camera answers to any arm,
    which is the least it must show when the scene moves.
    """
    issues = []
    for camera, motion in motions.items():
        arm = attached_to[camera]
        moved = travel.get(arm, 0.0) if arm else max(travel.values(), default=0.0)
        if (
            motion.pairs >= MIN_PAIRS
            and moved >= ARM_TRAVEL
            and motion.repeated_fraction >= STALL_FRACTION
        ):
            issues.append({"camera": camera, "arm": arm, "travel": moved, "motion": motion})
    return issues


class DatasetAutoQc:
    """Flag static middle frames and stalled cameras in one collected dataset.

    A scan only reports: an episode a human has not judged is marked for manual
    review with what the machine saw, and one a human already judged keeps that
    verdict with the machine opinion recorded beside it. Judging stays human.
    """

    def __init__(
        self,
        catalog: PlanCatalog,
        dataset_dir: Path,
        robot_type: str,
        stop: threading.Event,
    ) -> None:
        self.catalog = catalog
        self.dataset_dir = dataset_dir
        self.stop = stop
        self.io = LeRobotDatasetIO(dataset_dir)
        robot = ROBOT_REGISTRY.build(robot_type) if robot_type in ROBOT_REGISTRY else None
        self.attached_to = (
            {
                camera.observation_key: camera.attached_to
                for camera in robot.observation_schema.cameras
            }
            if robot
            else {}
        )
        self.groups = robot.actuator_groups if robot else ()
        info = json.loads((dataset_dir / "meta/info.json").read_text())
        self.fps = float(info.get("fps", 30) or 30)

    def run(self, progress: Callable[[int, int], None]) -> dict[str, Any]:
        """Scan every episode and record the failures it finds."""
        counts = {CAMERA_OFFLINE_REASON: 0, SHORT_EPISODE_REASON: 0, STATIC_FRAMES_REASON: 0}
        checked = flagged = reviewed = cleared = 0
        rows = self.catalog.episode_rows(self.dataset_dir)
        durations = [episode_seconds(row, self.fps) for row in rows]
        fence = short_episode_fence(durations) if durations else 0.0
        for position, row in enumerate(rows, 1):
            if self.stop.is_set():
                break
            progress(position, len(rows))
            checked += 1
            episode_index = int(row["episode_index"])
            issues = self._episode_issues(row, fence)
            if not issues:
                cleared += self.io.drop_auto_qc(episode_index)
                continue
            reason = issues[0][0]
            counts[reason] += 1
            note = "；".join(note for _, note in issues)
            if str(row.get("qc_verdict") or "").strip() in {"pass", "fail"}:
                # The verdict stays the reviewer's; the finding is filed beside it.
                reviewed += 1
                self.io.mark_qc(episode_index, "fail", note, reason, auto=True)
            else:
                flagged += 1
                self.io.mark_qc(episode_index, "unreviewed", note, MANUAL_REVIEW_REASON, auto=True)
        offline = counts[CAMERA_OFFLINE_REASON]
        static = counts[STATIC_FRAMES_REASON]
        short = counts[SHORT_EPISODE_REASON]
        summary = (
            f"检查 {checked} 条；标记 {flagged} 条待人工审核"
            f"（发现相机离线 {offline}、轨迹过短 {short}、静止帧 {static}）"
        )
        if reviewed:
            summary += f"；人工已判定 {reviewed} 条只记录机器意见"
        if cleared:
            summary += f"；清除 {cleared} 条失效机器判定"
        return {
            "summary": summary,
            "checked": checked,
            "flagged": flagged,
            "reviewed": reviewed,
            "cleared": cleared,
            **counts,
        }

    def _episode_issues(self, row: dict[str, Any], fence: float) -> list[tuple[str, str]]:
        """Automatic failures as (reason, note), most severe first."""
        episode_index = int(row["episode_index"])
        issues = [(CAMERA_OFFLINE_REASON, note) for note in self._camera_notes(episode_index)]
        seconds = episode_seconds(row, self.fps)
        if seconds < fence:
            issues.append(
                (
                    SHORT_EPISODE_REASON,
                    f"时长 {seconds:.1f}s（低于本数据集极端离群下限 {fence:.1f}s）",
                )
            )
        analysis = self.catalog.frame_analysis(self.dataset_dir, episode_index)
        if analysis["static_frames_excessive"]:
            issues.append(
                (STATIC_FRAMES_REASON, f"中间静止帧 {analysis['middle_static_frames']} 帧")
            )
        return issues

    def _camera_notes(self, episode_index: int) -> list[str]:
        videos = self.catalog.camera_videos(self.dataset_dir, episode_index, self.attached_to)
        if not videos:
            return []
        qpos = self._qpos(episode_index)
        motions = {camera: camera_motion(path) for camera, path in videos.items()}
        travel = group_travel(qpos, self.groups) if qpos is not None else {}
        return [
            f"{issue['camera']} 相邻帧重复 {issue['motion'].repeated_fraction:.0%}"
            f"（{issue['arm'] or '任一臂'} 行程 {issue['travel']:.2f} rad）"
            for issue in offline_cameras(self.attached_to, motions, travel)
        ]

    def _qpos(self, episode_index: int) -> np.ndarray | None:
        """Joint positions of one episode; None when the dataset has no qpos column."""
        path = self.io.episode_parquet(episode_index)
        names = set(pq.read_schema(path).names)
        key = next((name for name in QPOS_KEYS if name in names), None)
        if key is None:
            return None
        values = pq.read_table(path, columns=[key]).column(key).to_pylist()
        return np.asarray(values, dtype=np.float64)
