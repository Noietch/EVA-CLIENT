"""Automatic dataset QC: middle static frames, stalled cameras, unreachable joints.

A camera whose stream drops keeps publishing its last frame, so the recording
repeats one image while the joints keep travelling. Live cameras never repeat a
frame exactly: sensor and H.264 noise leave a nonzero mean absolute difference
between neighbouring frames, which is what separates a stalled stream from a
merely static scene.

A recorded trajectory is also held against the joint bounds its robot declares in
its URDF. The vis table maps every driven dataset dimension onto a URDF joint, so a
commanded position past that joint's range is visible in the data: the arm clamps
there and the take can never be replayed as recorded.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import cv2
import numpy as np
import pyarrow.parquet as pq

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from core.utils.lerobot import LeRobotDatasetIO
from robots.utils import VisPart
from tools.datasets.collection import (
    CAMERA_OFFLINE_REASON,
    JOINT_LIMIT_REASON,
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
# A declared joint bound is hard, so a position meaningfully past it is a real
# excursion. Below this, recordings carry only definition-level noise: sim clamping and
# float32 land within 5e-6 rad, and a robot's own zero offset with respect to the URDF
# bound stays under 5e-3 rad across collected episodes, while the excursions this check
# is after start at 5e-2 rad.
LIMIT_TOLERANCE = 0.01
# Findings that reject an episode outright instead of queueing it for a reviewer.
DIRECT_FAIL_REASONS = frozenset({JOINT_LIMIT_REASON})


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


def urdf_joint_bounds(path: Path) -> dict[str, tuple[float, float] | None]:
    """One URDF's actuated joints in cfg order, each with the bounds it declares.

    A continuous joint carries no position bounds and maps to None; it still holds its
    place, because the cfg vector is positional.
    """
    bounds: dict[str, tuple[float, float] | None] = {}
    for joint in ElementTree.parse(path).getroot().findall("joint"):
        limit = joint.find("limit")
        if joint.get("type") == "fixed":
            continue
        declared = limit is not None and limit.get("lower") is not None
        bounds[joint.get("name")] = (
            (float(limit.get("lower")), float(limit.get("upper"))) if declared else None
        )
    return bounds


def dataset_arm_limits(parts: tuple[VisPart, ...]) -> dict[int, tuple[str, float, float]]:
    """Dataset dimension -> (part.joint, lower, upper) for every driven arm joint.

    The vis table is the dataset-qpos -> URDF mapping: its segments emit the URDF's
    actuated joints in order, so the dimensions of a ``copy`` segment land on the
    joints the cursor has reached. Gripper dimensions carry the normalized opener
    rather than an arm angle and are left out, as is a part that maps qpos through a
    plain callable and therefore exposes no table.
    """
    limits: dict[int, tuple[str, float, float]] = {}
    for part in parts:
        joints = list(urdf_joint_bounds(part.urdf_path).items())
        cursor = 0
        for segment in part.segments:
            if "copy" in segment:
                start, end = segment["copy"]
                for offset, (name, bounds) in enumerate(joints[cursor : cursor + end - start]):
                    if bounds is not None:
                        limits[part.qpos_offset + start + offset] = (f"{part.name}.{name}", *bounds)
                cursor += end - start
            elif "fixed" in segment:
                cursor += len(segment["fixed"])
            elif "gripper" in segment:
                cursor += len(segment["fingers"])
    return limits


def limit_breaches(
    values: np.ndarray, limits: dict[int, tuple[str, float, float]]
) -> list[dict[str, Any]]:
    """Every arm dimension of one series that leaves its joint's declared range."""
    dims = sorted(limits)
    low = np.asarray([limits[dim][1] for dim in dims])
    high = np.asarray([limits[dim][2] for dim in dims])
    block = np.asarray(values, dtype=np.float64)[:, dims]
    over = np.maximum(block - high, low - block)
    breaches = []
    for index, dim in enumerate(dims):
        frames = np.flatnonzero(over[:, index] > LIMIT_TOLERANCE)
        if not frames.size:
            continue
        column = block[frames, index]
        above = bool(column.max() - high[index] >= low[index] - column.min())
        breaches.append(
            {
                "joint": limits[dim][0],
                "value": float(column.max() if above else column.min()),
                "limit": float(high[index] if above else low[index]),
                "above": above,
                "frames": int(frames.size),
                "first": int(frames[0]),
                "last": int(frames[-1]),
            }
        )
    return breaches


def limit_note(breaches: list[dict[str, Any]], label: str) -> str:
    """One series' out-of-range joints as a review note."""
    details = "、".join(
        f"{breach['joint']} {'最大' if breach['above'] else '最小'} {breach['value']:.4f}"
        f"（限 {breach['limit']:.4f}，{breach['frames']} 帧"
        f" {breach['first']}-{breach['last']}）"
        for breach in breaches
    )
    return f"{label}超出关节极限：{details}"


class DatasetAutoQc:
    """Flag static middle frames, stalled cameras, and unreachable joints in one dataset.

    A scan only reports: an episode a human has not judged is marked for manual
    review with what the machine saw, and one a human already judged keeps that
    verdict with the machine opinion recorded beside it. Judging stays human — the
    exception is a trajectory its robot cannot execute, which is rejected outright.
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
        self.arm_limits = dataset_arm_limits(robot.vis_config.parts) if robot else {}
        inferred = self.io.infer_keys()
        self.state_key = inferred["state"]["default"]
        self.action_key = inferred["action"]["default"]
        info = json.loads((dataset_dir / "meta/info.json").read_text())
        self.fps = float(info.get("fps", 30) or 30)

    def run(self, progress: Callable[[int, int], None]) -> dict[str, Any]:
        """Scan every episode and record the failures it finds."""
        counts = {
            JOINT_LIMIT_REASON: 0,
            CAMERA_OFFLINE_REASON: 0,
            SHORT_EPISODE_REASON: 0,
            STATIC_FRAMES_REASON: 0,
        }
        checked = flagged = failed = reviewed = cleared = 0
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
                continue
            if reason in DIRECT_FAIL_REASONS:
                # A trajectory the robot cannot execute rejects the take; a reviewer
                # overturns the machine verdict by judging the episode.
                failed += 1
                self.io.mark_qc(episode_index, "fail", note, reason, auto=True)
                continue
            flagged += 1
            self.io.mark_qc(episode_index, "unreviewed", note, MANUAL_REVIEW_REASON, auto=True)
        over_limit = counts[JOINT_LIMIT_REASON]
        offline = counts[CAMERA_OFFLINE_REASON]
        static = counts[STATIC_FRAMES_REASON]
        short = counts[SHORT_EPISODE_REASON]
        summary = (
            f"检查 {checked} 条；判红 {failed} 条、标记 {flagged} 条待人工审核"
            f"（发现关节超限 {over_limit}、相机离线 {offline}、轨迹过短 {short}、"
            f"静止帧 {static}）"
        )
        if reviewed:
            summary += f"；人工已判定 {reviewed} 条只记录机器意见"
        if cleared:
            summary += f"；清除 {cleared} 条失效机器判定"
        return {
            "summary": summary,
            "checked": checked,
            "flagged": flagged,
            "failed": failed,
            "reviewed": reviewed,
            "cleared": cleared,
            **counts,
        }

    def _episode_issues(self, row: dict[str, Any], fence: float) -> list[tuple[str, str]]:
        """Automatic failures as (reason, note), most severe first."""
        episode_index = int(row["episode_index"])
        series = self._series(episode_index)
        issues = [(JOINT_LIMIT_REASON, note) for note in self._limit_notes(series)]
        issues += [
            (CAMERA_OFFLINE_REASON, note) for note in self._camera_notes(episode_index, series)
        ]
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

    def _limit_notes(self, series: dict[str, np.ndarray]) -> list[str]:
        """Trajectories that leave the range of the joints they command."""
        if not self.arm_limits:
            return []
        notes = []
        for key, label in ((self.action_key, "动作"), (self.state_key, "状态")):
            breaches = limit_breaches(series[key], self.arm_limits) if key in series else []
            if breaches:
                notes.append(limit_note(breaches, label))
        return notes

    def _camera_notes(self, episode_index: int, series: dict[str, np.ndarray]) -> list[str]:
        videos = self.catalog.camera_videos(self.dataset_dir, episode_index, self.attached_to)
        if not videos:
            return []
        qpos = next((series[key] for key in QPOS_KEYS if key in series), None)
        motions = {camera: camera_motion(path) for camera, path in videos.items()}
        travel = group_travel(qpos, self.groups) if qpos is not None else {}
        return [
            f"{issue['camera']} 相邻帧重复 {issue['motion'].repeated_fraction:.0%}"
            f"（{issue['arm'] or '任一臂'} 行程 {issue['travel']:.2f} rad）"
            for issue in offline_cameras(self.attached_to, motions, travel)
        ]

    def _series(self, episode_index: int) -> dict[str, np.ndarray]:
        """Joint positions and commands of one episode, read once for every check."""
        path = self.io.episode_parquet(episode_index)
        names = set(pq.read_schema(path).names)
        keys = [
            key
            for key in dict.fromkeys((self.state_key, self.action_key, *QPOS_KEYS))
            if key in names
        ]
        if not keys:
            return {}
        table = pq.read_table(path, columns=keys)
        return {key: np.asarray(table.column(key).to_pylist(), dtype=np.float64) for key in keys}
