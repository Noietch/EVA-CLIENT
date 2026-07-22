#!/usr/bin/env python3
"""Verify async saving throughput with complete synthetic robot-scale data.

Run:
    PYTHONPATH=src python tests/manual/verify_async_save_perf.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger, sanitize_path_component
from core.types import (
    CollectionRawBatch,
    CollectionRawImage,
    CollectionRawSample,
    RawCollectionSnapshot,
)
from examples.hardware.r1_lite import fake_node
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot

FPS = 15
DURATION_S = 120
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
MAX_SAVE_S = 30.0
TASK = "async save performance"


class R1LiteFakeDataset:
    """Complete 2min/15Hz fake-node data source for recorder save verification.

    Args:
        fps: Fixed capture rate in Hz.
        duration_s: Episode duration in seconds.
        image_height: Camera image height.
        image_width: Camera image width.

    Attributes:
        frames: Number of samples, equal to fps * duration_s.
    """

    def __init__(
        self,
        fps: int,
        duration_s: int,
        image_height: int,
        image_width: int,
    ) -> None:
        self.fps = fps
        self.duration_s = duration_s
        self.image_height = image_height
        self.image_width = image_width
        self.frames = fps * duration_s
        self.camera_keys = tuple(fake_node.CAMERA_OBSERVATION_KEYS.values())
        self._camera_cache = self._build_camera_cache()
        self._state_qpos, self._action_qpos, self._state_eef, self._action_eef = (
            self._build_vectors()
        )

    def snapshot(self, frame_index: int, *, include_action: bool) -> RawCollectionSnapshot:
        """Return one raw snapshot matching R1 Lite fake-node cameras and vectors.

        Args:
            frame_index: Source frame index in [0, frames).
            include_action: Include action_qpos/action_eef streams for collection.

        Returns:
            RawCollectionSnapshot whose decode_raw closure exposes timestamped samples.
        """
        timestamp = self.timestamp(frame_index)

        def decode_raw() -> CollectionRawBatch:
            images = {
                camera_key: [
                    CollectionRawSample(
                        timestamp,
                        CollectionRawImage(
                            lambda key=camera_key, index=frame_index: self._camera_frame(
                                key, index
                            )
                        ),
                    )
                ]
                for camera_key in self.camera_keys
            }
            vectors = {
                "state_qpos:left_arm": [
                    CollectionRawSample(timestamp, self._state_qpos[frame_index, :7])
                ],
                "state_qpos:right_arm": [
                    CollectionRawSample(timestamp, self._state_qpos[frame_index, 7:])
                ],
                "state_eef:left_arm": [
                    CollectionRawSample(timestamp, self._state_eef[frame_index, :8])
                ],
                "state_eef:right_arm": [
                    CollectionRawSample(timestamp, self._state_eef[frame_index, 8:])
                ],
            }
            if include_action:
                vectors.update(
                    {
                        "action_qpos:left_arm": [
                            CollectionRawSample(timestamp, self._action_qpos[frame_index, :7])
                        ],
                        "action_qpos:right_arm": [
                            CollectionRawSample(timestamp, self._action_qpos[frame_index, 7:])
                        ],
                        "action_eef:left_arm": [
                            CollectionRawSample(timestamp, self._action_eef[frame_index, :8])
                        ],
                        "action_eef:right_arm": [
                            CollectionRawSample(timestamp, self._action_eef[frame_index, 8:])
                        ],
                    }
                )
            return CollectionRawBatch(images=images, vectors=vectors)

        return RawCollectionSnapshot(timestamp=timestamp, decode_raw=decode_raw)

    def action(self, frame_index: int) -> np.ndarray:
        """Return executed action_qpos [14] for rollout/eval raw snapshots."""
        return self._action_qpos[frame_index].copy()

    def timestamp(self, frame_index: int) -> float:
        return float(frame_index) / float(self.fps)

    def _build_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ticks = np.arange(self.frames, dtype=np.float32)[:, None]
        qpos_offsets = np.arange(14, dtype=np.float32)[None, :]
        eef_offsets = np.arange(16, dtype=np.float32)[None, :]
        state_qpos = 0.25 * np.sin(ticks * 0.017 + qpos_offsets * 0.11)
        action_qpos = state_qpos + 0.05 * np.cos(ticks * 0.013 + qpos_offsets * 0.07)
        state_qpos[:, 6] = np.where((np.arange(self.frames) // 45) % 2 == 0, 100.0, 0.0)
        state_qpos[:, 13] = np.where((np.arange(self.frames) // 60) % 2 == 0, 100.0, 0.0)
        action_qpos[:, 6] = state_qpos[:, 6]
        action_qpos[:, 13] = state_qpos[:, 13]
        state_eef = 0.4 * np.sin(ticks * 0.01 + eef_offsets * 0.05)
        action_eef = state_eef + 0.02 * np.cos(ticks * 0.015 + eef_offsets * 0.03)
        state_eef[:, 7] = state_qpos[:, 6]
        state_eef[:, 15] = state_qpos[:, 13]
        action_eef[:, 7] = action_qpos[:, 6]
        action_eef[:, 15] = action_qpos[:, 13]
        return (
            state_qpos.astype(np.float32),
            action_qpos.astype(np.float32),
            state_eef.astype(np.float32),
            action_eef.astype(np.float32),
        )

    def _build_camera_cache(self) -> dict[str, list[np.ndarray]]:
        frames: dict[str, list[np.ndarray]] = {}
        key_by_camera_name = fake_node.CAMERA_OBSERVATION_KEYS
        for camera_name, color in fake_node.CAMERA_COLORS.items():
            camera_key = key_by_camera_name[camera_name]
            frames[camera_key] = [
                self._make_frame(color, frame_index)
                for frame_index in range(fake_node.CAMERA_FRAME_CACHE_SIZE)
            ]
        return frames

    def _camera_frame(self, camera_key: str, frame_index: int) -> np.ndarray:
        frames = self._camera_cache[camera_key]
        return frames[frame_index % len(frames)]

    def _make_frame(self, color: tuple[int, int, int], frame_index: int) -> np.ndarray:
        image = np.empty((self.image_height, self.image_width, 3), dtype=np.uint8)
        image[:] = np.asarray(color, dtype=np.uint8)
        band_h = max(self.image_height // 18, 2)
        band_y = (frame_index * 7) % max(1, self.image_height - band_h + 1)
        image[band_y : band_y + band_h, :, :] = 255
        stripe_w = max(self.image_width // 32, 2)
        stripe_x = (frame_index * 13) % max(1, self.image_width - stripe_w + 1)
        image[:, stripe_x : stripe_x + stripe_w, :] //= 2
        return image


def build_robot() -> Robot:
    """Build R1 Lite fake-node robot metadata with 14 qpos/action dimensions."""
    left_joints = tuple(fake_node.LEFT_ARM_JOINTS) + ("left_gripper",)
    right_joints = tuple(fake_node.RIGHT_ARM_JOINTS) + ("right_gripper",)
    return Robot(
        name="r1_lite",
        actuator_groups=(
            ActuatorGroup("left_arm", 7, left_joints, gripper_index=6),
            ActuatorGroup("right_arm", 7, right_joints, gripper_index=6),
        ),
        initial_qpos=np.zeros(14, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=tuple(
                CameraSpec(camera_name, camera_key)
                for camera_name, camera_key in fake_node.CAMERA_OBSERVATION_KEYS.items()
            ),
            state_composition=("left_arm", "right_arm"),
        ),
    )


def dataset_keys() -> ConfigDict:
    """Recorder dataset keys matching R1 Lite rollout/eval output."""
    return ConfigDict(
        state_key="observations.state.qpos",
        eef_key="observations.state.eef",
        action_key="action",
        video_keys={
            "cam_high": "observation.images.cam_high",
            "cam_left_wrist": "observation.images.cam_left_wrist",
            "cam_right_wrist": "observation.images.cam_right_wrist",
        },
    )


def collection_config() -> ConfigDict:
    """R1 Lite collection schema used by the fake node E2E harness."""
    return ConfigDict(
        enabled=True,
        schema=ConfigDict(
            robot_type="r1_lite",
            min_episode_frames=10,
            arms={"left_arm": "left", "right_arm": "right"},
            cameras={
                "cam_high": "observation.images.cam_high",
                "cam_left_wrist": "observation.images.cam_left_wrist",
                "cam_right_wrist": "observation.images.cam_right_wrist",
            },
            columns={
                "qpos": "observations.state.qpos",
                "eef": "observations.state.eef",
                "action_qpos": "action.qpos",
                "action_eef": "action.eef",
            },
        ),
    )


def build_logger(
    path: Path,
    fake_data: R1LiteFakeDataset,
    *,
    collection: bool = False,
    eval_mode: bool = False,
) -> EpisodeLogger:
    """Build one async EpisodeLogger configured for full-size fake-node frames."""
    return EpisodeLogger(
        path,
        build_robot(),
        fps=fake_data.fps,
        dataset_keys=dataset_keys(),
        convert_bgr_to_rgb=False,
        collection=collection_config() if collection else None,
        async_save=True,
        save_queue_max=15,
        eval_mode=eval_mode,
        save_image_height=fake_data.image_height,
        save_image_width=fake_data.image_width,
    )


def run_collection(root: Path, fake_data: R1LiteFakeDataset, max_save_s: float) -> dict[str, Any]:
    """Record and asynchronously save one full fake collection episode."""
    logger = build_logger(root / "collection", fake_data, collection=True)
    logger.start_episode(TASK)
    for frame_index in range(fake_data.frames):
        logger.ingest_collection_snapshot(fake_data.snapshot(frame_index, include_action=True))
    elapsed = finish_and_wait(logger, max_save_s)
    dataset_dir = root / "collection" / sanitize_path_component(TASK)
    return verify_dataset("collection", dataset_dir, fake_data, elapsed, "action.qpos")


def run_raw_episode(
    label: str,
    root: Path,
    fake_data: R1LiteFakeDataset,
    max_save_s: float,
    *,
    eval_mode: bool,
) -> dict[str, Any]:
    """Record and asynchronously save one full fake rollout/eval episode."""
    logger = build_logger(root / label, fake_data, eval_mode=eval_mode)
    logger.start_episode(TASK)
    if eval_mode:
        logger.set_episode_meta(clip_id=f"{label}-clip", prompt=TASK, trial=1)
    for frame_index in range(fake_data.frames):
        logger.ingest_raw_episode_snapshot(
            fake_data.snapshot(frame_index, include_action=False),
            fake_data.action(frame_index),
        )
    elapsed = finish_and_wait(logger, max_save_s)
    return verify_dataset(label, root / label, fake_data, elapsed, "action")


def finish_and_wait(logger: EpisodeLogger, max_save_s: float) -> float:
    """End the episode and require the async save queue to drain within max_save_s."""
    started = time.perf_counter()
    saved = logger.end_episode()
    if not saved:
        raise AssertionError("episode was not queued for saving")
    if not logger.wait_for_saves(timeout=max_save_s):
        raise AssertionError(f"async save did not finish within {max_save_s:.1f}s")
    elapsed = time.perf_counter() - started
    if elapsed > max_save_s:
        raise AssertionError(f"async save took {elapsed:.3f}s, expected <= {max_save_s:.3f}s")
    return elapsed


def verify_dataset(
    label: str,
    dataset_dir: Path,
    fake_data: R1LiteFakeDataset,
    elapsed_s: float,
    action_column: str,
) -> dict[str, Any]:
    """Verify parquet rows, fixed timestamps, metadata, and videos for one dataset."""
    parquet = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not parquet.exists():
        raise AssertionError(f"{label} parquet missing: {parquet}")
    if not episodes_path.exists():
        raise AssertionError(f"{label} episodes.jsonl missing: {episodes_path}")
    table = pq.read_table(parquet)
    if table.num_rows != fake_data.frames:
        raise AssertionError(f"{label} rows={table.num_rows}, expected={fake_data.frames}")
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    expected = np.arange(fake_data.frames, dtype=np.float64) / float(fake_data.fps)
    if not np.allclose(timestamps, expected, atol=1e-6):
        max_error = float(np.max(np.abs(timestamps - expected)))
        raise AssertionError(f"{label} timestamp grid max error {max_error}")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise AssertionError(f"{label} metadata rows={len(rows)}, expected=1")
    if int(rows[0].get("length", -1)) != fake_data.frames:
        raise AssertionError(f"{label} metadata length={rows[0].get('length')}")
    actions = np.asarray(table.column(action_column).to_pylist(), dtype=np.float32)
    if actions.shape != (fake_data.frames, 14):
        expected_shape = (fake_data.frames, 14)
        raise AssertionError(f"{label} action shape={actions.shape}, expected={expected_shape}")
    if not math.isfinite(float(actions.std())) or float(actions.std()) <= 1e-4:
        raise AssertionError(f"{label} action column did not vary")
    for video_key in dataset_keys().video_keys.values():
        video = dataset_dir / "videos" / "chunk-000" / video_key / "episode_000000.mp4"
        if not video.exists() or video.stat().st_size <= 0:
            raise AssertionError(f"{label} video missing or empty: {video}")
    return {
        "label": label,
        "save_s": round(elapsed_s, 3),
        "rows": table.num_rows,
        "dataset_dir": str(dataset_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "manual" / "async_save_perf",
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration-s", type=int, default=DURATION_S)
    parser.add_argument("--image-height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--max-save-s", type=float, default=MAX_SAVE_S)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fake_data = R1LiteFakeDataset(
        fps=args.fps,
        duration_s=args.duration_s,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    results = [
        run_collection(run_dir, fake_data, args.max_save_s),
        run_raw_episode("rollout", run_dir, fake_data, args.max_save_s, eval_mode=False),
        run_raw_episode("eval", run_dir, fake_data, args.max_save_s, eval_mode=True),
    ]
    print(json.dumps({"frames": fake_data.frames, "results": results}, indent=2))


if __name__ == "__main__":
    main()
