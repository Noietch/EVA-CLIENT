"""Teleop collection episode writer — the EpisodeLogger sub-component that turns a
stream of raw teleop snapshots into one LeRobot v2.1 collection episode.

Decode/copy work runs on a background ingest thread so the capture loop never
blocks; ``end_episode`` drains it, validates each frame, and packs the per-frame
columns into a SaveJob the parent EpisodeLogger flushes to disk.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import queue
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq

from core.app.handlers.imaging import resize_direct
from core.recorder.episode import (
    _COLLECTION_VECTOR_FIELDS,
    QualityIssue,
    SaveJob,
    _read_jsonl,
    _StatAccumulator,
    _to_rgb_uint8,
)
from core.recorder.lerobot_meta import build_info
from core.types import Observation

if TYPE_CHECKING:
    from core.config import ConfigDict
    from core.recorder.episode import EpisodeLogger
    from core.types import RawCollectionSnapshot

logger = logging.getLogger(__name__)

# Collection vector fields fall into two taxonomy groups, each fixing both the
# expected dimension and the per-element feature names. Single source of truth for
# expected_dim() and _feature_names().
_JOINT_FIELDS = ("state_qpos", "action_qpos")
_EEF_FIELDS = ("state_eef", "action_eef")

# Schema column name (config / parquet key) -> Observation attribute name.
_COLUMN_TO_FIELD = {
    "qpos": "state_qpos",
    "eef": "state_eef",
    "action_qpos": "action_qpos",
    "action_eef": "action_eef",
}


def _is_hwc3(arr: np.ndarray) -> bool:
    """True when arr is a HWC image with 3 channels (the only shape we record)."""
    return arr.ndim == 3 and arr.shape[2] == 3


class CollectionEpisodeWriter:
    """Builds one teleop collection episode for its parent EpisodeLogger.

    Decode/copy work runs on a background ingest thread (fed via ``ingest``) so the
    capture loop never blocks; ``end_episode`` drains it, validates every frame's
    timestamps / images / vector dims (recording QualityIssue flags), and packs the
    per-frame columns into a SaveJob the parent flushes to a LeRobot v2.1 parquet.
    """

    _INGEST_HIGH_WATER = 300

    def __init__(self, episode_logger: EpisodeLogger, collection: ConfigDict) -> None:
        self._logger = episode_logger
        self._collection = collection
        self._schema = collection.schema
        self._records: list[Observation] = []
        self._quality_issues: list[QualityIssue] = []
        self._image_shapes: dict[str, tuple[int, int]] = {}
        self._ingest_queue: queue.Queue[RawCollectionSnapshot | None] = queue.Queue()
        self._ingest_thread: threading.Thread | None = None
        self._ingest_error: BaseException | None = None
        self._high_water_logged = False
        self._received_snapshots = 0
        self._skipped_before_start = 0
        self._skipped_no_action = 0
        self._min_capture_time: float | None = None

    def start_episode(self, min_capture_time: float | None = None) -> None:
        """Reset buffers and launch the background ingest thread for a new episode.

        Args:
            min_capture_time: Source timestamp cutoff; frames at or before it are
                treated as pre-start cache and skipped.
        """
        self._records = []
        self._quality_issues = []
        self._ingest_queue = queue.Queue()
        self._ingest_error = None
        self._high_water_logged = False
        self._received_snapshots = 0
        self._skipped_before_start = 0
        self._skipped_no_action = 0
        self._min_capture_time = min_capture_time
        self._ingest_thread = threading.Thread(
            target=self._ingest_loop, name="collection-ingest", daemon=True
        )
        self._ingest_thread.start()

    def ingest(self, snapshot: RawCollectionSnapshot) -> None:
        """Hand one pre-decode snapshot to the ingest thread. O(1), never blocks.

        The capture loop calls this; all decode/copy/convert work happens on the
        ingest thread. The queue is unbounded so a busy ingest never stalls the
        main loop or drops frames — sustained overload only grows memory, flagged
        once via a high-water warning.
        """
        if self._ingest_error is not None:
            raise RuntimeError("collection ingest thread failed") from self._ingest_error
        if self._is_before_episode_start(snapshot.timestamp):
            self._skipped_before_start += 1
            return
        self._ingest_queue.put(snapshot)
        self._received_snapshots += 1
        if not self._high_water_logged and self._ingest_queue.qsize() > self._INGEST_HIGH_WATER:
            self._high_water_logged = True
            logger.warning(
                "collection ingest backlog above %d frames; processing slower than capture rate",
                self._INGEST_HIGH_WATER,
            )

    def _ingest_loop(self) -> None:
        while True:
            snapshot = self._ingest_queue.get()
            try:
                if snapshot is None:
                    return
                frame = snapshot.decode()
                if self._is_before_episode_start(frame.timestamp):
                    self._received_snapshots = max(0, self._received_snapshots - 1)
                    self._skipped_before_start += 1
                    continue
                # Teleop action lags source frames at episode start (node publishes
                # no action until calibration completes); skip those rather than
                # record zero-action rows, matching the pre-async behavior.
                if frame.action_qpos is None:
                    self._skipped_no_action += 1
                    continue
                # Apply the optional per-frame hook (used by the EEF-projection
                # helper to fill frame.eef_uv) BEFORE record_frame.
                hook = getattr(self._logger, "_on_collection_frame", None)
                if hook is not None:
                    try:
                        hook(frame)
                    except Exception:  # never break ingest — uv is best-effort
                        logger.exception("collection frame hook failed")
                self.record_frame(frame, copy_images=False)
            except BaseException as exc:  # surface to end_episode; keep buffers intact
                self._ingest_error = exc
                logger.exception("collection ingest failed")
                return
            finally:
                self._ingest_queue.task_done()

    def _drain_ingest(self) -> None:
        """Block until the ingest thread has consumed every queued snapshot."""
        thread = self._ingest_thread
        if thread is None:
            return
        self._ingest_queue.put(None)
        thread.join()
        self._ingest_thread = None
        if self._ingest_error is not None:
            raise RuntimeError("collection ingest thread failed") from self._ingest_error

    def record_frame(self, frame: Observation, *, copy_images: bool = True) -> None:
        """Append one decoded frame and run per-frame QC checks.

        Args:
            frame: Decoded collection frame with HWC camera images and vector fields.
            copy_images: Copy image arrays before storing them. Public callers keep
                the default; raw snapshot ingest can pass False because decode()
                yields episode-private arrays.
        """
        if frame.timestamp is None or frame.state_qpos is None:
            raise ValueError("collection frame missing timestamp or state_qpos")
        frame_index = len(self._records)
        images = {
            key: np.asarray(value).copy() if copy_images else np.asarray(value)
            for key, value in frame.images.items()
        }
        vectors = {
            field: (
                None
                if getattr(frame, field) is None
                else np.asarray(getattr(frame, field), dtype=np.float32).copy()
            )
            for field in _COLLECTION_VECTOR_FIELDS
            if field != "state_qpos"
        }
        # Preserve eef_uv (per-camera pixel projection dict) alongside the vectors.
        uv_map = getattr(frame, "eef_uv", None)
        if uv_map:
            uv_copy = {
                cam: np.asarray(arr, dtype=np.float32).copy()
                for cam, arr in uv_map.items()
            }
        else:
            uv_copy = None
        record = Observation(
            timestamp=float(frame.timestamp),
            images=images,
            state_qpos=np.asarray(frame.state_qpos, dtype=np.float32).copy(),
            eef_uv=uv_copy,
            **vectors,
        )
        self._records.append(record)
        self._check_timestamp(float(frame.timestamp), frame_index)
        self._check_images(record, frame_index)

    def frame_counts(self) -> dict[str, int]:
        """Return live collection counters for UI status.

        received counts raw snapshots accepted by the main loop. recorded counts
        decoded rows that will be saved. pending bridges the ingest lag between
        those two so the UI does not show zero while frames are queued.
        """
        recorded = len(self._records)
        received = max(self._received_snapshots, recorded)
        pending = max(0, received - self._skipped_no_action - recorded)
        return {
            "received": received,
            "recorded": recorded,
            "pending": pending,
            "skipped_before_start": self._skipped_before_start,
            "skipped_no_action": self._skipped_no_action,
        }

    def _is_before_episode_start(self, timestamp: float | None) -> bool:
        if self._min_capture_time is None or timestamp is None or timestamp <= 0.0:
            return False
        return timestamp <= self._min_capture_time

    def expected_dim(self, field: str) -> int:
        """Expected vector length for a collection field (joints / eef).

        Raises ValueError for an unknown field name.
        """
        if field in _JOINT_FIELDS:
            return self._logger._robot.total_action_dim
        if field in _EEF_FIELDS:
            return 8 * len(self._schema.arms)
        raise ValueError(f"unknown collection field: {field}")

    def end_episode(self) -> SaveJob | None:
        """Drain ingest, validate frames, and pack the episode into a SaveJob.

        Returns:
            A SaveJob carrying the cleaned per-frame columns + episode meta row for
            the parent logger to flush, or None when no frames were recorded.
        """
        self._drain_ingest()
        if not self._records:
            logger.warning(
                "collection episode produced 0 frames; nothing saved "
                "(teleop action never arrived — calibration likely incomplete)"
            )
            return None
        if len(self._records) < self._schema.min_episode_frames:
            self._add_issue(
                "episode_too_short",
                f"episode has {len(self._records)} frames, expected at least "
                f"{self._schema.min_episode_frames}",
            )
        # measured fps stays in episodes.jsonl as a capture-jitter diagnostic only;
        # mp4 frame rate must match the synthesized timestamp grid (target fps).
        collection_fps = self._episode_average_fps()
        videos = self._snapshot_videos()

        task = self._logger._task or ""
        dataset_dir = self._logger._collection_dataset_dir(task)
        episode_index = self._logger._next_collection_episode_index(dataset_dir)
        task_index = 0
        n_frames = len(self._records)
        global_index = self._logger._next_collection_global_index(dataset_dir, n_frames)

        # LeRobot v2.1 requires each camera's mp4 frame count == data-table row count;
        # a dropped/corrupt frame is skipped in _snapshot_videos but still occupies a
        # parquet row, so the counts diverge and the episode crashes on load. Flag it.
        if self._logger._save_video:
            for video_key in self._schema.cameras.values():
                got = len(videos.get(video_key, [])) if videos else 0
                if got != n_frames:
                    self._add_issue(
                        "frame_count_mismatch",
                        f"video {video_key} has {got} frames but data table has {n_frames} rows",
                    )

        columns: dict[str, Any] = {}
        for column, output_key in self._schema.columns.items():
            field = _COLUMN_TO_FIELD[column]
            columns[output_key] = [
                self._clean_vector(field, getattr(record, field), frame_index).tolist()
                for frame_index, record in enumerate(self._records)
            ]

        # Optional eef_uv columns — one per camera whose calibration was live at
        # capture time. Absent from the schema/config; opt-in by simply having
        # calibration set. Written as observation.eef_uv.<cam> so downstream
        # trainers can pick them up per camera.
        uv_cams: dict[str, list[list[float]]] = {}
        for frame_index, record in enumerate(self._records):
            uv_map = getattr(record, "eef_uv", None) or {}
            for cam_key, uv_array in uv_map.items():
                col = uv_cams.setdefault(cam_key, [])
                # (n_arms, 2) -> flat list of 2*n_arms floats per row.
                arr = np.asarray(uv_array, dtype=np.float32).reshape(-1)
                # Backfill NaNs for earlier frames that were missing this camera.
                while len(col) < frame_index:
                    col.append([float("nan")] * arr.size)
                col.append(arr.tolist())
        for cam_key, col in uv_cams.items():
            while len(col) < n_frames:
                col.append([float("nan")] * (len(col[0]) if col else 2))
            columns[f"observation.eef_uv.{cam_key}"] = col
        # LeRobot requires timestamp[i] == i/fps (strictly equal-interval, tol 1e-4),
        # so synthesize it from the target fps. The real hardware capture time (raw,
        # jittery uptime seconds) is preserved out-of-band in ``capture_time``.
        fps = float(self._logger._fps)
        columns.update(
            {
                "timestamp": [i / fps for i in range(n_frames)],
                "capture_time": [
                    float(record.timestamp if record.timestamp is not None else 0.0)
                    for record in self._records
                ],
                "frame_index": list(range(n_frames)),
                "episode_index": [episode_index] * n_frames,
                "index": list(range(global_index, global_index + n_frames)),
                "task_index": [task_index] * n_frames,
            }
        )

        row: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": [task],
            "length": n_frames,
            "quality": "red" if self._quality_issues else "green",
            "quality_issues": [dataclasses.asdict(issue) for issue in self._quality_issues],
        }
        if collection_fps is not None:
            row["collection_fps"] = collection_fps
        if self._logger._episode_meta:
            row.update(self._logger._episode_meta)

        job = SaveJob(
            episode_index=episode_index,
            task=task,
            task_index=task_index,
            global_index=global_index,
            steps=[],
            episode_meta={},
            task_to_index={task: task_index},
            collection_columns=columns,
            collection_episode_row=row,
            videos=videos,
            video_fps=fps,
            dataset_dir=dataset_dir,
        )
        self._records = []
        self._quality_issues = []
        return job

    def cancel_episode(self) -> None:
        """Discard the in-flight episode: drain the ingest thread and drop all buffers."""
        try:
            self._drain_ingest()
        except BaseException:
            logger.exception("Error draining ingest thread during cancel")
            self._ingest_thread = None
        self._records = []
        self._quality_issues = []

    def finalize(self, dataset_dir: Path | None = None) -> None:
        if dataset_dir is None:
            return
        self._write_info_json(dataset_dir)
        self._write_stats_json(dataset_dir)

    def _check_timestamp(self, timestamp: float, frame_index: int) -> None:
        if frame_index == 0:
            return
        previous = self._records[frame_index - 1].timestamp
        assert previous is not None
        dt = timestamp - previous
        if dt <= 0.0:
            self._add_issue(
                "non_monotonic_timestamp",
                f"frame {frame_index} timestamp {timestamp} follows {previous}",
            )
            return

    def _episode_average_fps(self) -> float | None:
        if len(self._records) < 2:
            return None
        last = self._records[-1].timestamp
        first = self._records[0].timestamp
        assert last is not None and first is not None
        duration = last - first
        if duration <= 0.0:
            return None
        return (len(self._records) - 1) / duration

    def _save_size(self) -> tuple[int, int] | None:
        """Configured (height, width) for saved frames, or None to keep native size."""
        h = self._logger._save_image_height
        w = self._logger._save_image_width
        if h is None or w is None:
            return None
        return int(h), int(w)

    def _snapshot_videos(self) -> dict[str, list[np.ndarray]] | None:
        if not self._logger._save_video:
            return None
        size = self._save_size()
        videos: dict[str, list[np.ndarray]] = {}
        for camera_key, video_key in self._schema.cameras.items():
            frames = []
            for record in self._records:
                image = record.images.get(camera_key)
                if image is None:
                    continue
                arr = np.asarray(image)
                if not _is_hwc3(arr):
                    continue
                rgb = _to_rgb_uint8(arr, self._logger._convert_bgr_to_rgb)
                if size is not None:
                    rgb = resize_direct(rgb, size[0], size[1])
                frames.append(rgb)
            if frames:
                videos[video_key] = frames
        return videos or None

    def _check_images(self, record: Observation, frame_index: int) -> None:
        size = self._save_size()
        for camera_key, video_key in self._schema.cameras.items():
            image = record.images.get(camera_key)
            if image is None:
                self._add_issue(
                    "missing_camera",
                    f"frame {frame_index} missing configured camera {camera_key}",
                )
                continue
            arr = np.asarray(image)
            if not _is_hwc3(arr):
                self._add_issue(
                    "invalid_image_shape",
                    f"frame {frame_index} camera {camera_key} has shape {list(arr.shape)}",
                )
                continue
            rgb = _to_rgb_uint8(arr, self._logger._convert_bgr_to_rgb)
            if video_key not in self._image_shapes:
                self._image_shapes[video_key] = (
                    size if size is not None else (rgb.shape[0], rgb.shape[1])
                )

    def _clean_vector(self, field: str, value: np.ndarray | None, frame_index: int) -> np.ndarray:
        dim = self.expected_dim(field)
        if value is None:
            self._add_issue(
                "missing_configured_column",
                f"frame {frame_index} missing configured field {field}",
            )
            return np.zeros(dim, dtype=np.float32)

        arr = np.asarray(value, dtype=np.float32)
        original_shape = tuple(arr.shape)
        if arr.shape != (dim,):
            self._add_issue(
                "invalid_vector_dim",
                f"frame {frame_index} field {field} has shape {list(original_shape)}, "
                f"expected [{dim}]",
            )
            return np.zeros(dim, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            self._add_issue(
                "non_finite_value",
                f"frame {frame_index} field {field} contains non-finite values",
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float32, copy=True)

    def _write_info_json(self, dataset_dir: Path) -> None:
        episodes = _read_jsonl(self._logger._meta_path("episodes.jsonl", dataset_dir))
        total_episodes = len(episodes)
        total_frames = sum(int(e.get("length", 0)) for e in episodes)
        total_videos = total_episodes * len(self._schema.cameras) if self._logger._save_video else 0
        # fps must equal the integer target used to synthesize timestamps, not the
        # measured (jittery) average — LeRobot validates timestamp[i] == i/fps.
        fps = float(self._logger._fps)
        total_tasks = len(self._logger._load_existing_tasks(dataset_dir))
        info = build_info(
            robot_type=self._schema.robot_type or self._logger._robot.name,
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=total_tasks,
            total_videos=total_videos,
            fps=fps,
            features=self._build_features(fps),
        )
        with self._logger._meta_path("info.json", dataset_dir).open("w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def _build_features(self, fps: float) -> dict[str, Any]:
        features: dict[str, Any] = {}
        if self._logger._save_video:
            for video_key in self._schema.cameras.values():
                h, w = self._image_shapes.get(video_key, (480, 640))
                features[video_key] = {
                    "dtype": "video",
                    "shape": [h, w, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.height": h,
                        "video.width": w,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "video.fps": fps,
                        "video.channels": 3,
                        "has_audio": False,
                    },
                }
        for column, output_key in self._schema.columns.items():
            field = _COLUMN_TO_FIELD[column]
            features[output_key] = {
                "dtype": "float32",
                "shape": [self.expected_dim(field)],
                "names": self._feature_names(field),
            }
        for col in ("timestamp", "capture_time"):
            features[col] = {"dtype": "float32", "shape": [1], "names": None}
        for col in ("frame_index", "episode_index", "index", "task_index"):
            features[col] = {"dtype": "int64", "shape": [1], "names": None}
        return features

    def _write_stats_json(self, dataset_dir: Path) -> None:
        cached_stats = self._stats_from_episode_stats(dataset_dir)
        if cached_stats is not None:
            with self._logger._meta_path("stats.json", dataset_dir).open("w") as f:
                json.dump(cached_stats, f, indent=2)
            return

        accumulators = {
            output_key: _StatAccumulator() for output_key in self._schema.columns.values()
        }
        data_dir = dataset_dir / "data" / "chunk-000"
        for ep_path in sorted(data_dir.glob("*.parquet")):
            table = pq.read_table(str(ep_path))
            for output_key, accumulator in accumulators.items():
                if output_key in table.column_names:
                    accumulator.update(
                        np.array(table.column(output_key).to_pylist(), dtype=np.float64)
                    )

        stats: dict[str, Any] = {}
        for output_key, accumulator in accumulators.items():
            if accumulator.count:
                stats[output_key] = accumulator.result()
        with self._logger._meta_path("stats.json", dataset_dir).open("w") as f:
            json.dump(stats, f, indent=2)

    def _stats_from_episode_stats(self, dataset_dir: Path) -> dict[str, Any] | None:
        episodes = _read_jsonl(self._logger._meta_path("episodes.jsonl", dataset_dir))
        stats_rows = _read_jsonl(self._logger._meta_path("episodes_stats.jsonl", dataset_dir))
        if not episodes or not stats_rows:
            return None

        stats_by_episode = {
            int(row["episode_index"]): row.get("stats", {}) for row in stats_rows
        }
        episode_indices = {int(row["episode_index"]) for row in episodes}
        if set(stats_by_episode) != episode_indices:
            return None

        stats: dict[str, Any] = {}
        for output_key in self._schema.columns.values():
            combined = self._combine_episode_stats(
                stats_by_episode[int(row["episode_index"])].get(output_key)
                for row in episodes
            )
            if combined is None:
                return None
            stats[output_key] = combined
        return stats

    def _combine_episode_stats(
        self, rows: Iterable[dict[str, Any] | None]
    ) -> dict[str, list[float]] | None:
        count = 0
        min_value: np.ndarray | None = None
        max_value: np.ndarray | None = None
        total: np.ndarray | None = None
        total_squares: np.ndarray | None = None
        for row in rows:
            if row is None:
                return None
            n = int(row["count"][0])
            mean = np.asarray(row["mean"], dtype=np.float64)
            std = np.asarray(row["std"], dtype=np.float64)
            row_min = np.asarray(row["min"], dtype=np.float64)
            row_max = np.asarray(row["max"], dtype=np.float64)
            row_total = mean * n
            row_total_squares = (std * std + mean * mean) * n
            if min_value is None:
                min_value = row_min
                max_value = row_max
                total = row_total
                total_squares = row_total_squares
            else:
                min_value = np.minimum(min_value, row_min)
                assert max_value is not None and total is not None and total_squares is not None
                max_value = np.maximum(max_value, row_max)
                total = total + row_total
                total_squares = total_squares + row_total_squares
            count += n

        if count == 0 or min_value is None:
            return None
        assert max_value is not None and total is not None and total_squares is not None
        mean = total / count
        std = np.sqrt(np.maximum(total_squares / count - mean * mean, 0.0))
        return {
            "min": min_value.tolist(),
            "max": max_value.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    def _feature_names(self, field: str) -> list[str] | None:
        if field in _JOINT_FIELDS:
            return self._joint_names()
        if field in _EEF_FIELDS:
            return self._eef_names()
        return None

    def _joint_names(self) -> list[str]:
        names: list[str] = []
        for group in self._logger._robot.actuator_groups:
            prefix = (
                self._schema.arms[group.name] if group.name in self._schema.arms else group.name
            )
            joint_index = 0
            for i in range(group.dof):
                if group.gripper_index == i:
                    names.append(f"{prefix}.gripper")
                else:
                    names.append(f"{prefix}.j{joint_index}")
                    joint_index += 1
        return names

    def _eef_names(self) -> list[str]:
        suffixes = ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
        names: list[str] = []
        for group in self._logger._robot.actuator_groups:
            if group.name in self._schema.arms:
                prefix = self._schema.arms[group.name]
                names.extend(f"{prefix}.{suffix}" for suffix in suffixes)
        return names

    def _add_issue(self, code: str, detail: str) -> None:
        self._quality_issues.append(QualityIssue(severity="red", code=code, detail=detail))
