"""Teleop collection episode writer — the EpisodeLogger sub-component that turns a
stream of raw teleop snapshots into one LeRobot v2.1 collection episode.

Recording only stores raw snapshots; the save worker later extracts raw samples,
aligns the full episode, validates each frame, and packs the per-frame columns
into a SaveJob the parent EpisodeLogger flushes to disk.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq

from core.recorder.collection_alignment import (
    CollectionAlignmentReport,
    align_collection_samples,
    image_skew_tolerance_sec,
)
from core.recorder.episode import (
    _COLLECTION_VECTOR_FIELDS,
    QualityIssue,
    SaveJob,
    _read_jsonl,
    _StatAccumulator,
    _to_rgb_uint8,
)
from core.recorder.lerobot_meta import build_info
from core.types import CollectionRawBatch, Observation
from core.utils.images import resize_direct

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


@dataclasses.dataclass
class CollectionSavePayload:
    """Frozen collection episode input prepared by the save worker."""

    raw_snapshots: list[RawCollectionSnapshot]
    quality_issues: list[QualityIssue]
    min_capture_time: float | None
    max_capture_time: float | None


def _is_hwc3(arr: np.ndarray) -> bool:
    """True when arr is a HWC image with 3 channels (the only shape we record)."""
    return arr.ndim == 3 and arr.shape[2] == 3


class CollectionEpisodeWriter:
    """Builds one teleop collection episode for its parent EpisodeLogger.

    The capture path only stores RawCollectionSnapshot references. Raw extraction,
    alignment, validation, image decode, and video packing all run on the save
    worker so recording competes as little as possible with live robot/UI updates.
    """

    def __init__(self, episode_logger: EpisodeLogger, collection: ConfigDict) -> None:
        self._logger = episode_logger
        self._collection = collection
        self._schema = collection.schema
        self._records: list[Observation] = []
        self._quality_issues: list[QualityIssue] = []
        self._raw_batch = CollectionRawBatch()
        self._raw_snapshots: list[RawCollectionSnapshot] = []
        self._alignment_report: CollectionAlignmentReport | None = None
        self._image_shapes: dict[str, tuple[int, int]] = {}
        self._state_lock = threading.Lock()
        self._received_snapshots = 0
        self._skipped_before_start = 0
        self._min_capture_time: float | None = None
        self._max_capture_time: float | None = None

    def start_episode(self, min_capture_time: float | None = None) -> None:
        """Reset buffers for a new collection episode.

        Args:
            min_capture_time: Source timestamp cutoff; frames at or before it are
                treated as pre-start cache and skipped.
        """
        self._records = []
        self._quality_issues = []
        self._raw_batch = CollectionRawBatch(start_time=min_capture_time)
        self._raw_snapshots = []
        self._alignment_report = None
        self._received_snapshots = 0
        self._skipped_before_start = 0
        self._min_capture_time = min_capture_time
        self._max_capture_time = None

    def ingest(self, snapshot: RawCollectionSnapshot) -> None:
        """Store one pre-decode snapshot. O(1), never decodes raw messages.

        The capture loop calls this while recording. Raw extraction is intentionally
        deferred to the save worker so start-record does not add image/vector decode
        work to the live UI/control path.
        """
        if self._is_before_episode_start(snapshot.timestamp):
            with self._state_lock:
                self._skipped_before_start += 1
            return
        with self._state_lock:
            self._received_snapshots += 1
            if snapshot.timestamp > 0.0:
                self._max_capture_time = (
                    snapshot.timestamp
                    if self._max_capture_time is None
                    else max(self._max_capture_time, snapshot.timestamp)
                )
            self._raw_snapshots.append(snapshot)

    def _append_aligned_frame(self, frame: Observation) -> None:
        """Append one fixed-clock aligned frame and run per-frame QC checks.

        Args:
            frame: Aligned collection frame with HWC camera images and vector fields.
        """
        if frame.timestamp is None or frame.state_qpos is None:
            raise ValueError("collection frame missing timestamp or state_qpos")
        with self._state_lock:
            frame_index = len(self._records)
            images = {
                key: np.asarray(value) for key, value in frame.images.items()
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
            record = Observation(
                timestamp=float(frame.timestamp),
                images=images,
                state_qpos=np.asarray(frame.state_qpos, dtype=np.float32).copy(),
                **vectors,
            )
            self._records.append(record)
            self._check_timestamp(float(frame.timestamp), frame_index)
            self._check_images(record, frame_index)
            self._max_capture_time = (
                record.timestamp
                if self._max_capture_time is None
                else max(self._max_capture_time, record.timestamp)
            )

    def frame_counts(self) -> dict[str, int]:
        """Return live collection counters for UI status.

        received counts raw snapshots accepted by the main loop. recorded counts
        aligned rows that will be saved. pending bridges the ingest lag between
        those two so the UI does not show zero while frames are queued.
        """
        with self._state_lock:
            recorded = len(self._records)
            received = max(self._received_snapshots, recorded)
            pending = max(0, received - recorded)
            return {
                "received": received,
                "recorded": recorded,
                "pending": pending,
                "skipped_before_start": self._skipped_before_start,
            }

    def _is_before_episode_start(self, timestamp: float | None) -> bool:
        if self._min_capture_time is None or timestamp is None or timestamp <= 0.0:
            return False
        return timestamp <= self._min_capture_time

    def _merge_raw_batch(self, batch: CollectionRawBatch) -> None:
        with self._state_lock:
            if batch.start_time is not None:
                self._raw_batch.start_time = (
                    batch.start_time
                    if self._raw_batch.start_time is None
                    else max(self._raw_batch.start_time, batch.start_time)
                )
            if batch.end_time is not None:
                self._raw_batch.end_time = (
                    batch.end_time
                    if self._raw_batch.end_time is None
                    else max(self._raw_batch.end_time, batch.end_time)
                )
            for key, samples in batch.images.items():
                self._raw_batch.images.setdefault(key, []).extend(samples)
                self._track_raw_sample_times(samples)
            for key, samples in batch.vectors.items():
                self._raw_batch.vectors.setdefault(key, []).extend(samples)
                self._track_raw_sample_times(samples)

    def _track_raw_sample_times(self, samples: list[Any]) -> None:
        for sample in samples:
            self._max_capture_time = (
                sample.timestamp
                if self._max_capture_time is None
                else max(self._max_capture_time, sample.timestamp)
            )

    def _has_raw_samples(self) -> bool:
        return any(self._raw_batch.images.values()) or any(self._raw_batch.vectors.values())

    def _raw_sample_count(self) -> int:
        return sum(len(samples) for samples in self._raw_batch.images.values()) + sum(
            len(samples) for samples in self._raw_batch.vectors.values()
        )

    def _decode_raw_snapshots(self, snapshots: Iterable[RawCollectionSnapshot]) -> None:
        for snapshot in snapshots:
            self._merge_raw_batch(snapshot.decode_raw())

    def _image_skew_tolerance_sec(self) -> float:
        return image_skew_tolerance_sec(float(self._logger._fps))

    def _align_raw_records(self) -> None:
        self._records = []
        fields = tuple(_COLUMN_TO_FIELD[column] for column in self._schema.columns)
        frames, report = align_collection_samples(
            self._raw_batch,
            robot=self._logger._robot,
            camera_keys=tuple(self._schema.cameras.keys()),
            vector_fields=fields,
            fps=float(self._logger._fps),
            image_skew_sec=self._image_skew_tolerance_sec(),
        )
        self._alignment_report = report
        for issue in report.issues:
            self._add_issue(issue.code, issue.detail)
        for frame in frames:
            self._append_aligned_frame(frame)

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
        """Pack raw episode snapshots into a background SaveJob.

        Returns:
            A SaveJob carrying frozen raw snapshots for the parent logger's
            save worker, or None when no samples were recorded.
        """
        started = time.perf_counter()
        with self._state_lock:
            raw_snapshots = list(self._raw_snapshots)
            max_capture_time = self._max_capture_time
        has_raw = bool(raw_snapshots)
        if not has_raw:
            logger.warning(
                "collection episode produced 0 frames; nothing saved "
                "(no raw collection snapshots arrived)"
            )
            return None

        task = self._logger._task or ""
        dataset_dir = self._logger._collection_dataset_dir(task)
        episode_index = self._logger._next_collection_episode_index(dataset_dir)
        task_index = 0
        package_done = time.perf_counter()

        job = SaveJob(
            episode_index=episode_index,
            task=task,
            task_index=task_index,
            global_index=-1,
            steps=[],
            episode_meta=dict(self._logger._episode_meta),
            task_to_index={task: task_index},
            collection_payload=CollectionSavePayload(
                raw_snapshots=raw_snapshots,
                quality_issues=list(self._quality_issues),
                min_capture_time=self._min_capture_time,
                max_capture_time=max_capture_time,
            ),
            video_fps=float(self._logger._fps),
            dataset_dir=dataset_dir,
        )
        self._records = []
        self._quality_issues = []
        self._raw_batch = CollectionRawBatch()
        self._raw_snapshots = []
        self._alignment_report = None
        self._max_capture_time = None
        logger.info(
            "collection end_episode queued raw=%s raw_snapshots=%d: package=%.1fms "
            "total=%.1fms",
            has_raw,
            len(raw_snapshots),
            (package_done - started) * 1000.0,
            (package_done - started) * 1000.0,
        )
        return job

    def prepare_save_job(self, job: SaveJob) -> None:
        """Prepare a queued collection SaveJob on the save worker thread."""
        if job.collection_payload is None:
            return
        payload = job.collection_payload
        preparer = CollectionEpisodeWriter(self._logger, self._collection)
        preparer._quality_issues = list(payload.quality_issues)
        preparer._raw_batch = CollectionRawBatch(start_time=payload.min_capture_time)
        preparer._min_capture_time = payload.min_capture_time
        preparer._max_capture_time = payload.max_capture_time
        preparer._image_shapes = dict(self._image_shapes)
        preparer._decode_raw_snapshots(payload.raw_snapshots)
        if preparer._raw_batch.end_time is None:
            preparer._raw_batch.end_time = payload.max_capture_time
        preparer._build_prepared_save_job(job)
        self._image_shapes.update(preparer._image_shapes)

    def _build_prepared_save_job(self, job: SaveJob) -> None:
        started = time.perf_counter()
        self._align_raw_records()
        aligned = time.perf_counter()
        if not self._records:
            raise ValueError(self._alignment_failure_detail())
        if len(self._records) < self._schema.min_episode_frames:
            self._add_issue(
                "episode_too_short",
                f"episode has {len(self._records)} frames, expected at least "
                f"{self._schema.min_episode_frames}",
            )

        collection_fps = self._episode_average_fps()
        videos = self._snapshot_videos()
        video_prepared = time.perf_counter()
        n_frames = len(self._records)

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
        fps = float(self._logger._fps)
        dataset_dir = job.dataset_dir or self._logger._collection_dataset_dir(job.task)
        global_index = job.global_index
        if global_index < 0:
            global_index = self._logger._next_collection_global_index(dataset_dir, n_frames)
        columns.update(
            {
                "timestamp": [i / fps for i in range(n_frames)],
                "capture_time": [
                    float(record.timestamp if record.timestamp is not None else 0.0)
                    for record in self._records
                ],
                "frame_index": list(range(n_frames)),
                "episode_index": [job.episode_index] * n_frames,
                "index": list(range(global_index, global_index + n_frames)),
                "task_index": [job.task_index] * n_frames,
            }
        )

        row: dict[str, Any] = {
            "episode_index": job.episode_index,
            "tasks": [job.task],
            "length": n_frames,
            "quality": "red" if self._quality_issues else "green",
            "quality_issues": [dataclasses.asdict(issue) for issue in self._quality_issues],
        }
        if collection_fps is not None:
            row["collection_fps"] = collection_fps
        if self._alignment_report is not None:
            row.update(
                {
                    "alignment_fps": fps,
                    "alignment_grid_start": self._alignment_report.grid_start,
                    "alignment_grid_end": self._alignment_report.grid_end,
                    "alignment_image_skew_tolerance_sec": self._image_skew_tolerance_sec(),
                    "alignment_image_max_skew_sec": self._alignment_report.image_max_skew,
                    "alignment_image_stream_stats": (
                        self._alignment_report.image_stream_stats
                    ),
                }
            )
        if job.episode_meta:
            row.update(job.episode_meta)
        columns_packed = time.perf_counter()

        job.global_index = global_index
        job.collection_columns = columns
        job.collection_episode_row = row
        job.videos = videos
        job.video_fps = fps
        job.dataset_dir = dataset_dir
        job.collection_payload = None
        logger.info(
            "collection save worker prepared %d frames: align=%.1fms video=%.1fms "
            "columns=%.1fms total=%.1fms",
            n_frames,
            (aligned - started) * 1000.0,
            (video_prepared - aligned) * 1000.0,
            (columns_packed - video_prepared) * 1000.0,
            (columns_packed - started) * 1000.0,
        )

    def _alignment_failure_detail(self) -> str:
        issues = []
        if self._alignment_report is not None:
            issues = [
                f"{issue.code}: {issue.detail}" for issue in self._alignment_report.issues
            ]
        issue_text = "; ".join(issues) if issues else "none"
        return (
            "collection episode produced 0 frames; "
            f"alignment_issues={issue_text}; raw_streams={self._raw_stream_summary()}"
        )

    def _raw_stream_summary(self) -> str:
        parts = []
        for family, streams in (
            ("images", self._raw_batch.images),
            ("vectors", self._raw_batch.vectors),
        ):
            stream_parts = []
            for key in sorted(streams):
                samples = streams[key]
                if samples:
                    stream_parts.append(
                        f"{key}:{len(samples)}[{samples[0].timestamp:.6f},"
                        f"{samples[-1].timestamp:.6f}]"
                    )
                else:
                    stream_parts.append(f"{key}:0")
            parts.append(f"{family}=" + (",".join(stream_parts) if stream_parts else "none"))
        return " ".join(parts)

    def cancel_episode(self) -> None:
        """Discard the in-flight episode and drop all raw snapshot buffers."""
        self._records = []
        self._quality_issues = []
        self._raw_batch = CollectionRawBatch()
        self._raw_snapshots = []
        self._alignment_report = None
        self._max_capture_time = None

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
                if size is not None and rgb.shape[:2] != size:
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
            if video_key not in self._image_shapes:
                self._image_shapes[video_key] = (
                    size if size is not None else (arr.shape[0], arr.shape[1])
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
        stats = self._stats_from_episode_stats(dataset_dir)
        if stats is not None:
            with self._logger._meta_path("stats.json", dataset_dir).open("w") as f:
                json.dump(stats, f, indent=2)
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
