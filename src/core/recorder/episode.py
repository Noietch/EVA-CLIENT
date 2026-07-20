"""Episode logger — records each inference run as one LeRobot v2.1 episode.

One config -> one log dir -> one dataset. Each execution run appends one episode.

Two tables, deliberately different shapes, joined by ``absolute_step``:

1. Main table (``data/chunk-000/episode_xxxxxx.parquet``): LeRobot v2.1 standard,
   strictly 1 row per executed step. ``observation.<state>`` paired with the
   ``action`` actually sent to the robot, plus ``timestamp`` / ``frame_index`` /
   ``episode_index`` / ``task_index`` / ``index``. SFT training reads only this.
   Camera frames stream into per-camera mp4 via ``observation.images.<cam>``.

2. Debug sidecar (``meta/debug/episode_xxxxxx.parquet``): a LONG table, N rows
   per step allowed. Async/RTC overlapping chunks make one ``absolute_step`` carry
   several raw predictions (different ``chunk_index`` / ``inference_timestamp``),
   which is exactly what temporal-ensembling blends. Lives outside ``data/`` so
   training ignores it; joined back to the main table by ``absolute_step``.
"""

from __future__ import annotations

import bisect
import dataclasses
import datetime as _dt
import json
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.config import resolve_video_key
from core.recorder.collection_alignment import (
    align_collection_samples,
    image_skew_tolerance_sec,
)
from core.recorder.lerobot_meta import build_info, history_row
from core.types import (
    CollectionRawBatch,
    CollectionRawImage,
    CollectionRawSample,
    Observation,
    RolloutInterventionSegment,
)
from core.utils.images import resize_direct

if TYPE_CHECKING:
    from core.config import ConfigDict
    from core.types import RawCollectionSnapshot
    from robots.base import Robot

logger = logging.getLogger(__name__)

# Eval scoring metadata. Episode-level scalars (score / max_score / milestones / note /
# scored_at / duration_ms) live as top-level keys in meta/episodes.jsonl ONLY — they are
# not materialized into the parquet data table. They are declared in meta/info.json's
# "eval" section.
# episodes.jsonl keys the eval layer owns, surfaced in info.json's "eval" section.
_EVAL_META_KEYS = ("score", "max_score", "milestones", "note", "scored_at", "duration_ms")

# Observation vector fields (everything except timestamp/images) recorded per collection frame.
_COLLECTION_VECTOR_FIELDS = (
    "state_qpos",
    "state_eef",
    "action_qpos",
    "action_eef",
)
_INTERVENTION_VECTOR_FIELDS = (
    "state_qpos",
    "state_eef",
    "action_qpos",
    "action_eef",
)


def sanitize_path_component(text: str) -> str:
    """str -> filesystem-safe component (spaces/slashes collapsed, max 96 chars)."""
    sanitized = (text or "unset").strip().replace(" ", "_").replace("/", "-")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("._-")
    if sanitized == "":
        return "unset"
    return sanitized[:96]


@dataclasses.dataclass
class QualityIssue:
    """One quality-control flag raised while validating a collection episode.

    Fields:
        severity: Issue severity tag (e.g. "red").
        code: Machine-readable issue category (e.g. "missing_camera").
        detail: Human-readable description of the offending frame/field.
    """

    severity: str
    code: str
    detail: str


@dataclasses.dataclass
class SaveJob:
    """A self-contained snapshot of one finished episode, queued for background save.

    Carries everything the disk writers need (steps/debug rows or collection
    columns/videos, resolved indices, a frozen copy of the task table) so the
    worker thread never touches the live EpisodeLogger buffers. Non-collection
    mp4 writers are closed before SaveJob creation.
    """

    episode_index: int
    task: str
    task_index: int
    global_index: int
    steps: list[Observation]
    episode_meta: dict[str, Any]
    task_to_index: dict[str, int]
    collection_columns: dict[str, Any] | None = None
    collection_episode_row: dict[str, Any] | None = None
    collection_payload: Any | None = None
    raw_episode_payload: Any | None = None
    videos: dict[str, list[np.ndarray]] | None = None
    raw_video_samples: dict[str, list[CollectionRawSample]] | None = None
    video_fps: float | None = None
    dataset_dir: Path | None = None
    intervention_segments: list[RolloutInterventionSegment] = dataclasses.field(
        default_factory=list
    )
    reason: str = ""
    # When True the episodes.jsonl row + tasks.jsonl were already written before this
    # job's parquet/video flush; the worker must not append the row again.
    meta_written: bool = False
    status: str = "queued"  # queued -> saving -> saved | failed
    error: BaseException | None = None
    queued_wall_time: float = 0.0
    started_wall_time: float = 0.0
    finished_wall_time: float = 0.0


@dataclasses.dataclass(frozen=True)
class RawEpisodeFrameLabel:
    """Per-source-frame label copied into the fixed-clock episode rows."""

    timestamp: float
    intervention: bool
    segment_index: int


@dataclasses.dataclass
class RawEpisodeSavePayload:
    """Frozen raw eval/rollout episode input prepared at STOP time."""

    raw_batch: CollectionRawBatch
    frame_labels: list[RawEpisodeFrameLabel]
    raw_snapshots: list[RawEpisodeSnapshot] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RawEpisodeSnapshot:
    """One raw eval/rollout source snapshot plus the action executed at that tick."""

    snapshot: RawCollectionSnapshot
    action_qpos: np.ndarray | None
    intervention: bool = False
    segment_index: int = -1


class EpisodeLogger:
    """Records one inference run as a LeRobot v2.1 episode and appends it to the
    config-scoped dataset under ``log_dir``.

    Lifecycle::

        logger = EpisodeLogger(log_dir, robot, fps, dataset_keys)
        logger.start_episode(task="put the cup on the plate")
        # execution loop, every executed step:
        logger.record_step(obs, executed_action, timestamp=...)
        logger.set_episode_meta(score=3, milestones={...})   # eval only
        logger.end_episode()
        ...
        logger.finalize()   # stats.json + info.json once the dataset is done
    """

    def __init__(
        self,
        log_dir: str | Path,
        robot: Robot,
        fps: int,
        dataset_keys: ConfigDict,
        convert_bgr_to_rgb: bool = True,
        collection: ConfigDict | None = None,
        async_save: bool = False,
        save_queue_max: int = 15,
        eval_mode: bool = False,
        save_image_height: int | None = None,
        save_image_width: int | None = None,
        recording_space: str = "qpos",
        gripper_open: float | None = None,
        gripper_close: float | None = None,
        gripper_threshold: float | None = None,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._robot = robot
        self._fps = fps
        self._save_video = True
        self._keys = dataset_keys
        self._convert_bgr_to_rgb = convert_bgr_to_rgb
        self._collection = collection
        if recording_space not in {"qpos", "eef"}:
            raise ValueError(f"unsupported recording space: {recording_space}")
        self._recording_space = recording_space
        self._action_key = f"action.{recording_space}"
        self._gripper_open = gripper_open
        self._gripper_close = gripper_close
        self._gripper_threshold = gripper_threshold
        # Target saved-video resolution; both None keeps each camera's native size.
        self._save_image_height = save_image_height
        self._save_image_width = save_image_width
        # Eval datasets carry per-episode scoring (score/milestones/note/...) entered by
        # the operator. Those fields live in episodes.jsonl only; scoring waits for any
        # queued save to finish before patching the row.
        self._eval_mode = eval_mode
        from core.recorder.collection import CollectionEpisodeWriter  # local: break import cycle

        self._collection_writer = (
            CollectionEpisodeWriter(self, collection)
            if collection is not None and collection.schema.columns
            else None
        )

        self._camera_keys = [cam.observation_key for cam in robot.observation_schema.cameras]

        (self._log_dir / "meta").mkdir(parents=True, exist_ok=True)

        # dataset-wide running state (survives across episodes within one dataset)
        self._task_to_index: dict[str, int] = self._load_existing_tasks()
        self._episode_index = self._discover_next_episode_index()
        self._global_index = self._load_global_index()

        # per-episode buffers (reset by start_episode)
        self._active = False
        self._task: str | None = None
        self._steps: list[Observation] = []
        self._live_steps: list[Observation] = []
        self._episode_meta: dict[str, Any] = {}
        self._image_shape: tuple[int, int] | None = None  # (h, w) from first frame
        self._intervention_segments: list[RolloutInterventionSegment] = []
        self._raw_episode_batch = CollectionRawBatch()
        self._raw_episode_frame_labels: list[RawEpisodeFrameLabel] = []
        self._raw_episode_snapshots: list[RawEpisodeSnapshot] = []

        # async save queue: end_episode hands a frozen SaveJob snapshot to a background
        # worker so live capture/control can resume without waiting for alignment,
        # parquet, video, or meta writes.
        self._async_save = async_save
        self._save_queue_max = max(1, save_queue_max)
        # Per dataset_dir monotonic counters for collection episodes. Seeded from disk
        # on first use, then only advanced — allocation is decoupled from the async save
        # queue so a job that is mid-flush (already on disk but not yet popped off
        # _save_jobs) can never be counted twice and collide a later episode's index.
        self._collection_next_index: dict[Path, int] = {}
        self._collection_next_global: dict[Path, int] = {}
        self._save_jobs: list[SaveJob] = []
        self._save_worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._collection_history = (
            self._load_collection_history() if self._collection_writer is not None else []
        )
        self._completed_episodes = len(self._collection_history)
        self._save_durations: list[float] = []
        self._pending_episode_meta_by_clip: dict[str, dict[str, Any]] = {}

    # -- episode lifecycle ------------------------------------------------------

    def start_episode(self, task: str, collection_min_capture_time: float | None = None) -> None:
        """Begin a new episode.

        Args:
            task: Task label stored in the episode metadata.
            collection_min_capture_time: Collection-only source timestamp cutoff
                used to drop frames cached before START RECORD.
        """
        self._active = True
        self._task = task
        self._steps = []
        self._live_steps = []
        self._episode_meta = {}
        self._intervention_segments = []
        self._raw_episode_batch = CollectionRawBatch()
        self._raw_episode_frame_labels = []
        self._raw_episode_snapshots = []
        if self._collection_writer is not None:
            self._collection_writer.start_episode(collection_min_capture_time)
            return

    def record_step(
        self,
        obs: Observation,
        action: np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        """Main table: append obs[t]+action[t] and snapshot camera frames.

        Reads ``obs.state_qpos`` (raw joint state) and ``obs.state_eef`` (FK-derived
        end-effector state, or None in joint-only deployments) directly; the eef state
        is written as a second state column alongside the raw qpos. ``action`` is the
        action actually sent to the robot, stored on the buffered Observation.

        Args:
            obs: current-step observation with state_qpos / state_eef filled.
            action: [Da] executed action sent to the robot.
            timestamp: source capture time; defaults to wall clock when None.
        """
        if not self._active:
            return
        source_timestamp = self._record_step_timestamp(obs, timestamp)
        record = _copy_observation(obs)
        record.timestamp = source_timestamp
        record.action_qpos = np.asarray(action, dtype=np.float32).copy()
        self._steps.append(record)
        self._live_steps.append(_copy_observation(record))
        self._append_raw_episode_observation(
            self._raw_episode_batch,
            record,
            RawEpisodeFrameLabel(
                timestamp=source_timestamp,
                intervention=False,
                segment_index=-1,
            ),
            self._raw_episode_frame_labels,
        )

    def ingest_raw_episode_snapshot(
        self,
        snapshot: RawCollectionSnapshot,
        action: np.ndarray | None = None,
        state_qpos: np.ndarray | None = None,
        intervention: bool = False,
        segment_index: int = -1,
    ) -> None:
        """Store one pre-decode eval/rollout snapshot and executed action."""
        if not self._active:
            return
        action_qpos = None if action is None else np.asarray(action, dtype=np.float32).copy()
        self._raw_episode_snapshots.append(
            RawEpisodeSnapshot(
                snapshot=snapshot,
                action_qpos=action_qpos,
                intervention=intervention,
                segment_index=segment_index,
            )
        )
        if state_qpos is not None:
            self._live_steps.append(
                Observation(
                    images={},
                    state_qpos=np.asarray(state_qpos, dtype=np.float32).copy(),
                    action_qpos=None if action_qpos is None else action_qpos.copy(),
                    timestamp=float(snapshot.timestamp),
                )
            )

    def ingest_policy_action(
        self,
        action: np.ndarray,
        state_qpos: np.ndarray | None,
        timestamp: float,
        state_eef: np.ndarray | None = None,
        action_eef: np.ndarray | None = None,
    ) -> None:
        """Store a policy action independently from raw sensor snapshots."""
        if not self._active:
            return
        action_value = np.asarray(action, dtype=np.float32).copy()
        timestamp = float(timestamp)
        if state_qpos is None:
            self._raw_episode_batch.vectors.setdefault("action_qpos", []).append(
                CollectionRawSample(timestamp, action_value)
            )
            if self._recording_space == "eef" and action_eef is not None:
                self._raw_episode_batch.vectors.setdefault("action_eef", []).append(
                    CollectionRawSample(timestamp, np.asarray(action_eef, dtype=np.float32).copy())
                )
            if self._recording_space == "eef" and state_eef is not None:
                self._raw_episode_batch.vectors.setdefault("state_eef", []).append(
                    CollectionRawSample(timestamp, np.asarray(state_eef, dtype=np.float32).copy())
                )
            self._raw_episode_frame_labels.append(
                RawEpisodeFrameLabel(timestamp, False, -1)
            )
            self._extend_raw_episode_time_bounds(
                self._raw_episode_batch,
                self._raw_episode_batch.vectors["action_qpos"][-1:],
            )
            return
        frame = Observation(
            images={},
            state_qpos=np.asarray(state_qpos, dtype=np.float32).copy(),
            action_qpos=action_value,
            timestamp=timestamp,
        )
        if self._recording_space == "eef":
            frame.state_eef = (
                None if state_eef is None else np.asarray(state_eef, dtype=np.float32).copy()
            )
            frame.action_eef = (
                None if action_eef is None else np.asarray(action_eef, dtype=np.float32).copy()
            )
        self._append_raw_episode_observation(
            self._raw_episode_batch,
            frame,
            RawEpisodeFrameLabel(timestamp, False, -1),
            self._raw_episode_frame_labels,
        )

    def ingest_collection_snapshot(self, snapshot: RawCollectionSnapshot) -> None:
        """Hand one pre-decode snapshot to the collection writer (O(1))."""
        if not self._active or self._collection_writer is None:
            return
        self._collection_writer.ingest(snapshot)

    def set_episode_meta(self, **fields: Any) -> None:
        """Attach eval metadata (score, milestones, ...) to the current episode."""
        self._episode_meta.update(fields)

    def set_rollout_intervention_segments(self, segments: list[RolloutInterventionSegment]) -> None:
        """Attach accepted rollout intervention segments to the active episode."""
        self._intervention_segments = [
            _copy_intervention_segment(segment) for segment in segments if segment.frames
        ]

    def patch_episode_meta(self, episode_index: int, **fields: Any) -> bool:
        """Post-hoc: merge fields into an already-written episodes.jsonl row.

        Eval scores are entered after the episode's meta row is on disk, so this rewrites
        the matching row in place. Scores live ONLY in meta/episodes.jsonl (never in the
        parquet data table). Returns False when the episode index is not found.
        """
        path = self._meta_path("episodes.jsonl")
        rows = _read_jsonl(path)
        patched = False
        for row in rows:
            if int(row.get("episode_index", -1)) == episode_index:
                row.update(fields)
                patched = True
                break
        if patched:
            with path.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if self._eval_mode and ("score" in fields or "max_score" in fields):
                self._write_info_json()
        return patched

    def mark_collection_qc(
        self,
        task: str,
        episode_index: int,
        verdict: str,
        note: str,
    ) -> bool:
        """Write QC metadata for one fully saved collection episode.

        Args:
            task: Collection task whose task-specific dataset contains the episode.
            episode_index: Integer episode id in that dataset.
            verdict: ``"pass"``, ``"fail"``, or empty for a note-only update.
            note: QC note to replace the existing value.

        Returns:
            True when the saved episode row was updated, otherwise False.
        """
        if self._collection_writer is None:
            return False
        dataset_dir = self._collection_dataset_dir(task)
        path = self._meta_path("episodes.jsonl", dataset_dir)
        with self._lock:
            if any(
                job.episode_index == episode_index
                and (job.dataset_dir or self._log_dir) == dataset_dir
                for job in self._save_jobs
            ):
                return False
            rows = _read_jsonl(path)
            for row in rows:
                if int(row.get("episode_index", -1)) != episode_index:
                    continue
                if verdict:
                    row["qc_verdict"] = verdict
                row["qc_note"] = note
                replacement = path.with_suffix(f"{path.suffix}.qc.tmp")
                with replacement.open("w") as f:
                    for existing in rows:
                        f.write(json.dumps(existing, ensure_ascii=False) + "\n")
                replacement.replace(path)
                return True
        return False

    def patch_episode_meta_by_clip(self, clip_id: str, **fields: Any) -> int | None:
        """Merge eval metadata into an existing or still-saving episode for clip_id.

        Returns:
            Episode index when the clip is known, otherwise None.
        """
        if not clip_id:
            return None
        episode_index = self._episode_index_for_clip_id(clip_id)
        if episode_index is not None:
            self.patch_episode_meta(episode_index, **fields)
            return episode_index
        with self._lock:
            job = self._queued_job_for_clip_id_locked(clip_id)
            if job is None:
                return None
            job.episode_meta.update(fields)
            if job.collection_episode_row is not None:
                job.collection_episode_row.update(fields)
            self._pending_episode_meta_by_clip.setdefault(clip_id, {}).update(fields)
            return job.episode_index

    @property
    def current_episode_index(self) -> int:
        """Index the next end_episode() will assign (i.e. the in-flight episode)."""
        return self._episode_index

    @property
    def is_collection_enabled(self) -> bool:
        """True when this logger writes teleop collection episodes (vs. inference runs)."""
        return self._collection_writer is not None

    @property
    def has_active_episode(self) -> bool:
        """True while an episode is open (between start_episode and end/cancel)."""
        return self._active

    @property
    def active_frame_count(self) -> int:
        """Frames seen so far in the in-flight episode; 0 when none is active."""
        if not self._active:
            return 0
        if self._collection_writer is not None:
            return self._collection_writer.frame_counts()["received"]
        return max(
            len(self._live_steps),
            len(self._raw_episode_frame_labels),
            len(self._raw_episode_snapshots),
        )

    def end_episode(self) -> bool:
        """Finish the current episode.

        When async_save is on, the recorded buffers are snapshotted into a SaveJob
        and alignment/parquet/video/meta writing happens on a background worker so
        the caller can immediately start the next episode. Otherwise the job is
        written synchronously.

        Returns:
            True when the episode was saved/queued, False when it held no frames
            and nothing was written (lets the caller surface a banner).
        """
        if not self._active:
            return False
        if self._collection_writer is not None:
            job = self._collection_writer.end_episode()
            self._active = False
            self._task = None
            self._episode_meta = {}
            if job is None:
                return False
            if self._async_save:
                job.queued_wall_time = time.time()
                self._enqueue_save_job(job)
                return True
            job.queued_wall_time = time.time()
            job.started_wall_time = job.queued_wall_time
            self._write_collection_job(job)
            job.status = "saved"
            job.finished_wall_time = time.time()
            self._remember_finished_job(job)
            self._completed_episodes += 1
            return True
        if not self._has_raw_episode_samples():
            self._active = False
            self._task = None
            self._intervention_segments = []
            self._raw_episode_batch = CollectionRawBatch()
            self._raw_episode_frame_labels = []
            self._raw_episode_snapshots = []
            return False

        episode_index = self._episode_index
        task = self._task or ""
        task_index = self._resolve_task_index(task)
        episode_meta = dict(self._episode_meta)
        intervention_segments = [
            _copy_intervention_segment(segment)
            for segment in self._intervention_segments
            if segment.frames
        ]
        if intervention_segments:
            episode_meta["has_intervention"] = True
            episode_meta["intervention_segments"] = len(intervention_segments)
            episode_meta["intervention_frames"] = sum(
                len(segment.frames) for segment in intervention_segments
            )

        # Eval re-run: if this (prompt, trial) cell already has a recorded episode, reuse
        # its index and overwrite that episode's data IN PLACE (parquet + video + meta row)
        # instead of appending a new episode. Otherwise advance the dataset-wide counters so
        # the next episode can start while this one is still being written in the background.
        reuse_index = self._eval_existing_episode_index(episode_meta) if self._eval_mode else None
        if reuse_index is not None:
            episode_index = reuse_index
            global_index = self._existing_episode_global_start(episode_index)
            if global_index is None:
                global_index = -1
        else:
            global_index = -1
            self._episode_index += 1
        self._active = False
        raw_batch = self._copy_raw_episode_batch(self._raw_episode_batch)
        frame_labels = list(self._raw_episode_frame_labels)
        raw_snapshots = [
            RawEpisodeSnapshot(
                snapshot=sample.snapshot,
                action_qpos=None
                if sample.action_qpos is None
                else sample.action_qpos.copy(),
                intervention=sample.intervention,
                segment_index=sample.segment_index,
            )
            for sample in self._raw_episode_snapshots
        ]
        self._append_intervention_segments_to_raw_episode(
            raw_batch,
            frame_labels,
            intervention_segments,
        )

        episode_meta = dict(episode_meta)
        episode_meta.setdefault("recorded_at", _dt.datetime.now().isoformat(timespec="seconds"))
        job = SaveJob(
            episode_index=episode_index,
            task=task,
            task_index=task_index,
            global_index=global_index,
            steps=[],
            episode_meta=episode_meta,
            task_to_index=dict(self._task_to_index),
            raw_episode_payload=RawEpisodeSavePayload(raw_batch, frame_labels, raw_snapshots),
            video_fps=float(self._fps),
            queued_wall_time=time.time(),
        )
        self._steps = []
        self._live_steps = []
        self._episode_meta = {}
        self._intervention_segments = []
        self._raw_episode_batch = CollectionRawBatch()
        self._raw_episode_frame_labels = []
        self._raw_episode_snapshots = []
        self._task = None
        if self._async_save:
            self._enqueue_save_job(job)
            return True
        job.started_wall_time = job.queued_wall_time
        self._write_raw_episode_job(job)
        job.status = "saved"
        job.finished_wall_time = time.time()
        self._completed_episodes += 1
        self._remember_finished_job(job)
        return True

    def cancel_episode(self, reason: str = "cancelled") -> None:
        """Discard the in-flight episode: drop buffers, close + delete partial mp4,
        and roll back the episode index so the next start reuses it. Nothing is
        written to the dataset."""
        if not self._active:
            return
        if self._collection_writer is not None:
            self._collection_writer.cancel_episode()
            self._steps = []
            self._live_steps = []
            self._episode_meta = {}
            self._intervention_segments = []
            self._raw_episode_batch = CollectionRawBatch()
            self._raw_episode_frame_labels = []
            self._raw_episode_snapshots = []
            self._active = False
            self._task = None
            logger.info("Cancelled episode %d (%s)", self._episode_index, reason)
            return
        self._steps = []
        self._live_steps = []
        self._episode_meta = {}
        self._intervention_segments = []
        self._raw_episode_batch = CollectionRawBatch()
        self._raw_episode_frame_labels = []
        self._raw_episode_snapshots = []
        self._active = False
        self._task = None
        logger.info("Cancelled episode %d (%s)", self._episode_index, reason)

    def is_queue_full(self) -> bool:
        """True when the async save queue has no free slot (backpressure signal)."""
        with self._lock:
            return len(self._save_jobs) >= self._save_queue_max

    # -- async save queue -------------------------------------------------------

    def _enqueue_save_job(self, job: SaveJob) -> None:
        with self._lock:
            self._save_jobs.append(job)
            worker_alive = self._save_worker is not None and self._save_worker.is_alive()
        if not worker_alive:
            self._start_save_worker()

    def _start_save_worker(self) -> None:
        with self._lock:
            if self._save_worker is not None and self._save_worker.is_alive():
                return
            worker = threading.Thread(
                target=self._save_queue_worker, name="episode_save_worker", daemon=True
            )
            self._save_worker = worker
        worker.start()

    def _save_queue_worker(self) -> None:
        while True:
            with self._lock:
                job = next((j for j in self._save_jobs if j.status == "queued"), None)
                if job is None:
                    self._save_worker = None
                    return
                job.status = "saving"
                job.started_wall_time = time.time()
            try:
                self._write_job(job)
            except BaseException as exc:  # keep the worker alive across one bad job
                logger.exception("Failed saving episode %d", job.episode_index)
                with self._lock:
                    job.status = "failed"
                    job.error = exc
                    job.finished_wall_time = time.time()
            else:
                with self._lock:
                    job.status = "saved"
                    job.finished_wall_time = time.time()
                    self._completed_episodes += 1
                    self._remember_finished_job(job)
                    self._save_jobs = [j for j in self._save_jobs if j is not job]

    def _write_job(self, job: SaveJob) -> None:
        """Background-thread disk write from a frozen snapshot (no live buffers)."""
        if job.collection_columns is not None or job.collection_payload is not None:
            self._write_collection_job(job)
            return
        if job.raw_episode_payload is not None:
            self._write_raw_episode_job(job)
            return
        raise ValueError("save job missing raw payload")

    def _record_step_timestamp(self, obs: Observation, timestamp: float | None) -> float:
        if timestamp is not None:
            return float(timestamp)
        if obs.timestamp is not None:
            return float(obs.timestamp)
        if self._raw_episode_frame_labels:
            return self._raw_episode_frame_labels[-1].timestamp + 1.0 / float(self._fps)
        return 0.0

    def _append_raw_episode_observation(
        self,
        batch: CollectionRawBatch,
        frame: Observation,
        label: RawEpisodeFrameLabel,
        labels: list[RawEpisodeFrameLabel],
    ) -> None:
        timestamp = float(label.timestamp)
        for key, image in frame.images.items():
            batch.images.setdefault(key, []).append(
                CollectionRawSample(timestamp, np.asarray(image).copy())
            )
        for field in _COLLECTION_VECTOR_FIELDS:
            value = getattr(frame, field)
            if value is None:
                continue
            batch.vectors.setdefault(field, []).append(
                CollectionRawSample(
                    timestamp,
                    np.asarray(value, dtype=np.float32).copy(),
                )
            )
        if batch.start_time is None or timestamp < batch.start_time:
            batch.start_time = timestamp
        if batch.end_time is None or timestamp > batch.end_time:
            batch.end_time = timestamp
        labels.append(label)

    def _merge_raw_episode_batch(
        self,
        target: CollectionRawBatch,
        batch: CollectionRawBatch,
    ) -> None:
        if batch.start_time is not None:
            target.start_time = (
                batch.start_time
                if target.start_time is None
                else max(target.start_time, batch.start_time)
            )
        if batch.end_time is not None:
            target.end_time = (
                batch.end_time if target.end_time is None else max(target.end_time, batch.end_time)
            )
        for key, samples in batch.images.items():
            target.images.setdefault(key, []).extend(samples)
            self._extend_raw_episode_time_bounds(target, samples)
        for key, samples in batch.vectors.items():
            target.vectors.setdefault(key, []).extend(samples)
            self._extend_raw_episode_time_bounds(target, samples)

    def _extend_raw_episode_time_bounds(
        self,
        batch: CollectionRawBatch,
        samples: list[CollectionRawSample],
    ) -> None:
        for sample in samples:
            batch.start_time = (
                sample.timestamp
                if batch.start_time is None
                else min(batch.start_time, sample.timestamp)
            )
            batch.end_time = (
                sample.timestamp
                if batch.end_time is None
                else max(batch.end_time, sample.timestamp)
            )

    def _decode_raw_episode_snapshots(self, payload: RawEpisodeSavePayload) -> None:
        for raw in payload.raw_snapshots:
            batch = raw.snapshot.decode_raw()
            if raw.action_qpos is None:
                # Rollout actions/state are captured as a separate timestamped
                # stream. Raw snapshots contribute images only; mixing their
                # per-arm vector streams would shrink the common coverage window
                # to the latest arm cursor and drop rollout boundaries.
                batch.vectors.clear()
            if raw.action_qpos is not None:
                batch.vectors.setdefault("action_qpos", []).append(
                    CollectionRawSample(
                        raw.snapshot.timestamp,
                        raw.action_qpos.copy(),
                    )
                )
            self._merge_raw_episode_batch(payload.raw_batch, batch)
            payload.frame_labels.append(
                RawEpisodeFrameLabel(
                    timestamp=raw.snapshot.timestamp,
                    intervention=raw.intervention,
                    segment_index=raw.segment_index,
                )
            )
        payload.raw_snapshots = []

    def _append_intervention_segments_to_raw_episode(
        self,
        batch: CollectionRawBatch,
        labels: list[RawEpisodeFrameLabel],
        segments: list[RolloutInterventionSegment],
    ) -> None:
        for segment in segments:
            for frame in segment.frames:
                timestamp = (
                    float(frame.timestamp)
                    if frame.timestamp is not None
                    else self._next_raw_episode_timestamp(labels)
                )
                record = _copy_observation(frame)
                record.timestamp = timestamp
                if self._recording_space == "qpos":
                    record.state_eef = None
                    record.action_eef = None
                self._append_raw_episode_observation(
                    batch,
                    record,
                    RawEpisodeFrameLabel(
                        timestamp=timestamp,
                        intervention=True,
                        segment_index=segment.segment_index,
                    ),
                    labels,
                )

    def _next_raw_episode_timestamp(self, labels: list[RawEpisodeFrameLabel]) -> float:
        if labels:
            return labels[-1].timestamp + 1.0 / float(self._fps)
        return time.time()

    def _has_raw_episode_samples(self) -> bool:
        return (
            any(self._raw_episode_batch.vectors.values())
            or bool(self._raw_episode_snapshots)
            or any(segment.frames for segment in self._intervention_segments)
        )

    def _copy_raw_episode_batch(self, batch: CollectionRawBatch) -> CollectionRawBatch:
        return CollectionRawBatch(
            images={
                key: [
                    CollectionRawSample(sample.timestamp, _copy_raw_value(sample.value))
                    for sample in samples
                ]
                for key, samples in batch.images.items()
            },
            vectors={
                key: [
                    CollectionRawSample(sample.timestamp, _copy_raw_value(sample.value))
                    for sample in samples
                ]
                for key, samples in batch.vectors.items()
            },
            start_time=batch.start_time,
            end_time=batch.end_time,
        )

    def _raw_episode_vector_fields(self, batch: CollectionRawBatch) -> tuple[str, ...]:
        fields = []
        for field in _COLLECTION_VECTOR_FIELDS:
            if field in batch.vectors or any(key.startswith(f"{field}:") for key in batch.vectors):
                fields.append(field)
        if "state_qpos" not in fields:
            raise ValueError("raw episode missing state_qpos samples")
        if "action_qpos" not in fields:
            raise ValueError("raw episode missing action_qpos samples")
        return tuple(fields)

    def _raw_episode_image_skew_tolerance_sec(self) -> float:
        return image_skew_tolerance_sec(float(self._fps))

    def _write_raw_episode_job(self, job: SaveJob) -> None:
        if job.raw_episode_payload is not None:
            self._prepare_raw_episode_job(job)
        self._apply_pending_episode_meta_for_job(job)
        columns = job.collection_columns
        row = job.collection_episode_row
        if columns is None or row is None:
            raise ValueError("raw episode save job missing columns or episode row")
        path = self._parquet_path(job.episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), str(path))
        try:
            self._write_raw_sample_videos(
                job.episode_index,
                job.raw_video_samples,
                job.video_fps,
            )
        finally:
            job.raw_video_samples = None
        with self._lock:
            if self._eval_mode:
                self._upsert_episode_dict_row_locked(row)
            else:
                self._append_episode_dict_row_locked(row)
            self._write_tasks_jsonl_from(job.task_to_index)
        self._write_info_json()
        self._write_stats_json()
        self._clear_pending_episode_meta_for_job(job)

    def _exclude_raw_episode_ranges(
        self,
        frames: list[Observation],
        labels: list[RawEpisodeFrameLabel],
        ranges: list[dict[str, Any]],
        image_samples: dict[str, list[CollectionRawSample]],
    ) -> tuple[
        list[Observation],
        list[RawEpisodeFrameLabel],
        list[dict[str, Any]],
        dict[str, list[CollectionRawSample]],
    ]:
        kept_frames: list[Observation] = []
        kept_labels: list[RawEpisodeFrameLabel] = []
        kept_image_samples = {key: [] for key in image_samples}
        removed_counts = [0] * len(ranges)
        for frame_index, (frame, label) in enumerate(zip(frames, labels, strict=True)):
            assert frame.timestamp is not None
            timestamp = float(frame.timestamp)
            excluded_index = next(
                (
                    index
                    for index, interval in enumerate(ranges)
                    if float(interval["start_time"]) < timestamp
                    < float(interval["end_time"])
                ),
                None,
            )
            if excluded_index is not None:
                removed_counts[excluded_index] += 1
                continue
            kept_frames.append(frame)
            kept_labels.append(label)
            for key, samples in image_samples.items():
                kept_image_samples[key].append(samples[frame_index])
        details = []
        for interval, removed_frames in zip(ranges, removed_counts, strict=True):
            start_time = float(interval["start_time"])
            end_time = float(interval["end_time"])
            details.append(
                {
                    "reason": str(interval["reason"]),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_sec": round(end_time - start_time, 6),
                    "removed_frames": removed_frames,
                }
            )
        return kept_frames, kept_labels, details, kept_image_samples

    def _prepare_raw_episode_job(self, job: SaveJob) -> None:
        payload = job.raw_episode_payload
        if payload is None:
            return
        raw_snapshot_count = len(payload.raw_snapshots)
        self._decode_raw_episode_snapshots(payload)
        fields = self._raw_episode_vector_fields(payload.raw_batch)
        camera_keys = tuple(payload.raw_batch.images.keys())
        aligned_image_samples: dict[str, list[CollectionRawSample]] = {}
        frames, report = align_collection_samples(
            payload.raw_batch,
            robot=self._robot,
            camera_keys=camera_keys,
            vector_fields=fields,
            fps=float(self._fps),
            image_skew_sec=self._raw_episode_image_skew_tolerance_sec(),
            deferred_images=aligned_image_samples,
        )
        logger.info(
            "[ROLLOUT_SAVE] episode=%d raw_snapshots=%d labels=%d vectors=%s frames=%d "
            "grid_start=%.6f grid_end=%.6f",
            job.episode_index,
            raw_snapshot_count,
            len(payload.frame_labels),
            {key: len(samples) for key, samples in payload.raw_batch.vectors.items()},
            len(frames),
            report.grid_start,
            report.grid_end,
        )
        if not frames:
            raise ValueError("raw episode produced 0 frames")
        labels = self._labels_for_aligned_frames(frames, payload.frame_labels)
        excluded_ranges = list(job.episode_meta.pop("excluded_ranges", []))
        excluded_details: list[dict[str, Any]] = []
        if excluded_ranges:
            original_frame_count = len(frames)
            frames, labels, excluded_details, aligned_image_samples = (
                self._exclude_raw_episode_ranges(
                    frames,
                    labels,
                    excluded_ranges,
                    aligned_image_samples,
                )
            )
            logger.info(
                "[ROLLOUT_SAVE] episode=%d excluded_frames=%d ranges=%s",
                job.episode_index,
                original_frame_count - len(frames),
                excluded_details,
            )
            if not frames:
                raise ValueError("raw episode exclusions removed every aligned frame")
        issues = [QualityIssue("red", issue.code, issue.detail) for issue in report.issues]
        self._quantize_action_grippers(frames)
        logger.info(
            "[ROLLOUT_SAVE] episode=%d aligned_policy=%d aligned_intervention=%d "
            "first_capture=%.6f last_capture=%.6f",
            job.episode_index,
            sum(not label.intervention for label in labels),
            sum(label.intervention for label in labels),
            float(frames[0].timestamp or 0.0),
            float(frames[-1].timestamp or 0.0),
        )
        n_frames = len(frames)
        global_index = job.global_index
        if global_index < 0:
            global_index = self._next_episode_global_index(n_frames)
        fps = float(self._fps)
        action_field = "action_eef" if self._recording_space == "eef" else "action_qpos"
        action_values = [getattr(frame, action_field) for frame in frames]
        if any(action is None for action in action_values):
            raise ValueError(f"raw episode missing {action_field} values")
        columns: dict[str, Any] = {
            self._keys.state_key: [
                np.asarray(frame.state_qpos, dtype=np.float32).tolist() for frame in frames
            ],
            self._action_key: [
                np.asarray(action, dtype=np.float32).tolist() for action in action_values
            ],
            "timestamp": [i / fps for i in range(n_frames)],
            "capture_time": [
                float(frame.timestamp if frame.timestamp is not None else 0.0) for frame in frames
            ],
            "frame_index": list(range(n_frames)),
            "episode_index": [job.episode_index] * n_frames,
            "index": list(range(global_index, global_index + n_frames)),
            "task_index": [job.task_index] * n_frames,
            "intervention": [label.intervention for label in labels],
            "intervention_segment_index": [label.segment_index for label in labels],
            "control_source": [
                "intervention" if label.intervention else "policy" for label in labels
            ],
        }
        if any(frame.state_eef is not None for frame in frames):
            columns[self._keys.eef_key] = [
                None
                if frame.state_eef is None
                else np.asarray(frame.state_eef, dtype=np.float32).tolist()
                for frame in frames
            ]
        raw_video_samples = self._raw_video_samples(aligned_image_samples, camera_keys)
        if self._save_video and raw_video_samples:
            for video_key, samples in raw_video_samples.items():
                if len(samples) != n_frames:
                    issues.append(
                        QualityIssue(
                            "red",
                            "frame_count_mismatch",
                            f"video {video_key} has {len(samples)} frames but "
                            f"data table has {n_frames} rows",
                        )
                    )
        row: dict[str, Any] = {
            "episode_index": job.episode_index,
            "tasks": [job.task],
            "length": n_frames,
            "quality": "red" if issues else "green",
            "quality_issues": [dataclasses.asdict(issue) for issue in issues],
            "alignment_fps": fps,
            "alignment_grid_start": report.grid_start,
            "alignment_grid_end": report.grid_end,
            "alignment_image_skew_tolerance_sec": (self._raw_episode_image_skew_tolerance_sec()),
            "alignment_image_max_skew_sec": report.image_max_skew,
            "alignment_image_stream_stats": report.image_stream_stats,
        }
        episode_fps = self._raw_episode_average_fps(payload.raw_batch)
        if episode_fps is not None:
            row["episode_fps"] = episode_fps
        if job.episode_meta:
            row.update(job.episode_meta)
        if excluded_details:
            row["excluded_ranges"] = excluded_details
            row["excluded_frames"] = sum(
                int(interval["removed_frames"]) for interval in excluded_details
            )
        intervention_ranges = []
        range_start = None
        range_segment = -1
        for frame_index, label in enumerate([*labels, None]):
            if label is not None and label.intervention:
                if range_start is None:
                    range_start = frame_index
                    range_segment = label.segment_index
                elif label.segment_index != range_segment:
                    intervention_ranges.append(
                        {
                            "segment_index": range_segment,
                            "start_frame": range_start,
                            "end_frame": frame_index - 1,
                        }
                    )
                    range_start = frame_index
                    range_segment = label.segment_index
                continue
            if range_start is not None:
                intervention_ranges.append(
                    {
                        "segment_index": range_segment,
                        "start_frame": range_start,
                        "end_frame": frame_index - 1,
                    }
                )
                range_start = None
        row["has_intervention"] = bool(intervention_ranges)
        row["intervention_segments"] = len(intervention_ranges)
        row["intervention_frames"] = sum(label.intervention for label in labels)
        row["intervention_ranges"] = intervention_ranges
        job.global_index = global_index
        job.collection_columns = columns
        job.collection_episode_row = row
        job.raw_video_samples = raw_video_samples
        job.video_fps = fps
        job.raw_episode_payload = None

    def _quantize_action_grippers(self, frames: list[Observation]) -> None:
        """Map recorded qpos gripper actions to the configured open/close values."""
        if self._gripper_open is None or self._gripper_close is None:
            return
        threshold = self._gripper_threshold
        if threshold is None:
            threshold = (self._gripper_open + self._gripper_close) * 0.5
        high_is_open = self._gripper_open >= self._gripper_close
        for frame in frames:
            if frame.action_qpos is None:
                continue
            action = np.asarray(frame.action_qpos, dtype=np.float32).copy()
            for index in self._robot.gripper_indices:
                if index >= action.size:
                    continue
                is_open = action[index] >= threshold if high_is_open else action[index] <= threshold
                action[index] = self._gripper_open if is_open else self._gripper_close
            frame.action_qpos = action

    def _episode_index_for_clip_id(self, clip_id: str) -> int | None:
        match = None
        for row in _read_jsonl(self._meta_path("episodes.jsonl")):
            if row.get("clip_id") == clip_id and row.get("episode_index") is not None:
                match = int(row["episode_index"])
        return match

    def _queued_job_for_clip_id_locked(self, clip_id: str) -> SaveJob | None:
        for job in reversed(self._save_jobs):
            row = job.collection_episode_row or {}
            if job.episode_meta.get("clip_id") == clip_id or row.get("clip_id") == clip_id:
                return job
        return None

    def _apply_pending_episode_meta_for_job(self, job: SaveJob) -> None:
        clip_id = job.episode_meta.get("clip_id")
        if clip_id is None and job.collection_episode_row is not None:
            clip_id = job.collection_episode_row.get("clip_id")
        if not clip_id:
            return
        with self._lock:
            pending = dict(self._pending_episode_meta_by_clip.get(str(clip_id), {}))
        if not pending:
            return
        job.episode_meta.update(pending)
        if job.collection_episode_row is not None:
            job.collection_episode_row.update(pending)

    def _clear_pending_episode_meta_for_job(self, job: SaveJob) -> None:
        clip_id = job.episode_meta.get("clip_id")
        if clip_id is None and job.collection_episode_row is not None:
            clip_id = job.collection_episode_row.get("clip_id")
        if not clip_id:
            return
        with self._lock:
            self._pending_episode_meta_by_clip.pop(str(clip_id), None)

    def _labels_for_aligned_frames(
        self,
        frames: list[Observation],
        source_labels: list[RawEpisodeFrameLabel],
    ) -> list[RawEpisodeFrameLabel]:
        if not source_labels:
            return [
                RawEpisodeFrameLabel(
                    timestamp=float(frame.timestamp or 0.0),
                    intervention=False,
                    segment_index=-1,
                )
                for frame in frames
            ]
        labels = sorted(source_labels, key=lambda label: label.timestamp)
        times = [label.timestamp for label in labels]
        aligned = []
        for frame in frames:
            timestamp = float(frame.timestamp if frame.timestamp is not None else 0.0)
            index = bisect.bisect_right(times, timestamp) - 1
            if index < 0:
                index = 0
            label = labels[index]
            aligned.append(
                RawEpisodeFrameLabel(
                    timestamp=timestamp,
                    intervention=label.intervention,
                    segment_index=label.segment_index,
                )
            )
        return aligned

    def _raw_episode_average_fps(self, batch: CollectionRawBatch) -> float | None:
        samples = batch.vectors.get("state_qpos") or batch.vectors.get("action_qpos")
        if samples is None or len(samples) < 2:
            return None
        timestamps = sorted(float(sample.timestamp) for sample in samples)
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0.0:
            return None
        return (len(timestamps) - 1) / duration

    def _raw_video_samples(
        self,
        aligned_images: dict[str, list[CollectionRawSample]],
        camera_keys: tuple[str, ...],
    ) -> dict[str, list[CollectionRawSample]] | None:
        if not self._save_video:
            return None
        videos: dict[str, list[CollectionRawSample]] = {}
        for cam_key in camera_keys:
            video_key = resolve_video_key(self._keys, cam_key)
            if video_key is None:
                continue
            samples = aligned_images.get(cam_key, [])
            if samples:
                videos[video_key] = samples
        return videos or None

    def _write_raw_sample_videos(
        self,
        episode_index: int,
        videos: dict[str, list[CollectionRawSample]] | None,
        fps: float | None,
    ) -> None:
        if not videos:
            return
        video_fps = float(fps or self._fps)
        size = self._save_size()
        for video_key, samples in videos.items():
            path = (
                self._log_dir
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(path),
                fps=video_fps,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=["-preset", "ultrafast", "-movflags", "+faststart"],
            )
            active_value: Any | None = None
            active_frame: np.ndarray | None = None
            try:
                for sample in samples:
                    value = sample.value
                    if value is not active_value or active_frame is None:
                        if isinstance(active_value, CollectionRawImage):
                            active_value.release_decoded()
                        image = value.decode() if isinstance(value, CollectionRawImage) else value
                        rgb = _to_rgb_uint8(np.asarray(image), self._convert_bgr_to_rgb)
                        if size is not None and rgb.shape[:2] != size:
                            rgb = resize_direct(rgb, size[0], size[1])
                        active_value = value
                        active_frame = np.ascontiguousarray(rgb)
                        if self._image_shape is None:
                            self._image_shape = (active_frame.shape[0], active_frame.shape[1])
                    writer.append_data(active_frame)
            finally:
                if isinstance(active_value, CollectionRawImage):
                    active_value.release_decoded()
                writer.close()

    def _next_episode_global_index(self, n_frames: int) -> int:
        with self._lock:
            index = self._global_index
            self._global_index += n_frames
            return index

    def _write_collection_job(self, job: SaveJob) -> None:
        if job.collection_payload is not None:
            if self._collection_writer is None:
                raise ValueError("collection save job has payload but no collection writer")
            self._collection_writer.prepare_save_job(job)
        columns = job.collection_columns
        row = job.collection_episode_row
        if columns is None or row is None:
            raise ValueError("collection save job missing columns or episode row")
        dataset_dir = job.dataset_dir or self._log_dir
        path = self._parquet_path(job.episode_index, dataset_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), str(path))
        self._write_videos(job.episode_index, job.videos, job.video_fps, dataset_dir)
        episode_stats = {key: _vector_episode_stats(vals) for key, vals in columns.items()}
        if job.videos:
            for video_key, frames in job.videos.items():
                episode_stats[video_key] = _image_episode_stats(frames)
        with self._lock:
            self._append_episode_dict_row_locked(row, dataset_dir)
            self._append_episode_stats_locked(job.episode_index, episode_stats, dataset_dir)
            self._write_tasks_jsonl_from(job.task_to_index, dataset_dir)
        if self._collection_writer is not None:
            self._collection_writer.finalize(dataset_dir)

    def _write_videos(
        self,
        episode_index: int,
        videos: dict[str, list[np.ndarray]] | None,
        fps: float | None,
        dataset_dir: Path | None = None,
    ) -> None:
        if not videos:
            return
        root = dataset_dir or self._log_dir
        video_fps = float(fps or self._fps)
        for video_key, frames in videos.items():
            path = root / "videos" / "chunk-000" / video_key / f"episode_{episode_index:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(path),
                fps=video_fps,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=["-preset", "ultrafast", "-movflags", "+faststart"],
            )
            try:
                for frame in frames:
                    writer.append_data(np.ascontiguousarray(frame))
            finally:
                writer.close()

    def wait_for_saves(self, timeout: float | None = None) -> bool:
        """Block until the save queue drains. Returns True if it emptied in time."""
        worker = self._save_worker
        if worker is not None and worker.is_alive():
            worker.join(timeout)
        with self._lock:
            return not self._save_jobs

    def status_snapshot(self, task: str | None = None) -> dict[str, Any]:
        """Recording status for the web UI: pipeline state, history, and queue depth."""
        dataset_dir = (
            self._collection_dataset_dir(task) if self._collection_writer else self._log_dir
        )
        with self._lock:
            jobs = [
                job
                for job in self._save_jobs
                if not self._collection_writer or (job.dataset_dir or self._log_dir) == dataset_dir
            ]
            save_durations = list(self._save_durations)
            queue_size = len(jobs)
            global_queue_size = len(self._save_jobs)
            saving = any(j.status == "saving" for j in jobs)
            if self._collection_writer is not None:
                pending_episode_indices = {job.episode_index for job in jobs}
                history = [
                    row
                    for row in self._load_collection_history(dataset_dir)
                    if row["episode_index"] not in pending_episode_indices
                ]
            else:
                history = list(self._collection_history)
        current_episode_frames = 0
        current_episode_recorded_frames = 0
        current_episode_pending_frames = 0
        current_episode_skipped_before_start = 0
        active_for_task = self._active and (task is None or self._task == task)
        if active_for_task:
            if self._collection_writer is not None:
                counts = self._collection_writer.frame_counts()
                current_episode_frames = counts["received"]
                current_episode_recorded_frames = counts["recorded"]
                current_episode_pending_frames = counts["pending"]
                current_episode_skipped_before_start = counts["skipped_before_start"]
            else:
                current_episode_frames = self.active_frame_count
                current_episode_recorded_frames = self.active_frame_count
        if active_for_task:
            pipeline_state = "COLLECTING"
        elif global_queue_size >= self._save_queue_max:
            pipeline_state = "QUEUE_FULL"
        elif saving or jobs:
            pipeline_state = "SAVING"
        else:
            pipeline_state = "IDLE"
        queue = [self._save_job_summary(job) for job in jobs]
        terminal_jobs = sum(1 for job in jobs if job.status in ("saved", "failed"))
        known_jobs = len(history) + len(queue)
        done_jobs = len(history) + terminal_jobs
        progress = 0.0 if known_jobs == 0 else min(1.0, done_jobs / known_jobs)
        active_jobs = sum(1 for job in jobs if job.status in ("queued", "saving"))
        eta_sec = None
        if active_jobs > 0 and save_durations:
            eta_sec = round((sum(save_durations) / len(save_durations)) * active_jobs, 1)
        return {
            "pipeline_state": pipeline_state,
            "dataset_dir": str(dataset_dir),
            "collecting": active_for_task,
            "current_episode_frames": current_episode_frames,
            "current_episode_recorded_frames": current_episode_recorded_frames,
            "current_episode_pending_frames": current_episode_pending_frames,
            "current_episode_skipped_before_start": current_episode_skipped_before_start,
            "completed_episodes": (
                len(history) if self._collection_writer else self._completed_episodes
            ),
            "save_queue_size": queue_size,
            "save_queue_max": self._save_queue_max,
            "progress": progress,
            "eta_sec": eta_sec,
            "episodes": history,
            "queue": queue,
            "can_cancel": self._active,
        }

    def load_collection_replay_qpos(
        self, episode_index: int, task: str | None = None
    ) -> np.ndarray | None:
        """Load saved collection qpos for console replay.

        Args:
            episode_index: Integer episode id saved under data/chunk-000.

        Returns:
            qpos: [T, robot.total_action_dim] float32 array, or None when the
                current logger is not a collection logger or the episode is absent.
        """
        if self._collection is None or not self._collection.schema.columns:
            return None
        qpos_key = self._collection.schema.columns.get("state_qpos")
        if not qpos_key:
            return None
        dataset_dir = self._collection_dataset_dir(task)
        path = self._parquet_path(episode_index, dataset_dir=dataset_dir)
        if not path.exists():
            return None
        table = pq.read_table(str(path), columns=[qpos_key])
        return np.asarray(table.column(qpos_key).to_pylist(), dtype=np.float32)

    def load_collection_episode_fps(
        self, episode_index: int, task: str | None = None
    ) -> float | None:
        """Recorded collection_fps for one episode from episodes.jsonl; None if unset/absent."""
        if self._collection is None or not self._collection.schema.columns:
            return None
        dataset_dir = self._collection_dataset_dir(task)
        for row in _read_jsonl(self._meta_path("episodes.jsonl", dataset_dir)):
            if int(row.get("episode_index", -1)) != int(episode_index):
                continue
            fps = row.get("collection_fps")
            return None if fps is None else float(fps)
        return None

    def load_episode_series(self, episode_index: int) -> dict[str, Any] | None:
        """Load an episode's per-frame state/action time series for the viewer.

        Args:
            episode_index: Integer episode id saved under data/chunk-000.

        Returns:
            {"timestamp": [T], "state": [T, Ds], "action": [T, Da]} as plain lists,
            or None when the parquet is absent. ``state`` drives both the URDF
            replay (fed frame-by-frame to UrdfScene.transforms) and the per-dim
            time-series chart; ``action`` is an optional overlay on the chart.
        """
        path = self._parquet_path(episode_index)
        if not path.exists():
            return None
        state_key = self._keys.state_key
        action_key = self._action_key
        table = pq.read_table(str(path), columns=[state_key, action_key, "timestamp"])
        return {
            "timestamp": [float(t) for t in table.column("timestamp").to_pylist()],
            "state": table.column(state_key).to_pylist(),
            "action": table.column(action_key).to_pylist(),
        }

    def live_series(self, since: int = 0) -> dict[str, Any]:
        """In-memory per-frame state/action of the in-flight episode for live charts.

        Mirrors load_episode_series' shape so the frontend reuses one chart path,
        but reads the live ``self._steps`` buffer instead of disk — RUN feeds this
        continuously (record_step), so the console can plot and scrub the current
        run before it is ever saved.

        Args:
            since: Return only frames with index >= since for incremental polling.

        Returns:
            {"active": bool, "n": int (total frames), "timestamp": [k],
             "state": [k, Ds], "action": [k, Da]} where k = total - since.
        """
        steps = self._live_steps[since:] if since > 0 else self._live_steps[:]
        return {
            "active": self._active,
            "n": len(self._live_steps),
            "timestamp": [float(cast(float, s.timestamp)) for s in steps],
            "state": [s.state_qpos.tolist() for s in steps],
            "action": [cast(np.ndarray, s.action_qpos).tolist() for s in steps],
        }

    def list_episode_videos(self, episode_index: int) -> dict[str, Path]:
        """Map each camera's dataset video_key to its on-disk mp4 for one episode.

        Returns only cameras whose mp4 exists, so the viewer can offer camera
        switching without 404s. Empty when save_video was off or files are absent.
        """
        out: dict[str, Path] = {}
        for cam_key in self._camera_keys:
            video_key = resolve_video_key(self._keys, cam_key)
            if not video_key:
                continue
            path = self._video_path(episode_index, cam_key)
            if path.exists():
                out[video_key] = path
        return out

    def _remember_finished_job(self, job: SaveJob) -> None:
        self._collection_history.append(self._save_job_summary(job))
        self._collection_history = self._collection_history[-200:]
        if job.started_wall_time > 0.0 and job.finished_wall_time >= job.started_wall_time:
            self._save_durations.append(job.finished_wall_time - job.started_wall_time)
            self._save_durations = self._save_durations[-20:]

    def _save_job_summary(self, job: SaveJob) -> dict[str, Any]:
        row = job.collection_episode_row or {}
        return {
            "episode_index": job.episode_index,
            "length": int(row.get("length", len(job.steps))),
            "status": job.status,
            "quality": row.get("quality", "green"),
            "qc_verdict": row.get("qc_verdict", ""),
            "qc_note": row.get("qc_note", ""),
            "quality_issues": row.get("quality_issues", []),
            "error": "" if job.error is None else str(job.error),
        }

    def finalize(self) -> None:
        """Write meta/stats.json (per-feature min/max/mean/std) and meta/info.json."""
        if self._collection_writer is not None:
            self.wait_for_saves()
            self._collection_writer.finalize()
            return
        self.wait_for_saves()
        self._write_info_json()
        self._write_stats_json()

    def _steps_average_fps(self, steps: list[Observation]) -> float | None:
        if len(steps) < 2:
            return None
        duration = cast(float, steps[-1].timestamp) - cast(float, steps[0].timestamp)
        if duration <= 0.0:
            return None
        return (len(steps) - 1) / duration

    def _save_size(self) -> tuple[int, int] | None:
        h = self._save_image_height
        w = self._save_image_width
        if h is None or w is None:
            return None
        return int(h), int(w)

    # -- meta writers -----------------------------------------------------------

    def _eval_existing_episode_index(self, episode_meta: dict[str, Any]) -> int | None:
        """Episode index of a prior recording of this (prompt, trial) cell, or None.

        Used so an eval RE-RUN overwrites the same trial's data in place instead of
        appending. Matches on (prompt, trial) — the cell identity the operator scores —
        scanning meta/episodes.jsonl. Returns None when prompt/trial are absent (non-eval
        recordings) or the cell has no prior episode.
        """
        prompt = episode_meta.get("prompt")
        trial = episode_meta.get("trial")
        if prompt is None or trial is None:
            return None
        match: int | None = None
        for row in _read_jsonl(self._meta_path("episodes.jsonl")):
            if row.get("prompt") == prompt and row.get("trial") == trial:
                idx = row.get("episode_index")
                if idx is not None:
                    match = int(idx)  # last matching row wins
        return match

    def _existing_episode_global_start(self, episode_index: int) -> int | None:
        """Global frame offset (`index` col start) for an existing episode, or None.

        Sum of the lengths of all episodes recorded before this one, so a re-run's parquet
        `index` column resumes from the same dataset-global position the original used.
        """
        start = 0
        found = False
        for row in _read_jsonl(self._meta_path("episodes.jsonl")):
            idx = row.get("episode_index")
            if idx is None:
                continue
            if int(idx) == episode_index:
                found = True
                break
            start += int(row.get("length", 0))
        return start if found else None

    def _upsert_episode_row_locked(
        self, episode_index: int, task: str, length: int, episode_meta: dict[str, Any]
    ) -> None:
        """Replace the episodes.jsonl row for episode_index in place, or append if absent.

        A re-run rewrites the existing row entirely (no stale score/note carried over from
        the previous take); a first run just appends.
        """
        row: dict[str, Any] = {"episode_index": episode_index, "tasks": [task], "length": length}
        if episode_meta:
            row.update(episode_meta)
        path = self._meta_path("episodes.jsonl")
        rows = _read_jsonl(path)
        replaced = False
        for i, existing in enumerate(rows):
            if int(existing.get("episode_index", -1)) == episode_index:
                rows[i] = row
                replaced = True
                break
        if not replaced:
            self._append_episode_dict_row_locked(row)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _upsert_episode_dict_row_locked(self, row: dict[str, Any]) -> None:
        episode_index = int(row["episode_index"])
        path = self._meta_path("episodes.jsonl")
        rows = _read_jsonl(path)
        replaced = False
        for index, existing in enumerate(rows):
            if int(existing.get("episode_index", -1)) == episode_index:
                rows[index] = row
                replaced = True
                break
        if not replaced:
            self._append_episode_dict_row_locked(row)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for existing in rows:
                f.write(json.dumps(existing, ensure_ascii=False) + "\n")

    def _append_episode_dict_row_locked(
        self, row: dict[str, Any], dataset_dir: Path | None = None
    ) -> None:
        path = self._meta_path("episodes.jsonl", dataset_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append_episode_stats_locked(
        self, episode_index: int, stats: dict[str, Any], dataset_dir: Path | None = None
    ) -> None:
        path = self._meta_path("episodes_stats.jsonl", dataset_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(
                json.dumps({"episode_index": episode_index, "stats": stats}, ensure_ascii=False)
                + "\n"
            )

    def _write_tasks_jsonl_from(
        self, task_to_index: dict[str, int], dataset_dir: Path | None = None
    ) -> None:
        path = self._meta_path("tasks.jsonl", dataset_dir)
        items = sorted(task_to_index.items(), key=lambda kv: kv[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for task, idx in items:
                f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

    def _write_info_json(self) -> None:
        episodes = _read_jsonl(self._meta_path("episodes.jsonl"))
        total_episodes = len(episodes)
        total_frames = sum(int(e.get("length", 0)) for e in episodes)
        total_videos = total_episodes * len(self._camera_keys) if self._save_video else 0
        # fps must equal the integer target used to synthesize timestamps, not the
        # measured (jittery) average — LeRobot validates timestamp[i] == i/fps.
        fps = float(self._fps)
        info = build_info(
            robot_type=self._robot.name,
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=len(self._task_to_index),
            total_videos=total_videos,
            fps=fps,
            features=self._build_features(fps),
        )
        if self._eval_mode:
            info["eval"] = self._eval_info(episodes)
        with self._meta_path("info.json").open("w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def _eval_info(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """The info.json ``eval`` section: which keys carry scoring + how many are scored.

        ``episode_meta_keys`` are the episodes.jsonl scalars the eval layer writes; scoring
        lives only in meta/episodes.jsonl (no per-frame parquet columns). ``scored_episodes``
        counts rows that actually carry a score.
        """
        scored = [e for e in episodes if e.get("score") is not None]
        return {
            "episode_meta_keys": list(_EVAL_META_KEYS),
            "total_episodes": len(episodes),
            "scored_episodes": len(scored),
        }

    def _build_features(self, fps: float) -> dict[str, Any]:
        h, w = self._infer_image_shape()
        features: dict[str, Any] = {}
        if self._save_video:
            for cam_key in self._camera_keys:
                video_key = resolve_video_key(self._keys, cam_key)
                if video_key is None:
                    continue
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
        state_dim, action_dim = self._infer_vector_dims()
        features[self._keys.state_key] = {"dtype": "float32", "shape": [state_dim], "names": None}
        features[self._action_key] = {"dtype": "float32", "shape": [action_dim], "names": None}
        eef_dim = self._infer_eef_dim()
        if eef_dim is not None:
            features[self._keys.eef_key] = {
                "dtype": "float32",
                "shape": [eef_dim],
                "names": None,
            }
        for col in ("timestamp", "capture_time"):
            features[col] = {"dtype": "float32", "shape": [1], "names": None}
        for col in ("frame_index", "episode_index", "index", "task_index"):
            features[col] = {"dtype": "int64", "shape": [1], "names": None}
        features["intervention"] = {"dtype": "bool", "shape": [1], "names": None}
        features["intervention_segment_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }
        features["control_source"] = {"dtype": "string", "shape": [1], "names": None}
        return features

    def _write_stats_json(self) -> None:
        """Per-feature min/max/mean/std over the whole dataset (scans data parquet)."""
        state_acc = _StatAccumulator()
        action_acc = _StatAccumulator()
        eef_acc = _StatAccumulator()
        for ep_path in sorted((self._log_dir / "data" / "chunk-000").glob("episode_*.parquet")):
            table = pq.read_table(str(ep_path))
            state_acc.update(
                np.array(table.column(self._keys.state_key).to_pylist(), dtype=np.float64)
            )
            action_acc.update(
                np.array(table.column(self._action_key).to_pylist(), dtype=np.float64)
            )
            if self._keys.eef_key in table.column_names:
                eef_acc.update(
                    np.array(table.column(self._keys.eef_key).to_pylist(), dtype=np.float64)
                )
        stats: dict[str, Any] = {}
        if state_acc.count:
            stats[self._keys.state_key] = state_acc.result()
        if action_acc.count:
            stats[self._action_key] = action_acc.result()
        if eef_acc.count:
            stats[self._keys.eef_key] = eef_acc.result()
        with self._meta_path("stats.json").open("w") as f:
            json.dump(stats, f, indent=2)

    # -- dimension / shape inference --------------------------------------------

    def _infer_vector_dims(self) -> tuple[int, int]:
        if self._steps:
            action = (
                self._steps[0].action_eef
                if self._recording_space == "eef"
                else self._steps[0].action_qpos
            )
            if action is None:
                raise ValueError(f"recorded episode missing action.{self._recording_space}")
            return (
                self._steps[0].state_qpos.shape[0],
                action.shape[0],
            )
        action_dim = (
            8 * len(self._robot.arm_groups)
            if self._recording_space == "eef"
            else self._robot.total_action_dim
        )
        return self._robot.total_action_dim, action_dim

    def _infer_eef_dim(self) -> int | None:
        """Length of the eef state vector, or None when no episode recorded one.

        Returns:
            The eef vector dim read from the live buffer (preferred) or the first
            on-disk episode parquet that carries the eef column; None for joint-only
            datasets where state_eef was never present.
        """
        for step in self._steps:
            if step.state_eef is not None:
                return int(step.state_eef.shape[0])
        for ep_path in sorted((self._log_dir / "data" / "chunk-000").glob("episode_*.parquet")):
            table = pq.read_table(str(ep_path))
            if self._keys.eef_key in table.column_names:
                col = table.column(self._keys.eef_key).to_pylist()
                for row in col:
                    if row is not None:
                        return len(row)
        return None

    def _infer_image_shape(self) -> tuple[int, int]:
        if self._image_shape is not None:
            return self._image_shape
        configured = self._save_size()
        if configured is not None:
            return configured
        return 480, 640

    # -- dataset-resume discovery (so reruns append, not overwrite) -------------

    def _discover_next_episode_index(self, dataset_dir: Path | None = None) -> int:
        return len(_read_jsonl(self._meta_path("episodes.jsonl", dataset_dir)))

    def _load_existing_tasks(self, dataset_dir: Path | None = None) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for row in _read_jsonl(self._meta_path("tasks.jsonl", dataset_dir)):
            mapping[str(row["task"])] = int(row["task_index"])
        return mapping

    def _load_global_index(self, dataset_dir: Path | None = None) -> int:
        return sum(
            int(e.get("length", 0))
            for e in _read_jsonl(self._meta_path("episodes.jsonl", dataset_dir))
        )

    def _load_collection_history(self, dataset_dir: Path | None = None) -> list[dict[str, Any]]:
        history = []
        for row in _read_jsonl(self._meta_path("episodes.jsonl", dataset_dir)):
            history.append(history_row(row, len(history)))
        return history[-200:]

    def _collection_dataset_dir(self, task: str | None) -> Path:
        return self._log_dir / sanitize_path_component(task or "unset")

    def _resolve_task_index(self, task: str) -> int:
        if task not in self._task_to_index:
            self._task_to_index[task] = len(self._task_to_index)
        return self._task_to_index[task]

    def _next_collection_episode_index(self, dataset_dir: Path) -> int:
        with self._lock:
            if dataset_dir not in self._collection_next_index:
                self._collection_next_index[dataset_dir] = self._discover_next_episode_index(
                    dataset_dir
                )
            index = self._collection_next_index[dataset_dir]
            self._collection_next_index[dataset_dir] = index + 1
            return index

    def _next_collection_global_index(self, dataset_dir: Path, n_frames: int) -> int:
        with self._lock:
            if dataset_dir not in self._collection_next_global:
                self._collection_next_global[dataset_dir] = self._load_global_index(dataset_dir)
            index = self._collection_next_global[dataset_dir]
            self._collection_next_global[dataset_dir] = index + n_frames
            return index

    # -- path helpers (mirror transport.dataset layout so output is replayable) -

    def _parquet_path(
        self,
        episode_index: int,
        dataset_dir: Path | None = None,
    ) -> Path:
        root = dataset_dir or self._log_dir
        return root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"

    def _video_path(
        self,
        episode_index: int,
        cam_key: str,
        dataset_dir: Path | None = None,
    ) -> Path:
        root = dataset_dir or self._log_dir
        video_key = resolve_video_key(self._keys, cam_key)
        if video_key is None:
            raise ValueError(f"No video key for camera {cam_key!r}")
        return root / "videos" / "chunk-000" / video_key / f"episode_{episode_index:06d}.mp4"

    def _meta_path(self, name: str, dataset_dir: Path | None = None) -> Path:
        return (dataset_dir or self._log_dir) / "meta" / name


class _StatAccumulator:
    """Running min/max/sum/sumsq over float vectors -> LeRobot stats dict."""

    def __init__(self) -> None:
        self.count = 0
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._sum: np.ndarray | None = None
        self._sumsq: np.ndarray | None = None

    def update(self, arr: np.ndarray) -> None:
        """Fold one batch of rows into the running min/max/sum/sumsq.

        Args:
            arr: [N, D] float64 batch of feature vectors; empty batches are ignored.
        """
        if arr.size == 0:
            return
        batch_min = arr.min(axis=0)
        batch_max = arr.max(axis=0)
        batch_sum = arr.sum(axis=0)
        batch_sumsq = (arr * arr).sum(axis=0)
        if self._min is None or self._max is None or self._sum is None or self._sumsq is None:
            self._min, self._max = batch_min, batch_max
            self._sum, self._sumsq = batch_sum, batch_sumsq
        else:
            self._min = np.minimum(self._min, batch_min)
            self._max = np.maximum(self._max, batch_max)
            self._sum = self._sum + batch_sum
            self._sumsq = self._sumsq + batch_sumsq
        self.count += arr.shape[0]

    def result(self) -> dict[str, list[float]]:
        """Finalize the accumulated stats into a LeRobot dict.

        Returns:
            {"min", "max", "mean", "std"} each a length-D list of per-feature floats.
        """
        assert self._min is not None and self._max is not None
        assert self._sum is not None and self._sumsq is not None
        mean = self._sum / self.count
        var = np.maximum(self._sumsq / self.count - mean * mean, 0.0)
        std = np.sqrt(var)
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }


def _copy_observation(frame: Observation) -> Observation:
    images = {key: np.asarray(value).copy() for key, value in frame.images.items()}
    vectors = {}
    for field in _INTERVENTION_VECTOR_FIELDS:
        value = getattr(frame, field)
        vectors[field] = None if value is None else np.asarray(value, dtype=np.float32).copy()
    return Observation(
        timestamp=float(frame.timestamp if frame.timestamp is not None else 0.0),
        images=images,
        state_qpos=vectors["state_qpos"],
        state_eef=vectors["state_eef"],
        action_qpos=vectors["action_qpos"],
        action_eef=vectors["action_eef"],
    )


def _copy_raw_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _copy_intervention_segment(segment: RolloutInterventionSegment) -> RolloutInterventionSegment:
    return RolloutInterventionSegment(
        segment_index=segment.segment_index,
        start_policy_frame_index=segment.start_policy_frame_index,
        pre_intervention_qpos=np.asarray(segment.pre_intervention_qpos, dtype=np.float32).copy(),
        start_time=segment.start_time,
        frames=[_copy_observation(frame) for frame in segment.frames],
        resume_policy_frame_index=segment.resume_policy_frame_index,
        invalid_reason=segment.invalid_reason,
    )


def _vector_episode_stats(values: list[list[float]]) -> dict[str, Any]:
    """Per-feature min/max/mean/std/count for one episode's column (LeRobot v2.1).

    Args:
        values: [N, D] list of per-frame feature vectors (or [N, 1] for scalars).

    Returns:
        {"min", "max", "mean", "std"} each a length-D list, plus "count": [N].
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    std = np.sqrt(np.maximum((arr * arr).mean(axis=0) - mean * mean, 0.0))
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [n],
    }


def _image_episode_stats(frames: list[np.ndarray]) -> dict[str, Any]:
    """Per-channel min/max/mean/std/count over one episode's RGB frames (LeRobot v2.1).

    Args:
        frames: list of [H, W, 3] uint8 RGB frames.

    Returns:
        {"min", "max", "mean", "std"} each shaped [3, 1, 1] (per channel, normalized to
        0-1), plus "count": [N] where N is the frame count.
    """
    hist = np.zeros((3, 256), dtype=np.uint64)
    for frame in frames:
        arr = np.asarray(frame)
        for channel in range(3):
            hist[channel] += np.bincount(arr[..., channel].reshape(-1), minlength=256).astype(
                np.uint64
            )

    values = np.arange(256, dtype=np.float64)
    scale = 255.0
    counts = hist.sum(axis=1).astype(np.float64)
    nonzero = hist > 0
    min_value = np.array([np.flatnonzero(nonzero[channel])[0] for channel in range(3)])
    max_value = np.array([np.flatnonzero(nonzero[channel])[-1] for channel in range(3)])
    channel_sum = (hist * values).sum(axis=1)
    channel_sumsq = (hist * (values * values)).sum(axis=1)
    mean = channel_sum / counts / scale
    std = np.sqrt(np.maximum(channel_sumsq / counts / (scale * scale) - mean * mean, 0.0))

    def _ch(v: np.ndarray) -> list[list[list[float]]]:
        return [[[float(x)]] for x in v]  # [3, 1, 1]

    return {
        "min": _ch(min_value.astype(np.float64) / scale),
        "max": _ch(max_value.astype(np.float64) / scale),
        "mean": _ch(mean),
        "std": _ch(std),
        "count": [len(frames)],
    }


def _to_rgb_uint8(frame: np.ndarray, convert_bgr_to_rgb: bool) -> np.ndarray:
    """Ensure HWC uint8 RGB for imageio."""
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if convert_bgr_to_rgb and arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]
    return arr


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
