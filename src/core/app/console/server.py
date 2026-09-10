"""Stdlib HTTP server for the EVA unified console.

Drives the eva-client control mainline over HTTP by routing browser requests into
the existing command_queue (the same channel the legacy eval server uses). Adds
console-only routes for breakpoint single-step debugging, a live observation feed
(camera JPEGs + qpos), and the self-rendered 3D scene (per-mesh world transforms
from the robot's URDF, served via /api/scene + /api/meshes).
"""

from __future__ import annotations

import concurrent.futures
import copy
import csv
import dataclasses
import datetime as _dt
import gzip
import hashlib
import json
import logging
import multiprocessing
import os
import random
import re
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import imageio_ffmpeg
import numpy as np
import yaml
from tqdm import tqdm

from core.app.command_catalog import control_command_catalog
from core.app.console.collection_slots import (
    build_collection_slots,
    collection_slot_status,
    defer_active_slot,
    load_slot_state,
    select_collection_slot,
)
from core.app.console.dashboard import build_dashboard, discover_dashboard_upload_candidates
from core.app.console.transform_worker import build_transform_blob
from core.app.handlers import (
    _resolve_runtime_path,
    build_policy_observation,
    eval_episode_dir,
    eval_model_name,
    load_episode_history,
    rollout_hil_status,
    rollout_intervention_source,
    rollout_save_status,
    teleop_status,
)
from core.app.handlers.io import load_replay_dataset
from core.app.rl import build_rl_critic_observation, rl_live_series
from core.app.state import (
    RuntimeState,
    SessionMode,
    SessionState,
    format_task_label,
    resolve_inference_strategy_label,
)
from core.config import ConfigDict
from core.devices.service import DeviceSettings
from core.utils.dataset_upload import (
    DatasetUploadPlan,
    DatasetUploadSpec,
    record_dataset_upload_receipt,
    resolve_dataset_uploads,
    scan_dataset_directory,
    upload_dataset_directory,
)
from core.utils.lerobot import LeRobotDatasetIO
from core.utils.upload_plan import UploadProgress
from tools.conversion import (
    DATASET_EXPORT_FORMATS,
    DatasetExportProgress,
    export_dataset_by_quality,
)
from transport.base import ObservationSource
from transport.dataset import DatasetTransport

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Transport counts as online only if a SUB message arrived within this window; the
# frontend scene poll drains the reader ~12 Hz, so a live stream refreshes it well
# inside the window while a dead publisher lets it lapse.
_TRANSPORT_ONLINE_WINDOW_SECONDS = 2.0
_CAMERA_STREAM_FPS = 10.0
_EPISODE_HISTORY_DEFAULT_LIMIT = 128
_EPISODE_HISTORY_MAX_LIMIT = 512

# Transports that talk to a real arm: they track receipt freshness, so "no message
# yet" means offline (not merely "configured"). debug/dataset are not in this set.
_LIVE_TRANSPORT_TYPES = {"zmq", "ros1", "ros2"}
_VIDEO_CACHE_CONTROL = "public, max-age=3600"
_VIDEO_FASTSTART_CACHE_ENV = "EVA_VIDEO_CACHE_DIR"
_VIDEO_FASTSTART_LOCK = threading.Lock()
_VIDEO_POSTER_LOCK = threading.Lock()
_QUALITY_JOB_LIMIT = 16
_QUALITY_UPLOAD_FAILED_ERROR = "dataset upload failed"
_TRANSFORM_EXECUTOR: concurrent.futures.ProcessPoolExecutor | None = None
_TRANSFORM_EXECUTOR_LOCK = threading.Lock()
_TRACE_EVENT_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_REVIEW_DATA_CACHE_MAX = 8
_REVIEW_DATA_CACHE_LOCK = threading.RLock()
_SCENE_PLAN_CACHE_MAX = 8
_SCENE_PLAN_CACHE_LOCK = threading.RLock()
_SCENE_PLAN_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_COLLECTION_SLOT_PLAN_CACHE_MAX = 8
_COLLECTION_SLOT_PLAN_CACHE_LOCK = threading.RLock()
_COLLECTION_SLOT_PLAN_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()


@dataclasses.dataclass(frozen=True)
class _CachedReviewEpisode:
    series: dict
    video_keys: dict[str, str]
    scene_qpos: np.ndarray
    fps: int
    task: str


_REVIEW_DATA_CACHE: OrderedDict[tuple[str, int], _CachedReviewEpisode] = OrderedDict()


@dataclasses.dataclass
class _QualityUploadJob:
    job_id: str
    local_dir: str
    remote_dir: str
    backends: tuple[str, ...] = ()
    state: str = "queued"
    files_completed: int = 0
    files_total: int = 0
    bytes_completed: int = 0
    bytes_total: int = 0
    local_files: int = 0
    local_bytes: int = 0
    files_skipped: int = 0
    bytes_skipped: int = 0
    new_files: int = 0
    changed_files: int = 0
    files_to_delete: int = 0
    files_deleted: int = 0
    current_file: str = ""
    destination: str = ""
    error: str = ""
    skipped: bool = False
    _plan: DatasetUploadPlan | None = dataclasses.field(default=None, repr=False)

    def payload(self) -> dict[str, Any]:
        if self.bytes_total > 0:
            progress = self.bytes_completed / self.bytes_total
        elif self.files_total > 0:
            progress = self.files_completed / self.files_total
        else:
            progress = 1.0 if self.state == "completed" else 0.0
        payload = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if not field.name.startswith("_")
        }
        payload["progress"] = max(0.0, min(1.0, progress))
        return payload


@dataclasses.dataclass
class _QualityExportJob:
    job_id: str
    source_dir: str
    accepted_dir: str
    rejected_dir: str
    dataset_format: str
    state: str = "queued"
    episodes_completed: int = 0
    episodes_total: int = 0
    subset: str = ""
    source_episode_index: int | None = None
    accepted_episodes: int = 0
    rejected_episodes: int = 0
    accepted_frames: int = 0
    rejected_frames: int = 0
    error: str = ""

    def payload(self) -> dict[str, Any]:
        progress = (
            self.episodes_completed / self.episodes_total
            if self.episodes_total > 0
            else (1.0 if self.state == "completed" else 0.0)
        )
        return {
            **dataclasses.asdict(self),
            "progress": max(0.0, min(1.0, progress)),
        }


def _trace_json(value: Any, limit: int = 1600) -> str:
    """Compact, bounded JSON for diagnostic logs without large control payloads."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        encoded = repr(value)
    return encoded if len(encoded) <= limit else encoded[:limit] + "…"


def _post_body_summary(body: dict) -> dict:
    """Keep routing evidence while excluding large vectors and free-form text."""
    summary: dict[str, Any] = {}
    for key, value in body.items():
        if key in {"qpos", "actions", "state", "image", "images"}:
            summary[key] = f"<{len(value) if hasattr(value, '__len__') else '?'} values>"
        elif key in {"note", "annotation"}:
            summary[key] = f"<{len(str(value))} chars>"
        elif isinstance(value, (dict, list, tuple)):
            summary[key] = _trace_json(value, 240)
        else:
            summary[key] = value
    return summary


def _command_summary(command: str) -> str:
    if command.startswith(("web:manual_qpos:", "web:update_infer_params:")):
        return command.split(":", 2)[0] + ":" + command.split(":", 2)[1] + ":<payload>"
    return command if len(command) <= 240 else command[:240] + "…"


def _get_transform_executor() -> concurrent.futures.ProcessPoolExecutor:
    global _TRANSFORM_EXECUTOR
    with _TRANSFORM_EXECUTOR_LOCK:
        if _TRANSFORM_EXECUTOR is None:
            _TRANSFORM_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return _TRANSFORM_EXECUTOR


def _is_native_urdf_scene(scene: object) -> bool:
    cls = type(scene)
    return cls.__module__ == "robots.utils" and cls.__name__ == "UrdfScene"


def _compute_transform_blob(
    scene: object,
    qpos: np.ndarray,
    robot_type: str,
    gripper_open: float | None,
    gripper_close: float | None,
    batch_loading: threading.Event | None = None,
) -> bytes:
    """Compute a batch outside the web process when using the production scene."""
    qpos = np.ascontiguousarray(qpos, dtype=np.float32)
    started = time.monotonic()
    if _is_native_urdf_scene(scene):
        try:
            raw = (
                _get_transform_executor()
                .submit(
                    build_transform_blob,
                    robot_type,
                    gripper_open,
                    gripper_close,
                    qpos,
                )
                .result(timeout=120)
            )
            logger.info(
                "[TRANSFORM_BATCH] mode=worker frames=%d bytes=%d elapsed_ms=%.1f",
                len(qpos),
                len(raw),
                (time.monotonic() - started) * 1000.0,
            )
            return raw
        except Exception:
            logger.exception("[TRANSFORM_BATCH] worker failed; falling back in-process")

    if batch_loading is not None:
        batch_loading.set()
    try:
        raw = scene.all_transforms_blob(qpos)  # type: ignore[attr-defined]
    finally:
        if batch_loading is not None:
            batch_loading.clear()
    logger.info(
        "[TRANSFORM_BATCH] mode=in_process frames=%d bytes=%d elapsed_ms=%.1f",
        len(qpos),
        len(raw),
        (time.monotonic() - started) * 1000.0,
    )
    return raw


def _prewarm_transform_worker(config: ConfigDict, runtime: RuntimeState) -> None:
    """Start and initialize the isolated URDF worker without delaying web startup."""
    started = time.monotonic()
    empty = np.empty((0, int(runtime.robot.total_action_dim)), dtype=np.float32)
    future = _get_transform_executor().submit(
        build_transform_blob,
        str(config.robot.type),
        config.robot.gripper_open,
        config.robot.gripper_close,
        empty,
    )

    def report(done: concurrent.futures.Future) -> None:
        try:
            done.result()
        except Exception:
            logger.exception("[TRANSFORM_WORKER] prewarm failed")
            return
        logger.info(
            "[TRANSFORM_WORKER] prewarm complete elapsed_ms=%.1f",
            (time.monotonic() - started) * 1000.0,
        )

    future.add_done_callback(report)


def _mp4_atom_offsets(path: Path) -> dict[bytes, int]:
    offsets: dict[bytes, int] = {}
    size = path.stat().st_size
    with path.open("rb") as f:
        offset = 0
        while offset + 8 <= size:
            f.seek(offset)
            header = f.read(8)
            if len(header) != 8:
                break
            atom_size = int.from_bytes(header[:4], "big")
            atom_type = header[4:8]
            header_size = 8
            if atom_size == 1:
                large = f.read(8)
                if len(large) != 8:
                    break
                atom_size = int.from_bytes(large, "big")
                header_size = 16
            elif atom_size == 0:
                atom_size = size - offset
            if atom_size < header_size:
                break
            offsets.setdefault(atom_type, offset)
            offset += atom_size
    return offsets


def _mp4_needs_faststart(path: Path) -> bool:
    if path.suffix.lower() != ".mp4":
        return False
    offsets = _mp4_atom_offsets(path)
    moov = offsets.get(b"moov")
    mdat = offsets.get(b"mdat")
    return moov is not None and mdat is not None and moov > mdat


def _video_faststart_cache_path(path: Path) -> Path:
    stat = path.stat()
    digest = hashlib.sha256(
        f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    ).hexdigest()[:24]
    cache_root_env = os.environ.get(_VIDEO_FASTSTART_CACHE_ENV)
    cache_root = (
        Path(cache_root_env)
        if cache_root_env
        else _resolve_runtime_path("work_dirs/.video_faststart")
    )
    return cache_root / f"{path.stem}-{digest}.mp4"


def _video_poster_cache_path(path: Path) -> Path:
    stat = path.stat()
    digest = hashlib.sha256(
        f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    ).hexdigest()[:24]
    cache_root_env = os.environ.get(_VIDEO_FASTSTART_CACHE_ENV)
    cache_root = (
        Path(cache_root_env)
        if cache_root_env
        else _resolve_runtime_path("work_dirs/.video_faststart")
    )
    return cache_root / f"{path.stem}-{digest}.jpg"


def _ensure_faststart_video(path: Path) -> Path:
    if not _mp4_needs_faststart(path):
        return path
    out = _video_faststart_cache_path(path)
    with _VIDEO_FASTSTART_LOCK:
        if out.exists() and out.stat().st_size > 0 and not _mp4_needs_faststart(out):
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.mp4")
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(tmp),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
            raise RuntimeError(f"faststart remux failed for {path}: {stderr}") from exc
        tmp.replace(out)
        if _mp4_needs_faststart(out):
            raise RuntimeError(f"faststart remux did not move moov atom for {path}")
        return out


def _ensure_video_poster(path: Path) -> Path:
    out = _video_poster_cache_path(path)
    with _VIDEO_POSTER_LOCK:
        if out.exists() and out.stat().st_size > 0:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(path))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"could not read first video frame for {path}")
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise RuntimeError(f"could not encode first video frame for {path}")
        tmp = out.with_suffix(".tmp.jpg")
        tmp.write_bytes(buf.tobytes())
        tmp.replace(out)
        return out


def _pick_video(videos: dict[str, Path], cam: str) -> Path | None:
    # Named camera's mp4 when ``cam`` is given, else any one (the trajectory replay
    # is camera-agnostic). None when the dataset has no matching video.
    return videos.get(cam) if cam else next(iter(videos.values()), None)


def _match_video_key(candidates: list[str], cam: str) -> str | None:
    """First inferred candidate key naming this camera, or None.

    Args:
        candidates: image candidate keys from ``LeRobotDatasetIO.infer_keys``.
        cam: camera observation key (e.g. ``cam_high``).

    Returns:
        The matching candidate (``observation.images.<cam>`` or ``*.<cam>``), or None.
    """
    return next(
        (
            candidate
            for candidate in candidates
            if candidate == f"observation.images.{cam}" or candidate.endswith(f".{cam}")
        ),
        None,
    )


def _infer_replay_video_key(dataset_dir: Path, cam: str) -> str:
    try:
        inferred = LeRobotDatasetIO(dataset_dir).infer_keys()
    except Exception:
        return ""
    image = inferred.get("image") or {}
    candidates = image.get("candidates") or []
    match = next((c for c in candidates if c == cam or _match_video_key([c], cam)), None)
    return match or image.get("default") or ""


def _open_review_episode(
    ctx: ConsoleContext,
    dataset_dir: str | Path,
    episode_index: int,
) -> tuple[DatasetTransport, dict[str, str]]:
    """Open one saved episode without attaching it to shared console runtime state.

    Args:
        ctx: Active console context supplying robot and config definitions.
        dataset_dir: Resolved LeRobot dataset directory.
        episode_index: Zero-based saved episode index.

    Returns:
        source: Dataset transport positioned on the requested episode.
        video_keys: Camera observation keys mapped to dataset video keys.
    """
    runtime = ctx.runtime
    config = runtime.active_config or ctx.config
    inferred = LeRobotDatasetIO(dataset_dir).infer_keys()
    state_key = (inferred.get("state") or {}).get("default") or "observations.state.qpos"
    action_key = (inferred.get("action") or {}).get("default") or "action"
    image = inferred.get("image") or {}
    candidates = image.get("candidates") or []
    video_keys = {
        camera.observation_key: match
        for camera in runtime.robot.observation_schema.cameras
        if (match := _match_video_key(candidates, camera.observation_key)) is not None
    }
    if not video_keys and image.get("default"):
        camera = runtime.robot.observation_schema.cameras[0].observation_key
        video_keys[camera] = str(image["default"])

    review_config = copy.deepcopy(config)
    review_config.transport.dataset_dir = dataset_dir
    review_config.transport.episode_id = episode_index
    review_config.transport.dataset_keys = ConfigDict(
        state_key=state_key,
        action_key=action_key,
        video_keys=video_keys,
    )
    return (
        DatasetTransport(review_config, runtime.robot, dataset_dir, episode_index),
        video_keys,
    )


def _cached_review_episode(
    ctx: ConsoleContext,
    dataset_dir: str | Path,
    episode_index: int,
) -> _CachedReviewEpisode:
    """Read one stateless review episode once and cache its non-video data.

    Args:
        ctx: Active console context supplying robot and dataset key configuration.
        dataset_dir: Resolved LeRobot dataset directory.
        episode_index: Zero-based saved episode index.

    Returns:
        Cached state/action series, camera mapping, and scene qpos rows.
    """
    key = (str(Path(dataset_dir).resolve()), int(episode_index))
    with _REVIEW_DATA_CACHE_LOCK:
        cached = _REVIEW_DATA_CACHE.get(key)
        if cached is not None:
            _REVIEW_DATA_CACHE.move_to_end(key)
            return cached

    source, video_keys = _open_review_episode(ctx, dataset_dir, episode_index)
    try:
        series = source.series()
        scene_qpos = np.asarray(
            [source.get_scene_qpos(index) for index in range(source.n_steps)],
            dtype=np.float32,
        )
    finally:
        source.close()
    cached = _CachedReviewEpisode(
        series, dict(video_keys), scene_qpos, source.fps, source.current_task
    )
    with _REVIEW_DATA_CACHE_LOCK:
        existing = _REVIEW_DATA_CACHE.get(key)
        if existing is not None:
            _REVIEW_DATA_CACHE.move_to_end(key)
            return existing
        _REVIEW_DATA_CACHE[key] = cached
        while len(_REVIEW_DATA_CACHE) > _REVIEW_DATA_CACHE_MAX:
            _REVIEW_DATA_CACHE.popitem(last=False)
    return cached


def _ensure_ckpt_order(config: ConfigDict, runtime: RuntimeState) -> None:
    eval_cfg = config.eval
    if eval_cfg is None or not eval_cfg.checkpoints or runtime.ckpt_order:
        return
    runtime.ckpt_order = list(range(len(eval_cfg.checkpoints)))
    if eval_cfg.shuffle_ckpts:
        rng = random.Random(eval_cfg.shuffle_seed)
        rng.shuffle(runtime.ckpt_order)


def _resolve_initial_tab(config: ConfigDict) -> str:
    configured = str((config.get("console") or {}).get("initial_tab", "auto"))
    if configured != "auto":
        return configured
    return "eval" if config.eval else "debug"


def _collection_upload_payload(config: ConfigDict) -> dict[str, Any]:
    uploads = resolve_dataset_uploads(config.collection.storage)
    return {
        "configured": bool(uploads),
        "backends": [spec.backend for spec in uploads],
        "targets": [spec.display_root for spec in uploads],
    }


@dataclasses.dataclass
class ConsoleContext:
    """Shared state for the console request handler."""

    config: ConfigDict
    runtime: RuntimeState
    session: SessionState
    device_settings: DeviceSettings = dataclasses.field(default_factory=DeviceSettings)
    obs_reader: ObservationSource | None = None  # independent source for visualization
    camera_preview: Any = None
    scene: object | None = None  # UrdfScene, lazily built (None if URDF unavailable)
    scene_cache_key: object | None = None  # last qpos signature fed to scene.transforms
    scene_cache: dict | None = None  # cached transforms payload for that signature
    scene_log_key: object | None = None
    scene_batch_loading: threading.Event = dataclasses.field(default_factory=threading.Event)
    scene_pbar: tqdm | None = None  # collection-replay playback progress bar
    active_tab: str = "debug"
    output_dir: str = ""  # results root; debug results land in <output_dir>/console/
    preview: Any = None  # EpisodePreview for RESULT-tab playback (lazily set)
    quality_upload_lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    quality_upload_jobs: OrderedDict[str, _QualityUploadJob] = dataclasses.field(
        default_factory=OrderedDict
    )
    quality_export_jobs: OrderedDict[str, _QualityExportJob] = dataclasses.field(
        default_factory=OrderedDict
    )

    def ckpt_label(self, slot: int | None) -> str | None:
        """Anonymized operator label for a blind ckpt slot (0 -> "Model A"); None when slot None."""
        return None if slot is None else f"Model {chr(ord('A') + slot)}"

    def active_model_name(self) -> str:
        """Folder name of the active model's eval dataset (matches the recorder's rooting)."""
        return eval_model_name(self.config, self.runtime)

    def sync_collection_slot(self) -> dict[str, Any] | None:
        """Advance the backend collection cursor after an episode is queued."""
        return _sync_collection_slot_session(self)


def _run_quality_upload_scan(
    ctx: ConsoleContext,
    job_id: str,
    local_dir: Path,
    upload_specs: tuple[DatasetUploadSpec, ...],
) -> None:
    with ctx.quality_upload_lock:
        job = ctx.quality_upload_jobs[job_id]
        job.state = "scanning"

    try:
        plan = scan_dataset_directory(local_dir, upload_specs)
        if plan.files_to_upload == 0 and plan.files_to_delete == 0:
            record_dataset_upload_receipt(
                local_dir,
                destination=", ".join(
                    backend_plan.destination for _, backend_plan in plan.backends
                ),
                remote_dir=", ".join(spec.target for spec, _ in plan.backends),
            )
    except Exception:
        logger.exception("Failed to scan accepted collection dataset upload")
        with ctx.quality_upload_lock:
            job = ctx.quality_upload_jobs.get(job_id)
            if job is not None:
                job.state = "failed"
                job.error = "dataset upload scan failed"
        return

    with ctx.quality_upload_lock:
        job = ctx.quality_upload_jobs.get(job_id)
        if job is not None:
            job.state = "ready"
            job.files_total = plan.files_to_upload
            job.bytes_total = plan.bytes_to_upload
            job.local_files = plan.files_total
            job.local_bytes = plan.bytes_total
            job.files_skipped = plan.files_skipped
            job.bytes_skipped = plan.bytes_skipped
            job.new_files = plan.new_files
            job.changed_files = plan.changed_files
            job.files_to_delete = plan.files_to_delete
            job._plan = plan


def _run_quality_upload(ctx: ConsoleContext, job_id: str) -> None:
    with ctx.quality_upload_lock:
        job = ctx.quality_upload_jobs[job_id]
        plan = job._plan
        if plan is None:
            job.state = "failed"
            job.error = "dataset upload scan is required"
            return
        job.state = "running"
        local_dir = Path(job.local_dir)
        upload_specs = tuple(spec for spec, _ in plan.backends)

    def update_progress(progress: UploadProgress) -> None:
        with ctx.quality_upload_lock:
            active_job = ctx.quality_upload_jobs.get(job_id)
            if active_job is None:
                return
            active_job.files_completed = progress.files_completed
            active_job.files_total = progress.files_total
            active_job.bytes_completed = progress.bytes_completed
            active_job.bytes_total = progress.bytes_total
            active_job.current_file = progress.current_file

    try:
        result = upload_dataset_directory(
            local_dir,
            upload_specs,
            plan=plan,
            progress_callback=update_progress,
        )
        record_dataset_upload_receipt(
            local_dir,
            destination=result.destination,
            remote_dir=result.remote_dir,
        )
    except Exception as error:
        logger.exception("Failed to upload accepted collection dataset")
        with ctx.quality_upload_lock:
            job = ctx.quality_upload_jobs.get(job_id)
            if job is not None:
                job.state = "failed"
                job.error = (
                    "upload content changed after scan; scan again"
                    if "changed after scan" in str(error)
                    else _QUALITY_UPLOAD_FAILED_ERROR
                )
        return

    with ctx.quality_upload_lock:
        job = ctx.quality_upload_jobs.get(job_id)
        if job is not None:
            job.state = "completed"
            job.files_completed = result.files
            job.bytes_completed = result.bytes
            job.current_file = ""
            job.destination = result.destination
            job.remote_dir = result.remote_dir
            job.skipped = result.skipped
            job.files_deleted = result.files_deleted


def _quality_export_paths(dataset_dir: Path, dataset_format: str) -> tuple[Path, Path]:
    export_root = (
        dataset_dir.parent / "export"
        if dataset_dir.name == "raw"
        else dataset_dir.with_name(f"{dataset_dir.name}_export")
    )
    format_root = export_root / dataset_format
    return format_root / "accepted", format_root / "rejected"


def _run_quality_export(
    ctx: ConsoleContext,
    job_id: str,
    source_dir: Path,
    accepted_dir: Path,
    rejected_dir: Path,
    dataset_format: str,
) -> None:
    with ctx.quality_upload_lock:
        job = ctx.quality_export_jobs[job_id]
        job.state = "running"

    def update_progress(progress: DatasetExportProgress) -> None:
        with ctx.quality_upload_lock:
            active_job = ctx.quality_export_jobs.get(job_id)
            if active_job is None:
                return
            active_job.episodes_completed = progress.episodes_completed
            active_job.episodes_total = progress.episodes_total
            active_job.subset = progress.subset
            active_job.source_episode_index = progress.source_episode_index

    try:
        summary = export_dataset_by_quality(
            source_dir,
            accepted_dir,
            rejected_dir,
            dataset_format=dataset_format,
            replace_existing=True,
            progress_callback=update_progress,
        )
    except Exception as error:
        logger.exception("Failed to export collection quality split")
        with ctx.quality_upload_lock:
            job = ctx.quality_export_jobs.get(job_id)
            if job is not None:
                job.state = "failed"
                job.error = str(error)
        return

    with ctx.quality_upload_lock:
        job = ctx.quality_export_jobs.get(job_id)
        if job is not None:
            job.state = "completed"
            job.episodes_completed = summary.source_episodes
            job.episodes_total = summary.source_episodes
            job.subset = ""
            job.source_episode_index = None
            job.accepted_episodes = summary.accepted_episodes
            job.rejected_episodes = summary.rejected_episodes
            job.accepted_frames = summary.accepted_frames
            job.rejected_frames = summary.rejected_frames


def _serialize_prompt_config(prompt: Any) -> dict:
    # Per-prompt block shared by the EVAL config and the RESULT browser: bilingual
    # text plus the prompt's own milestones, each a (id, label) tuple in config.
    milestones = prompt.get("milestones") or ()
    return {
        "prompt_en": prompt.prompt_en,
        "prompt_zh": prompt.get("prompt_zh"),
        "milestones": [{"id": mid, "label": label} for mid, label in milestones],
    }


def _collection_set_for_prompt(config: ConfigDict, prompt: str) -> str | None:
    for dataset_name, entries in config.collection.tasks.items():
        if prompt in {str(entry[0]) for entry in entries}:
            return str(dataset_name)
    return None


def _trim_finished_jobs(jobs: OrderedDict[str, Any], limit: int = _QUALITY_JOB_LIMIT) -> None:
    while len(jobs) >= limit:
        oldest_id, oldest = next(iter(jobs.items()))
        if oldest.state in {"queued", "scanning", "running"}:
            break
        jobs.pop(oldest_id)


def _quality_job_payload(jobs: OrderedDict[str, Any], job_id: str) -> dict[str, Any] | None:
    job = jobs.get(job_id)
    return None if job is None else job.payload()


def _serialize_eval(ctx: ConsoleContext) -> dict:
    # Eval block for the EVAL/RESULT tabs. Empty dict when the config carries no eval,
    # which the frontend reads as "disable those tabs". Each checkpoint exposes its
    # anonymized slot label (Model A/B/...) plus its real ckpt name for the MODEL picker.
    eval_cfg = ctx.config.eval
    if eval_cfg is None:
        return {}
    order = ctx.runtime.ckpt_order or list(range(len(eval_cfg.checkpoints)))
    # The INIT panel is always available under eval: when no init_pose is configured
    # the start pose falls back to the robot's initial_qpos (home).
    init_pose_enabled = True
    # Native floats, not numpy scalars, so the qpos survives JSON serialization.
    home_pose = [float(x) for x in ctx.runtime.robot.initial_qpos]
    return {
        "trials_per_prompt": eval_cfg.trials_per_prompt,
        "cli_mode": eval_cfg.cli_mode,
        "init_pose_enabled": init_pose_enabled,
        "tasks": [
            {
                **_serialize_prompt_config(t),
                "init_pose": [
                    float(x)
                    for x in (t.get("init_pose") if t.get("init_pose") is not None else home_pose)
                ],
            }
            for t in eval_cfg.tasks
        ],
        "checkpoints": [
            {
                "slot": slot,
                "label": ctx.ckpt_label(slot),
                "name": eval_cfg.checkpoints[order[slot]].name,
            }
            for slot in range(len(order))
        ],
    }


def _serialize_rl(ctx: ConsoleContext) -> dict:
    """Return the configured RL workspace choices and storage contract."""
    config = ctx.runtime.active_config or ctx.config
    rl_cfg = config.rl
    if rl_cfg is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "backend_ready": True,
        "cli_mode": str(rl_cfg.cli_mode),
        "inference_strategy": str(rl_cfg.inference_strategy),
        "rollout_intervention_source": rollout_intervention_source(config),
        "tasks": [str(task) for task in rl_cfg.tasks],
        "policies": [
            {"slot": slot, "name": str(model.name)} for slot, model in enumerate(rl_cfg.policies)
        ],
        "critics": [
            {"slot": slot, "name": str(model.name), "type": str(model.type)}
            for slot, model in enumerate(rl_cfg.critics)
        ],
        "data": {
            "format": str(rl_cfg.data.format),
            "dataset_dir": _resolve_dataset_dir(str(rl_cfg.data.storage.log_dir)),
        },
    }


def _serialize_gripper_controls(ctx: ConsoleContext) -> list[dict[str, object]]:
    config = ctx.runtime.active_config or ctx.config
    controls: list[dict[str, object]] = []
    offset = 0
    for group in ctx.runtime.robot.actuator_groups:
        if group.gripper_index is not None:
            controls.append(
                {
                    "group": group.name,
                    "qpos_index": offset + group.gripper_index,
                    "initial_value": float(
                        ctx.runtime.robot.initial_qpos[offset + group.gripper_index]
                    ),
                    "open_value": float(config.robot.gripper_open),
                    "close_value": float(config.robot.gripper_close),
                }
            )
        offset += group.dof
    if not controls:
        return []
    if len(controls) == 1:
        controls[0].update(side="l", label="GRIPPER")
        return controls
    for control, side, label in zip(controls[:2], ("l", "r"), ("LEFT", "RIGHT"), strict=True):
        control.update(side=side, label=label)
    return controls[:2]


def _serialize_manual_qpos_limits(ctx: ConsoleContext) -> list[dict[str, float]]:
    config = ctx.runtime.active_config or ctx.config
    limits = [
        {"min": -3.2, "max": 3.2, "step": 0.005} for _ in range(ctx.runtime.robot.total_action_dim)
    ]
    gripper_min = float(min(config.robot.gripper_open, config.robot.gripper_close))
    gripper_max = float(max(config.robot.gripper_open, config.robot.gripper_close))
    gripper_step = float((gripper_max - gripper_min) / 1000.0)
    offset = 0
    for group in ctx.runtime.robot.actuator_groups:
        if group.gripper_index is not None:
            index = offset + group.gripper_index
            limits[index] = {"min": gripper_min, "max": gripper_max, "step": gripper_step}
        offset += group.dof
    return limits


def _resolve_dataset_dir(dataset_dir: str) -> str:
    if not dataset_dir:
        return dataset_dir
    return str(_resolve_runtime_path(dataset_dir))


def _scene_plan_root(config: ConfigDict | None = None, dataset: str | None = None) -> Path:
    task_set_dirs = (config.get("collection") or {}).get("task_set_dir") if config else None
    if isinstance(task_set_dirs, (list, tuple)) and task_set_dirs:
        roots = [_resolve_runtime_path(str(value)).expanduser().resolve() for value in task_set_dirs]
        root = next((path for path in roots if path.name == dataset), None) if dataset else roots[0]
        if root is None:
            raise ValueError("unknown collection task-set directory")
        if not root.is_dir():
            raise ValueError(f"collection task-set directory is missing: {root}")
        return root
    configured = os.environ.get("EVA_SCENE_PLAN_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if config is not None:
        task_set_dirs = ((config.get("collection") or {}).get("task_set_dir", "")) or ""
        if isinstance(task_set_dirs, (list, tuple)):
            task_set_dir = str(task_set_dirs[0]).strip() if task_set_dirs else ""
        else:
            task_set_dir = str(task_set_dirs).strip()
        if task_set_dir:
            task_set_root = _resolve_runtime_path(task_set_dir).expanduser().resolve()
            if task_set_root.is_dir():
                return task_set_root
    cwd_root = Path.cwd() / "work_dirs"
    repo_root = Path(__file__).resolve().parents[4] / "work_dirs"
    has_canonical_plan = any(
        (cwd_root / filename).is_file()
        for filename in ("layout.yaml", "scene.csv", "tasks.csv", "objects.csv")
    )
    return cwd_root if has_canonical_plan else repo_root


def _read_scene_plan_csv(root: Path, filename: str) -> list[dict[str, str]]:
    path = root / filename
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeError, csv.Error) as exc:
        logger.warning("Unable to read scene plan %s: %s", path, exc)
        return []


def _read_scene_plan_yaml(root: Path, filename: str) -> dict[str, Any]:
    path = root / filename
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        logger.warning("Unable to read scene plan %s: %s", path, exc)
        return {}


def _optional_scene_float(value: Any) -> float | None:
    try:
        text = "" if value is None else str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _scene_plan_placements(value: Any) -> list[dict[str, Any]]:
    try:
        payload = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Invalid scene placements JSON: %s", exc)
        return []
    if not isinstance(payload, list):
        return []
    placements = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id", "") or "").strip()
        position_ids = item.get("position_ids")
        if not object_id or not isinstance(position_ids, list):
            continue
        position_ids = [str(position_id or "").strip() for position_id in position_ids]
        position_ids = [position_id for position_id in position_ids if position_id]
        if position_ids:
            placements.append(
                {
                    "position_ids": position_ids,
                    "object_id": object_id,
                }
            )
    return placements


def _scene_plan_signature(root: Path) -> tuple[Any, ...]:
    signature: list[Any] = [str(root)]
    info = _read_scene_plan_yaml(root, "info.yaml")
    objects_file = str(info.get("objects_file", "objects.csv") or "objects.csv")
    paths = [root / name for name in ("info.yaml", "tasks.csv", "scene.csv", "layout.yaml")]
    paths.append(root / objects_file)
    for path in paths:
        try:
            stat = path.stat()
            signature.extend((int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            signature.extend((0, 0, 0))
    return tuple(signature)


def _scene_plan_positions(layout_doc: dict[str, Any]) -> list[dict[str, Any]]:
    positions = []
    unit = str(layout_doc.get("unit", "") or "")
    frame = str(layout_doc.get("coordinate_frame", "") or "")
    status = str(layout_doc.get("calibration_status", "unverified") or "unverified")
    for point in layout_doc.get("sampling_points", []):
        if not isinstance(point, dict):
            continue
        position_id = str(point.get("position_id", point.get("id", "")) or "").strip()
        if not position_id:
            continue
        positions.append(
            {
                "position_id": position_id,
                "x": _optional_scene_float(point.get("x")),
                "y": _optional_scene_float(point.get("y")),
                "unit": str(point.get("unit", unit) or unit),
                "coordinate_frame": str(point.get("coordinate_frame", frame) or frame),
                "calibration_status": str(point.get("calibration_status", status) or status),
            }
        )
    return positions


def _scene_plan_objects(root: Path) -> dict[str, dict[str, Any]]:
    payload = {}
    info = _read_scene_plan_yaml(root, "info.yaml")
    objects_file = str(info.get("objects_file", "objects.csv") or "objects.csv")
    objects_path = (root / objects_file).resolve()
    photo_root = objects_path.parent / "object_photos"
    for row in _read_scene_plan_csv(root, objects_file):
        object_id = str(row.get("object_id") or row.get("物品代码") or "").strip()
        if not object_id:
            continue
        photo_dir = str(
            row.get("photo_dir") or row.get("照片目录") or row.get("名称") or ""
        ).strip()
        photos = sorted((photo_root / photo_dir).glob("*")) if photo_dir else []
        photo = next(
            (
                path
                for path in photos
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ),
            None,
        )
        payload[object_id] = {
            "object_id": object_id,
            "name": str(
                row.get("object_name_zh") or row.get("名称") or row.get("object_name") or object_id
            ),
            "name_zh": str(row.get("object_name_zh") or row.get("名称") or ""),
            "name_en": str(row.get("object_name") or row.get("英文名") or ""),
            "color": str(row.get("color", "") or ""),
            "photo_url": f"/api/scene_plan/object-photo/{quote(object_id)}/{quote(photo.name)}?set={quote(root.name)}"
            if photo
            else "",
        }
    return payload


def _scene_plan_scenes(
    rows: list[dict[str, str]],
    objects: dict[str, dict[str, Any]],
    calibration_status: str,
) -> list[dict[str, Any]]:
    scenes: dict[str, dict[str, Any]] = {}
    for row in rows:
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id:
            continue
        scene = scenes.setdefault(
            scene_id,
            {
                "scene_id": scene_id,
                "placements": [],
                "placement_groups": [],
                "calibration_status": calibration_status,
            },
        )
        for group_index, placement in enumerate(_scene_plan_placements(row.get("placements"))):
            object_id = placement["object_id"]
            position_ids = placement["position_ids"]
            obj = objects.get(object_id, {"name": object_id, "color": ""})
            group_id = f"{scene_id}:{group_index}"
            scene["placement_groups"].append(
                {
                    "group_id": group_id,
                    "position_ids": position_ids,
                    "object_id": object_id,
                    "name": obj["name"],
                    "name_zh": obj.get("name_zh", ""),
                    "name_en": obj.get("name_en", ""),
                    "color": obj.get("color", ""),
                    "photo_url": obj.get("photo_url", ""),
                }
            )
            for position_index, position_id in enumerate(position_ids):
                scene["placements"].append(
                    {
                        "position_id": position_id,
                        "object_id": object_id,
                        "name": obj["name"],
                        "name_zh": obj.get("name_zh", ""),
                        "name_en": obj.get("name_en", ""),
                        "color": obj.get("color", ""),
                        "photo_url": obj.get("photo_url", ""),
                        "group_id": group_id,
                        "group_position_index": position_index,
                        "group_size": len(position_ids),
                        "group_position_ids": position_ids,
                    }
                )
    return list(scenes.values())


def _scene_plan_tasks(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        scene_ids = [value for value in str(row.get("scene_ids", "")).split(";") if value]
        if not task_id or not scene_ids:
            continue
        episode_counts = str(row.get("scene_epsiodes_count", "")).split(";")
        tasks.append(
            {
                "task_id": task_id,
                "action": str(row.get("action", "") or ""),
                "category": str(row.get("category", "") or ""),
                "operation_object": str(
                    row.get("operation_object") or row.get("操作对象") or ""
                ).strip(),
                "operation_object_ids": [
                    value.strip()
                    for value in str(row.get("operation_object_ids", "")).split(";")
                    if value.strip()
                ],
                "prompt_en": str(row.get("prompt_en", "") or ""),
                "prompt_zh": str(row.get("prompt_zh", "") or ""),
                "display_name": str(
                    row.get("prompt_zh") or row.get("prompt_en") or task_id
                ).strip(),
                "scene_ids": scene_ids,
                "scene_epsiodes_count": [int(value) for value in episode_counts if value.isdigit()],
                "total_epsiodes_count": int(row.get("total_epsiodes_count", 0) or 0),
            }
        )
    return tasks


def _build_scene_plan(root: Path) -> dict[str, Any]:
    """Parse one normalized scene catalog directory."""
    task_rows = _read_scene_plan_csv(root, "tasks.csv")
    layout_doc = _read_scene_plan_yaml(root, "layout.yaml")
    layout_unit = str(layout_doc.get("unit", "") or "")
    layout_frame = str(layout_doc.get("coordinate_frame", "") or "")
    layout_status = str(layout_doc.get("calibration_status", "unverified") or "unverified")
    positions = _scene_plan_positions(layout_doc)
    objects = _scene_plan_objects(root)
    raw_bounds = layout_doc.get("bounds")
    bounds = raw_bounds if isinstance(raw_bounds, dict) else {}
    return {
        "ok": True,
        "source": str(root / "tasks.csv") if task_rows else "",
        "layout_id": str(layout_doc.get("layout_id", "") or ""),
        "coordinate_frame": layout_frame,
        "bounds": {
            "width": _optional_scene_float(bounds.get("width")),
            "height": _optional_scene_float(bounds.get("height")),
            "unit": layout_unit,
        },
        "calibrated": bool(positions)
        and all(
            row["x"] is not None
            and row["y"] is not None
            and row["calibration_status"] == "verified"
            for row in positions
        ),
        "positions": positions,
        "objects": list(objects.values()),
        "scenes": _scene_plan_scenes(
            _read_scene_plan_csv(root, "scene.csv"), objects, layout_status
        ),
        "tasks": _scene_plan_tasks(task_rows),
    }


def _load_scene_plan(config: ConfigDict | None = None, dataset: str | None = None) -> dict[str, Any]:
    """Load and signature-cache the optional normalized collection scene plan."""
    root = _scene_plan_root(config, dataset)
    signature = _scene_plan_signature(root)
    with _SCENE_PLAN_CACHE_LOCK:
        cached = _SCENE_PLAN_CACHE.get(signature)
        if cached is not None:
            _SCENE_PLAN_CACHE.move_to_end(signature)
            return cached
    payload = _build_scene_plan(root)
    with _SCENE_PLAN_CACHE_LOCK:
        _SCENE_PLAN_CACHE[signature] = payload
        _SCENE_PLAN_CACHE.move_to_end(signature)
        while len(_SCENE_PLAN_CACHE) > _SCENE_PLAN_CACHE_MAX:
            _SCENE_PLAN_CACHE.popitem(last=False)
    return payload


def _serialize_config(ctx: ConsoleContext) -> dict:
    # Static config the frontend renders once: prompts, modes, strategies, robot.
    # Read the active (online-tuned) config so a reload reflects live param edits.
    from core.registry import STRATEGY_REGISTRY

    config = ctx.runtime.active_config or ctx.config
    strategies = [
        {
            "key": key,
            "type": spec.get("type", ""),
            "args": dict(spec.get("args") or {}),
            "fields": getattr(
                STRATEGY_REGISTRY.module_dict.get(spec.get("type", "")), "tune_fields", []
            ),
        }
        for key, spec in config.inference_strategies.items()
    ]
    is_replay = ctx.runtime.replay_source is not None
    n_episodes = ctx.runtime.replay_n_episodes
    if n_episodes == 0 and hasattr(ctx.runtime.transport, "n_episodes"):
        n_episodes = ctx.runtime.transport.n_episodes  # type: ignore[attr-defined]
    teleop_config = config.collection.get("teleop") or {}
    teleop_client_config = teleop_config.get("client") or {}
    controls_config = config.collection.get("controls") or {}
    is_vr = str(teleop_client_config.get("type", "")) == "vr_webxr"
    control_mode = "vr" if is_vr else "keyboard"
    controls = {
        str(action): dict(binding)
        for action, binding in (controls_config.get(control_mode) or {}).items()
    }
    control_groups = []
    if is_vr:
        for group_name, binding in (teleop_client_config.get("arms") or {}).items():
            hand = str((binding or {}).get("controller", ""))
            if hand not in {"left", "right"}:
                continue
            control_groups.append(
                {
                    "id": str(group_name),
                    "label": str((binding or {}).get("label") or group_name)
                    .replace("_", " ")
                    .upper(),
                    "control": f"{hand}.grip",
                    "binding": f"{hand}_arm_toggle",
                }
            )
    collection_storage = config.collection.storage
    collection_sftp = collection_storage.get("sftp") or {}
    return {
        "initial_tab": _resolve_initial_tab(ctx.config),
        "robot_type": config.robot.type,
        "device_selection": ctx.device_settings.workspace.initial_selection(config),
        "transport_type": config.transport.type,
        "policy": {
            "type": config.policy.type,
            "host": config.policy.host,
            "port": config.policy.port,
        },
        "tasks": list(config.inference_cfg.debug_tasks),
        "collect_tasks": {
            str(dataset_name): [[str(prompt), int(target)] for prompt, target in entries]
            for dataset_name, entries in config.collection.tasks.items()
        },
        "is_replay": is_replay,
        "manual_capable": config.transport.type in _LIVE_TRANSPORT_TYPES,
        "n_episodes": n_episodes,
        "modes": ["real", "sim", "step", "manual"],
        "camera_keys": [
            cam.observation_key for cam in ctx.runtime.robot.observation_schema.cameras
        ],
        "strategies": strategies,
        "gripper_controls": _serialize_gripper_controls(ctx),
        "manual_qpos_limits": _serialize_manual_qpos_limits(ctx),
        "manual_joint_groups": [
            {"name": group.name, "joints": list(group.joint_names)}
            for group in ctx.runtime.robot.actuator_groups
        ],
        "obs_mode": config.inference_cfg.obs_space.type,
        "policy_action_mode": config.inference_cfg.action_space.type,
        "publish_mode": "joint",
        "publish_rate": config.inference_cfg.publish_rate,
        "inference_rate": config.inference_cfg.inference_rate,
        "control_channel": {
            "enabled": bool((config.get("control_channel") or {}).get("enabled", False)),
            "host": str((config.get("control_channel") or {}).get("host", "127.0.0.1")),
            "port": int((config.get("control_channel") or {}).get("port", 5757)),
            "commands": control_command_catalog(),
        },
        "collection": {
            "enabled": bool(config.collection.schema.columns),
            "fps": None,
            "dataset_dir": _resolve_dataset_dir(
                config.collection.storage.log_dir if config.collection.schema.columns else ""
            ),
            "teleop": {
                "control_source": str(teleop_config.get("control_source", "transport")),
                "client_type": str(teleop_client_config.get("type", "")),
                "client": dict(teleop_client_config),
                "safety": dict(teleop_config.get("safety") or {}),
            },
            "controls": {
                "mode": control_mode,
                "bindings": controls,
                "groups": control_groups,
            },
            "upload": _collection_upload_payload(config),
            "sftp": {
                "configured": bool(
                    collection_sftp.get("host") and collection_sftp.get("remote_dir")
                ),
                "host": str(collection_sftp.get("host", "")),
                "port": int(collection_sftp.get("port", 22)),
                "remote_dir": str(collection_sftp.get("remote_dir", "")),
            },
        },
        "eval": _serialize_eval(ctx),
        "rl": _serialize_rl(ctx),
        "run_id": Path(ctx.output_dir).name,
    }


def _status_snapshot_for_poll(
    logger_obj: Any,
    task: str | None = None,
    collection_dataset: str | None = None,
    *,
    include_history: bool = False,
) -> dict[str, Any]:
    """Request a recorder snapshot, optionally including disk-backed history."""
    snapshot = dict(
        logger_obj.status_snapshot(
            task,
            include_history=include_history,
            collection_dataset=collection_dataset,
        )
    )
    if not include_history:
        snapshot.pop("episodes", None)
    return snapshot


def _cached_collection_slots(
    config: ConfigDict,
    scene_plan: dict[str, Any],
    dataset: str,
) -> list[Any]:
    entries = tuple(
        (str(entry[0]), int(entry[1])) for entry in config.collection.tasks.get(dataset, ())
    )
    key = (id(scene_plan), dataset, entries)
    with _COLLECTION_SLOT_PLAN_CACHE_LOCK:
        cached = _COLLECTION_SLOT_PLAN_CACHE.get(key)
        if cached is not None and cached[0] is scene_plan:
            _COLLECTION_SLOT_PLAN_CACHE.move_to_end(key)
            return list(cached[1])
    slots = tuple(build_collection_slots(config, scene_plan, dataset))
    with _COLLECTION_SLOT_PLAN_CACHE_LOCK:
        _COLLECTION_SLOT_PLAN_CACHE[key] = (scene_plan, slots)
        _COLLECTION_SLOT_PLAN_CACHE.move_to_end(key)
        while len(_COLLECTION_SLOT_PLAN_CACHE) > _COLLECTION_SLOT_PLAN_CACHE_MAX:
            _COLLECTION_SLOT_PLAN_CACHE.popitem(last=False)
    return list(slots)


def _collection_slots_snapshot(ctx: ConsoleContext, dataset: str) -> dict[str, Any]:
    """Resolve one dataset's fixed plan against saved and in-flight episodes."""
    config = ctx.runtime.active_config or ctx.config
    if dataset not in config.collection.tasks:
        raise ValueError("unknown collection dataset")
    scene_plan = _load_scene_plan(config, dataset)
    slots = _cached_collection_slots(config, scene_plan, dataset)
    logger_obj = ctx.runtime.episode_logger
    dataset_dir: Path | None = None
    queue: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    if logger_obj is not None and slots:
        status = _status_snapshot_for_poll(logger_obj, slots[0].task, dataset)
        raw_dataset_dir = str(status.get("dataset_dir") or "")
        dataset_dir = Path(raw_dataset_dir).resolve() if raw_dataset_dir else None
        queue = list(status.get("queue") or [])
        if dataset_dir is not None:
            episodes = list(load_episode_history(dataset_dir).get("episodes") or [])
    slot_state = load_slot_state(dataset_dir)
    rows, active, counts = collection_slot_status(slots, episodes, queue, slot_state)
    scenes: list[dict[str, str]] = []
    tasks: list[dict[str, str]] = []
    seen_scenes: set[str] = set()
    seen_tasks: set[str] = set()
    for row in rows:
        if row["scene_id"] not in seen_scenes:
            scenes.append({"id": row["scene_id"], "label": row["scene_label"]})
            seen_scenes.add(row["scene_id"])
        if row["task_id"] not in seen_tasks:
            tasks.append(
                {
                    "id": row["task_id"],
                    "label": row["task_zh"] or row["task"],
                    "prompt": row["task"],
                }
            )
            seen_tasks.add(row["task_id"])
    return {
        "dataset": dataset,
        "scene_plan": scene_plan,
        "dataset_dir": "" if dataset_dir is None else str(dataset_dir),
        "rows": rows,
        "active": active,
        "counts": counts,
        "scenes": scenes,
        "tasks": tasks,
        "slot_state": slot_state,
    }


def _apply_collection_slot_session(
    session: SessionState, slot: dict[str, Any] | None
) -> None:
    if slot is None:
        session.collection_scene_id = None
        session.collection_scene_round = None
        session.collection_slot_id = None
        session.collection_task_id = None
        return
    session.selected_collect_task = slot["task"]
    session.selected_collect_set = slot["dataset"]
    session.selected_collect_task_index = int(slot["task_index"])
    session.collection_scene_id = slot["scene_id"]
    session.collection_scene_round = int(slot["round_index"])
    session.collection_slot_id = slot["slot_id"]
    session.collection_task_id = slot["task_id"]


def _sync_collection_slot_session(ctx: ConsoleContext) -> dict[str, Any] | None:
    """Select the next unresolved slot without waiting for a browser poll."""
    dataset = str(ctx.session.selected_collect_set or "").strip()
    logger_obj = ctx.runtime.episode_logger
    if not dataset or logger_obj is None or bool(getattr(logger_obj, "has_active_episode", False)):
        return None
    snapshot = _collection_slots_snapshot(ctx, dataset)
    active = snapshot["active"]
    dataset_dir = str(snapshot["dataset_dir"] or "")
    if active is not None and dataset_dir:
        select_collection_slot(Path(dataset_dir), active["slot_id"], snapshot["slot_state"])
    _apply_collection_slot_session(ctx.session, active)
    logger.info(
        "[COLLECTION_SLOT_SYNC] dataset=%s active=%s",
        dataset,
        None if active is None else active["slot_id"],
    )
    return active


def _serialize_status(ctx: ConsoleContext, *, include_history: bool = False) -> dict:
    # Live snapshot polled by the frontend (~1 Hz). Mirrors the session/runtime
    # state machine so the UI can gate buttons and show telemetry.
    s = ctx.session
    r = ctx.runtime
    config = r.active_config or ctx.config
    hil_status = rollout_hil_status(config, r)
    elapsed_ms = 0
    if s.status.value == "running" and s.run_start_time > 0:
        elapsed_ms = int((time.monotonic() - s.run_start_time) * 1000)
    replay_qpos, replay_frame_index = _active_collection_replay_qpos(r)
    replay_total_frames = 0 if r.collection_replay_qpos is None else len(r.collection_replay_qpos)
    replay_playing = (
        r.collection_replay_qpos is not None
        and r.collection_replay_started > 0.0
        and replay_frame_index is not None
        and replay_frame_index < max(0, replay_total_frames - 1)
    )
    replay_source = r.replay_source
    replay_source_frame = None
    replay_source_total = None
    if replay_source is not None:
        raw_frame = getattr(replay_source, "frame_index", None)
        raw_total = getattr(replay_source, "n_steps", None)
        replay_source_frame = None if raw_frame is None else int(raw_frame)
        replay_source_total = None if raw_total is None else int(raw_total)
        if replay_source_total is not None and replay_source_total > 0:
            replay_source_frame = min(
                max(replay_source_frame or 0, int(s.step_index or 0)),
                replay_source_total - 1,
            )
    # "Online" means a message arrived recently, not just that the socket opened
    # (ZMQ connect() / ROS subscribe succeed even with no publisher). For a live
    # backend, since_recv is None until the first message lands — that is offline,
    # not "configured". Only sources that never track freshness (replay/synthetic)
    # fall back to the configured flag.
    reader = ctx.obs_reader or r.transport
    # Dedicated ZMQ visualization readers only update their freshness when consumed.
    # The status poll must therefore drain pending state even before camera/frame
    # streams have opened, otherwise MANUAL can wait forever despite live hardware.
    refresh_qpos = getattr(reader, "get_latest_qpos", None)
    if refresh_qpos is not None:
        refresh_qpos()
    since_recv = reader.seconds_since_last_recv()
    if since_recv is not None:
        transport_connected = since_recv <= _TRANSPORT_ONLINE_WINDOW_SECONDS
    elif config.transport.type in _LIVE_TRANSPORT_TYPES:
        transport_connected = False
    else:
        transport_connected = bool(config.transport.type)
    image_min_hz = reader.image_min_hz()
    critic_status = (
        r.rl_critic_runner.status()
        if r.rl_critic_runner is not None
        else {
            "connected": False,
            "metadata": {},
            "last_request_ms": 0.0,
            "last_error": r.rl_critic_error,
            "coalesced_requests": 0,
            "samples": 0,
        }
    )
    with r.rl_live_lock:
        policy_samples = sum(1 for sample in r.rl_live_samples if sample[3] == "policy")
        live_sample_count = len(r.rl_live_samples)
    intervention_samples = live_sample_count - policy_samples
    return {
        "robot_type": config.robot.type,
        "transport_type": config.transport.type,
        "transport_connected": transport_connected,
        "image_min_hz": image_min_hz,
        "policy_connected": r.policy is not None,
        "policy_error": r.last_policy_error,
        "cli_mode": s.mode.value,
        "session_status": s.status.value,
        "selected_task": s.selected_task,
        "selected_collect_task": s.selected_collect_task,
        "selected_collect_set": s.selected_collect_set,
        "selected_collect_task_index": s.selected_collect_task_index,
        "collection_scene_id": s.collection_scene_id,
        "collection_scene_round": s.collection_scene_round,
        "collection_slot_id": s.collection_slot_id,
        "collection_task_id": s.collection_task_id,
        "selected_strategy": (
            resolve_inference_strategy_label(config, r.selected_inference_strategy_key)
            if r.selected_inference_strategy_key is not None
            else None
        ),
        "is_setup_done": s.is_setup_done,
        "step_index": s.step_index,
        "chunk_index": s.chunk_index,
        "last_infer_ms": round(s.last_infer_ms, 1),
        "run_elapsed_ms": elapsed_ms,
        "follow_human_gripper": s.follow_human_gripper,
        "gripper_locks": dict(s.gripper_locks),
        "manual_dispatch": s.manual_dispatch,
        "manual_publish_active": s.manual_publish_active,
        "manual_qpos": (
            None if s.manual_qpos is None else np.asarray(s.manual_qpos, dtype=float).tolist()
        ),
        "pending_chunk": None if s.pending_real_chunk is None else len(s.pending_real_chunk),
        "current_episode": getattr(r.replay_source or r.transport, "episode_id", None),
        "last_error": s.last_error,
        "replay_dataset_dir": r.replay_dataset_dir,
        "replay_episode_id": r.replay_episode_id,
        "replay_n_episodes": r.replay_n_episodes,
        "replay_loaded": r.replay_source is not None,
        "replay_task": r.replay_task,
        "replay_frame_index": replay_source_frame,
        "replay_total_frames": replay_source_total,
        "replay_action_key": r.replay_action_key,
        "replay_action_mode": r.replay_action_mode,
        "replay_fps": r.replay_fps,
        "replay_exec_steps": r.replay_exec_steps,
        "setup_stage": r.setup_stage,
        "collection_replay": {
            "active": replay_qpos is not None,
            "playing": replay_playing,
            "episode_index": r.collection_replay_episode,
            "frame_index": replay_frame_index,
            "total_frames": replay_total_frames,
        },
        "collection_teleop_armed": r.collection_teleop_armed,
        "collection_teleop_active": r.collection_teleop_active,
        "teleop": teleop_status(r),
        "rollout_intervention_active": r.rollout_intervention_active,
        "rollout_intervention_enabled": r.rollout_intervention_enabled,
        "rollout_intervention_source": rollout_intervention_source(config),
        "hil_supported": hil_status.supported,
        "hil_active": hil_status.active,
        "hil_error": hil_status.error,
        "rollout_intervention_active_frames": (
            0
            if r.rollout_intervention_active_segment is None
            else len(r.rollout_intervention_active_segment.frames)
        ),
        "rollout_intervention_segments": len(r.rollout_intervention_segments),
        "rollout": rollout_save_status(config, r, include_history=include_history),
        "eval_recorder": (
            _status_snapshot_for_poll(
                r.episode_logger,
                format_task_label(s.selected_task),
                include_history=include_history,
            )
            if config.eval and r.episode_logger is not None
            else None
        ),
        "collect": (
            _status_snapshot_for_poll(
                r.episode_logger,
                format_task_label(s.selected_collect_task),
                s.selected_collect_set,
                include_history=include_history,
            )
            if r.episode_logger is not None
            else None
        ),
        "web_phase": r.web_phase,
        "init_ready": r.init_ready,
        "init_loaded": r.init_qpos is not None,
        "ckpt_slot": r.active_ckpt_slot,
        "ckpt_label": ctx.ckpt_label(r.active_ckpt_slot),
        "clip_id": r.current_clip_id,
        "cell": r.current_cell,
        "rl": {
            "active": r.rl_active,
            "selected_policy_slot": r.rl_selected_policy_slot,
            "selected_critic_slot": r.rl_selected_critic_slot,
            "critic_connected": critic_status["connected"],
            "critic_metadata": critic_status["metadata"],
            "critic_last_request_ms": critic_status["last_request_ms"],
            "critic_error": critic_status["last_error"],
            "coalesced_requests": critic_status["coalesced_requests"],
            "critic_samples": critic_status["samples"],
            "policy_samples": policy_samples,
            "intervention_samples": intervention_samples,
            "replay_episode": r.rl_replay_episode_id,
        },
    }


def _encode_jpeg(image: np.ndarray, convert_bgr_to_rgb: bool, quality: int = 60) -> bytes | None:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if not convert_bgr_to_rgb and arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buf.tobytes()


def _camera_jpeg_payload(
    reader: ObservationSource,
    key: str,
    convert_bgr_to_rgb: bool,
    previous_signature: tuple[Any, ...] | None = None,
) -> tuple[bytes | None, tuple[Any, ...] | None]:
    get_jpeg = getattr(reader, "get_camera_jpeg", None)
    if get_jpeg is not None:
        jpeg = get_jpeg(key)
        if jpeg is not None:
            return jpeg, ("jpeg", hashlib.blake2s(jpeg, digest_size=8).digest())

    get_one = getattr(reader, "get_camera_frame", None)
    if get_one is not None:
        image = get_one(key)
    else:
        frame = reader.get_frame()
        image = None if frame is None else frame.images.get(key)
    if image is None:
        return None, None
    arr = np.asarray(image)
    jpeg = _encode_jpeg(image, convert_bgr_to_rgb)
    if jpeg is None:
        return None, None
    sig = ("array", arr.shape, hashlib.blake2s(jpeg, digest_size=8).digest())
    return jpeg, sig


def _list_camera_keys(ctx: ConsoleContext) -> list[str]:
    # Cameras the visualization stream may expose, in schema order minus disabled ones.
    disabled = set(ctx.config.transport.disabled_cameras)
    reader = _camera_reader(ctx)
    available = getattr(reader, "available_camera_keys", None)
    if available is not None:
        return [key for key in available() if key not in disabled]
    return [
        cam.observation_key
        for cam in ctx.runtime.robot.observation_schema.cameras
        if cam.observation_key not in disabled
    ]


def _live_camera_keys(ctx: ConsoleContext) -> list[str]:
    reader = _camera_reader(ctx)
    get_keys = getattr(reader, "get_camera_keys", None)
    keys = list(get_keys()) if get_keys is not None else _list_camera_keys(ctx)
    get_frame = getattr(reader, "get_camera_frame", None)
    if get_frame is not None:
        # Empty MJPEG responses can occupy all browser HTTP/1 connection slots,
        # preventing Start/Stop requests from reaching the server.
        return [key for key in keys if get_frame(key) is not None]
    return keys


def _uses_replay_observation(ctx: ConsoleContext) -> bool:
    return ctx.active_tab == "replay" and ctx.runtime.replay_source is not None


def _camera_reader(ctx: ConsoleContext):
    preview = getattr(ctx, "camera_preview", None)
    if preview is not None and not _uses_replay_observation(ctx):
        return preview
    return _observation_reader(ctx)


def _observation_reader(ctx: ConsoleContext) -> ObservationSource:
    if _uses_replay_observation(ctx):
        return ctx.runtime.replay_source  # type: ignore[return-value]
    return ctx.obs_reader or ctx.runtime.transport  # type: ignore[return-value]


def _active_collection_replay_qpos(runtime: RuntimeState) -> tuple[np.ndarray | None, int | None]:
    qpos_seq = runtime.collection_replay_qpos
    if qpos_seq is None or len(qpos_seq) == 0:
        return None, None
    if runtime.collection_replay_started <= 0:
        return np.asarray(qpos_seq[0], dtype=float), 0
    elapsed = max(0.0, time.monotonic() - runtime.collection_replay_started)
    frame_index = min(int(elapsed * max(1, runtime.collection_replay_fps)), len(qpos_seq) - 1)
    return np.asarray(qpos_seq[frame_index], dtype=float), frame_index


def _collection_replay_progress(ctx: ConsoleContext, frame_index: int | None) -> None:
    """Drive a single-line tqdm bar for collection-replay playback, recreating it on rewind."""
    qpos_seq = ctx.runtime.collection_replay_qpos
    if frame_index is None or qpos_seq is None:
        _collection_replay_progress_close(ctx)
        return
    total = len(qpos_seq)
    n = min(frame_index + 1, total)
    pbar = ctx.scene_pbar
    if pbar is None or pbar.total != total or n < pbar.n:
        _collection_replay_progress_close(ctx)
        pbar = tqdm(total=total, desc="collection replay", unit="frame", leave=True)
        ctx.scene_pbar = pbar
    pbar.update(n - pbar.n)


def _collection_replay_progress_close(ctx: ConsoleContext) -> None:
    """Close and drop the active collection-replay progress bar, if any."""
    if ctx.scene_pbar is not None:
        ctx.scene_pbar.close()
        ctx.scene_pbar = None


def _qpos_debug_summary(qpos: np.ndarray | None) -> str:
    if qpos is None:
        return "none"
    arr = np.asarray(qpos, dtype=float).reshape(-1)
    head = ", ".join(f"{x:.4f}" for x in arr[: min(6, arr.size)])
    suffix = ", ..." if arr.size > 6 else ""
    return f"dim={arr.size} head=[{head}{suffix}] sum={arr.sum():.4f}"


def _serialize_frame(ctx: ConsoleContext) -> dict:
    # Live qpos + eef for the obs panel. Camera images are streamed separately over
    # MJPEG (/api/camera/<key>) so the heavy JPEG payload never gates this poll.
    reader = _observation_reader(ctx)
    source = ctx.runtime.replay_source if _uses_replay_observation(ctx) else None
    # Replay: report the recorded state of the very frame shown, so qpos never drifts
    # from the video. Live modes report the latest published qpos.
    collection_qpos, _ = _active_collection_replay_qpos(ctx.runtime)
    if collection_qpos is not None:
        qpos = collection_qpos
    elif source is not None and hasattr(source, "get_scene_qpos"):
        qpos = source.get_scene_qpos()  # type: ignore[attr-defined]
    else:
        qpos = reader.get_latest_qpos()
    out: dict = {
        "qpos": None if qpos is None else np.asarray(qpos, dtype=float).tolist(),
        "cameras": _live_camera_keys(ctx),
    }
    health = getattr(_camera_reader(ctx), "camera_health", None)
    if source is None and health is not None:
        enabled = set(_list_camera_keys(ctx))
        out["camera_health"] = {key: value for key, value in health().items() if key in enabled}
    return out


def _serialize_scene(ctx: ConsoleContext) -> dict:
    # Per-frame 3D state: dual-arm per-mesh world transforms from the latest qpos.
    # Cache by qpos signature so a static arm doesn't re-run forward kinematics each poll.
    if ctx.scene is None:
        return {"available": False, "arms": {}}
    # Whole-episode FK has priority over high-rate live scene polling. Return the last
    # rendered pose while the batch owns the mutable yourdfpy graph; otherwise dozens
    # of poll threads can starve the batch before it acquires the scene lock.
    if ctx.scene_batch_loading.is_set():
        return ctx.scene_cache or {"available": False, "arms": {}, "loading": True}
    reader = _observation_reader(ctx)
    session = ctx.session
    collection_qpos, collection_frame_index = _active_collection_replay_qpos(ctx.runtime)
    if collection_qpos is not None:
        qpos = collection_qpos
        source = f"collection_replay:{collection_frame_index}"
    # MANUAL mode: the solid arm follows live robot feedback. The hand-set command
    # is rendered separately as a ghost so entering START never snaps the solid arm
    # to the initial/target pose.
    elif session.mode is SessionMode.MANUAL and ctx.active_tab != "collect":
        state = reader.get_latest_qpos()
        if state is None:
            state = ctx.runtime.robot.initial_qpos
        command = session.manual_qpos
        return {
            "available": True,
            "arms": ctx.scene.transforms(state),  # type: ignore[attr-defined]
            "ghost": (
                ctx.scene.transforms(command)  # type: ignore[attr-defined]
                if command is not None
                else None
            ),
        }
    elif session.sim_preview_qpos is not None and ctx.active_tab != "collect":
        # SIM run/preview: render the last command directly. A replay source may
        # still be mounted after visiting REPLAY, but active SIM commands should
        # drive the console canvas.
        qpos = session.sim_preview_qpos
        source = "sim_preview_qpos"
    elif _uses_replay_observation(ctx) and hasattr(ctx.runtime.replay_source, "get_scene_qpos"):
        # Same frame index the camera/obs panel uses, expanded to the robot qpos layout.
        qpos = ctx.runtime.replay_source.get_scene_qpos()  # type: ignore[attr-defined]
        source = "replay_source.get_scene_qpos"
    elif _uses_replay_observation(ctx) and hasattr(ctx.runtime.replay_source, "get_obs_state"):
        # Same frame index the camera/obs panel uses, keeping the 3D arm in lockstep.
        qpos = ctx.runtime.replay_source.get_obs_state()  # type: ignore[attr-defined]
        source = "replay_source.get_obs_state"
    else:
        qpos = reader.get_latest_qpos()
        source = "reader.get_latest_qpos"
    if qpos is None:
        qpos = ctx.runtime.robot.initial_qpos
        source = "robot.initial_qpos"
    key = None if qpos is None else np.asarray(qpos, dtype=float).round(5).tobytes()
    replay_source = ctx.runtime.replay_source
    replay_frame = None if replay_source is None else getattr(replay_source, "frame_index", None)
    pending_len = None if session.pending_real_chunk is None else len(session.pending_real_chunk)
    log_key = (
        source,
        key,
        session.mode.value,
        session.status.value,
        session.step_index,
        session.chunk_index,
        pending_len,
        replay_frame,
    )
    if collection_qpos is not None:
        _collection_replay_progress(ctx, collection_frame_index)
    elif replay_source is not None and log_key != ctx.scene_log_key:
        logger.info(
            "[REPLAY_SCENE] source=%s mode=%s status=%s step=%d chunk=%d "
            "sim_preview=%s pending=%s replay_loaded=%s replay_frame=%s qpos=%s",
            source,
            session.mode.value,
            session.status.value,
            session.step_index,
            session.chunk_index,
            session.sim_preview_qpos is not None,
            pending_len,
            replay_source is not None,
            replay_frame,
            _qpos_debug_summary(qpos),
        )
        ctx.scene_log_key = log_key
    if key == ctx.scene_cache_key and ctx.scene_cache is not None:
        return ctx.scene_cache
    transforms = ctx.scene.transforms(qpos)  # type: ignore[attr-defined]
    payload = {"available": True, "arms": transforms}
    ctx.scene_cache_key = key
    ctx.scene_cache = payload
    return payload


def _serialize_manual_scene(ctx: ConsoleContext) -> dict:
    # MANUAL tab: two overlaid arms. "command" is the robot's live observed state
    # (solid arm); "ghost" is the staged target qpos from the sliders. Never cached:
    # both move.
    if ctx.scene is None:
        return {"available": False}
    state = ctx.runtime.transport.get_latest_qpos()
    if state is None:
        state = ctx.runtime.robot.initial_qpos
    command = ctx.session.manual_qpos
    if command is None:
        command = state
    return {
        "available": True,
        "command": ctx.scene.transforms(state),  # type: ignore[attr-defined]
        "ghost": ctx.scene.transforms(command),  # type: ignore[attr-defined]
    }


def _serialize_live_series(ctx: ConsoleContext, since: int) -> dict:
    # In-memory state/action of the in-flight run, for the live console charts.
    # eval/default recording uses episode_logger as the live step buffer; collection
    # has no executed-action series.
    episode_logger = ctx.runtime.episode_logger
    logger_obj = (
        episode_logger
        if episode_logger is not None and not episode_logger.is_collection_enabled
        else None
    )
    if logger_obj is None:
        return {"active": False, "n": 0, "timestamp": [], "state": [], "action": []}
    return logger_obj.live_series(since)


class ConsoleRequestHandler(BaseHTTPRequestHandler):
    """Routes browser requests to the command_queue + live state snapshots."""

    ctx: ConsoleContext  # set by start_console_server before constructing the server
    _mesh_cache: dict[tuple[int, str], tuple[bytes, bytes]] = {}
    # Whole-episode replay transforms, keyed by (scene, replay_source, episode, n_frames)
    # so an episode swap invalidates automatically. Bounded to a couple entries; the
    # lock serializes the expensive first compute so concurrent first GETs don't double it.
    _xf_cache: dict[tuple, tuple[bytes, bytes]] = {}
    _xf_lock = threading.Lock()
    # HTTP/1.1 keep-alive: the frontend fires ~20 JSON polls/sec; reusing one TCP
    # connection (instead of a fresh connect + handler thread per request) is the
    # single biggest console-latency win. Every response below carries an explicit
    # Content-Length (or is a never-ending MJPEG stream), so persistent connections
    # frame cleanly. timeout reaps idle keep-alive sockets so threads don't pile up.
    protocol_version = "HTTP/1.1"
    timeout = 30

    def _request_trace_id(self) -> str:
        supplied = self.headers.get("X-EVA-Trace-ID", "").strip()
        if supplied and len(supplied) <= 96 and _TRACE_EVENT_RE.fullmatch(supplied):
            return supplied
        return f"srv-{random.getrandbits(40):010x}"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Silence the default per-request access log (unless CONSOLE_ACCESS_LOG is set)."""
        if os.environ.get("CONSOLE_ACCESS_LOG"):
            logger.info("[ACCESS] %s", format % args)
        return

    def handle(self) -> None:
        """End quietly when the browser closes a keep-alive request mid-flight."""
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_empty(self, status: int) -> None:
        # HTTP/1.1 requires a framed body even for errors; without Content-Length a
        # keep-alive client would block waiting for a body that never comes.
        self._response_status = status
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _query_int(self, key: str, default: int) -> int:
        # One query param as an int (first value), falling back to ``default``.
        vals = parse_qs(urlparse(self.path).query).get(key)
        return int(vals[0]) if vals and vals[0] else default

    def _query_str(self, key: str, default: str = "") -> str:
        vals = parse_qs(urlparse(self.path).query).get(key)
        return vals[0] if vals and vals[0] else default

    def _send_json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._response_status = status
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path: Path) -> None:
        if not path.exists():
            self._send_empty(404)
            return
        body = path.read_bytes()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".woff2": "font/woff2",
            ".woff": "font/woff",
            ".ttf": "font/ttf",
        }.get(path.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_mesh(self, filename: str) -> None:
        # Serve one registered URDF geom from the already-loaded trimesh scene.
        scene = self.ctx.scene
        if scene is None:
            self._send_empty(404)
            return
        name = unquote(filename)
        use_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        cache = type(self)._mesh_cache
        cache_key = (id(scene), name)
        entry = cache.get(cache_key)
        if entry is None:
            raw = scene.mesh_bytes(name)  # type: ignore[attr-defined]
            if raw is None:
                self._send_empty(404)
                return
            entry = (raw, gzip.compress(raw))
            cache[cache_key] = entry
        body = entry[1] if use_gzip else entry[0]
        if len(body) == 0:
            self._send_empty(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def _enqueue(self, command: str) -> None:
        assert self.ctx.runtime.command_queue is not None
        logger.info(
            "[COMMAND_ENQUEUE] trace=%s command=%s",
            getattr(self, "_active_trace_id", "none"),
            _command_summary(command),
        )
        self.ctx.runtime.command_queue.put(command)

    def _invalidate_scene_cache(self) -> None:
        # Drop the cached FK transforms so the next /api/scene poll recomputes for the
        # newly loaded replay qpos instead of returning the previous frame.
        self.ctx.scene_cache_key = None
        self.ctx.scene_cache = None
        self.ctx.scene_log_key = None

    def _clear_collection_replay(self) -> None:
        # Leave collection/review replay: forget the loaded qpos so the scene falls back
        # to the live reader.
        self.ctx.runtime.collection_replay_qpos = None
        self.ctx.runtime.collection_replay_episode = None
        _collection_replay_progress_close(self.ctx)

    def _start_collection_replay(self, body: dict) -> None:
        runtime = self.ctx.runtime
        episode_logger = runtime.episode_logger
        if episode_logger is None:
            self._send_json(404, {"ok": False, "error": "collection logger unavailable"})
            return
        episode_index = int(body.get("episode", 0))
        task = format_task_label(self.ctx.session.selected_collect_task)
        collection_dataset = self.ctx.session.selected_collect_set
        qpos = episode_logger.load_collection_replay_qpos(
            episode_index,
            task,
            collection_dataset=collection_dataset,
        )
        if qpos is None or len(qpos) == 0:
            self._send_json(404, {"ok": False, "error": "collection episode unavailable"})
            return
        runtime.collection_replay_qpos = qpos
        runtime.collection_replay_episode = episode_index
        runtime.collection_replay_started = time.monotonic()
        replay_fps = episode_logger.load_collection_episode_fps(
            episode_index,
            task,
            collection_dataset=collection_dataset,
        )
        runtime.collection_replay_fps = max(
            1, int(round(replay_fps or self.ctx.config.inference_cfg.publish_rate))
        )
        self._invalidate_scene_cache()
        logger.info(
            "[COLLECT_REPLAY_LOAD] episode=%d frames=%d fps=%d first=%s last=%s",
            episode_index,
            int(qpos.shape[0]),
            runtime.collection_replay_fps,
            _qpos_debug_summary(qpos[0]),
            _qpos_debug_summary(qpos[-1]),
        )
        self._send_json(
            200,
            {"ok": True, "episode": episode_index, "frames": int(qpos.shape[0])},
        )

    def _post_review_episode(self, body: dict) -> None:
        dataset_dir = _resolve_dataset_dir(str(body.get("dataset_dir", "")).strip())
        episode_index = int(body.get("episode", 0))
        try:
            cached = _cached_review_episode(self.ctx, dataset_dir, episode_index)
            payload = {
                "ok": True,
                "episode": episode_index,
                "frames": len(cached.scene_qpos),
                "fps": cached.fps,
                "task": cached.task,
                "video_keys": cached.video_keys,
                **cached.series,
            }
        except Exception as error:
            self._send_json(404, {"ok": False, "error": str(error)})
            return
        if payload["frames"] <= 0:
            self._send_json(404, {"ok": False, "error": "episode has no frames"})
            return
        self._send_json(200, payload)

    def _eval_start(self, body: dict) -> None:
        # EVAL tab "RUN": bind a fresh clip_id + prompt/trial to this run. The recorder
        # stamps these onto the episode's meta at web:start (set_episode_meta), so the
        # recorded lerobot episode itself owns the trial identity — no separate
        # results.jsonl. An unrun trial simply has no episode row yet (and so can't be
        # scored), which is the gate we want.
        runtime = self.ctx.runtime
        clip_id = body.get("clip_id") or None
        prompt = body.get("prompt")
        trial = body.get("trial")
        runtime.current_clip_id = clip_id
        runtime.current_cell = {"prompt": prompt, "trial": trial} if clip_id else None
        self._enqueue("web:start")
        self._send_json(200, {"ok": True, "clip_id": clip_id})

    def _eval_score(self, body: dict) -> None:
        # Submit/overwrite a trial's score+milestones+note. The score is written ONLY into
        # the recorded lerobot episode (meta/episodes.jsonl + per-frame parquet columns +
        # info.json), keyed by clip_id -> episode_index. Gate: refuse to score a clip that
        # has no recorded episode yet (model still initializing, or the trial never ran).
        runtime = self.ctx.runtime
        clip_id = body.get("clip_id")
        if not clip_id:
            self._send_json(400, {"ok": False, "error": "clip_id required"})
            return
        if runtime.episode_logger is None:
            self._send_json(
                409,
                {"ok": False, "error": "no recorded episode for this trial yet — run it first"},
            )
            return
        fields = dict(
            prompt=body.get("prompt"),
            trial=body.get("trial"),
            score=body.get("score"),
            max_score=body.get("max_score"),
            milestones=body.get("milestones", {}),
            note=body.get("note", ""),
            scored_at=_dt.datetime.now().isoformat(timespec="seconds"),
            duration_ms=body.get("duration_ms", 0),
        )
        ep = runtime.episode_logger.patch_episode_meta_by_clip(
            str(clip_id),
            **fields,
        )
        if ep is None:
            self._sync_preview_to_active_model()
            if self.ctx.preview is not None:
                ep = self.ctx.preview.episode_index_by_clip().get(clip_id)
            if ep is None:
                self._send_json(
                    409,
                    {"ok": False, "error": "no recorded episode for this trial yet — run it first"},
                )
                return
            runtime.episode_logger.patch_episode_meta(int(ep), **fields)
        self._send_json(200, {"ok": True, "episode_index": ep})

    def _stream_camera(self, key: str) -> None:
        # Push raw JPEG frames over multipart/x-mixed-replace so the browser <img>
        # decodes them natively — no base64, no JSON, no per-frame request overhead.
        ctx = self.ctx
        if key not in _list_camera_keys(ctx):
            self._send_empty(404)
            return
        convert = ctx.config.transport.convert_bgr_to_rgb
        boundary = "evaframe"
        self.close_connection = True
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()
        period = 1.0 / _CAMERA_STREAM_FPS
        # The profiled bottleneck is bandwidth, not CPU: pushing 30 fps of a 720p +
        # a 480p JPEG is ~17 MB/s, and a paused/replay view re-sends one *static*
        # frame forever. Cap the stream at 10 FPS and encode + transmit only when
        # the pixels actually change (cheap subsampled signature), with a slow
        # heartbeat so idle proxies and late-joining <img> tags still get a frame.
        heartbeat = 2.0
        last_sig = None
        last_jpeg: bytes | None = None
        last_sent = 0.0
        try:
            while not ctx.runtime.transport.is_shutdown():
                tick = time.monotonic()
                # Re-resolve per tick so mounting/clearing a replay source mid-stream
                # swaps the feed without the client reopening the <img> connection.
                reader = _camera_reader(ctx)
                jpeg = None
                payload, sig = _camera_jpeg_payload(reader, key, convert, last_sig)
                if payload is not None and sig != last_sig:
                    jpeg = payload
                    last_sig = sig
                    last_jpeg = jpeg
                elif last_jpeg is not None and tick - last_sent >= heartbeat:
                    jpeg = last_jpeg
                if jpeg is not None:
                    part = (
                        b"--"
                        + boundary.encode()
                        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.write(part)
                    last_sent = tick
                rest = period - (time.monotonic() - tick)
                if rest > 0:
                    time.sleep(rest)
        except (BrokenPipeError, ConnectionResetError):
            # Browser closed the tab / swapped the <img> src — expected, end quietly.
            return

    def _stream_teleop_feedback(self) -> None:
        """Stream the small, high-rate VR control snapshot used by Collect hints."""
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private")
        self.send_header("Connection", "keep-alive")
        # Prevent buffering when the console is served behind nginx.
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.wfile.write(b"retry: 1000\n\n")
        self.wfile.flush()

        period = 1.0 / 60.0
        next_tick = time.monotonic()
        try:
            while not self.ctx.runtime.transport.is_shutdown():
                status = teleop_status(self.ctx.runtime) or {}
                payload = {
                    "connected": bool(status.get("connected", False)),
                    "pressed_controls": status.get("pressed_controls", []),
                    "hold_progress": status.get("hold_progress", {}),
                }
                body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                self.wfile.write(f"data: {body}\n\n".encode("utf-8"))
                self.wfile.flush()

                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    # Do not accumulate lag if the process was briefly busy.
                    next_tick = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            # Browser closed the EventSource or navigated away.
            return

    # --- routing ---

    def do_GET(self) -> None:
        """Route a GET to a stream, static asset, mesh, or registered JSON handler."""
        path = urlparse(self.path).path
        # Prefix routes (streams + static assets) come first; they own a path subtree.
        if path in {"/", "/index.html"}:
            self._send_static(STATIC_DIR / "index.html")
            return
        if path == "/task_preview.html":
            self._send_static(STATIC_DIR / "task_preview.html")
            return
        if path.startswith("/api/camera/"):
            self._stream_camera(unquote(path[len("/api/camera/") :]))
            return
        if (
            path.startswith("/fonts/")
            or path.startswith("/vendor/")
            or path.startswith("/css/")
            or path.startswith("/js/")
        ):
            self._serve_asset(path)
            return
        if path.startswith("/meshes/"):
            self._send_mesh(path[len("/meshes/") :])
            return
        if path.startswith("/api/scene_plan/object-photo/"):
            self._send_scene_plan_photo(path[len("/api/scene_plan/object-photo/") :])
            return
        handler = _GET_ROUTES.get(path)
        if handler is None:
            self._send_empty(404)
            return
        handler(self)

    # --- GET route handlers (exact-path; registered in _GET_ROUTES) ---

    def _serve_asset(self, path: str) -> None:
        # Static font/vendor file, guarded against path traversal outside STATIC_DIR.
        target = (STATIC_DIR / path.lstrip("/")).resolve()
        if STATIC_DIR.resolve() in target.parents:
            self._send_static(target)
        else:
            self._send_empty(404)

    def _get_config(self) -> None:
        self._send_json(200, _serialize_config(self.ctx))

    def _get_scene_plan(self) -> None:
        try:
            plan = _load_scene_plan(
                self.ctx.runtime.active_config or self.ctx.config,
                self._query_str("set").strip() or None,
            )
        except ValueError as error:
            self._send_json(400, {"ok": False, "error": str(error)})
            return
        self._send_json(200, plan)

    def _send_scene_plan_photo(self, suffix: str) -> None:
        parts = [unquote(part) for part in suffix.split("/")]
        if len(parts) != 2:
            self._send_empty(404)
            return
        try:
            root = _scene_plan_root(
                self.ctx.runtime.active_config or self.ctx.config,
                self._query_str("set").strip() or None,
            ).resolve()
        except ValueError:
            self._send_empty(404)
            return
        info = _read_scene_plan_yaml(root, "info.yaml")
        objects_file = str(info.get("objects_file", "objects.csv") or "objects.csv")
        row = next(
            (
                item
                for item in _read_scene_plan_csv(root, objects_file)
                if str(item.get("object_id") or item.get("物品代码") or "").strip() == parts[0]
            ),
            None,
        )
        if not row:
            self._send_empty(404)
            return
        objects_path = (root / objects_file).resolve()
        photo_dir = str(
            row.get("photo_dir") or row.get("照片目录") or row.get("名称") or ""
        ).strip()
        photo_root = (objects_path.parent / "object_photos").resolve()
        target = (photo_root / photo_dir / parts[1]).resolve()
        if photo_root not in target.parents or not target.is_file():
            self._send_empty(404)
            return
        self._send_static(target)

    def _get_collection_slots(self) -> None:
        dataset = self._query_str("dataset").strip()
        try:
            snapshot = _collection_slots_snapshot(self.ctx, dataset)
            requested_page = max(1, self._query_int("page", 1))
        except ValueError as error:
            self._send_json(400, {"ok": False, "error": str(error)})
            return
        scene_id = self._query_str("scene").strip()
        task_id = self._query_str("task").strip()
        show_all = self._query_str("all") == "1"
        rows = snapshot["rows"]
        if scene_id:
            rows = [row for row in rows if row["scene_id"] == scene_id]
        if task_id:
            rows = [row for row in rows if row["task_id"] == task_id]
        if not show_all and not scene_id and not task_id:
            rows = []
        page_size = 50
        active_slot_id = str((snapshot["active"] or {}).get("slot_id") or "")
        active_index = next(
            (index for index, row in enumerate(rows) if row["slot_id"] == active_slot_id),
            -1,
        )
        active_page = active_index // page_size + 1 if active_index >= 0 else None
        page_count = max(1, (len(rows) + page_size - 1) // page_size)
        page = min(requested_page, page_count)
        start = (page - 1) * page_size
        self._send_json(
            200,
            {
                "ok": True,
                "dataset": snapshot["dataset"],
                "scene_plan": snapshot["scene_plan"],
                "dataset_dir": snapshot["dataset_dir"],
                "active": snapshot["active"],
                "counts": snapshot["counts"],
                "scenes": snapshot["scenes"],
                "tasks": snapshot["tasks"],
                "slots": rows[start : start + page_size],
                "filtered_total": len(rows),
                "page": page,
                "page_size": page_size,
                "page_count": page_count,
                "active_page": active_page,
                "viewer_active": show_all or bool(scene_id or task_id),
            },
        )

    def _get_status(self) -> None:
        # This poll doubles as the client-liveness heartbeat the main loop watches.
        self.ctx.runtime.last_client_poll = time.monotonic()
        self._send_json(200, _serialize_status(self.ctx))

    def _get_episodes(self) -> None:
        """Serve saved episode rows separately from the watchdog status poll.

        ``since`` is an offset into the history list and ``limit`` bounds one
        response. The frontend and API clients use ``next_since`` plus ``cursor``
        for incremental pages; a prefix rewrite requests a reset in the same response.
        ``queue`` remains in this response because queued/in-flight rows are live
        status and are not present in ``episodes.jsonl`` yet.
        """
        scope = self._query_str("scope", "rollout").strip().lower()
        if scope not in {"collect", "rollout", "eval"}:
            self._send_json(400, {"ok": False, "error": "invalid episode history scope"})
            return
        try:
            since = max(0, self._query_int("since", 0))
            limit_value = self._query_int("limit", _EPISODE_HISTORY_DEFAULT_LIMIT)
        except ValueError:
            self._send_json(400, {"ok": False, "error": "invalid episode history cursor"})
            return
        limit = min(_EPISODE_HISTORY_MAX_LIMIT, max(0, limit_value))
        cursor = self._query_str("cursor").strip() or None

        runtime = self.ctx.runtime
        config = runtime.active_config or self.ctx.config
        queue: list[dict[str, Any]] = []
        dataset_dir: str = ""
        task_filter = None
        collection_set = None
        if scope == "rollout":
            status = rollout_save_status(config, runtime, include_history=False)
            dataset_dir = str(status.get("dataset_dir", ""))
            queue = list(status.get("queue") or [])
        else:
            logger_obj = runtime.episode_logger
            task_filter = (
                self._query_str("task")
                if scope == "collect"
                else format_task_label(self.ctx.session.selected_task)
            ) or None
            collection_set = self._query_str("set").strip() or None
            if logger_obj is not None:
                status = _status_snapshot_for_poll(
                    logger_obj,
                    task_filter,
                    collection_set if scope == "collect" else None,
                )
                dataset_dir = str(status.get("dataset_dir", ""))
                queue = list(status.get("queue") or [])
                if scope == "collect" and task_filter:
                    queue = [
                        row
                        for row in queue
                        if str(row.get("task") or row.get("prompt") or "") == task_filter
                    ]

        explicit_dir = self._query_str("dataset_dir")
        if explicit_dir:
            active_dataset_dir = dataset_dir
            dataset_dir = _resolve_dataset_dir(explicit_dir)
            if (
                not active_dataset_dir
                or Path(active_dataset_dir).resolve() != Path(dataset_dir).resolve()
            ):
                queue = []
        if not dataset_dir:
            self._send_json(
                200,
                {
                    "ok": True,
                    "scope": scope,
                    **({"set": collection_set, "task": task_filter} if scope == "collect" else {}),
                    "episodes": [],
                    "queue": [],
                    "total": 0,
                    "since": since,
                    "next_since": since,
                    "has_more": False,
                    "version": "",
                    "cursor": "",
                    "reset": bool(cursor or since),
                },
            )
            return

        history = load_episode_history(
            Path(dataset_dir),
            task=task_filter,
            since=since,
            limit=limit,
            cursor=cursor,
            exclude_episode_indices={
                int(row["episode_index"]) for row in queue if row.get("episode_index") is not None
            },
        )
        self._send_json(
            200,
            {
                "ok": True,
                "scope": scope,
                "dataset_dir": dataset_dir,
                **({"set": collection_set, "task": task_filter} if scope == "collect" else {}),
                "queue": queue,
                **history,
            },
        )

    def _get_dashboard(self) -> None:
        config = self.ctx.runtime.active_config or self.ctx.config
        roots = self._dashboard_roots(config)
        payload = build_dashboard(
            roots,
            start_date=self._query_str("from"),
            end_date=self._query_str("to"),
        )
        payload["upload"] = _collection_upload_payload(config)
        for candidate in payload["upload_candidates"]:
            specs = resolve_dataset_uploads(
                config.collection.storage,
                str(candidate["dataset"]),
            )
            candidate["remote_dirs"] = [spec.target for spec in specs]
        self._send_json(
            200,
            payload,
        )

    def _dashboard_roots(self, config: ConfigDict | None = None) -> tuple[Path, ...]:
        active_config = config or self.ctx.runtime.active_config or self.ctx.config
        work_root = Path(_resolve_dataset_dir(str(active_config.get("work_dir") or "work_dirs")))
        roots = [work_root]
        if self.ctx.output_dir:
            roots.append(Path(self.ctx.output_dir))
        return tuple(roots)

    def _get_collect_quality_upload(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        job_id = str((query.get("job_id") or [""])[0])
        with self.ctx.quality_upload_lock:
            payload = _quality_job_payload(self.ctx.quality_upload_jobs, job_id)
        if payload is None:
            self._send_json(404, {"ok": False, "error": "upload job not found"})
            return
        self._send_json(200, {"ok": True, **payload})

    def _get_collect_quality_export(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        job_id = str((query.get("job_id") or [""])[0])
        with self.ctx.quality_upload_lock:
            payload = _quality_job_payload(self.ctx.quality_export_jobs, job_id)
        if payload is None:
            self._send_json(404, {"ok": False, "error": "export job not found"})
            return
        self._send_json(200, {"ok": True, **payload})

    def _get_frame(self) -> None:
        self._send_json(200, _serialize_frame(self.ctx))

    def _get_scene(self) -> None:
        self._send_json(200, _serialize_scene(self.ctx))

    def _get_manual_scene(self) -> None:
        self._send_json(200, _serialize_manual_scene(self.ctx))

    def _get_meshes(self) -> None:
        scene = self.ctx.scene
        meshes = scene.static_meshes() if scene is not None else []  # type: ignore[attr-defined]
        arms = scene.arm_names if scene is not None else []  # type: ignore[attr-defined]
        self._send_json(200, {"available": scene is not None, "meshes": meshes, "arms": arms})

    def _get_results(self) -> None:
        # Results come straight from the active model's lerobot dataset (its
        # meta/episodes.jsonl), so a model's prior eval (scores included) auto-loads.
        self._sync_preview_to_active_model()
        records = self.ctx.preview.result_rows() if self.ctx.preview is not None else []
        self._send_json(
            200,
            {
                "run_id": Path(self.ctx.output_dir).name,
                "ckpt_slot": self.ctx.runtime.active_ckpt_slot,
                "model_name": self.ctx.active_model_name(),
                "records": records,
            },
        )

    def _get_live_series(self) -> None:
        since = self._query_int("since", 0)
        self._send_json(200, _serialize_live_series(self.ctx, max(since, 0)))

    def _get_rl_series(self) -> None:
        if not self._require_rl_active():
            return
        since = self._query_int("since", 0)
        critic_since = self._query_int("critic_since", 0)
        include_samples = self._query_str("samples", "1") != "0"
        self._send_json(
            200,
            rl_live_series(
                self.ctx.runtime,
                max(since, 0),
                max(critic_since, 0),
                include_samples=include_samples,
            ),
        )

    def _sync_preview_to_active_model(self) -> None:
        # Keep the preview pointed at the active model's dataset (it changes on ckpt
        # switch). The logger already roots there; mirror that path into the preview.
        if self.ctx.preview is None:
            return
        logger_obj = self.ctx.runtime.episode_logger
        if logger_obj is not None:
            self.ctx.preview.set_dataset_root(Path(logger_obj._log_dir))

    def _point_preview_for_request(self) -> None:
        # RESULT browsing may inspect a non-active model: an explicit ?model=<name> points
        # the preview at that per-model dataset; without it we fall back to the active model.
        model = self._query_str("model")
        if self.ctx.preview is not None and model:
            self.ctx.preview.set_dataset_root(eval_episode_dir(self.ctx.output_dir, model))
            return
        self._sync_preview_to_active_model()

    def _get_results_all(self) -> None:
        # Cross-model aggregation for the RESULT browser: one record list per model
        # discovered under the eval run root, plus the eval prompt/milestone config.
        # Each model's raw dataset lives at <output_dir>/<model>/episodes/raw.
        # whose meta/episodes.jsonl exists so prior evals auto-load.
        from core.app.console.episode_preview import read_result_rows
        from core.recorder.episode import sanitize_path_component

        eval_cfg = self.ctx.config.eval
        root = Path(self.ctx.output_dir)
        configured: list[str] = []
        if eval_cfg is not None and eval_cfg.checkpoints:
            configured = [c.name for c in eval_cfg.checkpoints]
        elif eval_cfg is not None:
            configured = [self.ctx.active_model_name() or "model"]
        seen: set[str] = set()
        models: list[dict] = []
        for name in configured:
            seen.add(sanitize_path_component(name))
            rows = read_result_rows(eval_episode_dir(str(root), name))
            models.append({"model_name": name, "records": rows})
        # Pick up any other recorded model dirs (e.g. a renamed ckpt from a prior run).
        if root.exists():
            for sub in sorted(p for p in root.iterdir() if p.is_dir()):
                if sub.name in seen or sub.name == "console":
                    continue
                rows = read_result_rows(sub / "episodes" / "raw")
                if rows:
                    models.append({"model_name": sub.name, "records": rows})
        tasks = []
        if eval_cfg is not None:
            tasks = [_serialize_prompt_config(t) for t in eval_cfg.tasks]
        self._send_json(
            200,
            {
                "run_id": Path(self.ctx.output_dir).name,
                "results_dir": str(self.ctx.output_dir),
                "trials_per_prompt": eval_cfg.trials_per_prompt if eval_cfg else 5,
                "tasks": tasks,
                "models": models,
            },
        )

    def _get_episode_series(self) -> None:
        self._point_preview_for_request()
        ep = self._query_int("episode_index", -1)
        series = self.ctx.preview.series(ep) if (self.ctx.preview and ep >= 0) else None
        if series is None:
            self._send_json(404, {"error": "episode series not found"})
            return
        self._send_json(200, series)

    def _get_replay_series(self) -> None:
        # Whole-episode state/action of the currently loaded replay dataset, so the
        # REPLAY charts plot and scrub on the same frame index as cameras + URDF.
        replay = self.ctx.runtime.replay_source
        if replay is None or not hasattr(replay, "series"):
            replay = self.ctx.runtime.transport
        if replay is None or not hasattr(replay, "series"):
            self._send_json(404, {"error": "no replay episode loaded"})
            return
        self._send_json(200, replay.series())  # type: ignore[attr-defined]

    def _get_replay_scene_frame(self) -> None:
        # Per-frame URDF world transforms of the loaded replay dataset at an arbitrary
        # frame, so the console's Three.js can render any frame on demand (the RESULT
        # tab's setFrame model) — driven by the single scrub clock, not the live poll.
        ctx = self.ctx
        frame = self._query_int("frame", 0)
        replay = ctx.runtime.replay_source
        if ctx.scene is None or replay is None or not hasattr(replay, "get_scene_qpos"):
            self._send_json(200, {"available": False, "arms": {}})
            return
        qpos = replay.get_scene_qpos(frame)  # type: ignore[attr-defined]
        arms = ctx.scene.transforms(qpos)  # type: ignore[attr-defined]
        self._send_json(200, {"available": True, "arms": arms})

    def _transform_range(self) -> tuple[int, int] | None:
        start = self._query_int("start", -1)
        count = self._query_int("count", -1)
        if start < 0 and count < 0:
            return None
        if start < 0 or count <= 0:
            self._send_json(400, {"available": False, "error": "invalid transform range"})
            return (-1, -1)
        return start, count

    def _send_transform_blob(self, raw: bytes, start: int, count: int, total: int) -> None:
        use_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        body = gzip.compress(raw) if use_gzip else raw
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-EVA-Transform-Start", str(start))
        self.send_header("X-EVA-Transform-Count", str(count))
        self.send_header("X-EVA-Transform-Total", str(total))
        self.send_header("Cache-Control", "no-store")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def _get_replay_transforms(self) -> None:
        # Whole-episode URDF world transforms in one binary blob, so the console can
        # play a replay locally (one fetch + local frame cursor) instead of an FK +
        # round-trip per frame. Computed once per episode and cached (gzip-aware).
        ctx = self.ctx
        scene = ctx.scene
        replay = ctx.runtime.replay_source
        if scene is None or replay is None or not hasattr(replay, "get_scene_qpos"):
            self._send_json(200, {"available": False})
            return
        n = int(getattr(replay, "n_steps", 0) or 0)
        if n <= 0:
            self._send_json(200, {"available": False})
            return
        transform_range = self._transform_range()
        if transform_range == (-1, -1):
            return
        if transform_range is not None:
            start, count = transform_range
            end = min(n, start + count)
            if start >= n:
                self._send_json(404, {"available": False, "error": "transform frame out of range"})
                return
            seq = np.stack(
                [
                    np.asarray(replay.get_scene_qpos(index), dtype=np.float32)  # type: ignore[attr-defined]
                    for index in range(start, end)
                ]
            )
            effective = ctx.runtime.active_config or ctx.config
            raw = _compute_transform_blob(
                scene,
                seq,
                str(effective.robot.type),
                effective.robot.gripper_open,
                effective.robot.gripper_close,
                ctx.scene_batch_loading,
            )
            self._send_transform_blob(raw, start, end - start, n)
            return
        key = (id(scene), id(replay), int(getattr(replay, "episode_id", -1)), n)
        with type(self)._xf_lock:
            cache = type(self)._xf_cache
            entry = cache.get(key)
            if entry is None:
                seq = np.stack(
                    [np.asarray(replay.get_scene_qpos(i), dtype=np.float32) for i in range(n)]  # type: ignore[attr-defined]
                )
                effective = ctx.runtime.active_config or ctx.config
                raw = _compute_transform_blob(
                    scene,
                    seq,
                    str(effective.robot.type),
                    effective.robot.gripper_open,
                    effective.robot.gripper_close,
                    ctx.scene_batch_loading,
                )
                entry = (raw, gzip.compress(raw))
                # Bound the cache to the two most recent episodes.
                if len(cache) >= 2:
                    cache.pop(next(iter(cache)))
                cache[key] = entry
        self._send_transform_blob(entry[0], 0, n, n)

    def _get_review_transforms(self) -> None:
        dataset_dir = _resolve_dataset_dir(self._query_str("dataset_dir"))
        episode_index = self._query_int("episode", -1)
        scene = self.ctx.scene
        if scene is None or episode_index < 0:
            self._send_json(404, {"ok": False, "error": "review transforms unavailable"})
            return
        try:
            cached = _cached_review_episode(self.ctx, dataset_dir, episode_index)
            n = len(cached.scene_qpos)
            if n <= 0:
                self._send_json(404, {"ok": False, "error": "episode has no frames"})
                return
            transform_range = self._transform_range()
            if transform_range == (-1, -1):
                return
            if transform_range is not None:
                start, count = transform_range
                end = min(n, start + count)
                if start >= n:
                    self._send_json(
                        404, {"available": False, "error": "transform frame out of range"}
                    )
                    return
                effective = self.ctx.runtime.active_config or self.ctx.config
                raw = _compute_transform_blob(
                    scene,
                    cached.scene_qpos[start:end],
                    str(effective.robot.type),
                    effective.robot.gripper_open,
                    effective.robot.gripper_close,
                    self.ctx.scene_batch_loading,
                )
                self._send_transform_blob(raw, start, end - start, n)
                return
            key = (id(scene), str(Path(dataset_dir).resolve()), episode_index, n)
            with type(self)._xf_lock:
                cache = type(self)._xf_cache
                entry = cache.get(key)
                if entry is None:
                    effective = self.ctx.runtime.active_config or self.ctx.config
                    raw = _compute_transform_blob(
                        scene,
                        cached.scene_qpos,
                        str(effective.robot.type),
                        effective.robot.gripper_open,
                        effective.robot.gripper_close,
                        self.ctx.scene_batch_loading,
                    )
                    entry = (raw, gzip.compress(raw))
                    if len(cache) >= 4:
                        cache.pop(next(iter(cache)))
                    cache[key] = entry
        except Exception as error:
            self._send_json(404, {"ok": False, "error": str(error)})
            return
        self._send_transform_blob(entry[0], 0, n, n)

    def _get_replay_video(self) -> None:
        # Stream the loaded replay dataset's recorded mp4 for one camera (HTTP Range),
        # so the REPLAY cam panel uses a native <video> the browser seeks frame-exactly.
        cam = self._query_str("cam")
        dataset_dir = self._query_str("dataset_dir")
        episode = self._query_int("episode", -1)
        video_key = self._query_str("video_key")
        if dataset_dir and episode >= 0:
            root = _resolve_runtime_path(dataset_dir)
            if not video_key:
                video_key = _infer_replay_video_key(root, cam)
            videos = LeRobotDatasetIO(root).episode_video_paths(
                episode, {cam or video_key: video_key}
            )
            video = _pick_video(videos, cam)
            if video is None:
                self._send_empty(404)
                return
            self._send_video(video)
            return

        replay = self.ctx.runtime.replay_source
        videos = replay.video_paths() if (replay and hasattr(replay, "video_paths")) else {}  # type: ignore[attr-defined]
        video = _pick_video(videos, cam)
        if video is None:
            self._send_empty(404)
            return
        self._send_video(video)

    def _get_replay_poster(self) -> None:
        cam = self._query_str("cam")
        dataset_dir = self._query_str("dataset_dir")
        episode = self._query_int("episode", -1)
        video_key = self._query_str("video_key")
        if dataset_dir and episode >= 0:
            root = _resolve_runtime_path(dataset_dir)
            if not video_key:
                video_key = _infer_replay_video_key(root, cam)
            videos = LeRobotDatasetIO(root).episode_video_paths(
                episode, {cam or video_key: video_key}
            )
            video = _pick_video(videos, cam)
            if video is None:
                self._send_empty(404)
                return
            self._send_video_poster(video)
            return

        replay = self.ctx.runtime.replay_source
        videos = replay.video_paths() if (replay and hasattr(replay, "video_paths")) else {}  # type: ignore[attr-defined]
        video = _pick_video(videos, cam)
        if video is None:
            self._send_empty(404)
            return
        self._send_video_poster(video)

    def _get_replay_frame(self) -> None:
        self._point_preview_for_request()
        ep = self._query_int("episode_index", -1)
        frame = self._query_int("frame", 0)
        preview = self.ctx.preview
        arms = preview.frame_transforms(ep, frame) if (preview and ep >= 0) else None
        self._send_json(200, {"available": arms is not None, "arms": arms or {}})

    def _get_episode_transforms(self) -> None:
        self._point_preview_for_request()
        ep = self._query_int("episode_index", -1)
        preview = self.ctx.preview
        transform_range = self._transform_range()
        if transform_range == (-1, -1):
            return
        start = count = None
        if transform_range is not None:
            start, count = transform_range
        raw = (
            preview.transforms_blob(ep, start=start, count=count) if (preview and ep >= 0) else None
        )
        if raw is None:
            self._send_json(404, {"available": False})
            return
        total = len(preview.series(ep)["state"]) if preview and ep >= 0 else 0
        actual_start = 0 if start is None else start
        actual_count = total - actual_start if count is None else min(count, total - actual_start)
        self._send_transform_blob(raw, actual_start, actual_count, total)

    def _get_episode_video(self) -> None:
        self._point_preview_for_request()
        ep = self._query_int("episode_index", -1)
        cam = self._query_str("cam")
        videos = self.ctx.preview.videos(ep) if (self.ctx.preview and ep >= 0) else {}
        video = _pick_video(videos, cam)
        if video is None:
            self._send_empty(404)
            return
        self._send_video(video)

    def _get_episode_cams(self) -> None:
        # Camera video_keys with an mp4 on disk for this episode, so the RESULT tab can
        # offer a camera switcher (the trajectory replay is camera-agnostic, but the
        # operator may want to inspect a specific view).
        self._point_preview_for_request()
        ep = self._query_int("episode_index", -1)
        videos = self.ctx.preview.videos(ep) if (self.ctx.preview and ep >= 0) else {}
        self._send_json(200, {"cams": list(videos.keys())})

    def _send_video(self, path: Path) -> None:
        # Stream an mp4 honoring HTTP Range so <video> can seek; whole-file otherwise.
        try:
            try:
                path = _ensure_faststart_video(path)
            except RuntimeError as exc:
                logger.error("%s", exc)
                self._send_json(500, {"error": str(exc)})
                return
            size = path.stat().st_size
            rng = self.headers.get("Range", "")
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", rng) if rng else None
            if m:
                s, e = m.group(1), m.group(2)
                if s:
                    start = int(s)
                    if e:
                        end = min(int(e), size - 1)
                elif e:
                    suffix_len = int(e)
                    start = max(size - suffix_len, 0)
                    end = size - 1
                else:
                    start = size
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            length = end - start + 1
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", _VIDEO_CACHE_CONTROL)
            self.send_header("ETag", f'"{path.stat().st_mtime_ns:x}-{size:x}"')
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_video_poster(self, path: Path) -> None:
        try:
            try:
                poster = _ensure_video_poster(path)
            except RuntimeError as exc:
                logger.error("%s", exc)
                self._send_json(500, {"error": str(exc)})
                return
            body = poster.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", _VIDEO_CACHE_CONTROL)
            self.send_header("ETag", f'"{poster.stat().st_mtime_ns:x}-{len(body):x}"')
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        """Route a POST to a constant-command enqueue or a body-driven handler."""
        path = urlparse(self.path).path
        body = self._read_json_body()
        trace_id = self._request_trace_id()
        self._active_trace_id = trace_id
        started = time.monotonic()
        self._response_status = None
        if path != "/api/client_trace":
            logger.info(
                "[HTTP_POST_BEGIN] trace=%s path=%s body=%s",
                trace_id,
                path,
                _trace_json(_post_body_summary(body)),
            )
        try:
            # Constant control verbs: enqueue and ack. Everything else routes to a method
            # that owns its own response (custom command arg, runtime mutation, or payload).
            cmd = _POST_COMMANDS.get(path)
            if cmd is not None:
                self._enqueue(cmd)
                self._send_json(200, {"ok": True})
                return
            handler = _POST_ROUTES.get(path)
            if handler is None:
                self._send_empty(404)
                return
            handler(self, body)
        except Exception:
            logger.exception("[HTTP_POST_ERROR] trace=%s path=%s", trace_id, path)
            raise
        finally:
            if path != "/api/client_trace":
                logger.info(
                    "[HTTP_POST_END] trace=%s path=%s status=%s elapsed_ms=%.1f",
                    trace_id,
                    path,
                    self._response_status,
                    (time.monotonic() - started) * 1000.0,
                )

    def _enqueue_ok(self, command: str) -> None:
        self._enqueue(command)
        self._send_json(200, {"ok": True})

    def _require_rl_active(self) -> bool:
        if self.ctx.active_tab == "rl":
            return True
        self._send_json(409, {"ok": False, "error": "RL tab is not active"})
        return False

    def _post_rl_select_task(self, body: dict) -> None:
        if not self._require_rl_active():
            return
        self._enqueue_ok(f"web:rl_select_task:{body.get('task', '')}")

    def _post_rl_select_policy(self, body: dict) -> None:
        if not self._require_rl_active():
            return
        self._enqueue_ok(f"web:rl_select_policy:{int(body['slot'])}")

    def _post_rl_select_critic(self, body: dict) -> None:
        if not self._require_rl_active():
            return
        self._enqueue_ok(f"web:rl_select_critic:{int(body['slot'])}")

    def _post_rl_command(self, command: str) -> None:
        if not self._require_rl_active():
            return
        self._enqueue_ok(command)

    def _post_rl_setup(self, _body: dict) -> None:
        self._post_rl_command("web:rl_setup")

    def _post_rl_run(self, _body: dict) -> None:
        self._post_rl_command("web:run")

    def _post_rl_intervene(self, _body: dict) -> None:
        self._post_rl_command("web:halt")

    def _post_rl_accept(self, _body: dict) -> None:
        self._post_rl_command("web:run")

    def _post_rl_abandon(self, _body: dict) -> None:
        self._post_rl_command("web:rollout_intervention_abandon")

    def _post_rl_hil_enabled(self, body: dict) -> None:
        if not self._require_rl_active():
            return
        enabled = "1" if bool(body.get("enabled", False)) else "0"
        self._enqueue_ok(f"web:rollout_intervention_enabled:{enabled}")

    def _post_rl_save(self, _body: dict) -> None:
        self._post_rl_command("web:rollout_save")

    def _post_rl_reset(self, _body: dict) -> None:
        self._post_rl_command("web:console_reset")

    def _post_rl_review_episode(self, body: dict) -> None:
        if not self._require_rl_active():
            return
        runtime = self.ctx.runtime
        with runtime.rl_replay_lock:
            runtime.rl_replay_generation += 1
            request_generation = runtime.rl_replay_generation
        dataset_dir = _resolve_dataset_dir(str(body.get("dataset_dir", "")).strip())
        episode_index = int(body.get("episode", 0))
        try:
            source, video_keys = _open_review_episode(self.ctx, dataset_dir, episode_index)
            series = source.series()
        except Exception as error:
            self._send_json(404, {"ok": False, "error": str(error)})
            return
        if source.n_steps <= 0:
            source.close()
            self._send_json(404, {"ok": False, "error": "episode has no frames"})
            return
        with runtime.rl_replay_lock:
            if request_generation != runtime.rl_replay_generation:
                source.close()
                self._send_json(409, {"ok": False, "error": "Replay request superseded"})
                return
            old_sources = runtime.rl_replay_sources
            runtime.rl_replay_sources = [source]
            runtime.rl_replay_source = source
            runtime.rl_replay_dataset_dir = dataset_dir
            runtime.rl_replay_episode_id = episode_index
            runtime.rl_replay_timestamps = [float(value) for value in series["timestamp"]]
            if runtime.rl_critic_runner is not None:
                runtime.rl_critic_runner.reset_series()
        for old_source in old_sources:
            old_source.close()
        self._send_json(
            200,
            {
                "ok": True,
                "episode": episode_index,
                "frames": source.n_steps,
                "fps": source.fps,
                "task": source.current_task,
                "video_keys": video_keys,
                **series,
            },
        )

    def _post_rl_replay_critic(self, body: dict) -> None:
        if not self._require_rl_active():
            return
        runtime = self.ctx.runtime
        runner = runtime.rl_critic_runner
        if runner is None:
            self._send_json(409, {"ok": False, "error": "Critic is not connected"})
            return
        with runtime.rl_replay_lock:
            source = runtime.rl_replay_source
            if source is None:
                self._send_json(409, {"ok": False, "error": "No RL replay episode is selected"})
                return
            frame_index = int(body["frame"])
            if frame_index < 0 or frame_index >= source.n_steps:
                self._send_json(400, {"ok": False, "error": "Replay frame is out of range"})
                return
            metadata = runtime.policy_metadata or {}
            horizon = runtime.rl_critic_action_horizon or max(1, int(metadata.get("chunk_size", 1)))
            timestamp = runtime.rl_replay_timestamps[frame_index]
            config = runtime.active_config or self.ctx.config
            replay_generation = runtime.rl_replay_generation

            def build_request() -> tuple[dict, np.ndarray]:
                with runtime.rl_replay_lock:
                    if (
                        replay_generation != runtime.rl_replay_generation
                        or source is not runtime.rl_replay_source
                    ):
                        raise RuntimeError("RL replay episode changed before Critic frame decode")
                    source.seek(frame_index)
                    frame = source.get_frame()
                    if frame is None:
                        raise RuntimeError(f"RL replay frame is unavailable: {frame_index}")
                    observation = build_policy_observation(
                        frame, source.current_task, config, runtime
                    )
                    trajectory = source.get_action_trajectory()
                    actions = trajectory[frame_index : frame_index + horizon]
                    if len(actions) < horizon:
                        actions = np.concatenate(
                            [actions, np.repeat(trajectory[-1:], horizon - len(actions), axis=0)],
                            axis=0,
                        )
                    return build_rl_critic_observation(runtime, observation), actions

        runner.submit_builder(build_request, timestamp, "replay")
        self._send_json(200, {"ok": True, "frame": frame_index})

    def _post_client_trace(self, body: dict) -> None:
        client_id = str(body.get("client_id", "unknown"))[:96]
        event = str(body.get("event", ""))[:96]
        seq = body.get("seq", 0)
        details = body.get("details")
        if not _TRACE_EVENT_RE.fullmatch(event):
            self._send_json(400, {"ok": False, "error": "invalid trace event"})
            return
        logger.info(
            "[UI_TRACE] client=%s seq=%s event=%s details=%s",
            client_id,
            seq,
            event,
            _trace_json(details if isinstance(details, dict) else {}),
        )
        self._send_json(200, {"ok": True})

    # --- POST route handlers (body-derived command arg / payload / mutation) ---

    def _post_select_task(self, body: dict) -> None:
        self._enqueue_ok(f"web:switch_task:{body.get('task', '')}")

    def _post_select_collect_task(self, body: dict) -> None:
        task = str(body.get("task", "")).strip()
        dataset = str(body.get("dataset", "")).strip()
        task_index_value = body.get("task_index")
        config = self.ctx.runtime.active_config or self.ctx.config
        locations = [
            (str(dataset_name), index)
            for dataset_name, entries in config.collection.tasks.items()
            for index, entry in enumerate(entries)
            if str(entry[0]) == task
        ]
        if not locations:
            self._send_json(400, {"ok": False, "error": "unknown collection task"})
            return
        if not dataset:
            dataset, task_index = locations[0]
        else:
            task_index = task_index_value
            entries = config.collection.tasks.get(dataset)
            if entries is None:
                self._send_json(400, {"ok": False, "error": "unknown collection dataset"})
                return
            valid_index = (
                isinstance(task_index, int)
                and not isinstance(task_index, bool)
                and 0 <= task_index < len(entries)
            )
            if not valid_index or str(entries[task_index][0]) != task:
                self._send_json(
                    400,
                    {"ok": False, "error": "collection task index does not match task"},
                )
                return
        logger_obj = self.ctx.runtime.episode_logger
        if logger_obj is not None and bool(getattr(logger_obj, "has_active_episode", False)):
            self._send_json(409, {"ok": False, "error": "cannot switch task while recording"})
            return
        self.ctx.session.selected_collect_task = task
        self.ctx.session.selected_collect_set = dataset
        self.ctx.session.selected_collect_task_index = task_index
        self.ctx.session.collection_scene_id = None
        self.ctx.session.collection_scene_round = None
        self.ctx.session.collection_slot_id = None
        self.ctx.session.collection_task_id = None
        self._send_json(
            200,
            {"ok": True, "task": task, "dataset": dataset, "task_index": task_index},
        )

    def _apply_collection_slot(self, slot: dict[str, Any] | None) -> None:
        _apply_collection_slot_session(self.ctx.session, slot)

    def _post_select_collection_slot(self, body: dict) -> None:
        dataset = str(body.get("dataset") or "").strip()
        slot_id = str(body.get("slot_id") or "").strip()
        logger_obj = self.ctx.runtime.episode_logger
        if logger_obj is not None and bool(getattr(logger_obj, "has_active_episode", False)):
            self._send_json(409, {"ok": False, "error": "cannot switch slot while recording"})
            return
        try:
            snapshot = _collection_slots_snapshot(self.ctx, dataset)
        except ValueError as error:
            self._send_json(400, {"ok": False, "error": str(error)})
            return
        selected = next((row for row in snapshot["rows"] if row["slot_id"] == slot_id), None)
        if selected is None:
            self._send_json(400, {"ok": False, "error": "unknown collection slot"})
            return
        if selected["state"] == "saving":
            self._send_json(409, {"ok": False, "error": "collection slot is not selectable"})
            return
        dataset_dir = str(snapshot["dataset_dir"] or "")
        if not dataset_dir:
            self._send_json(409, {"ok": False, "error": "collection recording is unavailable"})
            return
        select_collection_slot(
            Path(dataset_dir), slot_id, snapshot["slot_state"],
            episode_index=(selected.get("episode") or {}).get("episode_index"),
            manual=bool(body.get("manual", True)),
        )
        active = dict(selected)
        active["repair"] = active["state"] in {"complete", "deferred", "rejected"}
        active["state"] = "active"
        self._apply_collection_slot(active)
        self._send_json(
            200,
            {"ok": True, "active": active, "counts": snapshot["counts"]},
        )

    def _post_skip_collection_slot(self, body: dict) -> None:
        dataset = str(body.get("dataset") or "").strip()
        slot_id = str(body.get("slot_id") or "").strip()
        logger_obj = self.ctx.runtime.episode_logger
        if logger_obj is not None and bool(getattr(logger_obj, "has_active_episode", False)):
            self._send_json(409, {"ok": False, "error": "cannot skip slot while recording"})
            return
        try:
            snapshot = _collection_slots_snapshot(self.ctx, dataset)
        except ValueError as error:
            self._send_json(400, {"ok": False, "error": str(error)})
            return
        active = snapshot["active"]
        if active is None or active["slot_id"] != slot_id:
            self._send_json(409, {"ok": False, "error": "collection slot is no longer active"})
            return
        dataset_dir = str(snapshot["dataset_dir"] or "")
        if not dataset_dir:
            self._send_json(409, {"ok": False, "error": "collection recording is unavailable"})
            return
        defer_active_slot(Path(dataset_dir), slot_id, snapshot["slot_state"])
        updated = _collection_slots_snapshot(self.ctx, dataset)
        self._apply_collection_slot(updated["active"])
        self._send_json(
            200,
            {"ok": True, "active": updated["active"], "counts": updated["counts"]},
        )

    def _post_select_episode(self, body: dict) -> None:
        self._enqueue_ok(f"web:select_episode:{int(body.get('episode', 0))}")

    def _post_tab_switch(self, body: dict) -> None:
        self._clear_collection_replay()
        tab = str(body.get("tab", "") or "debug")
        self.ctx.active_tab = tab
        self.ctx.session.last_error = ""
        self._enqueue_ok(f"web:tab_switch:{tab}")

    def _post_collect_arm(self, body: dict) -> None:
        enabled = bool(body.get("enabled", False))
        if enabled:
            self.ctx.active_tab = "collect"
        self._enqueue_ok(f"web:collect_arm:{'on' if enabled else 'off'}")

    def _post_control_arm(self, body: dict) -> None:
        if self.ctx.active_tab != "manual":
            self._send_json(409, {"ok": False, "error": "Open DEVICE before enabling teleop"})
            return
        self._enqueue_ok(f"web:control_arm:{'on' if body.get('enabled') else 'off'}")

    def _get_device_settings(self) -> None:
        self._send_json(200, self.ctx.device_settings.status())

    def _post_device_command(self, body: dict) -> None:
        command = self.path.split("?", 1)[0].removeprefix("/api/")
        if command == "device_control":
            if self.ctx.active_tab != "manual":
                self._send_json(409, {"ok": False, "error": "Open DEVICE before enabling control"})
                return
            argument = "on" if body.get("enabled") else "off"
        else:
            argument = body.get("component") or ""
            if argument not in {"", "robot", "teleop", "camera"}:
                self._send_json(400, {"ok": False, "error": "Unknown device component"})
                return
        if command in {"device_start", "device_stop"}:
            try:
                request_id = self.ctx.device_settings.queue_command(
                    command.removeprefix("device_"), argument or None
                )
            except ValueError as error:
                self._send_json(409, {"ok": False, "error": str(error)})
                return
            self._enqueue(f"web:{command}:{argument}:{request_id}")
            self._send_json(200, {"ok": True, "request_id": request_id})
            return
        self._enqueue_ok(f"web:{command}:{argument}")

    def _post_device_open_pico(self, body: dict) -> None:
        try:
            self.ctx.device_settings.service.request("open_pico")
        except (ValueError, OSError, RuntimeError) as error:
            self._send_json(400, {"ok": False, "error": str(error)})
            return
        self._send_json(200, {"ok": True})

    def _post_device_kill_all(self, body: dict) -> None:
        try:
            result = self.ctx.device_settings.kill_all()
            result.pop("commands", None)
        except (ValueError, OSError, RuntimeError) as error:
            self._send_json(409, {"ok": False, "error": str(error)})
            return
        self._send_json(200, {"ok": True, **result})

    def _get_devices(self) -> None:
        workspace = self.ctx.device_settings.workspace
        config = self.ctx.runtime.active_config or self.ctx.config
        selected = workspace.initial_selection(config)
        for kind in selected:
            selected[kind] = self._query_str(kind, selected[kind])
        try:
            values = workspace.resolve(selected)
        except (ValueError, KeyError) as error:
            self._send_json(400, {"ok": False, "error": str(error)})
            return
        self._send_json(
            200,
            {
                "catalog": workspace.hardware.options(selected["robot"]),
                "selected": selected,
                "values": values,
            },
        )

    def _post_device_selection(self, body: dict) -> None:
        if self.ctx.device_settings.state in {"queued", "applying", "restarting"}:
            self._send_json(409, {"ok": False, "error": "A device update is already pending"})
            return
        self.ctx.device_settings.state = "queued"
        self._enqueue_ok("web:device_selection:" + json.dumps(body))

    def _post_camera_profile(self, body: dict) -> None:
        workspace = self.ctx.device_settings.workspace
        try:
            camera = body["camera"]
            filename = str(body.get("path", ""))
            if "content" in body:
                filename = str(workspace.import_profile(camera, body["content"]))
            data, settings = workspace.profile_values(camera, filename)
            self._send_json(200, {"path": filename, "data": data, "settings": settings})
        except (ValueError, TypeError, KeyError, OSError, yaml.YAMLError) as error:
            self._send_json(400, {"ok": False, "error": str(error)})

    def _post_load_replay_dataset(self, body: dict) -> None:
        self._clear_collection_replay()
        runtime = self.ctx.runtime
        session = self.ctx.session
        keys = body.get("keys") or {}
        dataset_dir = str(body.get("dataset_dir", "")).strip()
        episode = int(body.get("episode", 0))
        load_replay_dataset(
            dataset_dir,
            episode,
            runtime.active_config or self.ctx.config,
            session,
            runtime,
            state_key=str(keys.get("state", "")),
            action_key=str(keys.get("action", "")),
            video_keys=body.get("video_keys") or {},
            action_mode=str(body.get("action_mode", "")),
        )
        if session.last_error or runtime.replay_source is None:
            error = session.last_error or "replay dataset was not mounted"
            self._send_json(400, {"ok": False, "error": error})
            return
        replay_keys = runtime.replay_source._config.transport.dataset_keys  # type: ignore[attr-defined]
        self._send_json(
            200,
            {
                "ok": True,
                "dataset_dir": runtime.replay_dataset_dir,
                "episode": runtime.replay_episode_id,
                "n_episodes": runtime.replay_n_episodes,
                "frames": int(getattr(runtime.replay_source, "n_steps", 0) or 0),
                "fps": runtime.replay_fps,
                "task": runtime.replay_task,
                "state_key": str(replay_keys.state_key),
                "action_key": runtime.replay_action_key,
                "action_mode": runtime.replay_action_mode,
                "video_keys": dict(replay_keys.video_keys),
            },
        )

    def _post_set_replay_fps(self, body: dict) -> None:
        self._enqueue_ok(f"web:set_replay_fps:{int(body.get('fps', 10))}")

    def _post_replay_seek(self, body: dict) -> None:
        self._enqueue_ok(f"web:replay_seek:{int(body.get('frame', 0))}")

    def _post_select_mode(self, body: dict) -> None:
        self._enqueue_ok(f"web:select_mode:{body.get('mode', '')}")

    def _post_select_strategy(self, body: dict) -> None:
        self._enqueue_ok(f"web:select_strategy:{body.get('strategy', '')}")

    def _post_update_infer_params(self, body: dict) -> None:
        self._enqueue_ok("web:update_infer_params:" + json.dumps(body))

    def _post_rollout_intervention_enabled(self, body: dict) -> None:
        enabled = 1 if bool(body.get("enabled", False)) else 0
        self._enqueue_ok(f"web:rollout_intervention_enabled:{enabled}")

    def _post_operator_action(self, body: dict) -> None:
        intent = str(body.get("intent", "")).strip().lower()
        if intent not in {"start", "accept", "cancel"}:
            self._send_json(400, {"ok": False, "error": f"unsupported operator intent: {intent}"})
            return
        if intent == "start" and self.ctx.active_tab == "collect":
            task = str(body.get("task", "")).strip()
            dataset = str(body.get("dataset", "")).strip()
            task_index = body.get("task_index")
            session = self.ctx.session
            if (
                task != str(session.selected_collect_task or "")
                or dataset != str(session.selected_collect_set or "")
                or task_index != session.selected_collect_task_index
            ):
                self._send_json(
                    409,
                    {"ok": False, "error": "collection task selection is not confirmed"},
                )
                return
            self._post_collect_start(body)
            return
        self._enqueue_ok(f"web:operator_action:{intent}:ui")

    def _post_manual_qpos(self, body: dict) -> None:
        qpos = body.get("qpos", [])
        self._enqueue_ok("web:manual_qpos:" + ",".join(str(float(x)) for x in qpos))

    def _post_manual_send(self, _body: dict) -> None:
        self.ctx.session.manual_publish_active = True
        self._enqueue_ok("web:manual_send")

    def _post_manual_dispatch(self, body: dict) -> None:
        self._enqueue_ok(f"web:manual_dispatch:{body.get('dispatch', 'stage')}")

    def _post_gripper(self, body: dict) -> None:
        side = body.get("side", "l")
        state = body.get("state")
        if state in {"open", "close"}:
            lock = "1" if body.get("lock", True) else "0"
            self._enqueue_ok(f"web:gripper:{side}:{state}:{lock}")
            return
        value = body.get("value")
        arg = f"{side}:{value}" if value is not None else side
        self._enqueue_ok(f"web:gripper:{arg}")

    def _post_collect_start(self, body: dict) -> None:
        self._clear_collection_replay()
        if self.ctx.active_tab != "collect" or not self.ctx.runtime.collection_teleop_armed:
            self.ctx.session.last_error = "Collection teleop requires COLLECT activation"
            self._send_json(200, {"ok": False, "error": self.ctx.session.last_error})
            return
        session = self.ctx.session
        scene_id = str(body.get("scene_id") or session.collection_scene_id or "").strip()
        slot_id = str(body.get("slot_id") or session.collection_slot_id or "").strip()
        task_id = str(body.get("task_id") or session.collection_task_id or "").strip()
        raw_round = body.get("scene_round", session.collection_scene_round)
        scene_round = (
            raw_round
            if isinstance(raw_round, int) and not isinstance(raw_round, bool) and raw_round >= 0
            else None
        )
        if session.collection_slot_id and (
            slot_id != session.collection_slot_id
            or scene_id != session.collection_scene_id
            or task_id != session.collection_task_id
            or scene_round != session.collection_scene_round
        ):
            self._send_json(
                409,
                {"ok": False, "error": "collection slot selection is not confirmed"},
            )
            return
        if session.collection_slot_id:
            try:
                snapshot = _collection_slots_snapshot(
                    self.ctx, str(session.selected_collect_set or "")
                )
            except ValueError:
                snapshot = {"active": None}
            active = snapshot["active"]
            if active is None or active["slot_id"] != session.collection_slot_id:
                self._send_json(
                    409,
                    {"ok": False, "error": "collection slot is no longer active"},
                )
                return
        session.collection_scene_id = scene_id[:128] or None
        session.collection_slot_id = slot_id[:256] or None
        session.collection_task_id = task_id[:128] or None
        session.collection_scene_round = scene_round
        self._enqueue_ok("web:collect_start")

    def _post_exit_collection_replay(self, _body: dict) -> None:
        self._clear_collection_replay()
        self._invalidate_scene_cache()
        self._send_json(200, {"ok": True})

    def _post_inspect_dataset(self, body: dict) -> None:
        dataset_dir = _resolve_dataset_dir(str(body.get("dataset_dir", "")).strip())
        try:
            io = LeRobotDatasetIO(dataset_dir)
            keys = io.infer_keys()
            n_episodes = io.count_episodes()
            self._send_json(200, {"ok": True, "keys": keys, "n_episodes": n_episodes})
        except Exception as error:
            self._send_json(200, {"ok": False, "error": str(error)})

    def _post_qc_mark(self, body: dict) -> None:
        dataset_dir = _resolve_dataset_dir(str(body.get("dataset_dir", "")).strip())
        episode = int(body.get("episode", -1))
        verdict = str(body.get("verdict", ""))
        ok = LeRobotDatasetIO(dataset_dir).mark_qc(episode, verdict, str(body.get("note", "")))
        if ok:
            self._send_json(200, {"ok": True, "episode": episode, "verdict": verdict})
        else:
            self._send_json(404, {"ok": False, "error": "episode not found in dataset"})

    def _post_collect_qc_mark(self, body: dict) -> None:
        episode_value = body.get("episode")
        if (
            isinstance(episode_value, int)
            and not isinstance(episode_value, bool)
            and episode_value >= 0
        ):
            episode = episode_value
        elif isinstance(episode_value, str) and episode_value.isascii() and episode_value.isdigit():
            episode = int(episode_value)
        else:
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "collection episode must be a non-negative integer",
                },
            )
            return
        verdict = str(body.get("verdict", ""))
        if verdict not in {"", "pass", "fail"}:
            self._send_json(
                400,
                {"ok": False, "error": f"unsupported QC verdict: {verdict}"},
            )
            return
        logger_obj = self.ctx.runtime.episode_logger
        if logger_obj is None:
            self._send_json(
                409,
                {"ok": False, "error": "collection recording is unavailable"},
            )
            return
        task = str(body.get("task", ""))
        collection_dataset = str(body.get("dataset", "")).strip() or None
        config = self.ctx.runtime.active_config or self.ctx.config
        valid_target = any(
            (collection_dataset is None or dataset_name == collection_dataset)
            and any(str(entry[0]) == task for entry in entries)
            for dataset_name, entries in config.collection.tasks.items()
        )
        if not valid_target:
            self._send_json(
                409,
                {"ok": False, "error": "unknown collection review target"},
            )
            return
        ok = logger_obj.mark_collection_qc(
            task,
            episode,
            verdict,
            str(body.get("note", "")),
            collection_dataset=collection_dataset,
        )
        if not ok:
            self._send_json(
                409,
                {"ok": False, "error": "collection episode is not saved"},
            )
            return
        self._send_json(200, {"ok": True, "episode": episode, "verdict": verdict})

    def _active_collection_dataset(self, body: dict) -> Path | None:
        logger_obj = self.ctx.runtime.episode_logger
        if logger_obj is None:
            self._send_json(409, {"ok": False, "error": "collection recording is unavailable"})
            return None
        collection_dataset = str(body.get("dataset") or self.ctx.session.selected_collect_set or "").strip()
        config = self.ctx.runtime.active_config or self.ctx.config
        entries = config.collection.tasks.get(collection_dataset)
        if not entries:
            self._send_json(
                409,
                {"ok": False, "error": "unknown collection dataset"},
            )
            return None
        # Export belongs to the dataset, independently of the capture cursor.
        snapshot = _status_snapshot_for_poll(logger_obj, str(entries[0][0]), collection_dataset)
        raw_dataset_dir = str(snapshot.get("dataset_dir") or "")
        if not raw_dataset_dir:
            self._send_json(404, {"ok": False, "error": "collection dataset is unavailable"})
            return None
        dataset_dir = Path(raw_dataset_dir).resolve()
        if not dataset_dir.is_dir():
            self._send_json(404, {"ok": False, "error": "collection dataset is unavailable"})
            return None
        return dataset_dir

    def _requested_dataset_format(self, body: dict) -> str | None:
        dataset_format = body.get("dataset_format")
        if not isinstance(dataset_format, str) or not dataset_format:
            self._send_json(400, {"ok": False, "error": "dataset_format is required"})
            return None
        if dataset_format not in DATASET_EXPORT_FORMATS:
            self._send_json(400, {"ok": False, "error": "unsupported dataset export format"})
            return None
        return dataset_format

    def _post_collect_quality_export(self, body: dict) -> None:
        dataset_dir = self._active_collection_dataset(body)
        if dataset_dir is None:
            return
        dataset_format = self._requested_dataset_format(body)
        if dataset_format is None:
            return
        accepted_dir, rejected_dir = _quality_export_paths(dataset_dir, dataset_format)
        with self.ctx.quality_upload_lock:
            if any(
                existing.local_dir == str(accepted_dir)
                and existing.state in {"queued", "scanning", "running"}
                for existing in self.ctx.quality_upload_jobs.values()
            ):
                self._send_json(
                    409,
                    {"ok": False, "error": "dataset upload is already running"},
                )
                return
            for existing in self.ctx.quality_export_jobs.values():
                if (
                    existing.source_dir == str(dataset_dir)
                    and existing.dataset_format == dataset_format
                    and existing.state in {"queued", "running"}
                ):
                    self._send_json(
                        409,
                        {"ok": False, "error": "dataset export is already running"},
                    )
                    return
            _trim_finished_jobs(self.ctx.quality_export_jobs)
            job_id = uuid.uuid4().hex
            job = _QualityExportJob(
                job_id=job_id,
                source_dir=str(dataset_dir),
                accepted_dir=str(accepted_dir),
                rejected_dir=str(rejected_dir),
                dataset_format=dataset_format,
            )
            self.ctx.quality_export_jobs[job_id] = job

        threading.Thread(
            target=_run_quality_export,
            args=(
                self.ctx,
                job_id,
                dataset_dir,
                accepted_dir,
                rejected_dir,
                dataset_format,
            ),
            name=f"quality-export-{job_id[:8]}",
            daemon=True,
        ).start()
        self._send_json(202, {"ok": True, **job.payload()})

    def _post_collect_quality_upload(self, body: dict) -> None:
        dataset_dir = self._active_collection_dataset(body)
        if dataset_dir is None:
            return
        dataset_format = self._requested_dataset_format(body)
        if dataset_format is None:
            return
        local_dir, _ = _quality_export_paths(dataset_dir, dataset_format)
        with self.ctx.quality_upload_lock:
            exports = [
                job
                for job in self.ctx.quality_export_jobs.values()
                if job.source_dir == str(dataset_dir) and job.dataset_format == dataset_format
            ]
            latest_export = exports[-1] if exports else None
            if (
                latest_export is None
                or latest_export.state != "completed"
                or latest_export.accepted_dir != str(local_dir)
            ):
                self._send_json(
                    409,
                    {"ok": False, "error": "export the current dataset before upload"},
                )
                return
        dataset_name = self.ctx.session.selected_collect_set or (
            dataset_dir.parent.name if dataset_dir.name == "raw" else dataset_dir.name
        )
        self._post_quality_upload(
            body,
            dataset_dir=dataset_dir,
            local_dir=local_dir,
            dataset_format=dataset_format,
            dataset_name=dataset_name,
        )

    def _post_dashboard_upload(self, body: dict) -> None:
        dataset_format = self._requested_dataset_format(body)
        if dataset_format is None:
            return
        source_value = body.get("source_dir")
        if not isinstance(source_value, str) or not source_value.strip():
            self._send_json(400, {"ok": False, "error": "source_dir is required"})
            return
        try:
            source_dir = Path(source_value).resolve()
        except (OSError, ValueError):
            self._send_json(400, {"ok": False, "error": "invalid source_dir"})
            return
        candidate = next(
            (
                item
                for item in discover_dashboard_upload_candidates(self._dashboard_roots())
                if item["source_dir"] == str(source_dir)
                and item["dataset_format"] == dataset_format
            ),
            None,
        )
        if candidate is None:
            self._send_json(404, {"ok": False, "error": "accepted export is unavailable"})
            return
        self._post_quality_upload(
            body,
            dataset_dir=source_dir,
            local_dir=Path(str(candidate["accepted_dir"])),
            dataset_format=dataset_format,
            dataset_name=str(candidate["dataset"]),
        )

    def _post_quality_upload(
        self,
        body: dict,
        *,
        dataset_dir: Path,
        local_dir: Path,
        dataset_format: str,
        dataset_name: str,
    ) -> None:
        config = self.ctx.runtime.active_config or self.ctx.config
        dataset_dir = dataset_dir.resolve()
        local_dir = local_dir.resolve()
        marker_path = local_dir / "meta" / "quality_split.json"
        try:
            marker = json.loads(marker_path.read_text())
        except (OSError, ValueError):
            logger.warning("Invalid accepted collection export marker", exc_info=True)
            self._send_json(400, {"ok": False, "error": "invalid accepted export"})
            return
        if marker.get("subset") != "accepted":
            self._send_json(400, {"ok": False, "error": "only accepted exports can be uploaded"})
            return
        if marker.get("dataset_format") != dataset_format:
            self._send_json(400, {"ok": False, "error": "accepted export format mismatch"})
            return
        try:
            marker_source = Path(str(marker.get("source_dir", ""))).resolve()
        except (OSError, ValueError):
            marker_source = Path()
        if marker_source != dataset_dir:
            self._send_json(400, {"ok": False, "error": "accepted export source mismatch"})
            return

        upload_specs = resolve_dataset_uploads(config.collection.storage, dataset_name)
        if not upload_specs:
            self._send_json(409, {"ok": False, "error": "dataset upload is not configured"})
            return
        remote_dir = ", ".join(spec.target for spec in upload_specs)
        confirmed = body.get("confirmed") is True
        plan_id = str(body.get("plan_id", "") or "").strip()
        if confirmed:
            with self.ctx.quality_upload_lock:
                job = self.ctx.quality_upload_jobs.get(plan_id)
                if job is None:
                    self._send_json(404, {"ok": False, "error": "upload scan not found"})
                    return
                plan = job._plan
                if (
                    job.local_dir != str(local_dir)
                    or plan is None
                    or tuple(spec for spec, _ in plan.backends) != upload_specs
                ):
                    self._send_json(409, {"ok": False, "error": "upload scan no longer matches"})
                    return
                if job.state != "ready":
                    self._send_json(409, {"ok": False, "error": "upload scan is not ready"})
                    return
                if any(
                    existing is not job
                    and existing.local_dir == str(local_dir)
                    and existing.state in {"queued", "scanning", "running"}
                    for existing in self.ctx.quality_upload_jobs.values()
                ):
                    self._send_json(
                        409,
                        {"ok": False, "error": "dataset upload is already running"},
                    )
                    return
                job.state = "queued"
                job.error = ""
                job.files_completed = 0
                job.bytes_completed = 0
                job.current_file = ""

            threading.Thread(
                target=_run_quality_upload,
                args=(self.ctx, plan_id),
                name=f"quality-upload-{plan_id[:8]}",
                daemon=True,
            ).start()
            self._send_json(202, {"ok": True, **job.payload()})
            return
        if plan_id:
            self._send_json(400, {"ok": False, "error": "confirmed is required for upload"})
            return

        with self.ctx.quality_upload_lock:
            for existing in self.ctx.quality_upload_jobs.values():
                if existing.local_dir == str(local_dir) and existing.state in {
                    "queued",
                    "scanning",
                    "running",
                }:
                    self._send_json(
                        409,
                        {"ok": False, "error": "dataset upload scan is already running"},
                    )
                    return
            _trim_finished_jobs(self.ctx.quality_upload_jobs)
            job_id = uuid.uuid4().hex
            job = _QualityUploadJob(
                job_id=job_id,
                local_dir=str(local_dir),
                remote_dir=remote_dir,
                backends=tuple(spec.backend for spec in upload_specs),
            )
            self.ctx.quality_upload_jobs[job_id] = job

        threading.Thread(
            target=_run_quality_upload_scan,
            args=(self.ctx, job_id, local_dir, upload_specs),
            name=f"quality-upload-scan-{job_id[:8]}",
            daemon=True,
        ).start()
        self._send_json(202, {"ok": True, **job.payload()})

    def _post_annotate(self, body: dict) -> None:
        # Write/overwrite an episode's language annotation directly into the lerobot
        # dataset (episodes.jsonl key + per-frame parquet column + info.json feature).
        dataset_dir = _resolve_dataset_dir(str(body.get("dataset_dir", "")).strip())
        episode = int(body.get("episode", -1))
        annotation = str(body.get("annotation", ""))
        ok = LeRobotDatasetIO(dataset_dir).annotate_language(episode, annotation)
        if ok:
            self._send_json(200, {"ok": True, "episode": episode})
        else:
            self._send_json(404, {"ok": False, "error": "episode not found in dataset"})

    def _post_episode_annotation(self, body: dict) -> None:
        # Read back an episode's stored annotation so the editor prefills it (enabling edits).
        dataset_dir = _resolve_dataset_dir(str(body.get("dataset_dir", "")).strip())
        episode = int(body.get("episode", -1))
        annotation = LeRobotDatasetIO(dataset_dir).read_annotation(episode)
        self._send_json(200, {"ok": True, "episode": episode, "annotation": annotation})

    def _post_select_ckpt(self, body: dict) -> None:
        slot = int(body.get("slot", 0))
        self._enqueue(f"web:switch_ckpt:{slot}")
        self._send_json(200, {"ok": True, "slot": slot, "label": self.ctx.ckpt_label(slot)})

    def _post_init_move(self, body: dict) -> None:
        raw = body.get("qpos") or []
        qpos = [float(x) for x in raw]
        self._enqueue_ok("web:init_move:" + json.dumps({"qpos": qpos}))

    def _post_init_gripper(self, body: dict) -> None:
        action = "open" if str(body.get("action", "open")) == "open" else "close"
        side = str(body.get("side", "") or "")
        arg = f"{action}:{side}" if side in ("l", "r") else action
        self._enqueue_ok(f"web:init_gripper:{arg}")


# Exact-path GET routes (prefix routes — camera/assets/meshes — are handled inline in
# do_GET because they own a path subtree). Values are unbound handler methods.
_GET_ROUTES = {
    "/api/config": ConsoleRequestHandler._get_config,
    "/api/device_settings": ConsoleRequestHandler._get_device_settings,
    "/api/devices": ConsoleRequestHandler._get_devices,
    "/api/scene_plan": ConsoleRequestHandler._get_scene_plan,
    "/api/collection_slots": ConsoleRequestHandler._get_collection_slots,
    "/api/status": ConsoleRequestHandler._get_status,
    "/api/teleop/feedback": ConsoleRequestHandler._stream_teleop_feedback,
    "/api/episodes": ConsoleRequestHandler._get_episodes,
    "/api/dashboard": ConsoleRequestHandler._get_dashboard,
    "/api/dashboard_upload": ConsoleRequestHandler._get_collect_quality_upload,
    "/api/collect_quality_export": ConsoleRequestHandler._get_collect_quality_export,
    "/api/collect_quality_upload": ConsoleRequestHandler._get_collect_quality_upload,
    "/api/frame": ConsoleRequestHandler._get_frame,
    "/api/scene": ConsoleRequestHandler._get_scene,
    "/api/manual_scene": ConsoleRequestHandler._get_manual_scene,
    "/api/meshes": ConsoleRequestHandler._get_meshes,
    "/api/results": ConsoleRequestHandler._get_results,
    "/api/results_all": ConsoleRequestHandler._get_results_all,
    "/api/episode_series": ConsoleRequestHandler._get_episode_series,
    "/api/replay_series": ConsoleRequestHandler._get_replay_series,
    "/api/replay_scene_frame": ConsoleRequestHandler._get_replay_scene_frame,
    "/api/replay_transforms": ConsoleRequestHandler._get_replay_transforms,
    "/api/review_transforms": ConsoleRequestHandler._get_review_transforms,
    "/api/replay_video": ConsoleRequestHandler._get_replay_video,
    "/api/replay_poster": ConsoleRequestHandler._get_replay_poster,
    "/api/live_series": ConsoleRequestHandler._get_live_series,
    "/api/rl/series": ConsoleRequestHandler._get_rl_series,
    "/api/replay_frame": ConsoleRequestHandler._get_replay_frame,
    "/api/episode_transforms": ConsoleRequestHandler._get_episode_transforms,
    "/api/episode_video": ConsoleRequestHandler._get_episode_video,
    "/api/episode_cams": ConsoleRequestHandler._get_episode_cams,
}

# POST endpoints that just enqueue one constant command and ack {"ok": true}.
_POST_COMMANDS = {
    "/api/connect": "web:connect",
    "/api/disconnect": "web:disconnect",
    "/api/clear_replay": "web:clear_replay",
    "/api/setup": "web:setup",
    "/api/run": "web:run",
    "/api/halt": "web:halt",
    "/api/reset": "web:console_reset",
    "/api/step_infer": "web:step_infer",
    "/api/step_commit": "web:step_commit",
    "/api/step_cancel": "web:step_cancel",
    "/api/manual_home": "web:manual_home",
    "/api/collect_home": "web:collect_home",
    "/api/collect_stop": "web:collect_stop",
    "/api/collect_cancel": "web:collect_cancel",
    "/api/rollout_stop": "web:rollout_stop",
    "/api/rollout_save": "web:rollout_save",
    "/api/rollout_intervention_abandon": "web:rollout_intervention_abandon",
    "/api/warmup": "web:warmup",
    "/api/eval_stop": "web:stop",
    "/api/eval_cancel": "web:eval_cancel",
    "/api/eval_reset": "web:reset",
    "/api/init_done": "web:init_done",
}

# POST endpoints needing a body-derived command arg, a runtime mutation, or a custom
# response. Each handler owns its own _send_json. Values are unbound (self, body) methods.
_POST_ROUTES = {
    "/api/device_start": ConsoleRequestHandler._post_device_command,
    "/api/device_open_pico": ConsoleRequestHandler._post_device_open_pico,
    "/api/device_kill_all": ConsoleRequestHandler._post_device_kill_all,
    "/api/device_stop": ConsoleRequestHandler._post_device_command,
    "/api/device_control": ConsoleRequestHandler._post_device_command,
    "/api/client_trace": ConsoleRequestHandler._post_client_trace,
    "/api/rl/select_task": ConsoleRequestHandler._post_rl_select_task,
    "/api/rl/select_policy": ConsoleRequestHandler._post_rl_select_policy,
    "/api/rl/select_critic": ConsoleRequestHandler._post_rl_select_critic,
    "/api/rl/setup": ConsoleRequestHandler._post_rl_setup,
    "/api/rl/run": ConsoleRequestHandler._post_rl_run,
    "/api/rl/intervene": ConsoleRequestHandler._post_rl_intervene,
    "/api/rl/accept": ConsoleRequestHandler._post_rl_accept,
    "/api/rl/abandon": ConsoleRequestHandler._post_rl_abandon,
    "/api/rl/hil_enabled": ConsoleRequestHandler._post_rl_hil_enabled,
    "/api/rl/save": ConsoleRequestHandler._post_rl_save,
    "/api/rl/reset": ConsoleRequestHandler._post_rl_reset,
    "/api/rl/review_episode": ConsoleRequestHandler._post_rl_review_episode,
    "/api/rl/replay_critic": ConsoleRequestHandler._post_rl_replay_critic,
    "/api/collect_replay": ConsoleRequestHandler._start_collection_replay,
    "/api/exit_collect_replay": ConsoleRequestHandler._post_exit_collection_replay,
    "/api/review_episode": ConsoleRequestHandler._post_review_episode,
    "/api/select_task": ConsoleRequestHandler._post_select_task,
    "/api/select_collect_task": ConsoleRequestHandler._post_select_collect_task,
    "/api/select_collection_slot": ConsoleRequestHandler._post_select_collection_slot,
    "/api/skip_collection_slot": ConsoleRequestHandler._post_skip_collection_slot,
    "/api/select_episode": ConsoleRequestHandler._post_select_episode,
    "/api/tab_switch": ConsoleRequestHandler._post_tab_switch,
    "/api/collect_arm": ConsoleRequestHandler._post_collect_arm,
    "/api/control_arm": ConsoleRequestHandler._post_control_arm,
    "/api/device_selection": ConsoleRequestHandler._post_device_selection,
    "/api/camera_profile": ConsoleRequestHandler._post_camera_profile,
    "/api/inspect_dataset": ConsoleRequestHandler._post_inspect_dataset,
    "/api/qc_mark": ConsoleRequestHandler._post_qc_mark,
    "/api/collect_qc_mark": ConsoleRequestHandler._post_collect_qc_mark,
    "/api/collect_quality_export": ConsoleRequestHandler._post_collect_quality_export,
    "/api/collect_quality_upload": ConsoleRequestHandler._post_collect_quality_upload,
    "/api/dashboard_upload": ConsoleRequestHandler._post_dashboard_upload,
    "/api/annotate": ConsoleRequestHandler._post_annotate,
    "/api/episode_annotation": ConsoleRequestHandler._post_episode_annotation,
    "/api/load_replay_dataset": ConsoleRequestHandler._post_load_replay_dataset,
    "/api/set_replay_fps": ConsoleRequestHandler._post_set_replay_fps,
    "/api/replay_seek": ConsoleRequestHandler._post_replay_seek,
    "/api/select_mode": ConsoleRequestHandler._post_select_mode,
    "/api/select_strategy": ConsoleRequestHandler._post_select_strategy,
    "/api/update_infer_params": ConsoleRequestHandler._post_update_infer_params,
    "/api/rollout_intervention_enabled": ConsoleRequestHandler._post_rollout_intervention_enabled,
    "/api/operator_action": ConsoleRequestHandler._post_operator_action,
    "/api/manual_qpos": ConsoleRequestHandler._post_manual_qpos,
    "/api/manual_send": ConsoleRequestHandler._post_manual_send,
    "/api/manual_dispatch": ConsoleRequestHandler._post_manual_dispatch,
    "/api/gripper": ConsoleRequestHandler._post_gripper,
    "/api/collect_start": ConsoleRequestHandler._post_collect_start,
    "/api/select_ckpt": ConsoleRequestHandler._post_select_ckpt,
    "/api/init_move": ConsoleRequestHandler._post_init_move,
    "/api/init_gripper": ConsoleRequestHandler._post_init_gripper,
    "/api/eval_start": ConsoleRequestHandler._eval_start,
    "/api/score": ConsoleRequestHandler._eval_score,
}


def build_console_context(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    output_dir: str = "",
    *,
    with_scene: bool = True,
    with_obs_reader: bool = True,
    with_preview: bool = True,
) -> ConsoleContext:
    """Build the shared ConsoleContext and attach it to ``runtime.console_ctx``.

    The heavy pieces are opt-out so a headless run can skip them: ``with_scene``
    (URDF 3D scene / forward kinematics), ``with_obs_reader`` (an extra visualization
    SUB stream) and ``with_preview`` (RESULT-tab episode playback). Headless passes
    all False, keeping only the light state the ZMQ control-channel queries read.
    """
    scene: object | None = None
    if with_scene:
        try:
            from robots.utils import UrdfScene

            scene = UrdfScene(
                runtime.robot,
                gripper_open=config.robot.gripper_open,
                gripper_close=config.robot.gripper_close,
            )
        except Exception as exc:
            logger.warning("UrdfScene unavailable; 3D canvas disabled: %s", exc)
    _ensure_ckpt_order(config, runtime)
    ctx = ConsoleContext(
        config=config,
        runtime=runtime,
        session=session,
        obs_reader=runtime.transport.create_observation_reader() if with_obs_reader else None,
        scene=scene,
        active_tab=_resolve_initial_tab(config),
        output_dir=output_dir,
    )
    # RESULT-tab playback reads recorded episodes under <output_dir>/episodes.
    if with_obs_reader and config.transport.type != "dataset":
        workspace = ctx.device_settings.workspace
        selected = workspace.initial_selection(config)
        spec = workspace.hardware.options(selected["robot"])["camera"][selected["camera"]]
        if spec.get("separate"):
            from core.devices.camera import CameraPreview

            values = workspace.resolve(selected)
            ctx.camera_preview = CameraPreview(
                values["robot"]["settings"]["camera_endpoint"],
                list(config.collection.schema.cameras),
                runtime.transport.is_shutdown,
            )
    if with_preview and output_dir:
        from core.app.console.episode_preview import EpisodePreview

        ctx.preview = EpisodePreview(config, Path(output_dir))
    runtime.console_ctx = ctx
    return ctx


def start_console_server(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    port: int,
    host: str = "0.0.0.0",
    output_dir: str = "",
) -> threading.Thread:
    """Launch a daemon HTTP server thread bound to ``host:port``."""
    ctx = build_console_context(config, runtime, session, output_dir)
    ConsoleRequestHandler.ctx = ctx
    server = ThreadingHTTPServer((host, port), ConsoleRequestHandler)

    def serve() -> None:
        """Thread body: serve requests until the server is shut down."""
        display_host = "127.0.0.1" if host == "0.0.0.0" else host
        logger.info("Console web server: http://%s:%d", display_host, port)
        server.serve_forever()

    thread = threading.Thread(target=serve, name="eva-console-server", daemon=True)
    thread.start()
    if str(config.transport.type) in _LIVE_TRANSPORT_TYPES:
        _prewarm_transform_worker(config, runtime)
    return thread
