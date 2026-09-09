"""Observation building + episode/collection recording and eval dataset wiring."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.app.collection_capture import (
    prepare_collection_capture,
    start_collection_capture,
    stop_collection_capture,
)
from core.app.handlers.imaging import prepare_image
from core.app.handlers.teleop import (
    TELEOP_CONTROL_SOURCE_CLIENT,
    PublishedTeleopAction,
    activate_rollout_teleop,
    activate_teleop,
    deactivate_rollout_teleop,
    deactivate_teleop,
)
from core.app.handlers.utils import _resolve_runtime_path
from core.app.rl import record_rl_sample
from core.app.state import (
    RuntimeState,
    SessionMode,
    SessionState,
    SessionStatus,
    format_task_label,
)
from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger, sanitize_path_component
from core.recorder.lerobot_meta import history_row
from core.types import Observation, RolloutInterventionSegment
from transport.base import HilStatus

logger = logging.getLogger(__name__)


COLLECT_STEP_MAX_RAW_SNAPSHOTS = 16
ROLLOUT_STEP_MAX_RAW_SNAPSHOTS = 1
ROLLOUT_INTERVENTION_SOURCE_TRANSPORT = "transport"
ROLLOUT_INTERVENTION_SOURCE_CLIENT = "teleop_client"


# ``episodes.jsonl`` is append-only for normal saves, but it is also rewritten by
# QC/annotation updates. Cache the projected rows by file signature so the history
# endpoint can serve repeated reads without reparsing the whole dataset. The cache
# is deliberately process-local: the console server and the recorder share this
# process, while a changed mtime/size invalidates an entry automatically.
_EPISODE_HISTORY_CACHE_MAX = 32
_EPISODE_HISTORY_CACHE_LOCK = threading.RLock()


@dataclass
class _EpisodeHistoryCacheEntry:
    signature: tuple[int, int, int]
    version: str
    rows: list[dict[str, Any]]
    views: dict[str | None, tuple[list[dict[str, Any]], list[str]]]


_EPISODE_HISTORY_CACHE: dict[Path, _EpisodeHistoryCacheEntry] = {}
_EPISODE_HISTORY_COUNT_CACHE: dict[Path, tuple[tuple[int, int, int], str, int]] = {}


def _episode_history_signature(path: Path) -> tuple[int, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, 0, 0)
    return (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))


def _episode_history_version(signature: tuple[int, int, int]) -> str:
    return "-".join(f"{value:x}" for value in signature)


def _json_object(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _iter_json_objects(path: Path):
    try:
        stream = path.open()
    except OSError:
        return
    with stream:
        for line in stream:
            row = _json_object(line)
            if row is not None:
                yield row


def _read_episode_history_rows(path: Path) -> list[dict[str, Any]]:
    """Read and project one dataset history, tolerating a partial append line."""
    return [history_row(row, index) for index, row in enumerate(_iter_json_objects(path))]


def _episode_history_cursors(rows: list[dict[str, Any]]) -> list[str]:
    """Build stable cursors for every visible prefix in one pass."""
    digest = hashlib.blake2s(digest_size=12)
    cursors = [digest.hexdigest()]
    for row in rows:
        payload = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        digest.update(payload)
        digest.update(b"\n")
        cursors.append(digest.hexdigest())
    return cursors


def _count_episode_history(dataset_dir: Path) -> tuple[int, str]:
    """Count valid metadata rows without allocating the web history projection."""
    resolved = Path(dataset_dir).resolve()
    path = resolved / "meta" / "episodes.jsonl"
    signature = _episode_history_signature(path)
    with _EPISODE_HISTORY_CACHE_LOCK:
        cached_rows = _EPISODE_HISTORY_CACHE.get(resolved)
        if cached_rows is not None and cached_rows.signature == signature:
            return len(cached_rows.rows), cached_rows.version
        cached_count = _EPISODE_HISTORY_COUNT_CACHE.get(resolved)
        if cached_count is not None and cached_count[0] == signature:
            return cached_count[2], cached_count[1]
        count = sum(1 for _ in _iter_json_objects(path))
        version = _episode_history_version(signature)
        _EPISODE_HISTORY_COUNT_CACHE[resolved] = (signature, version, count)
        while len(_EPISODE_HISTORY_COUNT_CACHE) > _EPISODE_HISTORY_CACHE_MAX:
            _EPISODE_HISTORY_COUNT_CACHE.pop(next(iter(_EPISODE_HISTORY_COUNT_CACHE)))
        return count, version


def load_episode_history(
    dataset_dir: Path,
    *,
    task: str | None = None,
    since: int = 0,
    limit: int | None = None,
    cursor: str | None = None,
    exclude_episode_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Return a cached, offset-paginated episode history projection.

    ``task`` is applied before pagination, so ``since`` is the number of matching
    rows already held by the caller (an offset, not an episode id). This remains
    correct when episode ids have gaps or a QC update rewrites an existing row.
    The returned ``version`` lets a client reset the cursor when an old row was edited.
    """
    resolved = Path(dataset_dir).resolve()
    path = resolved / "meta" / "episodes.jsonl"
    signature = _episode_history_signature(path)
    if limit == 0 and task is None and not cursor and not exclude_episode_indices:
        total, version = _count_episode_history(resolved)
        offset = min(max(0, int(since)), total)
        return {
            "episodes": [],
            "total": total,
            "since": offset,
            "next_since": offset,
            "has_more": offset < total,
            "version": version,
            "cursor": "",
            "reset": False,
        }
    with _EPISODE_HISTORY_CACHE_LOCK:
        cached = _EPISODE_HISTORY_CACHE.get(resolved)
        if cached is None or cached.signature != signature:
            rows = _read_episode_history_rows(path)
            cached = _EpisodeHistoryCacheEntry(
                signature,
                _episode_history_version(signature),
                rows,
                {None: (rows, _episode_history_cursors(rows))},
            )
            _EPISODE_HISTORY_CACHE[resolved] = cached
            _EPISODE_HISTORY_COUNT_CACHE[resolved] = (
                signature,
                cached.version,
                len(rows),
            )
            while len(_EPISODE_HISTORY_CACHE) > _EPISODE_HISTORY_CACHE_MAX:
                _EPISODE_HISTORY_CACHE.pop(next(iter(_EPISODE_HISTORY_CACHE)))
            while len(_EPISODE_HISTORY_COUNT_CACHE) > _EPISODE_HISTORY_CACHE_MAX:
                _EPISODE_HISTORY_COUNT_CACHE.pop(next(iter(_EPISODE_HISTORY_COUNT_CACHE)))
        view = cached.views.get(task)
        if view is None:
            filtered_rows = [row for row in cached.rows if str(row.get("task", "")) == task]
            view = (filtered_rows, _episode_history_cursors(filtered_rows))
            cached.views[task] = view

    filtered_rows, cursors = view
    if exclude_episode_indices:
        filtered_rows = [
            row for row in filtered_rows if row.get("episode_index") not in exclude_episode_indices
        ]
        cursors = _episode_history_cursors(filtered_rows)
    total = len(filtered_rows)
    offset = min(max(0, int(since)), total)
    reset = bool(cursor) and cursor != cursors[offset]
    if reset:
        offset = 0
    if limit is None:
        end = total
    else:
        end = min(total, offset + max(0, int(limit)))
    return {
        "episodes": filtered_rows[offset:end],
        "total": total,
        "since": offset,
        "next_since": end,
        "has_more": end < total,
        "version": cached.version,
        "cursor": cursors[end],
        "reset": reset,
    }


def state_to_eef(config: ConfigDict, runtime: RuntimeState, qpos_state: np.ndarray) -> np.ndarray:
    """Map a qpos state vector to EEF space via forward kinematics when required.

    Shared by the observation, recording (eval/exec), and collection paths so all
    three derive EEF identically. Runs FK only in EEF mode when the robot declares
    no external EEF source; otherwise returns the input unchanged. All callers run
    on the main control loop, so the single cached solver is reused without locking.

    Args:
        qpos_state: joint-space state [qpos_dim] float32.

    Returns:
        EEF-space state [eef_dim] float32 (8D per arm: xyz + quat wxyz + gripper)
        when FK is applied, else the unchanged input.
    """
    if not config.inference_cfg.obs_space.is_eef():
        return qpos_state
    return _qpos_to_eef(config, runtime, qpos_state)


def _qpos_to_eef(config: ConfigDict, runtime: RuntimeState, qpos: np.ndarray) -> np.ndarray:
    from core.app.handlers.io import ensure_ik_solver

    solver = ensure_ik_solver(config, runtime)
    return solver.fk_chunk(qpos[np.newaxis, :])[0]


def _fill_record_eef(
    config: ConfigDict,
    runtime: RuntimeState,
    frame: Observation,
    action: np.ndarray,
) -> None:
    """Derive the EEF state/action columns on the main loop before recording.

    In EEF mode the recorder writes a second eef column alongside the raw qpos; when the
    robot has no external eef source the eef vectors are produced from qpos via forward
    kinematics here (on the main control loop, where the shared FK scratch is safe). In
    joint mode nothing is set. Fails fast when EEF derivation is required but qpos is
    absent.

    Args:
        frame: observation to fill in place; reads frame.state_qpos.
        action: [Da] executed/teleop action in joint space, mapped to action_eef.
    """
    obs_is_eef = config.inference_cfg.obs_space.is_eef()
    action_is_eef = config.inference_cfg.action_space.is_eef()
    if not obs_is_eef and not action_is_eef:
        return
    if frame.state_qpos is None:
        raise RuntimeError("EEF recording requires state_qpos but the frame has none")
    if obs_is_eef and frame.state_eef is None:
        frame.state_eef = state_to_eef(config, runtime, frame.state_qpos)
    if action_is_eef and frame.action_eef is None:
        frame.action_eef = _qpos_to_eef(
            config,
            runtime,
            np.asarray(action, dtype=np.float32),
        )


def build_policy_observation(
    frame: Observation,
    prompt: str,
    config: ConfigDict,
    runtime: RuntimeState,
) -> dict:
    """Build the policy-server observation dict from a raw transport frame.

    Each camera image is preprocessed (color/resize/layout per transport config) and
    combined with the robot state and prompt into the schema the policy expects. In EEF
    mode the frame's EEF state is preferred when present, else it is derived from
    state_qpos via forward kinematics; in joint mode state_qpos is used directly.

    Args:
        frame: Raw observation with per-camera images and the robot state vectors.
        prompt: Task instruction string passed to the policy.

    Returns:
        The observation dict produced by ``runtime.robot.build_observation``.
    """
    processed_images = {}
    for key, image in frame.images.items():
        processed_images[key] = prepare_image(
            image,
            config.transport.convert_bgr_to_rgb,
            config.transport.image_height,
            config.transport.image_width,
            resize_pad=config.transport.resize_pad,
            image_layout=config.transport.image_layout,
        )
    if config.inference_cfg.obs_space.is_eef():
        state = (
            frame.state_eef
            if frame.state_eef is not None
            else state_to_eef(config, runtime, frame.state_qpos)
        )
    else:
        state = frame.state_qpos
    return runtime.robot.build_observation(
        images=processed_images,
        state=state,
        prompt=prompt,
    )


def _is_collection_config(config: ConfigDict) -> bool:
    """A config is in collection mode when its collection schema declares columns.

    The collection section is always present (from defaults) but empty for
    deploy/eval configs; only real collection presets fill schema.columns.
    """
    coll = config.get("collection") or {}
    return bool((coll.get("schema") or {}).get("columns"))


def _recording_space(config: ConfigDict) -> str:
    inference_cfg = config.get("inference_cfg")
    if inference_cfg is None:
        return "qpos"
    action_space = inference_cfg.get("action_space")
    return "eef" if action_space is not None and action_space.is_eef() else "qpos"


def _gripper_recording_config(
    config: ConfigDict,
) -> tuple[float | None, float | None, float | None]:
    robot = config.get("robot") or {}
    return (
        robot.get("gripper_open"),
        robot.get("gripper_close"),
        robot.get("gripper_threshold"),
    )


def rollout_intervention_source(config: ConfigDict) -> str:
    teleop_cfg = (config.get("collection") or {}).get("teleop") or {}
    if str(teleop_cfg.get("control_source", "")) == TELEOP_CONTROL_SOURCE_CLIENT:
        return ROLLOUT_INTERVENTION_SOURCE_CLIENT
    rl_cfg = config.get("rl")
    if rl_cfg is not None:
        rl_intervention = rl_cfg.get("intervention") or {}
        source = rl_intervention.get("source")
        if source:
            return str(source)
    rollout_cfg = config.get("rollout") or {}
    rollout_intervention = rollout_cfg.get("intervention") or {}
    source = rollout_intervention.get("source")
    if source:
        return str(source)
    return ROLLOUT_INTERVENTION_SOURCE_TRANSPORT


def rollout_hil_status(config: ConfigDict, runtime: RuntimeState) -> HilStatus:
    source = rollout_intervention_source(config)
    if source != ROLLOUT_INTERVENTION_SOURCE_CLIENT:
        return runtime.transport.hil_status()
    execution = getattr(runtime, "teleop_execution", None)
    client = getattr(runtime, "teleop_client", None)
    if execution is None or execution.control_source != TELEOP_CONTROL_SOURCE_CLIENT:
        return HilStatus(supported=False, error="Teleop client control is not configured")
    if client is None:
        return HilStatus(supported=False, error="Teleop client is unavailable")
    status = client.status()
    error = status.source_error
    if not status.connected and not error:
        error = "Teleop client is not connected"
    return HilStatus(
        supported=True,
        active=bool(runtime.rollout_intervention_active and execution.active and status.connected),
        error=error,
    )


def resolve_storage(config: ConfigDict) -> ConfigDict | None:
    """Return the active recording storage block: ``eval.storage`` for eval configs,
    else ``collection.storage`` for collection configs, else None (no recording).
    """
    if config.get("eval"):
        return config.eval.storage
    if _is_collection_config(config):
        return config.collection.storage
    return None


def maybe_build_episode_logger(config: ConfigDict, runtime: RuntimeState) -> None:
    """Construct the collection EpisodeLogger once for a collection config.

    Collection configs carry a ``collection.storage`` block (fps / log_dir /
    save_queue_max). Eval recording is built separately by
    ``rebuild_eval_episode_logger`` reading ``eval.storage``. No-op for
    non-collection configs or when a logger already exists.
    """
    if not _is_collection_config(config) or runtime.episode_logger is not None:
        return
    collection = config.collection
    storage = collection.storage
    gripper_open, gripper_close, gripper_threshold = _gripper_recording_config(config)
    runtime.episode_logger = EpisodeLogger(
        log_dir=_resolve_runtime_path(storage.log_dir),
        robot=runtime.robot,
        fps=storage.fps,
        dataset_keys=config.transport.dataset_keys,
        convert_bgr_to_rgb=config.transport.convert_bgr_to_rgb,
        collection=collection,
        # Collection runs the disk write off the capture loop so saving never stalls
        # frame ingest.
        async_save=True,
        save_queue_max=storage.save_queue_max,
        eval_mode=False,
        save_image_height=storage.get("image_height"),
        save_image_width=storage.get("image_width"),
        recording_space=_recording_space(config),
        gripper_open=gripper_open,
        gripper_close=gripper_close,
        gripper_threshold=gripper_threshold,
        eef_reference_frame=config.robot.eef_reference_frame,
    )


def _rollout_storage(config: ConfigDict, runtime: RuntimeState | None = None) -> ConfigDict:
    if runtime is not None and runtime.rl_active and config.rl is not None:
        return config.rl.data.storage
    return config.rollout.storage


def rollout_save_log_dir(config: ConfigDict, runtime: RuntimeState | None = None) -> Path:
    """Return the rollout save dataset dir, separate from collection/eval datasets."""
    storage = _rollout_storage(config, runtime)
    if storage.log_dir:
        return _resolve_runtime_path(storage.log_dir)
    return _resolve_runtime_path(
        Path(config.get("work_dir") or "work_dirs") / "rollout" / config.robot.type
    )


def maybe_build_rollout_episode_logger(config: ConfigDict, runtime: RuntimeState) -> None:
    """Construct the rollout EpisodeLogger when explicit rollout saving is enabled."""
    storage = _rollout_storage(config, runtime)
    if not bool(storage.get("enabled", True)):
        logger_obj = runtime.rollout_episode_logger
        runtime.rollout_episode_logger = None
        if logger_obj is not None:
            if logger_obj.has_active_episode:
                logger_obj.cancel_episode("rollout storage disabled")
            logger_obj.finalize()
        return
    dataset_path = rollout_save_log_dir(config, runtime)
    logger_obj = runtime.rollout_episode_logger
    if logger_obj is not None and Path(logger_obj._log_dir).resolve() == dataset_path.resolve():
        return
    if logger_obj is not None:
        if logger_obj.has_active_episode:
            logger_obj.cancel_episode("rollout dataset changed")
        logger_obj.finalize()
    gripper_open, gripper_close, gripper_threshold = _gripper_recording_config(config)
    runtime.rollout_episode_logger = EpisodeLogger(
        log_dir=dataset_path,
        robot=runtime.robot,
        fps=storage.fps,
        dataset_keys=config.transport.dataset_keys,
        convert_bgr_to_rgb=config.transport.convert_bgr_to_rgb,
        collection=None,
        async_save=bool(storage.get("async_save", True)),
        save_queue_max=storage.save_queue_max,
        save_image_height=storage.get("image_height"),
        save_image_width=storage.get("image_width"),
        recording_space=_recording_space(config),
        gripper_open=gripper_open,
        gripper_close=gripper_close,
        gripper_threshold=gripper_threshold,
    )


def begin_rollout_save_episode(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> None:
    """Start an in-memory rollout that can be saved after stop/reset."""
    stop_collection_capture(runtime)
    maybe_build_rollout_episode_logger(config, runtime)
    logger_obj = runtime.rollout_episode_logger
    if logger_obj is None:
        return
    if logger_obj.has_active_episode:
        logger_obj.cancel_episode("superseded by new rollout")
    runtime.rollout_save_ready = False
    runtime.rollout_save_reason = ""
    runtime.rollout_intervention_pre_qpos = None
    runtime.rollout_intervention_active_segment = None
    runtime.rollout_intervention_segments = []
    runtime.rollout_intervention_next_segment_index = 0
    runtime.rollout_exclusion_active = None
    runtime.rollout_exclusions = []
    runtime.rollout_raw_snapshots = queue.Queue()
    runtime.rollout_policy_actions = []
    runtime.transport.start_policy_collection()
    runtime.transport.clear_collection_backlog()
    logger_obj.start_episode(task=format_task_label(session.selected_task))
    start_collection_capture(
        runtime,
        fps=config.inference_cfg.publish_rate,
        max_raw_snapshots_per_tick=ROLLOUT_STEP_MAX_RAW_SNAPSHOTS,
    )


def mark_rollout_save_ready(runtime: RuntimeState, reason: str) -> None:
    """Mark the active rollout as eligible for explicit save."""
    logger_obj = runtime.rollout_episode_logger
    if logger_obj is None or not logger_obj.has_active_episode:
        return
    buffered_frames = (
        logger_obj.active_frame_count
        + len(runtime.rollout_policy_actions)
        + runtime.rollout_raw_snapshots.qsize()
        + sum(len(segment.frames) for segment in runtime.rollout_intervention_segments)
    )
    if buffered_frames <= 0:
        logger_obj.cancel_episode("empty rollout")
        runtime.rollout_save_ready = False
        runtime.rollout_save_reason = ""
        return
    runtime.rollout_save_ready = True
    runtime.rollout_save_reason = reason


def _close_rollout_exclusion(runtime: RuntimeState, end_time: float) -> None:
    active = runtime.rollout_exclusion_active
    if active is None:
        return
    reason, start_time = active
    runtime.rollout_exclusions.append((reason, start_time, end_time))
    runtime.rollout_exclusion_active = None
    logger.info(
        "[ROLLOUT_EXCLUSION] reason=%s start=%.6f end=%.6f duration_ms=%.1f",
        reason,
        start_time,
        end_time,
        (end_time - start_time) * 1000.0,
    )


def save_rollout_episode(runtime: RuntimeState, session: SessionState) -> bool:
    """Persist the stopped rollout through its dedicated EpisodeLogger.

    Args:
        runtime: Active runtime containing the rollout logger and transport.
        session: Session receiving any save error.

    Returns:
        True when the episode was queued or written, otherwise False.
    """
    logger_obj = runtime.rollout_episode_logger
    if logger_obj is None:
        session.last_error = "Rollout save is disabled"
        return False
    if not runtime.rollout_save_ready or not logger_obj.has_active_episode:
        session.last_error = "Stop or reset before saving the rollout"
        return False
    stop_collection_capture(runtime)
    release_memory = getattr(logger_obj, "release_unused_memory", None)
    if release_memory is not None:
        release_memory()
    runtime.transport.stop_collection()
    _close_rollout_exclusion(runtime, time.time())
    intervention_ranges = []
    for segment in runtime.rollout_intervention_segments:
        if not segment.frames:
            continue
        start = segment.frames[0].timestamp
        end = segment.frames[-1].timestamp
        if start is None or end is None:
            continue
        intervention_ranges.append((segment.segment_index, float(start), float(end)))
    snapshot_count = 0
    while True:
        try:
            snapshot = runtime.rollout_raw_snapshots.get_nowait()
        except queue.Empty:
            break
        intervention = False
        segment_index = -1
        for candidate_index, start, end in intervention_ranges:
            if start <= snapshot.timestamp <= end:
                intervention = True
                segment_index = candidate_index
                break
        logger_obj.ingest_raw_episode_snapshot(
            snapshot,
            intervention=intervention,
            segment_index=segment_index,
        )
        snapshot_count += 1
    for timestamp, action, state_qpos, state_eef, action_eef in runtime.rollout_policy_actions:
        logger_obj.ingest_policy_action(
            action,
            state_qpos,
            timestamp,
            state_eef=state_eef,
            action_eef=action_eef,
        )
    logger.info(
        "[ROLLOUT_CAPTURE] finalize policy_actions=%d raw_snapshots=%d intervention_frames=%d",
        len(runtime.rollout_policy_actions),
        snapshot_count,
        sum(len(segment.frames) for segment in runtime.rollout_intervention_segments),
    )
    if runtime.rollout_policy_actions:
        logger.info(
            "[ROLLOUT_CAPTURE] policy_time_range start=%.6f end=%.6f",
            runtime.rollout_policy_actions[0][0],
            runtime.rollout_policy_actions[-1][0],
        )
    if logger_obj.active_frame_count <= 0:
        logger_obj.cancel_episode("empty rollout")
        runtime.rollout_save_ready = False
        runtime.rollout_save_reason = ""
        session.last_error = "No rollout frames to save"
        return False
    logger_obj.set_rollout_intervention_segments(runtime.rollout_intervention_segments)
    if runtime.rollout_exclusions:
        logger_obj.set_episode_meta(
            excluded_ranges=[
                {
                    "reason": reason,
                    "start_time": start_time,
                    "end_time": end_time,
                }
                for reason, start_time, end_time in runtime.rollout_exclusions
            ]
        )
    saved = logger_obj.end_episode()
    if not saved:
        session.last_error = "No rollout frames to save"
        return False
    runtime.rollout_save_ready = False
    runtime.rollout_save_reason = ""
    runtime.rollout_intervention_segments = []
    runtime.rollout_intervention_next_segment_index = 0
    runtime.rollout_exclusion_active = None
    runtime.rollout_exclusions = []
    runtime.rollout_raw_snapshots = queue.Queue()
    runtime.rollout_policy_actions = []
    session.last_error = ""
    return True


def discard_rollout_episode(runtime: RuntimeState) -> None:
    """Discard the complete in-flight rollout without writing an episode."""
    logger_obj = runtime.rollout_episode_logger
    stop_collection_capture(runtime)
    if logger_obj is not None and logger_obj.has_active_episode:
        logger_obj.cancel_episode("user reset")
    runtime.transport.stop_collection()
    runtime.rollout_save_ready = False
    runtime.rollout_save_reason = ""
    runtime.rollout_intervention_pre_qpos = None
    runtime.rollout_intervention_active_segment = None
    runtime.rollout_intervention_segments = []
    runtime.rollout_intervention_next_segment_index = 0
    runtime.rollout_exclusion_active = None
    runtime.rollout_exclusions = []
    runtime.rollout_raw_snapshots = queue.Queue()
    runtime.rollout_policy_actions = []


def cancel_eval_episode(runtime: RuntimeState) -> None:
    """Cancel the active evaluation episode without resetting its execution endpoint."""
    stop_collection_capture(runtime)
    logger_obj = runtime.episode_logger
    if logger_obj is not None and logger_obj.is_evaluation and logger_obj.has_active_episode:
        logger_obj.cancel_episode("external eval cancel")
    runtime.transport.stop_collection()


def _load_saved_episode_history(dataset_dir: Path) -> list[dict[str, Any]]:
    """Return the complete projected history, cached by ``episodes.jsonl`` signature."""
    return load_episode_history(dataset_dir)["episodes"]


def _rollout_intervention_save_status(runtime: RuntimeState) -> dict[str, Any]:
    active_segment = runtime.rollout_intervention_active_segment
    accepted_segments = runtime.rollout_intervention_segments
    return {
        "active_intervention_frames": 0 if active_segment is None else len(active_segment.frames),
        "accepted_intervention_segments": len(accepted_segments),
        "save_blocked_by_intervention": runtime.rollout_intervention_active,
    }


def rollout_save_status(
    config: ConfigDict,
    runtime: RuntimeState,
    *,
    include_history: bool = True,
) -> dict[str, Any]:
    """Return queue/progress state for the rollout save panel.

    The high-frequency console status poll passes ``include_history=False`` so
    serializing the live pipeline never reads or embeds the complete history. The
    default remains the original full snapshot for direct callers and tests.
    """
    dataset_path = rollout_save_log_dir(config, runtime)
    if include_history:
        saved_history = _load_saved_episode_history(dataset_path)
        completed_episodes = len(saved_history)
    else:
        saved_history = []
        completed_episodes = load_episode_history(dataset_path, limit=0)["total"]
    logger_obj = runtime.rollout_episode_logger
    if logger_obj is not None and Path(logger_obj._log_dir).resolve() != dataset_path.resolve():
        logger_obj = None
    storage = _rollout_storage(config, runtime)
    snapshot = {
        "enabled": bool(storage.get("enabled", True)),
        "dataset_dir": str(dataset_path),
        "pipeline_state": "IDLE",
        "collecting": False,
        "current_episode_frames": 0,
        "completed_episodes": completed_episodes,
        "save_queue_size": 0,
        "save_queue_max": storage.save_queue_max,
        "progress": 1.0 if completed_episodes else 0.0,
        "eta_sec": None,
        "queue": [],
        "save_ready": False,
        "reason": "",
    }
    if include_history:
        snapshot["episodes"] = saved_history
    if not snapshot["enabled"]:
        snapshot["pipeline_state"] = "DISABLED"
        snapshot["completed_episodes"] = 0
        snapshot["progress"] = 0.0
    elif logger_obj is not None:
        snapshot = logger_obj.status_snapshot(include_history=include_history)
        snapshot["enabled"] = True
        snapshot["dataset_dir"] = str(dataset_path)
        snapshot["completed_episodes"] = completed_episodes
        if include_history:
            snapshot["episodes"] = saved_history
        else:
            snapshot.pop("episodes", None)
        buffered_frames = (
            logger_obj.active_frame_count
            + len(runtime.rollout_policy_actions)
            + runtime.rollout_raw_snapshots.qsize()
        )
        snapshot["save_ready"] = bool(runtime.rollout_save_ready and buffered_frames > 0)
        snapshot["reason"] = runtime.rollout_save_reason
        if snapshot["save_ready"]:
            snapshot["pipeline_state"] = "READY_TO_SAVE"
            snapshot["collecting"] = False
    snapshot.update(_rollout_intervention_save_status(runtime))
    return snapshot


def enable_default_recording(config: ConfigDict, output_dir: str) -> ConfigDict:
    """Root an eval config's per-trial recording at the model's dataset dir.

    Result preview (URDF replay, camera, per-dim chart) needs every trial's qpos +
    camera mp4 on disk. Eval roots each model's dataset at
    ``<output_dir>/<model_name>/episodes`` (see ``eval_episode_dir``) so a model's
    prior episodes auto-resume. The computed dir lands in ``eval.storage.log_dir``.
    No-op for non-eval configs (collection/plain console manage their own storage).
    """
    if not config.eval:
        return config
    new_config = copy.deepcopy(config)
    new_config.eval.storage.log_dir = str(
        eval_episode_dir(output_dir, eval_model_name(config, None))
    )
    return new_config


def eval_model_name(config: ConfigDict, runtime: RuntimeState | None) -> str:
    """Folder name for the active model's eval dataset.

    Blind multi-ckpt: the real ckpt name of the active slot. Single anonymous model
    (no checkpoints): the first ckpt name if the eval lists one, else ``"model"``.
    """
    eval_cfg = config.eval
    if eval_cfg is None:
        return "model"
    if eval_cfg.checkpoints:
        slot = (
            0 if runtime is None or runtime.active_ckpt_slot is None else runtime.active_ckpt_slot
        )
        order = (
            runtime.ckpt_order
            if (runtime and runtime.ckpt_order)
            else list(range(len(eval_cfg.checkpoints)))
        )
        return eval_cfg.checkpoints[order[slot]].name
    return "model"


def eval_episode_dir(output_dir: str, model_name: str) -> Path:
    """Return the raw per-model eval dataset, isolated from future exports."""
    return Path(output_dir) / sanitize_path_component(model_name) / "episodes" / "raw"


def rebuild_eval_episode_logger(config: ConfigDict, runtime: RuntimeState) -> None:
    """Point the episode logger at the active model's per-model dataset.

    Called at bootstrap and on every ckpt switch. Finalizes the previous logger (flush
    info.json/stats) and builds a fresh one rooted at the new model's dir, which the
    EpisodeLogger naturally resumes (next index = existing episode count) so prior eval
    episodes and their embedded scores carry over.
    """
    if not config.eval:
        return
    storage = config.eval.storage
    target = eval_episode_dir(runtime.eval_output_dir, eval_model_name(config, runtime))
    prev = runtime.episode_logger
    if prev is not None and Path(prev._log_dir) == target:
        return
    if prev is not None:
        try:
            prev.finalize()
        except Exception:
            logger.exception("Failed finalizing previous eval logger")
    gripper_open, gripper_close, gripper_threshold = _gripper_recording_config(config)
    runtime.episode_logger = EpisodeLogger(
        log_dir=target,
        robot=runtime.robot,
        fps=storage.fps,
        dataset_keys=config.transport.dataset_keys,
        convert_bgr_to_rgb=config.transport.convert_bgr_to_rgb,
        collection=None,
        async_save=True,
        save_queue_max=storage.save_queue_max,
        eval_mode=True,
        save_image_height=storage.get("image_height"),
        save_image_width=storage.get("image_width"),
        recording_space=_recording_space(config),
        gripper_open=gripper_open,
        gripper_close=gripper_close,
        gripper_threshold=gripper_threshold,
    )


def start_episode(runtime: RuntimeState, session: SessionState) -> None:
    """Begin one episode = one inference run (status -> RUNNING)."""
    if runtime.episode_logger is not None:
        prepare_collection_capture(runtime, runtime.episode_logger)
        runtime.episode_logger.start_episode(task=format_task_label(session.selected_task))


def end_episode(runtime: RuntimeState) -> None:
    """Close the current episode, flushing parquet + mp4 to the dataset."""
    stop_collection_capture(runtime)
    if runtime.episode_logger is not None:
        runtime.episode_logger.end_episode()


def start_rollout_intervention(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> bool:
    """Enable teleop takeover for a paused normal rollout."""
    if runtime.rollout_intervention_active:
        return True
    if not runtime.rollout_intervention_enabled:
        session.last_error = "Rollout HIL is off"
        return False
    source = rollout_intervention_source(config)
    hil_status = rollout_hil_status(config, runtime)
    if not hil_status.supported or hil_status.error:
        session.last_error = hil_status.error or (
            "Teleop client is unavailable"
            if source == ROLLOUT_INTERVENTION_SOURCE_CLIENT
            else "Transport does not support HIL"
        )
        return False
    pre_qpos = runtime.transport.get_latest_qpos()
    if pre_qpos is None:
        session.last_error = "Cannot start rollout intervention without joint feedback"
        return False
    # Keep the raw capture runner active through HIL so camera streams remain continuous.
    if runtime.infer_strategy is not None:
        runtime.infer_strategy.reset()
    if source == ROLLOUT_INTERVENTION_SOURCE_CLIENT:
        if not activate_rollout_teleop(config, runtime, session):
            return False
    else:
        started = runtime.transport.start_hil_control(runtime.hil_control_mode)
        if not started.supported or not started.active or started.error:
            session.last_error = started.error or "HIL takeover did not activate"
            return False
    intervention_start_time = time.time()
    _close_rollout_exclusion(runtime, intervention_start_time)
    runtime.rollout_intervention_pre_qpos = np.asarray(pre_qpos, dtype=np.float32).copy()
    runtime.rollout_intervention_active_segment = RolloutInterventionSegment(
        segment_index=runtime.rollout_intervention_next_segment_index,
        start_policy_frame_index=session.step_index,
        pre_intervention_qpos=runtime.rollout_intervention_pre_qpos.copy(),
        start_time=intervention_start_time,
    )
    runtime.rollout_intervention_next_segment_index += 1
    runtime.rollout_intervention_active = True
    session.last_error = ""
    logger.info(
        "Rollout teleop intervention started policy_actions=%d queued_raw_snapshots=%d",
        len(runtime.rollout_policy_actions),
        runtime.rollout_raw_snapshots.qsize(),
    )
    return True


def stop_rollout_intervention(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    *,
    required: bool,
) -> bool:
    """Disable rollout teleop takeover before policy resume or cleanup."""
    if not runtime.rollout_intervention_active:
        return True
    if rollout_intervention_source(config) == ROLLOUT_INTERVENTION_SOURCE_CLIENT:
        try:
            deactivate_rollout_teleop(runtime)
        except Exception as error:
            session.last_error = f"Teleop takeover did not stop: {error}"
            if required:
                return False
            logger.warning("Rollout teleop cleanup failed during optional stop: %s", error)
            return False
    else:
        stopped = runtime.transport.stop_hil_control()
        if required and (stopped.active or stopped.error):
            session.last_error = stopped.error or "HIL takeover did not stop"
            return False
    runtime.rollout_intervention_active = False
    logger.info(
        "Rollout teleop intervention stopped queued_raw_snapshots=%d",
        runtime.rollout_raw_snapshots.qsize(),
    )
    _ = config, required
    return True


def _copy_observation(frame: Observation) -> Observation:
    images = {key: np.asarray(value).copy() for key, value in frame.images.items()}
    vectors = {}
    for field in (
        "state_qpos",
        "state_eef",
        "action_qpos",
        "action_eef",
    ):
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


def _record_rollout_intervention_frame(
    runtime: RuntimeState, session: SessionState, frame: Observation
) -> bool:
    segment = runtime.rollout_intervention_active_segment
    if segment is None:
        return False
    if frame.action_qpos is None and frame.action_eef is None:
        return False
    if frame.state_qpos is None and frame.state_eef is None:
        segment.invalid_reason = "Intervention frame has no state"
        session.last_error = segment.invalid_reason
        return False
    copied = _copy_observation(frame)
    segment.frames.append(copied)
    state = copied.state_qpos if copied.state_qpos is not None else copied.state_eef
    action = copied.action_qpos if copied.action_qpos is not None else copied.action_eef
    record_rl_sample(
        runtime,
        state,
        action,
        "intervention",
        timestamp=copied.timestamp,
        segment_index=segment.segment_index,
    )
    return True


def record_rollout_intervention_step(runtime: RuntimeState, session: SessionState) -> bool:
    """Record available teleop frames while normal rollout intervention is active."""
    if not runtime.rollout_intervention_active:
        return False
    segment = runtime.rollout_intervention_active_segment
    if segment is None or segment.invalid_reason:
        return False
    frame = runtime.transport.get_hil_frame()
    if frame is None:
        return False
    return _record_rollout_intervention_frame(runtime, session, frame)


def record_client_rollout_intervention_step(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    published: PublishedTeleopAction,
) -> bool:
    """Record one client-driven RL intervention step into the active rollout segment."""
    if not runtime.rollout_intervention_active:
        return False
    segment = runtime.rollout_intervention_active_segment
    if segment is None or segment.invalid_reason:
        return False
    frame = runtime.transport.get_frame()
    if frame is None:
        return False
    recorded = _copy_observation(frame)
    if recorded.timestamp is None or recorded.timestamp <= 0.0:
        recorded.timestamp = time.time()
    recorded.action_qpos = published.qpos.copy()
    if recorded.state_qpos is None:
        latest_qpos = runtime.transport.get_latest_qpos()
        if latest_qpos is not None:
            recorded.state_qpos = np.asarray(latest_qpos, dtype=np.float32).copy()
    try:
        _fill_record_eef(config, runtime, recorded, published.qpos)
    except Exception as error:
        segment.invalid_reason = f"Intervention frame EEF derivation failed: {error}"
        session.last_error = segment.invalid_reason
        return False
    return _record_rollout_intervention_frame(runtime, session, recorded)


def accept_rollout_intervention_segment(runtime: RuntimeState, session: SessionState) -> bool:
    """Keep the active intervention segment so rollout SAVE can write it later."""
    segment = runtime.rollout_intervention_active_segment
    if segment is None:
        runtime.rollout_intervention_pre_qpos = None
        return True
    if segment.invalid_reason:
        session.last_error = segment.invalid_reason
        return False
    segment.resume_policy_frame_index = session.step_index
    if segment.frames:
        runtime.rollout_intervention_segments.append(segment)
    runtime.rollout_intervention_active_segment = None
    runtime.rollout_intervention_pre_qpos = None
    return True


def discard_rollout_intervention_segment(runtime: RuntimeState) -> None:
    """Drop the active intervention segment without saving it."""
    runtime.rollout_intervention_active_segment = None
    runtime.rollout_intervention_pre_qpos = None


def rollback_rollout_intervention(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> bool:
    """Return the robot to the qpos captured before the active intervention."""
    target = runtime.rollout_intervention_pre_qpos
    if target is None:
        session.last_error = "No pre-intervention qpos to roll back to"
        return False
    current_qpos = runtime.transport.get_latest_qpos()
    if current_qpos is None:
        session.last_error = "Cannot roll back intervention without joint feedback"
        return False
    from core.app.handlers.control import (
        MANUAL_MAX_QPOS_STEP,
        consume_motion_interrupt,
        poll_motion_commands,
        publish_action,
    )
    from core.app.handlers.imaging import build_linear_trajectory

    current = np.asarray(current_qpos, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    gripper_mask = np.asarray(runtime.robot.gripper_mask, dtype=bool)
    joint_delta = np.abs(target - current)[~gripper_mask]
    max_delta = float(np.max(joint_delta)) if joint_delta.size else 0.0
    steps = max(1, int(np.ceil(max_delta / MANUAL_MAX_QPOS_STEP)))
    trajectory = build_linear_trajectory(current, target, steps + 1)[1:]
    trajectory[:, gripper_mask] = target[gripper_mask]
    rate = runtime.transport.create_rate(config.inference_cfg.publish_rate)
    for action in trajectory:
        if poll_motion_commands(config, runtime, session):
            consume_motion_interrupt(session, SessionStatus.READY)
            session.last_error = "Intervention rollback interrupted"
            return False
        if runtime.transport.has_sim_publishers():
            publish_action(runtime, action, target="sim")
        publish_action(runtime, action, target="real")
        session.sim_preview_qpos = np.asarray(action, dtype=np.float32).copy()
        rate.sleep()
    return True


def collect_start_teleop(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    """Enter collect teleoperation without opening a recording episode."""
    runtime.collection_replay_qpos = None
    runtime.collection_replay_episode = None
    if not getattr(runtime, "collection_teleop_armed", False):
        session.last_error = "Collection teleop requires COLLECT activation"
        logger.warning("collect_start_teleop refused: activation gate is off")
        return False
    if runtime.collection_teleop_active:
        session.mode = SessionMode.COLLECT
        if session.status is not SessionStatus.RUNNING:
            session.status = SessionStatus.READY
        return True
    if not runtime.transport.supports_collection():
        session.last_error = "Transport does not support collection frames"
        return False
    runtime.transport.reset_hil_control()
    runtime.transport.set_hil_relay_enabled(True)
    runtime.transport.start_collection()
    runtime.collection_teleop_active = True
    runtime.last_collection_timestamp = None
    session.mode = SessionMode.COLLECT
    if session.status is not SessionStatus.RUNNING:
        session.status = SessionStatus.READY
    session.last_error = ""
    logger.info("Collection teleop started")
    return True


def collect_stop_teleop(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """Leave collect teleoperation; callers close any active recording episode first."""
    deactivate_teleop(config, runtime, session)


def _collection_target_meta(session: SessionState) -> dict[str, object]:
    """Return the optional scene target for the next episode."""
    fields = {
        "scene_id": session.collection_scene_id,
        "scene_round": session.collection_scene_round,
        "slot_id": session.collection_slot_id,
        "task_id": session.collection_task_id,
    }
    return {key: value for key, value in fields.items() if value is not None and value != ""}


def collect_start(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    """Begin one teleop collection episode. Backpressure: refuse to start when the
    async save queue is full so we never grow memory unbounded. Returns True when
    an episode was opened."""
    if runtime.episode_logger is None:
        session.last_error = "Recording is disabled (log.enabled is false)"
        return False
    if runtime.episode_logger.is_queue_full():
        session.last_error = "Save queue is full; wait for a slot"
        logger.warning("collect_start refused: save queue full")
        return False
    runtime.collection_replay_qpos = None
    runtime.collection_replay_episode = None
    if runtime.episode_logger.is_collection_enabled:
        if not activate_teleop(config, runtime, session):
            return False
        logger_obj = runtime.episode_logger
        try:
            collection_min_capture_time = runtime.transport.clear_collection_backlog()
            logger_obj.start_episode(
                task=format_task_label(session.selected_collect_task),
                collection_min_capture_time=collection_min_capture_time,
                collection_dataset=session.selected_collect_set,
            )
            target_meta = _collection_target_meta(session)
            if target_meta:
                logger_obj.set_episode_meta(**target_meta)
            control_source = str(
                (config.collection.teleop or {}).get("control_source", "transport")
            )
            if control_source == "transport":
                start_collection_capture(
                    runtime,
                    fps=config.inference_cfg.publish_rate,
                    max_raw_snapshots_per_tick=COLLECT_STEP_MAX_RAW_SNAPSHOTS,
                )
        except Exception as error:
            # The episode may have been opened before capture startup failed. Cancel it
            # first, then use the same idempotent teleop shutdown as normal deactivation.
            cleanup_errors: list[Exception] = []
            try:
                episode_active = bool(getattr(logger_obj, "has_active_episode", False))
            except Exception as active_error:
                episode_active = False
                cleanup_errors.append(active_error)
            if episode_active:
                try:
                    logger_obj.cancel_episode("collection start failed")
                except Exception as cancel_error:
                    cleanup_errors.append(cancel_error)
                    logger.error(
                        "Failed to cancel collection episode after start failure: %s",
                        cancel_error,
                        exc_info=True,
                    )
            try:
                # This single path stops capture (if it started), disables the relay,
                # and stops transport collection. The input worker remains available
                # for a later ARM command.
                deactivate_teleop(config, runtime, session)
            except Exception as deactivate_error:
                cleanup_errors.append(deactivate_error)
            message = f"Collection start failed: {error}"
            if cleanup_errors:
                message += "; cleanup failed: " + "; ".join(
                    str(cleanup_error) for cleanup_error in cleanup_errors
                )
            session.last_error = message
            logger.exception("Collection start failed")
            return False
    else:
        runtime.episode_logger.start_episode(task=format_task_label(session.selected_collect_task))
        target_meta = _collection_target_meta(session)
        if target_meta:
            runtime.episode_logger.set_episode_meta(**target_meta)
    session.step_index = 0
    session.mode = SessionMode.COLLECT
    session.status = SessionStatus.RUNNING
    logger.info("Collection episode started")
    return True


def collect_step(config: ConfigDict, runtime: RuntimeState) -> bool:
    """Record one synchronized frame during collection.

    The action_qpos field rides along the transport observation stream, so it is
    already aligned with state/images. The collector only records frames here; it
    does not publish robot commands. Returns True when a frame was recorded.
    """
    if runtime.episode_logger is None:
        return False
    if runtime.episode_logger.is_collection_enabled:
        return False
    frame = runtime.transport.get_frame()
    if frame is None:
        return False
    action = frame.action_qpos
    if action is None:
        # No teleop command aligned to this frame yet; skip until one arrives.
        return False
    _fill_record_eef(config, runtime, frame, action)
    runtime.episode_logger.record_step(frame, action)
    return True


def ingest_client_teleop_action(
    config: ConfigDict,
    runtime: RuntimeState,
    published: PublishedTeleopAction,
) -> bool:
    """Pair one client action with an execution-endpoint observation timestamp.

    ``PublishedTeleopAction.timestamp`` belongs to this process's monotonic clock,
    while raw snapshots may be stamped by another process or host. Pairing both
    streams on ``snapshot.timestamp`` keeps collection alignment in one clock domain.
    The call remains single-shot and non-blocking.
    """
    del config
    logger_obj = runtime.episode_logger
    if (
        logger_obj is None
        or not logger_obj.is_collection_enabled
        or not logger_obj.has_active_episode
    ):
        return False
    snapshot = runtime.transport.acquire_collection_raw()
    if snapshot is None:
        return False
    runtime.last_collection_timestamp = float(snapshot.timestamp)
    logger_obj.ingest_collection_action_snapshot(snapshot, published.qpos)
    return True


def collect_stop(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> bool:
    """End the collection episode and leave teleop active at its current pose."""
    saved = False
    if runtime.episode_logger is not None and runtime.episode_logger.is_collection_enabled:
        stop_collection_capture(runtime)
        try:
            saved = runtime.episode_logger.end_episode()
        finally:
            session.status = SessionStatus.READY
        if not saved:
            diagnostics = runtime.transport.collection_diagnostics()
            if diagnostics:
                session.last_error = f"Episode saved 0 frames; {diagnostics}"
            else:
                session.last_error = (
                    "Episode saved 0 frames (teleop action never arrived); nothing recorded"
                )
            logger.warning(session.last_error)
    else:
        if runtime.episode_logger is not None:
            saved = runtime.episode_logger.end_episode()
        session.status = SessionStatus.READY
    if saved:
        console_ctx = getattr(runtime, "console_ctx", None)
        sync_collection_slot = getattr(console_ctx, "sync_collection_slot", None)
        if callable(sync_collection_slot):
            try:
                sync_collection_slot()
            except Exception:
                logger.exception("Failed to advance collection slot after queuing episode")
    logger.info("Collection episode stopped; leaving hardware at current pose")
    return saved


def collect_cancel(runtime: RuntimeState, session: SessionState) -> None:
    """Discard the in-flight collection episode (no save, partial files removed)."""
    if runtime.episode_logger is not None and runtime.episode_logger.is_collection_enabled:
        stop_collection_capture(runtime)
        try:
            runtime.episode_logger.cancel_episode("user cancel")
        finally:
            session.status = SessionStatus.READY
    else:
        if runtime.episode_logger is not None:
            runtime.episode_logger.cancel_episode("user cancel")
        session.status = SessionStatus.READY
    logger.info("Collection episode cancelled")


def _active_loggers(runtime: RuntimeState) -> list:
    """Return loggers with an open episode."""
    return [
        logger_obj
        for logger_obj in (runtime.episode_logger, runtime.rollout_episode_logger)
        if logger_obj is not None and logger_obj.has_active_episode
    ]


def record_executed_action(
    config: ConfigDict,
    session: SessionState,
    action: np.ndarray,
    runtime: RuntimeState | None = None,
) -> None:
    """Capture one raw eval/rollout sample paired with the executed action."""
    _ = config, session
    if runtime is None:
        return
    state_qpos = (
        runtime.transport.get_latest_qpos()
        if hasattr(runtime.transport, "get_latest_qpos")
        else None
    )
    record_rl_sample(runtime, state_qpos, action, "policy")
    loggers = _active_loggers(runtime)
    if not loggers:
        return
    rollout_logger = runtime.rollout_episode_logger
    runner = getattr(runtime, "collection_capture_runner", None)
    rollout_timestamp = None
    if rollout_logger in loggers:
        rollout_timestamp = runtime.last_collection_timestamp
        if rollout_timestamp is None:
            rollout_timestamp = time.time()
        _close_rollout_exclusion(runtime, rollout_timestamp)
    if rollout_logger in loggers and runner is None:
        assert rollout_logger is not None
        snapshot = runtime.transport.acquire_collection_raw()
        while snapshot is None:
            snapshot = runtime.transport.acquire_collection_raw()
        rollout_logger.ingest_raw_episode_snapshot(snapshot, action, state_qpos)
        return
    if rollout_logger in loggers:
        assert rollout_timestamp is not None
        timestamp = rollout_timestamp
        state_eef = None
        action_eef = None
        if config.inference_cfg.obs_space.is_eef() and state_qpos is not None:
            state_eef = state_to_eef(config, runtime, np.asarray(state_qpos, dtype=np.float32))
        if config.inference_cfg.action_space.is_eef():
            action_eef = _qpos_to_eef(
                config,
                runtime,
                np.asarray(action, dtype=np.float32),
            )
        runtime.rollout_policy_actions.append(
            (
                timestamp,
                np.asarray(action, dtype=np.float32).copy(),
                state_qpos,
                state_eef,
                action_eef,
            )
        )
        count = len(runtime.rollout_policy_actions)
        if count == 1 or count % 100 == 0:
            logger.info("[ROLLOUT_CAPTURE] policy_action_count=%d timestamp=%.6f", count, timestamp)
    for logger_obj in loggers:
        if logger_obj is rollout_logger:
            continue
        snapshot = runtime.transport.acquire_collection_raw()
        if snapshot is None:
            logger.warning("Collection raw snapshot missing at action capture")
            continue
        if state_qpos is None:
            logger_obj.ingest_raw_episode_snapshot(snapshot, action)
        else:
            logger_obj.ingest_raw_episode_snapshot(snapshot, action, state_qpos)


__all__ = [
    "COLLECT_STEP_MAX_RAW_SNAPSHOTS",
    "ROLLOUT_STEP_MAX_RAW_SNAPSHOTS",
    "start_collection_capture",
    "stop_collection_capture",
    "state_to_eef",
    "_fill_record_eef",
    "build_policy_observation",
    "resolve_storage",
    "load_episode_history",
    "maybe_build_episode_logger",
    "rollout_save_log_dir",
    "maybe_build_rollout_episode_logger",
    "begin_rollout_save_episode",
    "mark_rollout_save_ready",
    "save_rollout_episode",
    "discard_rollout_episode",
    "cancel_eval_episode",
    "_load_saved_episode_history",
    "rollout_save_status",
    "enable_default_recording",
    "eval_model_name",
    "eval_episode_dir",
    "rebuild_eval_episode_logger",
    "start_episode",
    "end_episode",
    "start_rollout_intervention",
    "stop_rollout_intervention",
    "rollout_hil_status",
    "rollout_intervention_source",
    "record_rollout_intervention_step",
    "record_client_rollout_intervention_step",
    "accept_rollout_intervention_segment",
    "discard_rollout_intervention_segment",
    "rollback_rollout_intervention",
    "collect_start_teleop",
    "collect_stop_teleop",
    "collect_start",
    "collect_step",
    "ingest_client_teleop_action",
    "collect_stop",
    "collect_cancel",
    "_active_loggers",
    "record_executed_action",
]
