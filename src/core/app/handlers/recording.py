"""Observation building + episode/collection recording and eval dataset wiring."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.app.handlers.imaging import prepare_image
from core.app.handlers.utils import _resolve_runtime_path
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
from core.types import Observation

logger = logging.getLogger(__name__)


COLLECT_STEP_MAX_RAW_SNAPSHOTS = 16


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
    from core.app.handlers.io import ensure_ik_solver

    solver = ensure_ik_solver(config, runtime)
    return solver.fk_chunk(qpos_state[np.newaxis, :])[0]


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
    if not config.inference_cfg.obs_space.is_eef():
        return
    if frame.state_qpos is None:
        raise RuntimeError("EEF recording requires state_qpos but the frame has none")
    if frame.state_eef is None:
        frame.state_eef = state_to_eef(config, runtime, frame.state_qpos)
    if frame.action_eef is None:
        frame.action_eef = state_to_eef(config, runtime, np.asarray(action, dtype=np.float32))


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
    )


def _load_saved_episode_history(dataset_dir: Path) -> list[dict[str, Any]]:
    history = []
    path = dataset_dir / "meta" / "episodes.jsonl"
    if not path.exists():
        return history
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            history.append(history_row(row, len(history)))
    return history


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
        return eval_cfg.checkpoints[slot].name
    return "model"


def eval_episode_dir(output_dir: str, model_name: str) -> Path:
    """<output_dir>/<model_name>/episodes — the per-model eval lerobot dataset root."""
    return Path(output_dir) / sanitize_path_component(model_name) / "episodes"


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
    runtime.episode_logger = EpisodeLogger(
        log_dir=target,
        robot=runtime.robot,
        fps=storage.fps,
        dataset_keys=config.transport.dataset_keys,
        convert_bgr_to_rgb=config.transport.convert_bgr_to_rgb,
        collection=None,
        async_save=False,  # eval patches scores post-flush; needs synchronous writes
        save_queue_max=storage.save_queue_max,
        eval_mode=True,
    )


def start_episode(runtime: RuntimeState, session: SessionState) -> None:
    """Begin one episode = one inference run (status -> RUNNING)."""
    if runtime.episode_logger is not None:
        runtime.episode_logger.start_episode(task=format_task_label(session.selected_task))


def end_episode(runtime: RuntimeState) -> None:
    """Close the current episode, flushing parquet + mp4 to the dataset."""
    if runtime.episode_logger is not None:
        runtime.episode_logger.end_episode()


def collect_start_teleop(runtime: RuntimeState, session: SessionState) -> bool:
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
    runtime.transport.start_collection()
    runtime.collection_teleop_active = True
    runtime.last_collection_timestamp = None
    session.mode = SessionMode.COLLECT
    if session.status is not SessionStatus.RUNNING:
        session.status = SessionStatus.READY
    session.last_error = ""
    logger.info("Collection teleop started")
    return True


def collect_stop_teleop(runtime: RuntimeState, session: SessionState) -> None:
    """Leave collect teleoperation; callers close any active recording episode first."""
    if runtime.collection_teleop_active:
        try:
            runtime.transport.stop_collection()
        finally:
            runtime.collection_teleop_active = False
            runtime.last_collection_timestamp = None
    if session.mode is SessionMode.COLLECT and session.status is not SessionStatus.RUNNING:
        session.status = SessionStatus.UNSET
    logger.info("Collection teleop stopped")


def collect_start(runtime: RuntimeState, session: SessionState) -> bool:
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
        if not collect_start_teleop(runtime, session):
            return False
        collection_min_capture_time = runtime.transport.clear_collection_backlog()
        runtime.episode_logger.start_episode(
            task=format_task_label(session.selected_collect_task),
            collection_min_capture_time=collection_min_capture_time,
        )
    else:
        runtime.episode_logger.start_episode(task=format_task_label(session.selected_collect_task))
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
        recorded = False
        for _ in range(COLLECT_STEP_MAX_RAW_SNAPSHOTS):
            snap = runtime.transport.acquire_collection_raw()
            if snap is None:
                return recorded
            runtime.episode_logger.ingest_collection_snapshot(snap)
            recorded = True
        return recorded
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


def collect_stop(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """End the collection episode and leave teleop active at its current pose."""
    if runtime.episode_logger is not None and runtime.episode_logger.is_collection_enabled:
        try:
            saved = runtime.episode_logger.end_episode()
        finally:
            session.status = SessionStatus.READY
        if not saved:
            session.last_error = (
                "Episode saved 0 frames (teleop action never arrived); nothing recorded"
            )
    else:
        if runtime.episode_logger is not None:
            runtime.episode_logger.end_episode()
        session.status = SessionStatus.READY
    logger.info("Collection episode stopped; leaving hardware at current pose")


def collect_cancel(runtime: RuntimeState, session: SessionState) -> None:
    """Discard the in-flight collection episode (no save, partial files removed)."""
    if runtime.episode_logger is not None and runtime.episode_logger.is_collection_enabled:
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
    """The episode logger currently built (skips None)."""
    return [logger_obj for logger_obj in (runtime.episode_logger,) if logger_obj is not None]


def record_executed_action(
    config: ConfigDict,
    session: SessionState,
    action: np.ndarray,
    runtime: RuntimeState | None = None,
) -> None:
    """Main table: pair the freshly-captured obs[t] with the executed action[t]."""
    if runtime is None:
        return
    loggers = _active_loggers(runtime)
    if not loggers:
        return
    frame = runtime.transport.get_frame()
    if frame is None:
        return
    _fill_record_eef(config, runtime, frame, action)
    for logger_obj in loggers:
        logger_obj.record_step(frame, action)
    _ = session


__all__ = [
    "COLLECT_STEP_MAX_RAW_SNAPSHOTS",
    "state_to_eef",
    "_fill_record_eef",
    "build_policy_observation",
    "resolve_storage",
    "maybe_build_episode_logger",
    "_load_saved_episode_history",
    "enable_default_recording",
    "eval_model_name",
    "eval_episode_dir",
    "rebuild_eval_episode_logger",
    "start_episode",
    "end_episode",
    "collect_start_teleop",
    "collect_stop_teleop",
    "collect_start",
    "collect_step",
    "collect_stop",
    "collect_cancel",
    "_active_loggers",
    "record_executed_action",
]
