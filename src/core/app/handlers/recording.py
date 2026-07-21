"""Observation building + episode/collection recording and eval dataset wiring."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.app.collection_capture import start_collection_capture, stop_collection_capture
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
from core.types import Observation, RolloutInterventionSegment

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
        fallback_image_height=config.transport.image_height,
        fallback_image_width=config.transport.image_width,
    )


def rollout_save_log_dir(config: ConfigDict) -> Path:
    """Return the rollout save dataset dir, separate from collection/eval datasets."""
    storage = config.rollout.storage
    if storage.log_dir:
        return _resolve_runtime_path(storage.log_dir)
    return _resolve_runtime_path(
        Path(config.get("work_dir") or "work_dirs") / "rollout" / config.robot.type
    )


def maybe_build_rollout_episode_logger(config: ConfigDict, runtime: RuntimeState) -> None:
    """Construct the rollout EpisodeLogger when explicit rollout saving is enabled."""
    storage = config.rollout.storage
    if not storage.enabled or runtime.rollout_episode_logger is not None:
        return
    runtime.rollout_episode_logger = EpisodeLogger(
        log_dir=rollout_save_log_dir(config),
        robot=runtime.robot,
        fps=storage.fps,
        dataset_keys=config.transport.dataset_keys,
        convert_bgr_to_rgb=config.transport.convert_bgr_to_rgb,
        collection=None,
        async_save=True,
        save_queue_max=storage.save_queue_max,
        save_image_height=storage.get("image_height"),
        save_image_width=storage.get("image_width"),
    )


def begin_rollout_save_episode(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> None:
    """Start an in-memory rollout that can be saved after stop/reset."""
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
    logger_obj.start_episode(task=format_task_label(session.selected_task))
    _stamp_scene_meta(runtime, logger_obj)


def mark_rollout_save_ready(runtime: RuntimeState, reason: str) -> None:
    """Mark the active rollout as eligible for explicit save."""
    logger_obj = runtime.rollout_episode_logger
    if logger_obj is None or not logger_obj.has_active_episode:
        return
    if logger_obj.active_frame_count <= 0:
        logger_obj.cancel_episode("empty rollout")
        runtime.rollout_save_ready = False
        runtime.rollout_save_reason = ""
        return
    runtime.rollout_save_ready = True
    runtime.rollout_save_reason = reason


def save_rollout_episode(runtime: RuntimeState, session: SessionState) -> None:
    """Persist the stopped rollout through its dedicated EpisodeLogger."""
    logger_obj = runtime.rollout_episode_logger
    if logger_obj is None:
        session.last_error = "Rollout save is disabled"
        return
    if not runtime.rollout_save_ready or not logger_obj.has_active_episode:
        session.last_error = "Stop or reset before saving the rollout"
        return
    if logger_obj.active_frame_count <= 0:
        logger_obj.cancel_episode("empty rollout")
        runtime.rollout_save_ready = False
        runtime.rollout_save_reason = ""
        session.last_error = "No rollout frames to save"
        return
    logger_obj.set_rollout_intervention_segments(runtime.rollout_intervention_segments)
    saved = logger_obj.end_episode()
    if not saved:
        session.last_error = "No rollout frames to save"
        return
    runtime.rollout_save_ready = False
    runtime.rollout_save_reason = ""
    runtime.rollout_intervention_segments = []
    runtime.rollout_intervention_next_segment_index = 0
    session.last_error = ""


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


def _rollout_intervention_save_status(runtime: RuntimeState) -> dict[str, Any]:
    active_segment = runtime.rollout_intervention_active_segment
    accepted_segments = runtime.rollout_intervention_segments
    return {
        "active_intervention_frames": 0 if active_segment is None else len(active_segment.frames),
        "accepted_intervention_segments": len(accepted_segments),
        "save_blocked_by_intervention": runtime.rollout_intervention_active,
    }


def rollout_save_status(config: ConfigDict, runtime: RuntimeState) -> dict[str, Any]:
    """Return queue/progress state for the rollout save panel."""
    dataset_path = rollout_save_log_dir(config)
    dataset_dir = str(dataset_path)
    saved_history = _load_saved_episode_history(dataset_path)
    logger_obj = runtime.rollout_episode_logger
    storage = config.rollout.storage
    if not storage.enabled:
        return {
            "enabled": False,
            "dataset_dir": dataset_dir,
            "pipeline_state": "DISABLED",
            "collecting": False,
            "current_episode_frames": 0,
            "completed_episodes": 0,
            "save_queue_size": 0,
            "save_queue_max": storage.save_queue_max,
            "progress": 0.0,
            "eta_sec": None,
            "episodes": saved_history,
            "queue": [],
            "save_ready": False,
            "reason": "",
            **_rollout_intervention_save_status(runtime),
        }
    if logger_obj is None:
        return {
            "enabled": True,
            "dataset_dir": dataset_dir,
            "pipeline_state": "IDLE",
            "collecting": False,
            "current_episode_frames": 0,
            "completed_episodes": len(saved_history),
            "save_queue_size": 0,
            "save_queue_max": storage.save_queue_max,
            "progress": 1.0 if saved_history else 0.0,
            "eta_sec": None,
            "episodes": saved_history,
            "queue": [],
            "save_ready": False,
            "reason": "",
            **_rollout_intervention_save_status(runtime),
        }
    snapshot = logger_obj.status_snapshot()
    snapshot["enabled"] = True
    snapshot["dataset_dir"] = dataset_dir
    snapshot["episodes"] = saved_history
    snapshot["completed_episodes"] = len(saved_history)
    snapshot["save_ready"] = bool(runtime.rollout_save_ready and logger_obj.active_frame_count > 0)
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
        async_save=True,
        save_queue_max=storage.save_queue_max,
        eval_mode=True,
        save_image_height=storage.get("image_height"),
        save_image_width=storage.get("image_width"),
    )


def start_episode(runtime: RuntimeState, session: SessionState) -> None:
    """Begin one episode = one inference run (status -> RUNNING)."""
    if runtime.episode_logger is not None:
        runtime.episode_logger.start_episode(task=format_task_label(session.selected_task))
        _stamp_scene_meta(runtime, runtime.episode_logger)


def end_episode(runtime: RuntimeState) -> None:
    """Close the current episode, flushing parquet + mp4 to the dataset."""
    if runtime.episode_logger is not None:
        runtime.episode_logger.end_episode()


def start_rollout_intervention(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> bool:
    """Enable teleop takeover for a paused normal rollout."""
    if runtime.rollout_intervention_active:
        return True
    if not config.rollout.intervention.enabled:
        return False
    if not runtime.transport.supports_collection():
        session.last_error = "Rollout teleop intervention recording is not available"
        return False
    pre_qpos = runtime.transport.get_latest_qpos()
    if pre_qpos is None:
        session.last_error = "Cannot start rollout intervention without joint feedback"
        return False
    runtime.transport.reset_hil_control()
    runtime.transport.start_collection()
    runtime.transport.clear_collection_backlog()
    if runtime.infer_strategy is not None:
        runtime.infer_strategy.reset()
    runtime.rollout_intervention_pre_qpos = np.asarray(pre_qpos, dtype=np.float32).copy()
    runtime.rollout_intervention_active_segment = RolloutInterventionSegment(
        segment_index=runtime.rollout_intervention_next_segment_index,
        start_policy_frame_index=session.step_index,
        pre_intervention_qpos=runtime.rollout_intervention_pre_qpos.copy(),
    )
    runtime.rollout_intervention_next_segment_index += 1
    runtime.rollout_intervention_active = True
    session.last_error = ""
    logger.info("Rollout teleop intervention started")
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
    runtime.transport.stop_collection()
    runtime.rollout_intervention_active = False
    logger.info("Rollout teleop intervention stopped")
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
    segment.frames.append(_copy_observation(frame))
    return True


def record_rollout_intervention_step(runtime: RuntimeState, session: SessionState) -> bool:
    """Record available teleop frames while normal rollout intervention is active."""
    if not runtime.rollout_intervention_active:
        return False
    segment = runtime.rollout_intervention_active_segment
    if segment is None or segment.invalid_reason:
        return False
    frame = runtime.transport.get_collection_frame()
    if frame is None:
        return False
    return _record_rollout_intervention_frame(runtime, session, frame)


def accept_rollout_intervention_segment(
    runtime: RuntimeState, session: SessionState
) -> bool:
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


def collect_start_teleop(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> bool:
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
    stop_collection_capture(runtime)
    runtime.transport.stop_collection()
    runtime.collection_teleop_active = False
    runtime.last_collection_timestamp = None
    if session.mode is SessionMode.COLLECT and session.status is not SessionStatus.RUNNING:
        session.status = SessionStatus.UNSET
    logger.info("Collection teleop stopped")
    _ = config


def _stamp_scene_meta(
    runtime: RuntimeState, episode_logger: EpisodeLogger | None = None
) -> None:
    """Attach simulator scene metadata to the current episode for offline scene rebuild.

    Only simulator nodes publish reconstruction metadata; real-robot transports omit
    it, so this is a no-op when the transport exposes no scene status.
    """
    logger_obj = runtime.episode_logger if episode_logger is None else episode_logger
    if logger_obj is None:
        return
    get_status = getattr(runtime.transport, "get_task_status", None)
    if get_status is None:
        return
    status = get_status()
    scene_fields = {
        key: status[key]
        for key in ("scene_index", "seed", "task_name", "robot_name", "split")
        if status.get(key) is not None
    }
    if status.get("scene_state") is not None:
        scene_fields["scene"] = status["scene_state"]
    if scene_fields:
        logger_obj.set_episode_meta(**scene_fields)


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
        if not collect_start_teleop(config, runtime, session):
            return False
        collection_min_capture_time = runtime.transport.clear_collection_backlog()
        runtime.episode_logger.start_episode(
            task=format_task_label(session.selected_collect_task),
            collection_min_capture_time=collection_min_capture_time,
        )
        _stamp_scene_meta(runtime)
        start_collection_capture(
            runtime,
            fps=config.inference_cfg.publish_rate,
            max_raw_snapshots_per_tick=COLLECT_STEP_MAX_RAW_SNAPSHOTS,
        )
    else:
        runtime.episode_logger.start_episode(task=format_task_label(session.selected_collect_task))
        _stamp_scene_meta(runtime)
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


def collect_stop(config: ConfigDict, runtime: RuntimeState, session: SessionState) -> None:
    """End the collection episode and leave teleop active at its current pose."""
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
            runtime.episode_logger.end_episode()
        session.status = SessionStatus.READY
    logger.info("Collection episode stopped; leaving hardware at current pose")


def advance_collection_tasks_after_save(
    config: ConfigDict, runtime: RuntimeState, session: SessionState
) -> None:
    """Advance the collection task after a configured number of saved trials.

    Save completion arrives from the logger's worker thread.  This function is
    intentionally called from the application loop so the session and web status
    are changed on their normal owner thread.
    """
    logger_obj = runtime.episode_logger
    if logger_obj is None:
        return
    drain = getattr(logger_obj, "drain_saved_collection_tasks", None)
    if drain is None:
        return
    saved_tasks = drain()
    if not saved_tasks:
        return

    trials_per_task = config.collection.get("trials_per_task", 0)
    if not isinstance(trials_per_task, int) or isinstance(trials_per_task, bool):
        return
    if trials_per_task <= 0:
        return
    tasks = list(config.collection.get("tasks") or ())
    if not tasks:
        return
    completed = getattr(logger_obj, "completed_collection_episodes", None)
    if completed is None:
        return

    for saved_task in saved_tasks:
        # Do not override a deliberate operator selection made while an older
        # episode was still flushing in the background.
        if session.selected_collect_task != saved_task:
            continue
        if completed(saved_task) < trials_per_task:
            continue
        try:
            task_index = tasks.index(saved_task)
        except ValueError:
            continue
        if task_index + 1 >= len(tasks):
            logger.info(
                "Collection target reached for final task %s (%d trials)",
                format_task_label(saved_task),
                trials_per_task,
            )
            continue
        next_task = tasks[task_index + 1]
        session.selected_collect_task = next_task
        session.last_error = ""
        logger.info(
            "Collection target reached for %s (%d trials); next task -> %s",
            format_task_label(saved_task),
            trials_per_task,
            format_task_label(next_task),
        )


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
    """The episode logger currently built (skips None)."""
    return [
        logger_obj
        for logger_obj in (runtime.episode_logger, runtime.rollout_episode_logger)
        if logger_obj is not None
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
    loggers = _active_loggers(runtime)
    if not loggers:
        return
    snapshot = runtime.transport.acquire_collection_raw()
    if snapshot is None:
        return
    state_qpos = runtime.transport.get_latest_qpos()
    for logger_obj in loggers:
        logger_obj.ingest_raw_episode_snapshot(snapshot, action, state_qpos)


__all__ = [
    "COLLECT_STEP_MAX_RAW_SNAPSHOTS",
    "state_to_eef",
    "_fill_record_eef",
    "build_policy_observation",
    "resolve_storage",
    "maybe_build_episode_logger",
    "rollout_save_log_dir",
    "maybe_build_rollout_episode_logger",
    "begin_rollout_save_episode",
    "mark_rollout_save_ready",
    "save_rollout_episode",
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
    "record_rollout_intervention_step",
    "accept_rollout_intervention_segment",
    "discard_rollout_intervention_segment",
    "rollback_rollout_intervention",
    "collect_start_teleop",
    "collect_stop_teleop",
    "collect_start",
    "collect_step",
    "collect_stop",
    "advance_collection_tasks_after_save",
    "collect_cancel",
    "_active_loggers",
    "record_executed_action",
]
