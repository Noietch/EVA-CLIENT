"""Stdlib HTTP server for the EVA unified console.

Drives the eva-client control mainline over HTTP by routing browser requests into
the existing command_queue (the same channel the legacy eval server uses). Adds
console-only routes for breakpoint single-step debugging, a live observation feed
(camera JPEGs + qpos), and the self-rendered 3D scene (per-mesh world transforms
from the robot's URDF, served via /api/scene + /api/meshes).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import gzip
import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from tqdm import tqdm

from core.app.handlers import (
    _resolve_runtime_path,
    eval_episode_dir,
    eval_model_name,
    manual_publish_rate,
)
from core.app.state import (
    RuntimeState,
    SessionMode,
    SessionState,
    format_task_label,
    resolve_inference_strategy_label,
)
from core.config import ConfigDict
from core.utils.lerobot import LeRobotDatasetIO
from transport.base import ObservationSource
from transport.dataset import DatasetTransport

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Transport counts as online only if a SUB message arrived within this window; the
# frontend scene poll drains the reader ~12 Hz, so a live stream refreshes it well
# inside the window while a dead publisher lets it lapse.
_TRANSPORT_ONLINE_WINDOW_SECONDS = 2.0

# Transports that talk to a real arm: they track receipt freshness, so "no message
# yet" means offline (not merely "configured"). debug/dataset are not in this set.
_LIVE_TRANSPORT_TYPES = {"zmq", "ros1", "ros2"}


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


@dataclasses.dataclass
class ConsoleContext:
    """Shared state for the console request handler."""

    config: ConfigDict
    runtime: RuntimeState
    session: SessionState
    obs_reader: ObservationSource | None = None  # independent source for visualization
    scene: object | None = None  # UrdfScene, lazily built (None if URDF unavailable)
    scene_cache_key: object | None = None  # last qpos signature fed to scene.transforms
    scene_cache: dict | None = None  # cached transforms payload for that signature
    scene_log_key: object | None = None
    scene_pbar: tqdm | None = None  # collection-replay playback progress bar
    active_tab: str = "debug"
    output_dir: str = ""  # results root; debug results land in <output_dir>/console/
    preview: Any = None  # EpisodePreview for RESULT-tab playback (lazily set)
    calibrate: Any = None  # calibrate_session.CalibrateSession | None (lazy on /start)
    config_path: Any = None  # pathlib.Path | None; source config path for SAVE + reload

    def ckpt_label(self, slot: int | None) -> str | None:
        """Anonymized operator label for a blind ckpt slot (0 -> "Model A"); None when slot None."""
        return None if slot is None else f"Model {chr(ord('A') + slot)}"

    def active_model_name(self) -> str:
        """Folder name of the active model's eval dataset (matches the recorder's rooting)."""
        return eval_model_name(self.config, self.runtime)


def _serialize_prompt_config(prompt: Any) -> dict:
    # Per-prompt block shared by the EVAL config and the RESULT browser: bilingual
    # text plus the prompt's own milestones, each a (id, label) tuple in config.
    milestones = prompt.get("milestones") or ()
    return {
        "prompt_en": prompt.prompt_en,
        "prompt_zh": prompt.get("prompt_zh"),
        "milestones": [{"id": mid, "label": label} for mid, label in milestones],
    }


def _serialize_eval(ctx: ConsoleContext) -> dict:
    # Eval block for the EVAL/RESULT tabs. Empty dict when the config carries no eval,
    # which the frontend reads as "disable those tabs". Each checkpoint exposes its
    # anonymized slot label (Model A/B/...) plus its real ckpt name for the MODEL picker.
    eval_cfg = ctx.config.eval
    if eval_cfg is None:
        return {}
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
                "name": ckpt.name,
            }
            for slot, ckpt in enumerate(eval_cfg.checkpoints)
        ],
    }


def _serialize_gripper_controls(ctx: ConsoleContext) -> list[dict[str, str]]:
    gripper_groups = [g for g in ctx.runtime.robot.actuator_groups if g.gripper_index is not None]
    if len(gripper_groups) == 0:
        return []
    if len(gripper_groups) == 1:
        return [{"side": "l", "label": "GRIPPER", "group": gripper_groups[0].name}]
    return [
        {"side": "l", "label": "LEFT", "group": gripper_groups[0].name},
        {"side": "r", "label": "RIGHT", "group": gripper_groups[1].name},
    ]


def _resolve_dataset_dir(dataset_dir: str) -> str:
    if not dataset_dir:
        return dataset_dir
    return str(_resolve_runtime_path(dataset_dir))


def _serialize_calibration(ctx: ConsoleContext) -> dict:
    """Wire-form of cfg.calibration.cameras keyed by camera name. Uncalibrated
    cameras (present in the observation schema but not in the calibration block)
    are included with ``calibrated: False`` so the frontend can list them with a
    Missing badge. K/dist/T_cam_link are converted to nested lists so json.dumps
    works. Consumed by the DEBUG CAMERAS panel and the 3D-stage frustum overlay.
    """
    config = ctx.runtime.active_config or ctx.config
    calib = getattr(config, "calibration", None) or {}
    cams = calib.get("cameras") if isinstance(calib, dict) else getattr(calib, "cameras", {})
    cams = cams or {}
    out: dict[str, dict] = {}
    for spec in ctx.runtime.robot.observation_schema.cameras:
        out[spec.name] = {"calibrated": False}
    for name, cc in cams.items():
        entry: dict = {"calibrated": True}
        K = getattr(cc, "K", None)
        dist = getattr(cc, "dist", None)
        attach_link = getattr(cc, "attach_link", "")
        T_cam_link = getattr(cc, "T_cam_link", None)
        image_size = getattr(cc, "image_size", None)
        entry["K"] = K.tolist() if K is not None else None
        entry["dist"] = dist.tolist() if dist is not None else None
        entry["attach_link"] = str(attach_link or "")
        entry["T_cam_link"] = T_cam_link.tolist() if T_cam_link is not None else None
        entry["image_size"] = list(image_size) if image_size else None
        out[name] = entry
    return out


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
    return {
        "robot_type": config.robot.type,
        "transport_type": config.transport.type,
        "policy": {
            "type": config.policy.type,
            "host": config.policy.host,
            "port": config.policy.port,
        },
        "tasks": list(config.inference_cfg.debug_tasks),
        "collect_tasks": list(config.collection.tasks),
        "is_replay": is_replay,
        "manual_capable": config.transport.type in _LIVE_TRANSPORT_TYPES,
        "n_episodes": n_episodes,
        "modes": ["real", "sim", "step", "manual"],
        "camera_keys": [
            cam.observation_key for cam in ctx.runtime.robot.observation_schema.cameras
        ],
        "strategies": strategies,
        "gripper_controls": _serialize_gripper_controls(ctx),
        "obs_mode": config.inference_cfg.obs_space.type,
        "policy_action_mode": config.inference_cfg.action_space.type,
        "publish_mode": "joint",
        "publish_rate": config.inference_cfg.publish_rate,
        "manual_publish_rate": manual_publish_rate(config),
        "inference_rate": config.inference_cfg.inference_rate,
        "collection": {
            "enabled": bool(config.collection.schema.columns),
            "fps": None,
            "dataset_dir": _resolve_dataset_dir(
                config.collection.storage.log_dir if config.collection.schema.columns else ""
            ),
        },
        "eval": _serialize_eval(ctx),
        "run_id": Path(ctx.output_dir).name,
    }


def _serialize_status(ctx: ConsoleContext) -> dict:
    # Live snapshot polled by the frontend (~1 Hz). Mirrors the session/runtime
    # state machine so the UI can gate buttons and show telemetry.
    s = ctx.session
    r = ctx.runtime
    config = r.active_config or ctx.config
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
    since_recv = reader.seconds_since_last_recv()
    if since_recv is not None:
        transport_connected = since_recv <= _TRANSPORT_ONLINE_WINDOW_SECONDS
    elif config.transport.type in _LIVE_TRANSPORT_TYPES:
        transport_connected = False
    else:
        transport_connected = bool(config.transport.type)
    return {
        "robot_type": config.robot.type,
        "transport_type": config.transport.type,
        "transport_connected": transport_connected,
        "policy_connected": r.policy is not None,
        "policy_error": r.last_policy_error,
        "cli_mode": s.mode.value,
        "session_status": s.status.value,
        "selected_task": s.selected_task,
        "selected_collect_task": s.selected_collect_task,
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
        "collect": (
            r.episode_logger.status_snapshot(format_task_label(s.selected_collect_task))
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


def _compute_T_gripper_base(
    ctx: "ConsoleContext", attach_link: str
) -> np.ndarray | None:
    """Compute the current gripper->base transform for hand-eye calibration.

    Reads the transport's latest joint positions, runs FK via the robot's built-
    in kinematics solver, and returns a 4x4 SE(3) matrix (identity fallback with
    None when qpos or solver are unavailable). ``attach_link`` selects which arm
    frame to report — currently unused because the solver returns a single-arm
    frame; kept as a signature hook for multi-arm robots.
    """
    del attach_link
    transport = ctx.runtime.transport
    qpos = getattr(transport, "get_latest_qpos", lambda: None)()
    if qpos is None:
        return None
    try:
        kin = ctx.runtime.robot.build_kinematics(
            initial_qpos_groups=ctx.runtime.robot.initial_qpos_by_group()
        )
    except Exception as exc:
        logger.warning("kinematics unavailable for hand-eye T_gripper_base: %s", exc)
        return None
    if kin is None:
        return None
    fk = getattr(kin, "fk_arm", None) or getattr(kin, "fk_chunk", None)
    if fk is None:
        return None
    try:
        pose = fk(np.asarray(qpos, dtype=np.float32))
    except Exception as exc:
        logger.debug("FK failed for T_gripper_base: %s", exc)
        return None
    arr = np.asarray(pose)
    if arr.shape[-2:] != (4, 4):
        return None
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float64)


def _apply_calibrate_overlay(
    ctx: ConsoleContext, key: str, image: np.ndarray
) -> tuple[np.ndarray, int]:
    """Draw ChArUco detection on ``image`` when the calibrate session is live.

    Returns ``(possibly_annotated_image, overlay_sig)``. When no session is
    active or the camera isn't participating, returns the original image and 0.
    The signature is bumped each time a fresh detection is recorded so the
    MJPEG cache re-encodes; the sensor buffer itself is never mutated.
    """
    session = getattr(ctx, "calibrate", None)
    if session is None or not getattr(session, "active", False):
        return image, 0
    if key not in session.camera_keys:
        return image, 0
    try:
        from tools.calibration import detect_board, draw_overlay
    except Exception as exc:
        logger.warning("calibrate overlay disabled: %s", exc)
        return image, 0
    arr_bgr = np.asarray(image)
    if arr_bgr.ndim == 3 and arr_bgr.shape[2] == 3 and ctx.config.transport.convert_bgr_to_rgb:
        arr_bgr = cv2.cvtColor(arr_bgr, cv2.COLOR_RGB2BGR)
    try:
        detection = detect_board(arr_bgr, session.board)
    except Exception as exc:
        logger.debug("charuco detection failed: %s", exc)
        return image, 0
    session.detections[key] = detection
    session.overlay_tick += 1
    if not detection.detected:
        return image, session.overlay_tick
    copy = arr_bgr.copy()
    intrinsic = session.intrinsics.get(key)
    K = intrinsic.K if intrinsic is not None else None
    dist = intrinsic.dist if intrinsic is not None else None
    draw_overlay(copy, detection, K=K, dist=dist, board_spec=session.board)
    if ctx.config.transport.convert_bgr_to_rgb:
        copy = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
    return copy, session.overlay_tick


def _list_camera_keys(ctx: ConsoleContext) -> list[str]:
    # Cameras the visualization stream may expose, in schema order minus disabled ones.
    disabled = set(ctx.config.transport.disabled_cameras)
    reader = _observation_reader(ctx)
    available = getattr(reader, "available_camera_keys", None)
    if available is not None:
        return [key for key in available() if key not in disabled]
    return [
        cam.observation_key
        for cam in ctx.runtime.robot.observation_schema.cameras
        if cam.observation_key not in disabled
    ]


def _live_camera_keys(ctx: ConsoleContext) -> list[str]:
    reader = _observation_reader(ctx)
    get_keys = getattr(reader, "get_camera_keys", None)
    if get_keys is not None:
        return list(get_keys())
    return _list_camera_keys(ctx)


def _uses_replay_observation(ctx: ConsoleContext) -> bool:
    return ctx.active_tab == "replay" and ctx.runtime.replay_source is not None


def _observation_reader(ctx: ConsoleContext) -> ObservationSource:
    if _uses_replay_observation(ctx):
        return ctx.runtime.replay_source  # type: ignore[return-value]
    return ctx.obs_reader or ctx.runtime.transport


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
    return out


def _serialize_scene(ctx: ConsoleContext) -> dict:
    # Per-frame 3D state: dual-arm per-mesh world transforms from the latest qpos.
    # Cache by qpos signature so a static arm doesn't re-run forward kinematics each poll.
    if ctx.scene is None:
        return {"available": False, "arms": {}}
    reader = _observation_reader(ctx)
    session = ctx.session
    collection_qpos, collection_frame_index = _active_collection_replay_qpos(ctx.runtime)
    if collection_qpos is not None:
        qpos = collection_qpos
        source = f"collection_replay:{collection_frame_index}"
    # MANUAL mode: the solid arm follows the hand-set command qpos. Bypass the cache
    # since manual_qpos moves freely with the sliders.
    elif session.mode is SessionMode.MANUAL:
        return {
            "available": True,
            "arms": ctx.scene.transforms(session.manual_qpos),  # type: ignore[attr-defined]
        }
    elif session.sim_preview_qpos is not None:
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
        qpos = episode_logger.load_collection_replay_qpos(episode_index, task)
        if qpos is None or len(qpos) == 0:
            self._send_json(404, {"ok": False, "error": "collection episode unavailable"})
            return
        runtime.collection_replay_qpos = qpos
        runtime.collection_replay_episode = episode_index
        runtime.collection_replay_started = time.monotonic()
        replay_fps = episode_logger.load_collection_episode_fps(episode_index, task)
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

    def _start_episode_review(self, body: dict) -> None:
        runtime = self.ctx.runtime
        config = runtime.active_config or self.ctx.config
        dataset_dir = _resolve_dataset_dir(str(body.get("dataset_dir", "")).strip())
        episode_index = int(body.get("episode", 0))
        try:
            inferred = LeRobotDatasetIO(dataset_dir).infer_keys()
            state_key = (inferred.get("state") or {}).get("default") or "observations.state.qpos"
            action_key = (inferred.get("action") or {}).get("default") or "action"
            image = inferred.get("image") or {}
            candidates = image.get("candidates") or []
            video_keys: dict[str, str] = {}
            for camera in runtime.robot.observation_schema.cameras:
                cam_key = camera.observation_key
                match = _match_video_key(candidates, cam_key)
                if match is not None:
                    video_keys[cam_key] = match
            if not video_keys and image.get("default"):
                video_keys[runtime.robot.observation_schema.cameras[0].observation_key] = str(
                    image["default"]
                )
            import copy
            review_config = copy.deepcopy(config)
            review_config.transport.dataset_dir = dataset_dir
            review_config.transport.episode_id = episode_index
            review_config.transport.dataset_keys = ConfigDict(
                state_key=state_key,
                action_key=action_key,
                video_keys=video_keys,
            )
            source = DatasetTransport(review_config, runtime.robot, dataset_dir, episode_index)
            qpos = np.asarray(
                [source.get_scene_qpos(i) for i in range(source.n_steps)], dtype=np.float32
            )
            fps = source.fps
            task = source.current_task
            source.close()
        except Exception as error:
            self._send_json(404, {"ok": False, "error": str(error)})
            return
        if len(qpos) == 0:
            self._send_json(404, {"ok": False, "error": "episode has no frames"})
            return
        runtime.collection_replay_qpos = qpos
        runtime.collection_replay_episode = episode_index
        runtime.collection_replay_started = 0.0
        runtime.collection_replay_fps = max(
            1, int(round(fps or self.ctx.config.inference_cfg.publish_rate))
        )
        self._invalidate_scene_cache()
        self._send_json(
            200,
            {
                "ok": True,
                "episode": episode_index,
                "frames": int(qpos.shape[0]),
                "fps": runtime.collection_replay_fps,
                "task": task,
                "video_keys": video_keys,
            },
        )

    def _start_review_replay(self, body: dict) -> None:
        runtime = self.ctx.runtime
        episode_index = int(body.get("episode", runtime.collection_replay_episode or 0))
        if (
            runtime.collection_replay_qpos is None
            or runtime.collection_replay_episode != episode_index
        ):
            self._send_json(404, {"ok": False, "error": "review episode unavailable"})
            return
        runtime.collection_replay_started = time.monotonic()
        self._invalidate_scene_cache()
        self._send_json(200, {"ok": True, "episode": episode_index})

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
        self._sync_preview_to_active_model()
        ep = None
        if self.ctx.preview is not None:
            ep = self.ctx.preview.episode_index_by_clip().get(clip_id)
        if ep is None or runtime.episode_logger is None:
            self._send_json(
                409,
                {"ok": False, "error": "no recorded episode for this trial yet — run it first"},
            )
            return
        runtime.episode_logger.patch_episode_meta(
            int(ep),
            prompt=body.get("prompt"),
            trial=body.get("trial"),
            score=body.get("score"),
            max_score=body.get("max_score"),
            milestones=body.get("milestones", {}),
            note=body.get("note", ""),
            scored_at=_dt.datetime.now().isoformat(timespec="seconds"),
            duration_ms=body.get("duration_ms", 0),
        )
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
        period = 1.0 / 30.0
        # The profiled bottleneck is bandwidth, not CPU: pushing 30 fps of a 720p +
        # a 480p JPEG is ~17 MB/s, and a paused/replay view re-sends one *static*
        # frame forever. Encode + transmit only when the pixels actually change
        # (cheap subsampled signature), with a slow heartbeat so idle proxies and
        # late-joining <img> tags still get a frame. Live cameras keep streaming at
        # full rate because sensor noise changes the signature every frame.
        heartbeat = 2.0
        last_sig = None
        last_jpeg: bytes | None = None
        last_sent = 0.0
        try:
            while not ctx.runtime.transport.is_shutdown():
                tick = time.monotonic()
                # Re-resolve per tick so mounting/clearing a replay source mid-stream
                # swaps the feed without the client reopening the <img> connection.
                reader = _observation_reader(ctx)
                get_one = getattr(reader, "get_camera_frame", None)
                if get_one is not None:
                    image = get_one(key)
                else:
                    frame = reader.get_frame()
                    image = None if frame is None else frame.images.get(key)
                jpeg = None
                if image is not None:
                    image, overlay_sig = _apply_calibrate_overlay(ctx, key, image)
                    arr = np.asarray(image)
                    sig = (arr.shape, int(arr[::32, ::32].sum(dtype=np.int64)), overlay_sig)
                    if sig != last_sig:
                        jpeg = _encode_jpeg(image, convert)
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

    # --- routing ---

    def do_GET(self) -> None:
        """Route a GET to a stream, static asset, mesh, or registered JSON handler."""
        path = urlparse(self.path).path
        # Prefix routes (streams + static assets) come first; they own a path subtree.
        if path in {"/", "/index.html"}:
            self._send_static(STATIC_DIR / "index.html")
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

    def _get_calibration(self) -> None:
        self._send_json(200, _serialize_calibration(self.ctx))

    def _get_status(self) -> None:
        # This poll doubles as the client-liveness heartbeat the main loop watches.
        self.ctx.runtime.last_client_poll = time.monotonic()
        self._send_json(200, _serialize_status(self.ctx))

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
        # Each model's dataset lives at <output_dir>/<model>/episodes; scan every subdir
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
            rows = read_result_rows(root / sanitize_path_component(name) / "episodes")
            models.append({"model_name": name, "records": rows})
        # Pick up any other recorded model dirs (e.g. a renamed ckpt from a prior run).
        if root.exists():
            for sub in sorted(p for p in root.iterdir() if p.is_dir()):
                if sub.name in seen or sub.name == "console":
                    continue
                rows = read_result_rows(sub / "episodes")
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
        key = (id(scene), id(replay), int(getattr(replay, "episode_id", -1)), n)
        with type(self)._xf_lock:
            cache = type(self)._xf_cache
            entry = cache.get(key)
            if entry is None:
                seq = np.stack(
                    [np.asarray(replay.get_scene_qpos(i), dtype=np.float32) for i in range(n)]  # type: ignore[attr-defined]
                )
                raw = scene.all_transforms_blob(seq)  # type: ignore[attr-defined]
                entry = (raw, gzip.compress(raw))
                # Bound the cache to the two most recent episodes.
                if len(cache) >= 2:
                    cache.pop(next(iter(cache)))
                cache[key] = entry
        use_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        body = entry[1] if use_gzip else entry[0]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

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
        raw = preview.transforms_blob(ep) if (preview and ep >= 0) else None
        if raw is None:
            self._send_json(200, {"available": False})
            return
        use_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        body = gzip.compress(raw) if use_gzip else raw
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

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
            self.send_header("Cache-Control", "no-store")
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

    def do_POST(self) -> None:
        """Route a POST to a constant-command enqueue or a body-driven handler."""
        path = urlparse(self.path).path
        body = self._read_json_body()
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

    def _enqueue_ok(self, command: str) -> None:
        self._enqueue(command)
        self._send_json(200, {"ok": True})

    # --- POST route handlers (body-derived command arg / payload / mutation) ---

    def _post_select_task(self, body: dict) -> None:
        self._enqueue_ok(f"web:switch_task:{body.get('task', '')}")

    def _post_select_collect_task(self, body: dict) -> None:
        self.ctx.session.selected_collect_task = str(body.get("task", ""))
        self._send_json(200, {"ok": True})

    def _post_select_episode(self, body: dict) -> None:
        self._enqueue_ok(f"web:select_episode:{int(body.get('episode', 0))}")

    def _post_tab_switch(self, body: dict) -> None:
        self._clear_collection_replay()
        tab = str(body.get("tab", "") or "debug")
        armed = tab == "collect" and bool(body.get("collect_teleop_armed", False))
        self.ctx.active_tab = tab
        self.ctx.runtime.collection_teleop_armed = armed
        self.ctx.session.last_error = ""
        command_tab = "collect" if armed else ("debug" if tab == "collect" else tab)
        self._enqueue_ok(f"web:tab_switch:{command_tab}")

    def _post_load_replay_dataset(self, body: dict) -> None:
        self._clear_collection_replay()
        self._enqueue_ok("web:load_replay_dataset:" + json.dumps(body))

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

    def _post_update_manual_params(self, body: dict) -> None:
        self._enqueue_ok("web:update_manual_params:" + json.dumps(body))

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

    def _post_collect_start(self, _body: dict) -> None:
        self._clear_collection_replay()
        if self.ctx.active_tab != "collect" or not self.ctx.runtime.collection_teleop_armed:
            self.ctx.session.last_error = "Collection teleop requires COLLECT activation"
            self._send_json(200, {"ok": False, "error": self.ctx.session.last_error})
            return
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

    def _post_calibration_reload(self, _body: dict) -> None:
        # PR#1 stub: re-serialize what's already loaded. Full hot-reload is PR#3.
        self._send_json(200, {"ok": True, "cameras": _serialize_calibration(self.ctx)})

    # --- CALIBRATE workspace (MANUAL tab) ---

    def _get_calibrate_session(self) -> None:
        from core.app.calibrate_session import serialize_session

        self._send_json(200, serialize_session(getattr(self.ctx, "calibrate", None)))

    def _post_calibrate_start(self, _body: dict) -> None:
        from core.app.calibrate_session import (
            new_calibrate_session,
            serialize_session,
        )
        from tools.calibration import CharucoBoardSpec

        # Use observation_key (the string the transport/MJPEG stream and /api/config expose),
        # not spec.name — the frontend, MJPEG stream and observation reader all key by it.
        keys = tuple(
            spec.observation_key for spec in self.ctx.runtime.robot.observation_schema.cameras
        )
        board = getattr(self.ctx.calibrate, "board", None) if self.ctx.calibrate is not None else None
        self.ctx.calibrate = new_calibrate_session(board or CharucoBoardSpec(), keys)
        for spec in self.ctx.runtime.robot.observation_schema.cameras:
            if spec.calibration is not None:
                self.ctx.calibrate.attach_links[spec.observation_key] = spec.calibration.attach_link
        # Seed pose list: config override wins over robot's built-in default.
        cfg_poses = list(self.ctx.config.get("calibration_poses") or [])
        if cfg_poses:
            self.ctx.calibrate.poses = [
                np.asarray(p, dtype=np.float64).reshape(-1).tolist() for p in cfg_poses
            ]
        else:
            robot_poses = self.ctx.runtime.robot.default_calibration_poses()
            self.ctx.calibrate.poses = [list(map(float, np.asarray(p).reshape(-1))) for p in robot_poses]
        self.ctx.calibrate.pose_capture_state = [False] * len(self.ctx.calibrate.poses)
        self.ctx.calibrate.current_pose_index = 0 if self.ctx.calibrate.poses else -1
        self._send_json(200, serialize_session(self.ctx.calibrate))

    def _post_calibrate_mode(self, body: dict) -> None:
        from core.app.calibrate_session import MODES, serialize_session

        session = self.ctx.calibrate
        if session is None:
            self._send_json(409, {"ok": False, "error": "session not started"})
            return
        mode = str(body.get("mode") or "sim").lower()
        if mode not in MODES:
            self._send_json(400, {"ok": False, "error": f"mode must be one of {MODES}"})
            return
        session.mode = mode
        self._send_json(200, serialize_session(session))

    def _post_calibrate_poses(self, body: dict) -> None:
        """Body = {op, index?, qpos?}. op ∈ {add, replace, delete, select}.

        add: append a new pose at end (uses qpos or current live robot qpos).
        replace: replace pose at index with qpos.
        delete: remove pose at index.
        select: set current_pose_index to index.
        """
        from core.app.calibrate_session import serialize_session

        session = self.ctx.calibrate
        if session is None:
            self._send_json(409, {"ok": False, "error": "session not started"})
            return
        op = str(body.get("op") or "")
        idx = int(body.get("index", -1))
        raw_qpos = body.get("qpos")
        if op == "add":
            if raw_qpos is None:
                transport = self.ctx.runtime.transport
                qpos = getattr(transport, "get_latest_qpos", lambda: None)()
                if qpos is None:
                    self._send_json(503, {"ok": False, "error": "no live qpos to add"})
                    return
                qpos = np.asarray(qpos, dtype=np.float32).reshape(-1).tolist()
            else:
                qpos = [float(x) for x in raw_qpos]
            session.poses.append(qpos)
            session.pose_capture_state.append(False)
        elif op == "replace":
            if idx < 0 or idx >= len(session.poses):
                self._send_json(400, {"ok": False, "error": "index out of range"})
                return
            if raw_qpos is None:
                self._send_json(400, {"ok": False, "error": "replace needs qpos"})
                return
            session.poses[idx] = [float(x) for x in raw_qpos]
            session.pose_capture_state[idx] = False
        elif op == "delete":
            if idx < 0 or idx >= len(session.poses):
                self._send_json(400, {"ok": False, "error": "index out of range"})
                return
            del session.poses[idx]
            del session.pose_capture_state[idx]
            if session.current_pose_index >= len(session.poses):
                session.current_pose_index = len(session.poses) - 1
        elif op == "select":
            if idx < -1 or idx >= len(session.poses):
                self._send_json(400, {"ok": False, "error": "index out of range"})
                return
            session.current_pose_index = idx
        else:
            self._send_json(400, {"ok": False, "error": f"unknown op '{op}'"})
            return
        self._send_json(200, serialize_session(session))

    def _post_calibrate_move_to(self, body: dict) -> None:
        """SIM: no-op (frontend renders the ghost). REAL: dispatch via manual_send."""
        from core.app.calibrate_session import serialize_session

        session = self.ctx.calibrate
        if session is None:
            self._send_json(409, {"ok": False, "error": "session not started"})
            return
        idx = int(body.get("index", session.current_pose_index))
        if idx < 0 or idx >= len(session.poses):
            self._send_json(400, {"ok": False, "error": "index out of range"})
            return
        session.current_pose_index = idx
        dispatched = False
        if session.mode == "real":
            qpos = session.poses[idx]
            try:
                self._enqueue(
                    "web:manual_send:" + json.dumps({"qpos": [float(x) for x in qpos]})
                )
                dispatched = True
            except Exception as exc:
                logger.warning("move_to real dispatch failed: %s", exc)
                session.last_error = str(exc)
        payload = serialize_session(session)
        payload["ok"] = True
        payload["dispatched"] = dispatched
        self._send_json(200, payload)

    def _post_calibrate_reset(self, _body: dict) -> None:
        from core.app.calibrate_session import serialize_session

        self.ctx.calibrate = None
        self._send_json(200, serialize_session(None))

    def _post_calibrate_board(self, body: dict) -> None:
        from core.app.calibrate_session import serialize_session
        from tools.calibration import CharucoBoardSpec

        session = self.ctx.calibrate
        if session is None:
            self._send_json(409, {"ok": False, "error": "session not started"})
            return
        board = CharucoBoardSpec(
            dict_name=str(body.get("dict") or session.board.dict_name),
            cols=int(body.get("cols") or session.board.cols),
            rows=int(body.get("rows") or session.board.rows),
            square_length=float(body.get("square") or session.board.square_length),
            marker_length=float(body.get("marker") or session.board.marker_length),
        )
        session.board = board
        self._send_json(200, serialize_session(session))

    def _post_calibrate_intrinsic_source(self, body: dict) -> None:
        from core.app.calibrate_session import INTRINSIC_SOURCES, serialize_session

        session = self.ctx.calibrate
        if session is None:
            self._send_json(409, {"ok": False, "error": "session not started"})
            return
        key = str(body.get("camera_key") or "")
        source = str(body.get("source") or "")
        if source not in INTRINSIC_SOURCES:
            self._send_json(400, {"ok": False, "error": f"source must be one of {INTRINSIC_SOURCES}"})
            return
        session.ensure_camera(key)
        session.intrinsic_source[key] = source
        # Also let the operator flip attach_link here so scene vs hand-eye can be chosen.
        if "attach_link" in body:
            session.attach_links[key] = str(body.get("attach_link") or "")
        self._send_json(200, serialize_session(session))

    def _post_calibrate_capture(self, body: dict) -> None:
        from core.app.calibrate_session import capture_sample, serialize_session

        session = self.ctx.calibrate
        if session is None:
            self._send_json(409, {"ok": False, "error": "session not started"})
            return
        # SIM mode blocks CAPTURE: user explicitly asked for SIM to be preview-only.
        if session.mode == "sim":
            self._send_json(409, {"ok": False, "error": "SIM mode — switch to REAL to capture"})
            return
        key = str(body.get("camera_key") or "")
        if key not in session.camera_keys:
            self._send_json(400, {"ok": False, "error": f"unknown camera '{key}'"})
            return
        reader = _observation_reader(self.ctx)
        image = None
        get_one = getattr(reader, "get_camera_frame", None)
        if get_one is not None:
            image = get_one(key)
        if image is None:
            frame = reader.get_frame()
            image = None if frame is None else frame.images.get(key)
        if image is None:
            self._send_json(503, {"ok": False, "error": f"no frame available for '{key}'"})
            return
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3 and self.ctx.config.transport.convert_bgr_to_rgb:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        # Compute T_gripper_base via FK of the current qpos. This is what feeds
        # cv2.calibrateHandEye downstream — without it we'd never build pose pairs.
        T_gripper_base = _compute_T_gripper_base(self.ctx, session.attach_links.get(key, ""))
        encode = lambda img: _encode_jpeg(img, convert_bgr_to_rgb=False) or b""
        ok, _sample, msg = capture_sample(session, key, arr, T_gripper_base, encode)
        # Mark the current pose as captured if the operator moved to a preset.
        if ok and 0 <= session.current_pose_index < len(session.pose_capture_state):
            session.pose_capture_state[session.current_pose_index] = True
        payload = serialize_session(session)
        payload["capture_ok"] = ok
        payload["capture_msg"] = msg
        self._send_json(200, payload)

    def _post_calibrate_solve(self, body: dict) -> None:
        from core.app.calibrate_session import serialize_session, solve_all

        session = self.ctx.calibrate
        if session is None:
            self._send_json(409, {"ok": False, "error": "session not started"})
            return
        method = str(body.get("method") or session.method or "TSAI").upper()
        session.method = method
        transport = self.ctx.runtime.transport

        def _sdk(key: str):
            fn = getattr(transport, "get_intrinsics", None)
            return None if fn is None else fn(key)

        solve_all(session, _sdk, method=method)
        self._send_json(200, serialize_session(session))

    def _post_calibrate_save(self, _body: dict) -> None:
        from core.app.calibrate_session import (
            compose_save_payload,
            rebuild_camera_specs,
            serialize_session,
        )
        from core.config import _coerce_calibration
        from tools.calibration import write_raiden_json

        session = self.ctx.calibrate
        if session is None:
            self._send_json(409, {"ok": False, "error": "session not started"})
            return
        entries = compose_save_payload(session, self.ctx.runtime.robot)
        if not entries:
            self._send_json(400, {"ok": False, "error": "no solved cameras to save"})
            return
        cfg_path = self.ctx.config_path or Path("./calibration_results.json")
        dest = Path(cfg_path).expanduser().parent / "calibration_results.json"
        write_raiden_json(dest, entries)
        session.saved_path = str(dest)
        # Refresh the config so downstream reads see the new values immediately.
        if self.ctx.config.get("calibration") is None:
            from core.cfg import ConfigDict as _CD

            self.ctx.config.calibration = _CD({"cameras": {}, "results_path": str(dest)})
        else:
            self.ctx.config.calibration.results_path = str(dest)
            self.ctx.config.calibration.cameras = {}
        _coerce_calibration(self.ctx.config, cfg_path)
        rebuild_camera_specs(self.ctx)
        payload = serialize_session(session)
        payload["ok"] = True
        payload["cameras"] = _serialize_calibration(self.ctx)
        self._send_json(200, payload)

    def _get_calibrate_print_board(self) -> None:
        from tools.calibration import CharucoBoardSpec, build_charuco_pdf
        from urllib.parse import parse_qs, urlparse

        board = getattr(getattr(self.ctx, "calibrate", None), "board", None) or CharucoBoardSpec()
        query = parse_qs(urlparse(self.path).query)
        paper = (query.get("paper") or ["a4"])[0]
        try:
            pdf = build_charuco_pdf(board, paper=str(paper))
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(pdf)))
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="charuco_{board.dict_name}_{board.cols}x{board.rows}.pdf"',
        )
        self.end_headers()
        self.wfile.write(pdf)

    def _post_calibrate_preview(self, body: dict) -> None:
        """Return the FK transforms for an arbitrary qpos so the frontend can
        render a ghost URDF at the target pose. Never touches live state.
        """
        raw = body.get("qpos") or []
        if not raw:
            self._send_json(400, {"ok": False, "error": "qpos required"})
            return
        qpos = np.asarray(raw, dtype=np.float32).reshape(-1)
        if self.ctx.scene is None:
            self._send_json(503, {"ok": False, "error": "URDF unavailable"})
            return
        try:
            transforms = self.ctx.scene.transforms(qpos)  # type: ignore[attr-defined]
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"FK failed: {exc}"})
            return
        self._send_json(200, {"available": True, "arms": transforms})


# Exact-path GET routes (prefix routes — camera/assets/meshes — are handled inline in
# do_GET because they own a path subtree). Values are unbound handler methods.
_GET_ROUTES = {
    "/api/config": ConsoleRequestHandler._get_config,
    "/api/calibration": ConsoleRequestHandler._get_calibration,
    "/api/calibrate/session": ConsoleRequestHandler._get_calibrate_session,
    "/api/calibrate/print_board": ConsoleRequestHandler._get_calibrate_print_board,
    "/api/status": ConsoleRequestHandler._get_status,
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
    "/api/replay_video": ConsoleRequestHandler._get_replay_video,
    "/api/live_series": ConsoleRequestHandler._get_live_series,
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
    "/api/collect_stop": "web:collect_stop",
    "/api/collect_cancel": "web:collect_cancel",
    "/api/warmup": "web:warmup",
    "/api/eval_stop": "web:stop",
    "/api/eval_reset": "web:reset",
    "/api/init_done": "web:init_done",
}

# POST endpoints needing a body-derived command arg, a runtime mutation, or a custom
# response. Each handler owns its own _send_json. Values are unbound (self, body) methods.
_POST_ROUTES = {
    "/api/collect_replay": ConsoleRequestHandler._start_collection_replay,
    "/api/exit_collect_replay": ConsoleRequestHandler._post_exit_collection_replay,
    "/api/review_episode": ConsoleRequestHandler._start_episode_review,
    "/api/review_replay_start": ConsoleRequestHandler._start_review_replay,
    "/api/select_task": ConsoleRequestHandler._post_select_task,
    "/api/select_collect_task": ConsoleRequestHandler._post_select_collect_task,
    "/api/select_episode": ConsoleRequestHandler._post_select_episode,
    "/api/tab_switch": ConsoleRequestHandler._post_tab_switch,
    "/api/inspect_dataset": ConsoleRequestHandler._post_inspect_dataset,
    "/api/qc_mark": ConsoleRequestHandler._post_qc_mark,
    "/api/annotate": ConsoleRequestHandler._post_annotate,
    "/api/episode_annotation": ConsoleRequestHandler._post_episode_annotation,
    "/api/load_replay_dataset": ConsoleRequestHandler._post_load_replay_dataset,
    "/api/set_replay_fps": ConsoleRequestHandler._post_set_replay_fps,
    "/api/replay_seek": ConsoleRequestHandler._post_replay_seek,
    "/api/select_mode": ConsoleRequestHandler._post_select_mode,
    "/api/select_strategy": ConsoleRequestHandler._post_select_strategy,
    "/api/update_infer_params": ConsoleRequestHandler._post_update_infer_params,
    "/api/update_manual_params": ConsoleRequestHandler._post_update_manual_params,
    "/api/manual_qpos": ConsoleRequestHandler._post_manual_qpos,
    "/api/manual_send": ConsoleRequestHandler._post_manual_send,
    "/api/manual_dispatch": ConsoleRequestHandler._post_manual_dispatch,
    "/api/gripper": ConsoleRequestHandler._post_gripper,
    "/api/collect_start": ConsoleRequestHandler._post_collect_start,
    "/api/select_ckpt": ConsoleRequestHandler._post_select_ckpt,
    "/api/init_move": ConsoleRequestHandler._post_init_move,
    "/api/init_gripper": ConsoleRequestHandler._post_init_gripper,
    "/api/calibration/reload": ConsoleRequestHandler._post_calibration_reload,
    "/api/calibrate/start": ConsoleRequestHandler._post_calibrate_start,
    "/api/calibrate/reset": ConsoleRequestHandler._post_calibrate_reset,
    "/api/calibrate/board": ConsoleRequestHandler._post_calibrate_board,
    "/api/calibrate/mode": ConsoleRequestHandler._post_calibrate_mode,
    "/api/calibrate/poses": ConsoleRequestHandler._post_calibrate_poses,
    "/api/calibrate/move_to": ConsoleRequestHandler._post_calibrate_move_to,
    "/api/calibrate/intrinsic_source": ConsoleRequestHandler._post_calibrate_intrinsic_source,
    "/api/calibrate/capture": ConsoleRequestHandler._post_calibrate_capture,
    "/api/calibrate/solve": ConsoleRequestHandler._post_calibrate_solve,
    "/api/calibrate/save": ConsoleRequestHandler._post_calibrate_save,
    "/api/calibrate/preview": ConsoleRequestHandler._post_calibrate_preview,
    "/api/eval_start": ConsoleRequestHandler._eval_start,
    "/api/score": ConsoleRequestHandler._eval_score,
}


def start_console_server(
    config: ConfigDict,
    runtime: RuntimeState,
    session: SessionState,
    port: int,
    host: str = "0.0.0.0",
    output_dir: str = "",
    config_path: Path | None = None,
) -> threading.Thread:
    """Launch a daemon HTTP server thread bound to ``host:port``."""
    scene: object | None = None
    try:
        from robots.utils import UrdfScene

        scene = UrdfScene(
            runtime.robot,
            gripper_open=config.robot.gripper_open,
            gripper_close=config.robot.gripper_close,
        )
    except Exception as exc:
        logger.warning("UrdfScene unavailable; 3D canvas disabled: %s", exc)
    ctx = ConsoleContext(
        config=config,
        runtime=runtime,
        session=session,
        obs_reader=runtime.transport.create_observation_reader(),
        scene=scene,
        output_dir=output_dir,
        config_path=config_path,
    )
    # RESULT-tab playback reads recorded episodes under <output_dir>/episodes.
    if output_dir:
        from core.app.console.episode_preview import EpisodePreview

        ctx.preview = EpisodePreview(config, Path(output_dir))
    ConsoleRequestHandler.ctx = ctx
    server = ThreadingHTTPServer((host, port), ConsoleRequestHandler)

    def serve() -> None:
        """Thread body: serve requests until the server is shut down."""
        display_host = "127.0.0.1" if host == "0.0.0.0" else host
        logger.info("Console web server: http://%s:%d", display_host, port)
        server.serve_forever()

    thread = threading.Thread(target=serve, name="eva-console-server", daemon=True)
    thread.start()
    return thread
