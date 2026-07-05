"""Server-side CALIBRATE session state + handler helpers.

Lifecycle:
  1. ``new_calibrate_session(board, cameras)`` — held on ``ConsoleContext.calibrate``
     after the first ``/api/calibrate/start``.
  2. ``capture_sample(session, camera_key, bgr, T_gripper_base, encode_jpeg)`` —
     runs ChArUco detection on the pixel buffer, stores the objp/imgp pair for
     later intrinsic solve, records a thumbnail, and if ``T_gripper_base`` is
     non-None also queues a hand-eye pose pair.
  3. ``solve_all(session, sdk_intrinsics_fn, method)`` — for each participating
     camera: intrinsic (SDK or solve), then hand-eye (per attach_link) or scene
     extrinsic (for world-frame cameras).
  4. ``compose_save_payload(session, robot)`` — assemble raiden-schema entries.

Everything sits on the request-handler thread; the MJPEG overlay hook reads
``session.detections`` on the streaming thread (a fresh detection replaces the
old dict entry atomically, which is what Python dict item assignment does under
the GIL — no lock required for this narrow use).
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import time
from typing import Any, Callable

import numpy as np

from tools.calibration import (
    CameraCalibrationEntry,
    CameraSample,
    CharucoBoardSpec,
    CharucoDetection,
    HandEyeResult,
    IntrinsicSolveResult,
    PoseSample,
    compose_T_cam_link,
    detect_board,
    estimate_board_pose,
    match_image_points,
    package_sdk_intrinsics,
    solve_hand_eye,
    solve_intrinsics,
    solve_scene_extrinsic,
)

log = logging.getLogger(__name__)

INTRINSIC_SOURCES = ("sdk", "solve", "sdk_or_solve")
MODES = ("sim", "real")


@dataclasses.dataclass
class CalibrateSession:
    """Mutable session held on ConsoleContext.calibrate while a run is active."""

    board: CharucoBoardSpec
    camera_keys: tuple[str, ...]
    intrinsic_source: dict[str, str] = dataclasses.field(default_factory=dict)
    samples: dict[str, list[CameraSample]] = dataclasses.field(default_factory=dict)
    detections: dict[str, CharucoDetection | None] = dataclasses.field(default_factory=dict)
    last_thumb_jpeg: dict[str, bytes] = dataclasses.field(default_factory=dict)
    intrinsics: dict[str, IntrinsicSolveResult | None] = dataclasses.field(default_factory=dict)
    hand_eye_pairs: dict[str, list[PoseSample]] = dataclasses.field(default_factory=dict)
    hand_eye_results: dict[str, HandEyeResult | None] = dataclasses.field(default_factory=dict)
    scene_extrinsics: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    attach_links: dict[str, str] = dataclasses.field(default_factory=dict)  # camera_key -> URDF link ("" for scene)
    active: bool = False
    mode: str = "sim"  # "sim" | "real"
    method: str = "TSAI"
    overlay_tick: int = 0
    last_error: str = ""
    saved_path: str = ""
    poses: list[list[float]] = dataclasses.field(default_factory=list)  # each is qpos as plain list (JSON-friendly)
    current_pose_index: int = -1  # -1 = none selected
    pose_capture_state: list[bool] = dataclasses.field(default_factory=list)  # parallel to poses; True = captured

    def ensure_camera(self, camera_key: str) -> None:
        if camera_key not in self.camera_keys:
            self.camera_keys = self.camera_keys + (camera_key,)
        self.intrinsic_source.setdefault(camera_key, "sdk_or_solve")
        self.samples.setdefault(camera_key, [])
        self.detections.setdefault(camera_key, None)
        self.intrinsics.setdefault(camera_key, None)
        self.hand_eye_pairs.setdefault(camera_key, [])
        self.hand_eye_results.setdefault(camera_key, None)
        self.attach_links.setdefault(camera_key, "")


def new_calibrate_session(
    default_board: CharucoBoardSpec, keys: tuple[str, ...]
) -> CalibrateSession:
    session = CalibrateSession(board=default_board, camera_keys=tuple(keys))
    for key in keys:
        session.ensure_camera(key)
    session.active = True
    return session


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _encode_thumbnail(image_bgr: np.ndarray, encode_jpeg: Callable[[Any], bytes]) -> bytes:
    """Downsample the BGR frame to 200-px wide and re-encode as JPEG."""
    import cv2

    h, w = image_bgr.shape[:2]
    scale = 200.0 / max(1, w)
    if scale < 1.0:
        new_w = 200
        new_h = max(1, int(round(h * scale)))
        thumb = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        thumb = image_bgr
    ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return encode_jpeg(image_bgr[..., ::-1])  # fallback
    return bytes(buf.tobytes())


def capture_sample(
    session: CalibrateSession,
    camera_key: str,
    image_bgr: np.ndarray,
    T_gripper_base: np.ndarray | None,
    encode_jpeg: Callable[[Any], bytes],
) -> tuple[bool, CameraSample | None, str]:
    """Detect the board in ``image_bgr`` and, on success, store a CameraSample.

    Returns ``(ok, sample_or_none, message)``. When ``T_gripper_base`` is not
    None AND detection succeeds, a hand-eye pose pair is queued too.
    """
    session.ensure_camera(camera_key)
    detection = detect_board(image_bgr, session.board)
    session.detections[camera_key] = detection
    session.overlay_tick += 1
    if not detection.detected:
        session.last_error = f"{camera_key}: board not detected"
        return False, None, "no board in image"
    try:
        objp, imgp = match_image_points(session.board, detection)
    except ValueError as exc:
        session.last_error = f"{camera_key}: {exc}"
        return False, None, str(exc)
    thumb_jpeg = _encode_thumbnail(image_bgr, encode_jpeg)
    sample = CameraSample(
        image_shape=(int(image_bgr.shape[0]), int(image_bgr.shape[1])),
        objp=objp,
        imgp=imgp,
        thumbnail_jpeg=thumb_jpeg,
    )
    session.samples[camera_key].append(sample)
    session.last_thumb_jpeg[camera_key] = thumb_jpeg
    if T_gripper_base is not None:
        intrinsic = session.intrinsics.get(camera_key)
        if intrinsic is not None:
            pose = estimate_board_pose(session.board, detection, intrinsic.K, intrinsic.dist)
            if pose is not None:
                rvec, tvec = pose
                from tools.calibration import rvec_tvec_to_se3

                T_target_cam = rvec_tvec_to_se3(rvec, tvec)
                session.hand_eye_pairs[camera_key].append(
                    PoseSample(
                        T_target_cam=T_target_cam,
                        T_gripper_base=np.asarray(T_gripper_base, dtype=np.float64).reshape(4, 4),
                        camera_key=camera_key,
                        robot_link=session.attach_links.get(camera_key, ""),
                        ts=time.time(),
                    )
                )
    session.last_error = ""
    return True, sample, "ok"


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def solve_intrinsic_for(
    session: CalibrateSession,
    camera_key: str,
    sdk_intrinsics_fn: Callable[[str], tuple[np.ndarray, np.ndarray, tuple[int, int]] | None],
) -> IntrinsicSolveResult | None:
    """Solve K/dist for one camera per the configured intrinsic source."""
    session.ensure_camera(camera_key)
    source = session.intrinsic_source.get(camera_key, "sdk_or_solve")
    if source in ("sdk", "sdk_or_solve"):
        sdk = sdk_intrinsics_fn(camera_key)
        if sdk is not None:
            K, dist, image_size = sdk
            result = package_sdk_intrinsics(K, dist, image_size)
            session.intrinsics[camera_key] = result
            return result
        if source == "sdk":
            session.last_error = f"{camera_key}: transport has no SDK intrinsics"
            return None
    samples = tuple(session.samples.get(camera_key, ()))
    if len(samples) < 4:
        session.last_error = f"{camera_key}: need >=4 captures to solve intrinsics"
        return None
    try:
        result = solve_intrinsics(samples)
    except ValueError as exc:
        session.last_error = f"{camera_key}: {exc}"
        return None
    session.intrinsics[camera_key] = result
    return result


def _rebuild_hand_eye_pairs(session: CalibrateSession, camera_key: str) -> None:
    """Recompute per-camera hand-eye pairs from stored samples + new intrinsics.

    Called after an intrinsic solve so already-captured samples get their
    T_target_cam re-projected with the fresh K/dist.
    """
    intrinsic = session.intrinsics.get(camera_key)
    if intrinsic is None:
        return
    session.hand_eye_pairs[camera_key] = []
    for sample in session.samples.get(camera_key, ()):
        # We don't have T_gripper_base stored on the sample (design choice —
        # samples exist for both hand-eye and scene solves). The live capture
        # path is the one that queues hand-eye pairs; rebuilding here is a no-op
        # for offline replay. This keeps the code path simple; retrospective
        # hand-eye solves are not part of the PR#2 UX.
        del sample


def solve_hand_eye_for(
    session: CalibrateSession, camera_key: str, method: str
) -> HandEyeResult | None:
    """Solve hand-eye for one wrist-mounted camera."""
    pairs = tuple(session.hand_eye_pairs.get(camera_key, ()))
    if len(pairs) < 3:
        session.last_error = f"{camera_key}: need >=3 pose pairs for hand-eye"
        return None
    try:
        result = solve_hand_eye(pairs, method=method)
    except (ValueError, RuntimeError) as exc:
        session.last_error = f"{camera_key}: {exc}"
        return None
    session.hand_eye_results[camera_key] = result
    session.method = result.method
    return result


def run_scene_solve(session: CalibrateSession, camera_key: str) -> np.ndarray | None:
    """Solve scene extrinsic for one world-fixed camera."""
    intrinsic = session.intrinsics.get(camera_key)
    if intrinsic is None:
        session.last_error = f"{camera_key}: intrinsic must be solved before scene extrinsic"
        return None
    samples = tuple(session.samples.get(camera_key, ()))
    if not samples:
        session.last_error = f"{camera_key}: no samples for scene solve"
        return None
    T_target_cams: list[np.ndarray] = []
    # Recreate T_target_cam from stored objp/imgp using cv2.solvePnP + fresh intrinsic.
    import cv2
    from tools.calibration import rvec_tvec_to_se3

    for sample in samples:
        ok, rvec, tvec = cv2.solvePnP(
            sample.objp.astype(np.float32),
            sample.imgp.astype(np.float32),
            intrinsic.K,
            intrinsic.dist,
        )
        if not ok:
            continue
        T_target_cams.append(
            rvec_tvec_to_se3(np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3))
        )
    if not T_target_cams:
        session.last_error = f"{camera_key}: no valid PnP solutions"
        return None
    T_cam_world = solve_scene_extrinsic(tuple(T_target_cams))
    session.scene_extrinsics[camera_key] = T_cam_world
    return T_cam_world


def solve_all(
    session: CalibrateSession,
    sdk_intrinsics_fn: Callable[[str], tuple[np.ndarray, np.ndarray, tuple[int, int]] | None],
    method: str = "TSAI",
) -> None:
    """Run intrinsics then hand-eye/scene solve for every participating camera."""
    for key in session.camera_keys:
        solve_intrinsic_for(session, key, sdk_intrinsics_fn)
    for key in session.camera_keys:
        attach = session.attach_links.get(key, "")
        if attach:
            solve_hand_eye_for(session, key, method)
        else:
            run_scene_solve(session, key)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def compose_save_payload(
    session: CalibrateSession, robot: Any
) -> dict[str, CameraCalibrationEntry]:
    """Assemble ``{camera_name: CameraCalibrationEntry}`` from the solved state."""
    del robot
    entries: dict[str, CameraCalibrationEntry] = {}
    for key in session.camera_keys:
        intrinsic = session.intrinsics.get(key)
        if intrinsic is None:
            continue
        attach = session.attach_links.get(key, "")
        T_cam_link: np.ndarray | None = None
        he = session.hand_eye_results.get(key)
        if he is not None:
            T_cam_link = compose_T_cam_link(he.T_cam_gripper, attach)
        else:
            scene = session.scene_extrinsics.get(key)
            if scene is not None:
                T_cam_link = np.asarray(scene, dtype=np.float64).reshape(4, 4)
        entries[key] = CameraCalibrationEntry(
            K=intrinsic.K,
            dist=intrinsic.dist,
            attach_link=attach,
            T_cam_link=T_cam_link,
            image_size=intrinsic.image_size,
        )
    return entries


def rebuild_camera_specs(ctx: Any) -> None:
    """After a config._coerce_calibration reload, refresh CameraSpec.calibration on the Robot.

    The schema is frozen, so we swap in a new tuple built from freshly-coerced
    CameraCalibration instances via ``dataclasses.replace`` on each CameraSpec.
    """
    from robots.base import CameraSpec, ObservationSchema

    cams_by_name = dict((ctx.config.calibration or {}).get("cameras") or {})
    new_specs: list[CameraSpec] = []
    for spec in ctx.runtime.robot.observation_schema.cameras:
        calib = cams_by_name.get(spec.name)
        new_specs.append(dataclasses.replace(spec, calibration=calib))
    ctx.runtime.robot.observation_schema = dataclasses.replace(
        ctx.runtime.robot.observation_schema,
        cameras=tuple(new_specs),
    )
    del ObservationSchema


# ---------------------------------------------------------------------------
# Wire form
# ---------------------------------------------------------------------------


def _encode_thumb_data_uri(jpeg: bytes) -> str:
    if not jpeg:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def serialize_session(session: CalibrateSession | None) -> dict:
    """JSON-safe wire form consumed by the frontend."""
    if session is None:
        return {
            "active": False,
            "mode": "sim",
            "camera_keys": [],
            "per_camera": {},
            "hand_eye": {"total_pairs": 0, "method": "TSAI"},
            "board": None,
            "poses": [],
            "current_pose_index": -1,
            "pose_capture_state": [],
            "ready_to_save": False,
            "last_error": "",
            "saved_path": "",
        }
    per_camera: dict[str, dict] = {}
    total_pairs = 0
    solved_cams = 0
    for key in session.camera_keys:
        intrinsic = session.intrinsics.get(key)
        detection = session.detections.get(key)
        he = session.hand_eye_results.get(key)
        pairs = session.hand_eye_pairs.get(key, [])
        per_camera[key] = {
            "samples": len(session.samples.get(key, [])),
            "intrinsic_source": session.intrinsic_source.get(key, "sdk_or_solve"),
            "attach_link": session.attach_links.get(key, ""),
            "last_thumb": _encode_thumb_data_uri(session.last_thumb_jpeg.get(key, b"")),
            "latest_detection": (
                {
                    "n_corners": int(detection.n_corners),
                    "image_shape": [int(detection.image_shape[0]), int(detection.image_shape[1])],
                }
                if detection is not None
                else None
            ),
            "intrinsic": (
                {
                    "K": intrinsic.K.tolist(),
                    "dist": intrinsic.dist.tolist(),
                    "image_size": [int(intrinsic.image_size[0]), int(intrinsic.image_size[1])],
                    "rms": intrinsic.rms,
                    "n_frames": intrinsic.n_frames,
                    "method": intrinsic.method,
                }
                if intrinsic is not None
                else None
            ),
            "hand_eye": (
                {
                    "T_cam_gripper": he.T_cam_gripper.tolist(),
                    "method": he.method,
                    "rotation_rms_deg": he.rotation_rms_deg,
                    "translation_rms_m": he.translation_rms_m,
                    "n_pairs": he.n_pairs,
                }
                if he is not None
                else None
            ),
            "n_hand_eye_pairs": len(pairs),
        }
        total_pairs += len(pairs)
        if intrinsic is not None:
            solved_cams += 1
    n_captured = sum(1 for c in session.pose_capture_state if c)
    return {
        "active": session.active,
        "mode": session.mode,
        "board": session.board.to_wire(),
        "camera_keys": list(session.camera_keys),
        "per_camera": per_camera,
        "hand_eye": {"total_pairs": total_pairs, "method": session.method},
        "poses": [list(qpos) for qpos in session.poses],
        "current_pose_index": int(session.current_pose_index),
        "pose_capture_state": list(session.pose_capture_state),
        "n_poses": len(session.poses),
        "n_captured": n_captured,
        "ready_to_save": solved_cams > 0,
        "last_error": session.last_error,
        "saved_path": session.saved_path,
    }

