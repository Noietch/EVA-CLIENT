"""Camera calibration solver — self-contained ChArUco + hand-eye + scene-extrinsic.

Everything the console CALIBRATE tab depends on lives here: board spec, the
ChArUco detector, intrinsic + hand-eye + scene solvers, a tiny SE(3) helper set,
and the raiden-schema JSON writer. ``build_charuco_pdf`` also renders a
printable PDF of the board so the operator can drop it straight onto letterhead.

Clean-room reimplementation of the raiden calibration pipeline. Only the OpenCV
public API and the persisted JSON schema field names are reused.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import struct
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


DEFAULT_DICT_NAMES: tuple[str, ...] = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_7X7_250",
)


# ---------------------------------------------------------------------------
# SE(3) helpers — no scipy dependency, so this module works in pure-numpy envs.
# ---------------------------------------------------------------------------


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Return T^{-1} for a 4x4 rigid transform without a full matrix inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=T.dtype)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def compose_se3(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> 3-vector so(3) log (axis * angle). Safe near pi."""
    trace = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(trace))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-6:
        d = np.diag(R)
        i = int(np.argmax(d))
        axis = np.zeros(3, dtype=np.float64)
        axis[i] = np.sqrt(max((d[i] + 1.0) * 0.5, 0.0))
        for j in range(3):
            if j != i:
                axis[j] = R[i, j] / (2.0 * axis[i]) if axis[i] > 1e-8 else 0.0
        return axis * angle
    factor = angle / (2.0 * np.sin(angle))
    return factor * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
    )


def so3_exp(rvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = rvec / angle
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rvec_tvec_to_se3(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = so3_exp(np.asarray(rvec, dtype=np.float64).reshape(3))
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


# ---------------------------------------------------------------------------
# Board spec + OpenCV factory
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CharucoBoardSpec:
    """Geometry + dictionary of a printed ChArUco calibration board."""

    dict_name: str = "DICT_5X5_100"
    cols: int = 5
    rows: int = 7
    square_length: float = 0.03
    marker_length: float = 0.023

    def to_wire(self) -> dict[str, Any]:
        return {
            "dict": self.dict_name,
            "cols": self.cols,
            "rows": self.rows,
            "square": self.square_length,
            "marker": self.marker_length,
        }


def build_cv_board(spec: CharucoBoardSpec):
    """Build a ``cv2.aruco.CharucoBoard`` matching ``spec``. Raises on bad dict."""
    import cv2

    dict_id = getattr(cv2.aruco, spec.dict_name, None)
    if dict_id is None:
        raise ValueError(
            f"unknown ArUco dictionary '{spec.dict_name}'; try one of {DEFAULT_DICT_NAMES}"
        )
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.CharucoBoard(
        (spec.cols, spec.rows),
        spec.square_length,
        spec.marker_length,
        dictionary,
    )


# ---------------------------------------------------------------------------
# Detector + overlay
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CharucoDetection:
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    marker_corners: tuple | None
    marker_ids: np.ndarray | None
    n_corners: int
    image_shape: tuple[int, int]

    @property
    def detected(self) -> bool:
        return self.n_corners >= 4


def detect_board(image_bgr: np.ndarray, spec: CharucoBoardSpec) -> CharucoDetection:
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    shape = (int(image_bgr.shape[0]), int(image_bgr.shape[1]))
    board = build_cv_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    n = 0 if charuco_ids is None else int(len(charuco_ids))
    return CharucoDetection(
        charuco_corners=charuco_corners if n else None,
        charuco_ids=charuco_ids if n else None,
        marker_corners=tuple(marker_corners) if marker_corners is not None else None,
        marker_ids=marker_ids,
        n_corners=n,
        image_shape=shape,
    )


def match_image_points(
    spec: CharucoBoardSpec, detection: CharucoDetection
) -> tuple[np.ndarray, np.ndarray]:
    if not detection.detected:
        raise ValueError("detection has fewer than 4 corners")
    assert detection.charuco_corners is not None and detection.charuco_ids is not None
    board = build_cv_board(spec)
    # cv2 stubs annotate `detectedCorners` as `Sequence[MatLike]`, but the runtime
    # accepts a single ndarray. Casting silences pyright without changing behavior.
    from typing import Any as _Any

    corners_any: _Any = detection.charuco_corners
    objp, imgp = board.matchImagePoints(corners_any, detection.charuco_ids)
    if objp is None or imgp is None or len(objp) == 0:
        raise ValueError("matchImagePoints returned no points")
    return (
        np.asarray(objp, dtype=np.float64).reshape(-1, 3),
        np.asarray(imgp, dtype=np.float64).reshape(-1, 2),
    )


def estimate_board_pose(
    spec: CharucoBoardSpec,
    detection: CharucoDetection,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not detection.detected:
        return None
    import cv2

    objp, imgp = match_image_points(spec, detection)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, np.asarray(K), np.asarray(dist).reshape(-1))
    if not ok:
        return None
    return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)


def draw_overlay(
    image_bgr_copy: np.ndarray,
    detection: CharucoDetection,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
    board_spec: CharucoBoardSpec | None = None,
) -> np.ndarray:
    """Draw detected corners into ``image_bgr_copy`` in place (caller passes a copy)."""
    import cv2

    if detection.marker_corners and detection.marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(
            image_bgr_copy, list(detection.marker_corners), detection.marker_ids
        )
    if detection.detected:
        assert detection.charuco_corners is not None and detection.charuco_ids is not None
        cv2.aruco.drawDetectedCornersCharuco(
            image_bgr_copy, detection.charuco_corners, detection.charuco_ids, (0, 255, 0)
        )
        if K is not None and dist is not None and board_spec is not None:
            pose = estimate_board_pose(board_spec, detection, K, dist)
            if pose is not None:
                rvec, tvec = pose
                axis_len = 0.5 * min(board_spec.cols, board_spec.rows) * board_spec.square_length
                cv2.drawFrameAxes(
                    image_bgr_copy,
                    np.asarray(K),
                    np.asarray(dist).reshape(-1),
                    rvec,
                    tvec,
                    axis_len,
                )
    return image_bgr_copy


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


_MIN_FRAMES = 4
_MIN_CORNERS = 8


@dataclasses.dataclass(frozen=True)
class CameraSample:
    image_shape: tuple[int, int]
    objp: np.ndarray
    imgp: np.ndarray
    thumbnail_jpeg: bytes


@dataclasses.dataclass(frozen=True)
class IntrinsicSolveResult:
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]
    rms: float
    per_frame_rms: tuple[float, ...]
    n_frames: int
    method: str


def solve_intrinsics(samples: tuple[CameraSample, ...]) -> IntrinsicSolveResult:
    valid = tuple(s for s in samples if len(s.imgp) >= _MIN_CORNERS)
    if len(valid) < _MIN_FRAMES:
        raise ValueError(
            f"need at least {_MIN_FRAMES} frames with >= {_MIN_CORNERS} corners; got {len(valid)}"
        )
    import cv2

    image_h, image_w = valid[0].image_shape
    obj_points = [s.objp.astype(np.float32) for s in valid]
    img_points = [s.imgp.astype(np.float32) for s in valid]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        (image_w, image_h),
        np.eye(3, dtype=np.float64),
        np.zeros(5, dtype=np.float64),
        flags=0,
    )
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    per_frame: list[float] = []
    for s, r, t in zip(valid, rvecs, tvecs):
        proj, _ = cv2.projectPoints(s.objp.astype(np.float32), r, t, K, dist)
        proj = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((proj - s.imgp) ** 2, axis=1))))
        per_frame.append(err)
    return IntrinsicSolveResult(
        K=np.asarray(K, dtype=np.float64),
        dist=dist,
        image_size=(int(image_w), int(image_h)),
        rms=float(rms),
        per_frame_rms=tuple(per_frame),
        n_frames=len(valid),
        method="solve",
    )


def package_sdk_intrinsics(
    K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int]
) -> IntrinsicSolveResult:
    return IntrinsicSolveResult(
        K=np.asarray(K, dtype=np.float64).reshape(3, 3),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        image_size=(int(image_size[0]), int(image_size[1])),
        rms=0.0,
        per_frame_rms=(),
        n_frames=0,
        method="sdk",
    )


# ---------------------------------------------------------------------------
# Hand-eye
# ---------------------------------------------------------------------------


HANDEYE_METHOD_NAMES: tuple[str, ...] = ("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS")


@dataclasses.dataclass(frozen=True)
class PoseSample:
    T_target_cam: np.ndarray
    T_gripper_base: np.ndarray
    camera_key: str
    robot_link: str
    ts: float


@dataclasses.dataclass(frozen=True)
class HandEyeResult:
    T_cam_gripper: np.ndarray
    method: str
    rotation_rms_deg: float
    translation_rms_m: float
    n_pairs: int


def _method_flags() -> dict[str, int]:
    import cv2

    flags: dict[str, int] = {}
    for name in HANDEYE_METHOD_NAMES:
        val = getattr(cv2, f"CALIB_HAND_EYE_{name}", None)
        if val is not None:
            flags[name] = int(val)
    return flags


def _residual_rms(
    T_cam_gripper: np.ndarray, pairs: tuple[PoseSample, ...]
) -> tuple[float, float]:
    if len(pairs) < 2:
        return 0.0, 0.0
    estimates = [p.T_gripper_base @ T_cam_gripper @ p.T_target_cam for p in pairs]
    ref = estimates[0]
    rot_errs: list[float] = []
    trans_errs: list[float] = []
    for est in estimates[1:]:
        delta = est @ invert_se3(ref)
        angle = float(np.linalg.norm(so3_log(delta[:3, :3])))
        rot_errs.append(np.degrees(angle))
        trans_errs.append(float(np.linalg.norm(delta[:3, 3])))
    return (
        float(np.sqrt(np.mean(np.square(rot_errs)))),
        float(np.sqrt(np.mean(np.square(trans_errs)))),
    )


def solve_hand_eye(
    pairs: tuple[PoseSample, ...], method: str = "TSAI"
) -> HandEyeResult:
    if len(pairs) < 3:
        raise ValueError(f"hand-eye needs at least 3 pose pairs; got {len(pairs)}")
    import cv2

    flags = _method_flags()
    chosen = method.upper()
    if chosen not in flags:
        log.warning("hand-eye method %s unavailable; falling back to PARK", chosen)
        chosen = "PARK"
    if chosen not in flags:
        raise RuntimeError("OpenCV build lacks all hand-eye methods")

    R_gripper2base = [p.T_gripper_base[:3, :3] for p in pairs]
    t_gripper2base = [p.T_gripper_base[:3, 3:4] for p in pairs]
    R_target2cam = [p.T_target_cam[:3, :3] for p in pairs]
    t_target2cam = [p.T_target_cam[:3, 3:4] for p in pairs]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=flags[chosen],
    )
    T_cam_gripper = np.eye(4, dtype=np.float64)
    T_cam_gripper[:3, :3] = np.asarray(R_cam2gripper, dtype=np.float64)
    T_cam_gripper[:3, 3] = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)
    rot_rms, trans_rms = _residual_rms(T_cam_gripper, pairs)
    return HandEyeResult(
        T_cam_gripper=T_cam_gripper,
        method=chosen,
        rotation_rms_deg=rot_rms,
        translation_rms_m=trans_rms,
        n_pairs=len(pairs),
    )


def compose_T_cam_link(T_cam_gripper: np.ndarray, attach_link: str) -> np.ndarray:
    del attach_link
    return np.asarray(T_cam_gripper, dtype=np.float64).reshape(4, 4)


# ---------------------------------------------------------------------------
# Scene extrinsic
# ---------------------------------------------------------------------------


def _average_se3(mats: tuple[np.ndarray, ...]) -> np.ndarray:
    if not mats:
        raise ValueError("no matrices to average")
    if len(mats) == 1:
        return mats[0].astype(np.float64)
    trans = np.stack([np.asarray(m)[:3, 3] for m in mats], axis=0).mean(axis=0)
    R_mean = np.asarray(mats[0])[:3, :3].astype(np.float64)
    for _ in range(20):
        residuals = np.stack(
            [so3_log(R_mean.T @ np.asarray(m)[:3, :3]) for m in mats], axis=0
        )
        step = residuals.mean(axis=0)
        if float(np.linalg.norm(step)) < 1e-9:
            break
        R_mean = R_mean @ so3_exp(step)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mean
    T[:3, 3] = trans
    return T


def solve_scene_extrinsic(
    target_cam_poses: tuple[np.ndarray, ...],
    T_board_world: np.ndarray | None = None,
) -> np.ndarray:
    if not target_cam_poses:
        raise ValueError("solve_scene_extrinsic needs at least one pose")
    T_world = T_board_world if T_board_world is not None else np.eye(4, dtype=np.float64)
    cam_in_board = tuple(invert_se3(np.asarray(P, dtype=np.float64)) for P in target_cam_poses)
    T_cam_board = _average_se3(cam_in_board)
    return T_cam_board @ invert_se3(T_world) if T_board_world is not None else T_cam_board


# ---------------------------------------------------------------------------
# raiden-schema JSON writer
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CameraCalibrationEntry:
    K: np.ndarray
    dist: np.ndarray
    attach_link: str = ""
    T_cam_link: np.ndarray | None = None
    image_size: tuple[int, int] | None = None


def _entry_to_dict(entry: CameraCalibrationEntry) -> dict:
    return {
        "K": np.asarray(entry.K, dtype=np.float64).reshape(3, 3).tolist(),
        "dist": np.asarray(entry.dist, dtype=np.float64).reshape(-1).tolist(),
        "attach_link": str(entry.attach_link),
        "T_cam_link": (
            np.asarray(entry.T_cam_link, dtype=np.float64).reshape(4, 4).tolist()
            if entry.T_cam_link is not None
            else None
        ),
        "image_size": (
            [int(entry.image_size[0]), int(entry.image_size[1])]
            if entry.image_size is not None
            else None
        ),
    }


def _dict_to_entry(raw: dict) -> CameraCalibrationEntry:
    T = raw.get("T_cam_link")
    size = raw.get("image_size")
    return CameraCalibrationEntry(
        K=np.asarray(raw["K"], dtype=np.float64).reshape(3, 3),
        dist=np.asarray(raw["dist"], dtype=np.float64).reshape(-1),
        attach_link=str(raw.get("attach_link") or ""),
        T_cam_link=(np.asarray(T, dtype=np.float64).reshape(4, 4) if T is not None else None),
        image_size=(int(size[0]), int(size[1])) if size else None,
    )


def write_raiden_json(path: Path, entries: dict[str, CameraCalibrationEntry]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cameras": {name: _entry_to_dict(e) for name, e in entries.items()}}
    fd, tmp = tempfile.mkstemp(
        prefix=".calibration_results.", suffix=".json.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_raiden_json(path: Path) -> dict[str, CameraCalibrationEntry]:
    raw = json.loads(Path(path).read_text())
    return {name: _dict_to_entry(entry) for name, entry in (raw.get("cameras") or {}).items()}


# ---------------------------------------------------------------------------
# Printable ChArUco PDF
# ---------------------------------------------------------------------------


# PDF paper sizes in points (1 pt = 1/72 inch).
_PAPER_SIZES = {
    "a4": (595.276, 841.890),
    "letter": (612.0, 792.0),
}


def _png_bytes_from_gray(gray: np.ndarray) -> bytes:
    """Encode a HxW uint8 grayscale array as a minimal PNG. Avoids opencv PNG deps."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("expected HxW uint8 image")

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    h, w = gray.shape
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
    # Prepend a filter byte 0 to each row.
    raw = bytearray()
    for row in gray:
        raw.append(0)
        raw.extend(row.tobytes())
    idat = zlib.compress(bytes(raw), level=9)
    return header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def build_charuco_pdf(spec: CharucoBoardSpec, paper: str = "a4") -> bytes:
    """Render a printable single-page PDF of ``spec``.

    Encodes the board as a Flate-compressed PNG-in-PDF /Image XObject and places
    it centered on a page sized for ``paper``. The physical size of the board on
    the page reflects ``spec.square_length`` and ``spec.cols x rows`` — so a
    printed page at 100% scale matches the spec's declared geometry.
    """
    import cv2

    paper_key = paper.lower()
    if paper_key not in _PAPER_SIZES:
        raise ValueError(f"unknown paper size '{paper}'; try {list(_PAPER_SIZES)}")
    page_w_pt, page_h_pt = _PAPER_SIZES[paper_key]

    board = build_cv_board(spec)
    # 300 dpi rasterization; the printer will scale down to the placement rect.
    # Board size in inches (from meters) -> pixel size.
    board_w_m = spec.cols * spec.square_length
    board_h_m = spec.rows * spec.square_length
    dpi = 300
    px_w = max(200, int(round(board_w_m / 0.0254 * dpi)))
    px_h = max(200, int(round(board_h_m / 0.0254 * dpi)))
    gray = board.generateImage((px_w, px_h), marginSize=int(dpi * 0.15), borderBits=1)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    png_bytes = _png_bytes_from_gray(gray)

    # Physical placement on the page (points).
    board_w_pt = board_w_m / 0.0254 * 72.0
    board_h_pt = board_h_m / 0.0254 * 72.0
    # 25pt margin from top for header caption.
    x0 = (page_w_pt - board_w_pt) / 2.0
    y0 = (page_h_pt - board_h_pt) / 2.0 - 20.0

    # Assemble the PDF. Kept to the bare minimum PDF 1.4 skeleton with one page,
    # one image XObject (raster of the ChArUco board), and one small text stream
    # for the caption. No external dependencies.
    # Build objects and record their byte offsets for the xref.
    caption_lines = [
        f"ChArUco board — {spec.dict_name}",
        f"{spec.cols} x {spec.rows}  square {spec.square_length*1000:.1f} mm  marker {spec.marker_length*1000:.1f} mm",
        "Print at 100% scale. Measure a square with calipers to confirm size before use.",
    ]

    def _content_stream() -> bytes:
        parts = [f"q\n{board_w_pt:.3f} 0 0 {board_h_pt:.3f} {x0:.3f} {y0:.3f} cm\n/Img0 Do\nQ\n"]
        parts.append("BT\n/F1 10 Tf\n")
        y = page_h_pt - 50.0
        for line in caption_lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            parts.append(f"1 0 0 1 {x0:.3f} {y:.3f} Tm ({escaped}) Tj\n")
            y -= 14
        parts.append("ET\n")
        return "".join(parts).encode("latin-1", errors="replace")

    stream = _content_stream()

    # Objects (each rendered as bytes).
    objects: list[bytes] = []

    def obj(n: int, body: str) -> bytes:
        return f"{n} 0 obj\n{body}\nendobj\n".encode("latin-1")

    # 1: Catalog
    objects.append(obj(1, "<< /Type /Catalog /Pages 2 0 R >>"))
    # 2: Pages
    objects.append(obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    # 3: Page
    page_body = (
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w_pt:.3f} {page_h_pt:.3f}]"
        f" /Resources << /XObject << /Img0 4 0 R >> /Font << /F1 6 0 R >> >>"
        f" /Contents 5 0 R >>"
    )
    objects.append(obj(3, page_body))
    # 4: Image XObject (raw PNG bytes embedded via /Filter /FlateDecode + PNG predictor).
    # Simpler: embed as raw pixel data with /Filter /FlateDecode, no PNG chunks.
    raw_pixels = zlib.compress(gray.tobytes(), level=9)
    xobj_dict = (
        f"<< /Type /XObject /Subtype /Image /Width {px_w} /Height {px_h}"
        f" /ColorSpace /DeviceGray /BitsPerComponent 8"
        f" /Filter /FlateDecode /Length {len(raw_pixels)} >>"
        f"\nstream\n"
    ).encode("latin-1") + raw_pixels + b"\nendstream"
    objects.append(f"4 0 obj\n".encode("latin-1") + xobj_dict + b"\nendobj\n")
    del png_bytes  # not needed; we embedded raw pixels
    # 5: Content stream
    stream_body = (
        f"<< /Length {len(stream)} >>\nstream\n"
    ).encode("latin-1") + stream + b"\nendstream"
    objects.append(f"5 0 obj\n".encode("latin-1") + stream_body + b"\nendobj\n")
    # 6: Font
    objects.append(obj(6, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    header = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"
    body = bytearray(header)
    offsets: list[int] = [0]
    for obj_bytes in objects:
        offsets.append(len(body))
        body.extend(obj_bytes)
    xref_pos = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    body.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        body.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    body.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(body)


# ---------------------------------------------------------------------------
# EEF -> pixel projection
# ---------------------------------------------------------------------------


def project_eef_uv(
    eef_world: np.ndarray,
    T_world_link: np.ndarray,
    T_cam_link: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Project one or more world-frame EEF positions into a camera's pixel plane.

    Args:
        eef_world: (3,) or (N, 3) EEF xyz in the world (base) frame.
        T_world_link: (4, 4) transform from the link (URDF attach_link) to the
            world frame (i.e. FK output for the link the camera is bolted to).
        T_cam_link: (4, 4) transform from the link frame to the camera frame,
            as stored in the raiden JSON.
        K: (3, 3) intrinsic matrix.
        dist: (D,) distortion coefficients (any length OpenCV accepts).

    Returns:
        ``(2,)`` or ``(N, 2)`` float32 pixel coordinates. Points behind the
        camera (Z_cam <= 0) become ``NaN, NaN`` so downstream filters can drop
        them cleanly.
    """
    import cv2

    T_world_link = np.asarray(T_world_link, dtype=np.float64).reshape(4, 4)
    T_cam_link = np.asarray(T_cam_link, dtype=np.float64).reshape(4, 4)
    T_cam_world = T_cam_link @ invert_se3(T_world_link)

    eef = np.asarray(eef_world, dtype=np.float64)
    single = eef.ndim == 1
    if single:
        eef = eef.reshape(1, 3)
    else:
        eef = eef.reshape(-1, 3)

    homo = np.concatenate([eef, np.ones((eef.shape[0], 1), dtype=np.float64)], axis=1)
    cam_pts = (T_cam_world @ homo.T).T[:, :3]
    behind = cam_pts[:, 2] <= 0
    front = ~behind

    uv = np.full((eef.shape[0], 2), np.nan, dtype=np.float32)
    if front.any():
        pts = cam_pts[front]
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        proj, _ = cv2.projectPoints(
            pts.astype(np.float32), rvec, tvec, np.asarray(K), np.asarray(dist).reshape(-1)
        )
        uv[front] = np.asarray(proj, dtype=np.float32).reshape(-1, 2)
    return uv[0] if single else uv


# ---------------------------------------------------------------------------
# Public exports for `from tools.calibration import ...`
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_DICT_NAMES",
    "HANDEYE_METHOD_NAMES",
    "CameraCalibrationEntry",
    "CameraSample",
    "CharucoBoardSpec",
    "CharucoDetection",
    "HandEyeResult",
    "IntrinsicSolveResult",
    "PoseSample",
    "build_charuco_pdf",
    "build_cv_board",
    "compose_T_cam_link",
    "compose_se3",
    "detect_board",
    "draw_overlay",
    "estimate_board_pose",
    "invert_se3",
    "match_image_points",
    "package_sdk_intrinsics",
    "project_eef_uv",
    "read_raiden_json",
    "rvec_tvec_to_se3",
    "so3_exp",
    "so3_log",
    "solve_hand_eye",
    "solve_intrinsics",
    "solve_scene_extrinsic",
    "write_raiden_json",
    "_average_se3",
    "_method_flags",
    "_residual_rms",
]


# ---------------------------------------------------------------------------
# Time helpers used elsewhere (kept for API parity with old modules).
# ---------------------------------------------------------------------------


def _now() -> float:  # small hook so tests can freeze time if needed
    return time.time()
