"""Orbbec SDK camera source for the Franka ZMQ execution node."""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import threading
import time
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FRANKA_CAMERA_KEYS: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_FRANKA_ORBBEC_CAMERAS: dict[str, str | None] = {
    "cam_high": "CL8R353009V",
    "cam_right_wrist": "CP053530008B",
    "cam_left_wrist": None,
}


@dataclasses.dataclass(frozen=True)
class OrbbecCameraSpec:
    """One Orbbec color camera mapped to one EVA image observation key.

    Args:
        image_key: EVA observation image key, e.g. ``cam_high``.
        serial: Orbbec device serial. Mutually exclusive with ``device_index``.
        device_index: Orbbec device index when serial is not available.
        resolution: requested color resolution as ``(width, height)``; ``None``
            lets the SDK choose its default profile.
        fps: requested color stream FPS when a resolution is set.
        color_format: requested SDK color format.
        timeout_ms: frame wait timeout in milliseconds.
        retry_interval_s: delay before retrying a failed camera start/read loop.
    """

    image_key: str
    serial: str | None = None
    device_index: int | None = None
    resolution: tuple[int, int] | None = None
    fps: int = 30
    color_format: str = "RGB"
    timeout_ms: int = 1000
    retry_interval_s: float = 5.0


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    """Parse a WIDTHxHEIGHT resolution string or ``default``.

    Args:
        value: resolution text such as ``1280x720``; ``None``/``default`` means
            the SDK default profile.

    Returns:
        resolution: ``(width, height)`` or ``None``.
    """
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"", "default", "none"}:
        return None
    width_text, height_text = text.split("x", maxsplit=1)
    width = int(width_text)
    height = int(height_text)
    if width <= 0 or height <= 0:
        raise ValueError("Orbbec resolution dimensions must be positive")
    return width, height


def parse_orbbec_camera_specs(
    specs: list[str],
    resolution: tuple[int, int] | None = None,
    fps: int = 30,
    color_format: str = "RGB",
    timeout_ms: int = 1000,
) -> tuple[OrbbecCameraSpec, ...]:
    """Parse repeated ``KEY=SERIAL`` or ``KEY=index:N`` camera specs.

    Args:
        specs: CLI specs mapping EVA image keys to Orbbec serials or indices.
        resolution: default resolution applied to each camera.
        fps: default color FPS applied to each camera.
        color_format: default SDK color format applied to each camera.
        timeout_ms: frame wait timeout per camera.

    Returns:
        cameras: parsed camera specs in CLI order.
    """
    cameras: list[OrbbecCameraSpec] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected Orbbec camera spec KEY=SERIAL_OR_INDEX, got {spec!r}")
        image_key, selector = spec.split("=", 1)
        image_key = image_key.strip()
        selector = selector.strip()
        if not image_key:
            raise ValueError(f"Missing EVA image key in Orbbec camera spec {spec!r}")
        if not selector:
            raise ValueError(f"Missing device selector in Orbbec camera spec {spec!r}")

        serial: str | None = selector
        device_index: int | None = None
        if selector.startswith("index:"):
            serial = None
            device_index = int(selector.removeprefix("index:"))
        cameras.append(
            OrbbecCameraSpec(
                image_key=image_key,
                serial=serial,
                device_index=device_index,
                resolution=resolution,
                fps=fps,
                color_format=color_format,
                timeout_ms=timeout_ms,
            )
        )
    return tuple(cameras)


def default_franka_orbbec_camera_specs(
    resolution: tuple[int, int] | None = None,
    fps: int = 30,
    color_format: str = "RGB",
    timeout_ms: int = 1000,
) -> tuple[OrbbecCameraSpec, ...]:
    """Return the known Franka workcell Orbbec cameras.

    Args:
        resolution: default resolution applied to each camera.
        fps: default color FPS applied to each camera.
        color_format: default SDK color format applied to each camera.
        timeout_ms: frame wait timeout per camera.

    Returns:
        cameras: default high/top and right-wrist camera specs.
    """
    return tuple(
        OrbbecCameraSpec(
            image_key=image_key,
            serial=serial,
            resolution=resolution,
            fps=fps,
            color_format=color_format,
            timeout_ms=timeout_ms,
        )
        for image_key, serial in DEFAULT_FRANKA_ORBBEC_CAMERAS.items()
    )


def merge_orbbec_camera_specs(
    defaults: tuple[OrbbecCameraSpec, ...],
    overrides: tuple[OrbbecCameraSpec, ...],
    disabled_cameras: tuple[str, ...] = (),
) -> tuple[OrbbecCameraSpec, ...]:
    """Merge default and CLI camera specs, then drop disabled image keys.

    Args:
        defaults: default camera specs.
        overrides: CLI-provided camera specs; same image_key overrides defaults.
        disabled_cameras: EVA image keys intentionally absent in this deployment.

    Returns:
        cameras: enabled camera specs in stable insertion order.
    """
    disabled = set(disabled_cameras)
    merged: dict[str, OrbbecCameraSpec] = {}
    for spec in defaults + overrides:
        if spec.image_key not in FRANKA_CAMERA_KEYS:
            raise ValueError(
                f"Unknown Franka camera key {spec.image_key!r}; "
                f"expected one of {FRANKA_CAMERA_KEYS}"
            )
        merged[spec.image_key] = spec
    return tuple(spec for key, spec in merged.items() if key not in disabled)


def _preload_bundled_sdk_library() -> None:
    spec = find_spec("pyorbbecsdk")
    if spec is None or spec.origin is None:
        return
    package_dir = Path(spec.origin).parent
    for library_name in ("libOrbbecSDK.so.2", "libOrbbecSDK.so"):
        library_path = package_dir / library_name
        if library_path.exists():
            ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            return


def get_orbbec_sdk() -> ModuleType:
    try:
        _preload_bundled_sdk_library()
        return import_module("pyorbbecsdk")
    except Exception as exc:
        raise RuntimeError(
            "pyorbbecsdk could not be imported. Activate the Franka hardware "
            "environment before running examples/hardware/franka/run_hardware.sh."
        ) from exc


def _enum_member(enum_cls: object, value: object) -> object:
    if not isinstance(value, str):
        return value
    name = value.upper()
    if hasattr(enum_cls, name):
        return getattr(enum_cls, name)
    raise ValueError(f"{value!r} is not a valid {enum_cls!r} member")


def _u8_data(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.uint8, copy=False).ravel()
    return np.frombuffer(data, dtype=np.uint8)


def frame_to_bgr_image(frame: Any, sdk: ModuleType) -> np.ndarray:
    """Convert a pyorbbecsdk color frame into an OpenCV BGR image.

    Args:
        frame: SDK color frame.
        sdk: imported ``pyorbbecsdk`` module.

    Returns:
        image: HWC BGR uint8 array.
    """
    fmt = frame.get_format()
    width = frame.get_width()
    height = frame.get_height()
    data = _u8_data(frame.get_data())

    if fmt == sdk.OBFormat.RGB:
        image = data.reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if hasattr(sdk.OBFormat, "BGR") and fmt == sdk.OBFormat.BGR:
        return np.ascontiguousarray(data.reshape((height, width, 3)))
    if hasattr(sdk.OBFormat, "RGBA") and fmt == sdk.OBFormat.RGBA:
        image = data.reshape((height, width, 4))
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if hasattr(sdk.OBFormat, "BGRA") and fmt == sdk.OBFormat.BGRA:
        image = data.reshape((height, width, 4))
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if fmt == sdk.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode Orbbec MJPG color frame")
        return image
    if fmt == sdk.OBFormat.YUYV or (hasattr(sdk.OBFormat, "YUY2") and fmt == sdk.OBFormat.YUY2):
        yuyv = data.reshape((height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    if hasattr(sdk.OBFormat, "UYVY") and fmt == sdk.OBFormat.UYVY:
        uyvy = data.reshape((height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if fmt == sdk.OBFormat.I420:
        i420 = data.reshape((height * 3 // 2, width))
        return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
    if hasattr(sdk.OBFormat, "YV12") and fmt == sdk.OBFormat.YV12:
        yv12 = data.reshape((height * 3 // 2, width))
        return cv2.cvtColor(yv12, cv2.COLOR_YUV2BGR_YV12)
    if fmt == sdk.OBFormat.NV12:
        nv12 = data.reshape((height * 3 // 2, width))
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    if fmt == sdk.OBFormat.NV21:
        nv21 = data.reshape((height * 3 // 2, width))
        return cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV21)
    if (hasattr(sdk.OBFormat, "Y8") and fmt == sdk.OBFormat.Y8) or (
        hasattr(sdk.OBFormat, "GRAY") and fmt == sdk.OBFormat.GRAY
    ):
        gray = data.reshape((height, width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported Orbbec color frame format: {fmt}")


class _OrbbecCameraWorker:
    def __init__(self, spec: OrbbecCameraSpec) -> None:
        self.spec = spec
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._latest: np.ndarray | None = None
        self._state = "starting"
        self._last_frame_time: float | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"orbbec-{spec.image_key}",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def status(self) -> str:
        with self._lock:
            state = self._state
            last_frame_time = self._last_frame_time
        if last_frame_time is None:
            return state
        age_s = time.monotonic() - last_frame_time
        return f"{state}(age={age_s:.1f}s)"

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_camera_loop()
            except Exception as exc:
                self._set_state("retrying")
                logger.warning(
                    "Orbbec camera %s stopped; retrying in %.1f s: %s",
                    self.spec.image_key,
                    self.spec.retry_interval_s,
                    exc,
                )
                self._stop.wait(self.spec.retry_interval_s)

    def _run_camera_loop(self) -> None:
        self._set_state("connecting")
        sdk = get_orbbec_sdk()
        ctx = sdk.Context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No Orbbec camera is connected")

        if self.spec.serial is not None:
            device = devices.get_device_by_serial_number(self.spec.serial)
            if device is None:
                raise RuntimeError(f"Orbbec camera serial {self.spec.serial!r} was not found")
        else:
            assert self.spec.device_index is not None
            if self.spec.device_index < 0 or self.spec.device_index >= len(devices):
                raise RuntimeError(f"Orbbec camera index {self.spec.device_index} is out of range")
            device = devices.get_device_by_index(self.spec.device_index)

        info = device.get_device_info()
        serial = info.get_serial_number()
        pipeline = sdk.Pipeline(device)
        stream_config = sdk.Config()
        profile = self._select_color_profile(sdk, pipeline)
        stream_config.enable_stream(profile)
        pipeline.start(stream_config)
        self._set_state("online")
        logger.info("Started Orbbec camera %s serial=%s", self.spec.image_key, serial)
        try:
            while not self._stop.is_set():
                frameset = pipeline.wait_for_frames(self.spec.timeout_ms)
                if frameset is None:
                    continue
                color_frame = frameset.get_color_frame()
                if color_frame is None:
                    continue
                image = frame_to_bgr_image(color_frame, sdk)
                with self._lock:
                    self._latest = image
                    self._last_frame_time = time.monotonic()
        finally:
            pipeline.stop()

    def _select_color_profile(self, sdk: ModuleType, pipeline: Any) -> Any:
        profile_list = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        if self.spec.resolution is None:
            return profile_list.get_default_video_stream_profile()
        fmt = _enum_member(sdk.OBFormat, self.spec.color_format)
        try:
            return profile_list.get_video_stream_profile(
                self.spec.resolution[0],
                self.spec.resolution[1],
                fmt,
                self.spec.fps,
            )
        except Exception as exc:
            logger.warning(
                "Orbbec camera %s falling back to default profile after %sx%s %s@%s failed: %s",
                self.spec.image_key,
                self.spec.resolution[0],
                self.spec.resolution[1],
                self.spec.color_format,
                self.spec.fps,
                exc,
            )
            return profile_list.get_default_video_stream_profile()


class OrbbecCameraCache:
    """Background multi-camera cache using the Orbbec Python SDK."""

    def __init__(self, camera_specs: tuple[OrbbecCameraSpec, ...]) -> None:
        self._workers = [_OrbbecCameraWorker(spec) for spec in camera_specs]
        if self._workers:
            logger.info("Started %d Orbbec camera worker(s)", len(self._workers))
        else:
            logger.info("Orbbec camera backend selected with no cameras configured")

    def snapshot(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for worker in self._workers:
            image = worker.snapshot()
            if image is not None:
                images[worker.spec.image_key] = image
        return images

    def hardware_status(self) -> dict[str, str]:
        return {worker.spec.image_key: worker.status() for worker in self._workers}

    def close(self) -> None:
        for worker in self._workers:
            worker.stop()
