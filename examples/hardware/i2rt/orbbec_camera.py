"""Orbbec color-camera cache for the isolated I2RT execution node."""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import multiprocessing
import os
import signal
import time
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class OrbbecCameraSpec:
    """One Orbbec color camera mapped to an EVA image observation key."""

    image_key: str
    serial: str | None = None
    device_index: int | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    color_format: str = "MJPG"
    timeout_ms: int = 1000
    retry_interval_s: float = 3.0
    warmup_frames: int = 30


def parse_orbbec_camera_specs(
    values: list[str],
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    color_format: str = "MJPG",
    timeout_ms: int = 1000,
    warmup_frames: int = 30,
) -> tuple[OrbbecCameraSpec, ...]:
    """Parse repeated ``IMAGE_KEY=SERIAL`` or ``IMAGE_KEY=index:N`` mappings."""
    if width <= 0 or height <= 0 or fps <= 0 or timeout_ms <= 0:
        raise ValueError("Orbbec width, height, fps, and timeout must be positive")
    if warmup_frames < 0:
        raise ValueError("Orbbec warmup frame count must be non-negative")

    cameras: list[OrbbecCameraSpec] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected Orbbec mapping KEY=SERIAL_OR_INDEX, got {value!r}")
        image_key, selector = (part.strip() for part in value.split("=", 1))
        if not image_key or not selector:
            raise ValueError(f"Invalid Orbbec camera mapping {value!r}")
        if image_key in seen:
            raise ValueError(f"Duplicate Orbbec camera key: {image_key}")
        seen.add(image_key)

        serial: str | None = selector
        device_index: int | None = None
        if selector.startswith("index:"):
            serial = None
            device_index = int(selector.removeprefix("index:"))
            if device_index < 0:
                raise ValueError("Orbbec device index must be non-negative")
        cameras.append(
            OrbbecCameraSpec(
                image_key=image_key,
                serial=serial,
                device_index=device_index,
                width=width,
                height=height,
                fps=fps,
                color_format=color_format,
                timeout_ms=timeout_ms,
                warmup_frames=warmup_frames,
            )
        )
    return tuple(cameras)


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
            "pyorbbecsdk is unavailable in the I2RT environment; rerun "
            "examples/hardware/i2rt/setup_sdk.sh"
        ) from exc


def list_orbbec_devices() -> list[dict[str, str]]:
    """Return connected Orbbec device identity and USB connection information."""
    sdk = get_orbbec_sdk()
    context = sdk.Context()
    devices = context.query_devices()
    result: list[dict[str, str]] = []
    for index in range(len(devices)):
        info = devices.get_device_by_index(index).get_device_info()
        result.append(
            {
                "name": str(info.get_name()),
                "serial": str(info.get_serial_number()),
                "connection_type": str(info.get_connection_type()),
            }
        )
    return result


def _enum_member(enum_cls: object, value: str) -> object:
    name = value.upper()
    if hasattr(enum_cls, name):
        return getattr(enum_cls, name)
    raise ValueError(f"{value!r} is not a valid Orbbec color format")


def _u8_data(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.uint8, copy=False).ravel()
    return np.frombuffer(data, dtype=np.uint8)


def select_orbbec_device(devices: Any, spec: OrbbecCameraSpec) -> Any:
    if spec.serial is not None:
        try:
            device = devices.get_device_by_serial_number(spec.serial)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open Orbbec camera {spec.image_key} serial={spec.serial}; "
                "stop other camera processes and check the USB connection"
            ) from exc
        if device is None:
            raise RuntimeError(f"Orbbec camera serial {spec.serial!r} was not found")
        return device

    assert spec.device_index is not None
    if spec.device_index >= len(devices):
        raise RuntimeError(f"Orbbec camera index {spec.device_index} is out of range")
    return devices.get_device_by_index(spec.device_index)


def select_color_profile(sdk: ModuleType, pipeline: Any, spec: OrbbecCameraSpec) -> Any:
    profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
    color_format = _enum_member(sdk.OBFormat, spec.color_format)
    return profiles.get_video_stream_profile(
        spec.width,
        spec.height,
        color_format,
        spec.fps,
    )


def enable_auto_exposure(device: Any, sdk: ModuleType) -> bool:
    """Explicitly enable color auto exposure when the camera supports it."""
    prop = getattr(sdk.OBPropertyID, "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", None)
    if prop is None:
        return False
    permission = sdk.OBPermissionType.PERMISSION_WRITE
    read_write = sdk.OBPermissionType.PERMISSION_READ_WRITE
    if not any(device.is_property_supported(prop, item) for item in (read_write, permission)):
        return False
    device.set_bool_property(prop, True)
    return True


def frame_to_bgr_image(frame: Any, sdk: ModuleType) -> np.ndarray:
    """Convert an Orbbec color frame into a contiguous OpenCV BGR image."""
    fmt = frame.get_format()
    width = frame.get_width()
    height = frame.get_height()
    data = _u8_data(frame.get_data())

    if fmt == sdk.OBFormat.RGB:
        return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)
    if hasattr(sdk.OBFormat, "BGR") and fmt == sdk.OBFormat.BGR:
        return np.ascontiguousarray(data.reshape((height, width, 3)))
    if fmt == sdk.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode Orbbec MJPG color frame")
        return image
    if fmt == sdk.OBFormat.YUYV or (hasattr(sdk.OBFormat, "YUY2") and fmt == sdk.OBFormat.YUY2):
        return cv2.cvtColor(data.reshape((height, width, 2)), cv2.COLOR_YUV2BGR_YUY2)
    if hasattr(sdk.OBFormat, "UYVY") and fmt == sdk.OBFormat.UYVY:
        return cv2.cvtColor(data.reshape((height, width, 2)), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == sdk.OBFormat.I420:
        return cv2.cvtColor(data.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_I420)
    if fmt == sdk.OBFormat.NV12:
        return cv2.cvtColor(data.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV12)
    if fmt == sdk.OBFormat.NV21:
        return cv2.cvtColor(data.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV21)
    raise ValueError(f"Unsupported Orbbec color frame format: {fmt}")


_STATE_NAMES = ("starting", "connecting", "warming", "online", "retrying", "stopped")


def _set_process_state(state: Any, name: str) -> None:
    state.value = _STATE_NAMES.index(name)


def _capture_orbbec_camera(
    spec: OrbbecCameraSpec,
    frame_buffer: Any,
    frame_lock: Any,
    frame_count: Any,
    last_frame_time: Any,
    state: Any,
    stop: Any,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    try:
        os.nice(5)
    except OSError:
        logger.debug("Could not lower Orbbec camera process priority", exc_info=True)
    while not stop.is_set():
        try:
            _run_orbbec_capture_loop(
                spec,
                frame_buffer,
                frame_lock,
                frame_count,
                last_frame_time,
                state,
                stop,
            )
        except Exception as exc:
            _set_process_state(state, "retrying")
            logger.warning(
                "Orbbec camera %s stopped; retrying in %.1f s: %s",
                spec.image_key,
                spec.retry_interval_s,
                exc,
            )
            stop.wait(spec.retry_interval_s)
    _set_process_state(state, "stopped")


def _run_orbbec_capture_loop(
    spec: OrbbecCameraSpec,
    frame_buffer: Any,
    frame_lock: Any,
    frame_count: Any,
    last_frame_time: Any,
    state: Any,
    stop: Any,
) -> None:
    _set_process_state(state, "connecting")
    sdk = get_orbbec_sdk()
    context = sdk.Context()
    devices = context.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No Orbbec camera is connected")
    device = select_orbbec_device(devices, spec)
    serial = device.get_device_info().get_serial_number()
    auto_exposure = enable_auto_exposure(device, sdk)
    pipeline = sdk.Pipeline(device)
    stream_config = sdk.Config()
    try:
        profile = select_color_profile(sdk, pipeline, spec)
    except Exception as exc:
        logger.warning(
            "Orbbec camera %s falling back to its default color profile after "
            "%sx%s %s@%s failed: %s",
            spec.image_key,
            spec.width,
            spec.height,
            spec.color_format,
            spec.fps,
            exc,
        )
        profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        profile = profiles.get_default_video_stream_profile()
    stream_config.enable_stream(profile)
    pipeline.start(stream_config)
    _set_process_state(state, "warming")
    logger.info(
        "Started Orbbec camera %s serial=%s auto_exposure=%s warmup_frames=%d",
        spec.image_key,
        serial,
        auto_exposure,
        spec.warmup_frames,
    )
    remaining_warmup = spec.warmup_frames
    shared_image = np.frombuffer(frame_buffer, dtype=np.uint8).reshape(
        (spec.height, spec.width, 3)
    )
    try:
        while not stop.is_set():
            frameset = pipeline.wait_for_frames(spec.timeout_ms)
            if frameset is None:
                continue
            color_frame = frameset.get_color_frame()
            if color_frame is None:
                continue
            if remaining_warmup > 0:
                remaining_warmup -= 1
                continue
            image = frame_to_bgr_image(color_frame, sdk)
            if image.shape != shared_image.shape:
                image = cv2.resize(image, (spec.width, spec.height))
            with frame_lock:
                np.copyto(shared_image, image)
                frame_count.value += 1
                last_frame_time.value = time.monotonic()
                _set_process_state(state, "online")
    finally:
        pipeline.stop()


class _OrbbecCameraWorker:
    def __init__(self, spec: OrbbecCameraSpec) -> None:
        self.spec = spec
        context = multiprocessing.get_context("spawn")
        self._frame_buffer = context.RawArray("B", spec.width * spec.height * 3)
        self._frame_lock = context.Lock()
        self._frame_count = context.Value("Q", 0, lock=False)
        self._last_frame_time = context.Value("d", 0.0, lock=False)
        self._state = context.Value("i", 0, lock=False)
        self._stop = context.Event()
        self._process = context.Process(
            target=_capture_orbbec_camera,
            args=(
                spec,
                self._frame_buffer,
                self._frame_lock,
                self._frame_count,
                self._last_frame_time,
                self._state,
                self._stop,
            ),
            name=f"orbbec-{spec.image_key}",
            daemon=True,
        )
        self._process.start()

    def snapshot(self) -> np.ndarray | None:
        with self._frame_lock:
            if self._frame_count.value == 0:
                return None
            image = np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape(
                (self.spec.height, self.spec.width, 3)
            )
            return image.copy()

    def status(self) -> str:
        state = _STATE_NAMES[self._state.value]
        if not self._process.is_alive() and not self._stop.is_set():
            state = "stopped"
        last_frame_time = self._last_frame_time.value
        if last_frame_time == 0.0:
            return state
        return f"{state}(age={time.monotonic() - last_frame_time:.1f}s)"

    def stop(self) -> None:
        self._stop.set()
        self._process.join(timeout=max(3.0, self.spec.timeout_ms / 1000.0 + 2.0))
        if self._process.is_alive():
            logger.warning("Terminating unresponsive Orbbec camera process %s", self.spec.image_key)
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._process.close()


class OrbbecCameraCache:
    """Background multi-camera cache using the Orbbec Python SDK."""

    def __init__(self, camera_specs: tuple[OrbbecCameraSpec, ...]) -> None:
        self._workers = [_OrbbecCameraWorker(spec) for spec in camera_specs]
        logger.info("Started %d Orbbec camera worker(s)", len(self._workers))

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
