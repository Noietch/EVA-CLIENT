"""Intel RealSense D405 color-camera cache for the I2RT execution node."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CameraSpec:
    image_key: str
    serial: str | None = None
    device_index: int | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    timeout_ms: int = 1000
    retry_interval_s: float = 3.0


def get_realsense_sdk() -> ModuleType:
    try:
        return import_module("pyrealsense2")
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is unavailable in the I2RT environment; rerun "
            "examples/hardware/i2rt/setup_sdk.sh"
        ) from exc


def parse_camera_specs(
    values: list[str],
    *,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    timeout_ms: int = 1000,
) -> tuple[CameraSpec, ...]:
    """Parse repeated ``IMAGE_KEY=SERIAL`` or ``IMAGE_KEY=index:N`` mappings."""
    specs: list[CameraSpec] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected D405 mapping KEY=SERIAL_OR_INDEX, got {value!r}")
        image_key, selector = (part.strip() for part in value.split("=", 1))
        if not image_key or not selector:
            raise ValueError(f"Invalid D405 camera mapping {value!r}")
        if image_key in seen:
            raise ValueError(f"Duplicate D405 camera key: {image_key}")
        serial: str | None = selector
        device_index: int | None = None
        if selector.startswith("index:"):
            serial = None
            device_index = int(selector.removeprefix("index:"))
            if device_index < 0:
                raise ValueError("RealSense device index must be non-negative")
        specs.append(
            CameraSpec(
                image_key=image_key,
                serial=serial,
                device_index=device_index,
                width=width,
                height=height,
                fps=fps,
                timeout_ms=timeout_ms,
            )
        )
        seen.add(image_key)
    return tuple(specs)


def list_realsense_devices() -> list[dict[str, str]]:
    rs = get_realsense_sdk()
    devices: list[dict[str, str]] = []
    for device in rs.context().query_devices():
        devices.append(
            {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "product_line": device.get_info(rs.camera_info.product_line),
            }
        )
    return devices


def _resolve_serial(rs: ModuleType, spec: CameraSpec) -> str:
    devices = rs.context().query_devices()
    device = None
    if spec.serial is not None:
        for candidate in devices:
            if candidate.get_info(rs.camera_info.serial_number) == spec.serial:
                device = candidate
                break
        if device is None:
            raise RuntimeError(f"RealSense serial {spec.serial} is not connected")
    else:
        assert spec.device_index is not None
        if spec.device_index >= len(devices):
            raise RuntimeError(
                f"RealSense index {spec.device_index} is out of range ({len(devices)} detected)"
            )
        device = devices[spec.device_index]

    serial = device.get_info(rs.camera_info.serial_number)
    name = device.get_info(rs.camera_info.name)
    if "D405" not in name.upper():
        raise RuntimeError(f"serial {serial} is {name!r}, not an Intel RealSense D405")
    return serial


class _D405Worker:
    def __init__(self, spec: CameraSpec, rs: ModuleType) -> None:
        self.spec = spec
        self._rs = rs
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._latest: np.ndarray | None = None
        self._state = "starting"
        self._last_frame_time: float | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"d405-{spec.image_key}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            pipeline: Any = None
            try:
                self._set_state("connecting")
                serial = _resolve_serial(self._rs, self.spec)
                pipeline = self._rs.pipeline()
                config = self._rs.config()
                config.enable_device(serial)
                width = self.spec.width
                height = self.spec.height
                fps = self.spec.fps
                if width is not None and height is not None and fps is not None:
                    config.enable_stream(
                        self._rs.stream.color,
                        width,
                        height,
                        self._rs.format.bgr8,
                        fps,
                    )
                else:
                    config.enable_stream(self._rs.stream.color)
                pipeline.start(config)
                self._set_state("online")
                logger.info("Started D405 camera %s serial=%s", self.spec.image_key, serial)
                while not self._stop.is_set():
                    frames = pipeline.wait_for_frames(self.spec.timeout_ms)
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    image = np.asanyarray(color_frame.get_data())
                    if image.ndim != 3 or image.shape[2] != 3:
                        raise RuntimeError(f"unexpected D405 color frame shape {image.shape}")
                    with self._lock:
                        self._latest = np.ascontiguousarray(image, dtype=np.uint8)
                        self._last_frame_time = time.monotonic()
            except Exception as exc:
                self._set_state("retrying")
                logger.warning(
                    "D405 camera %s unavailable; retrying in %.1f s: %s",
                    self.spec.image_key,
                    self.spec.retry_interval_s,
                    exc,
                )
                self._stop.wait(self.spec.retry_interval_s)
            finally:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def status(self) -> str:
        with self._lock:
            state = self._state
            last_frame_time = self._last_frame_time
        if last_frame_time is None:
            return state
        return f"{state}(age={time.monotonic() - last_frame_time:.1f}s)"

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class RealSenseCameraCache:
    def __init__(self, specs: tuple[CameraSpec, ...]) -> None:
        rs = get_realsense_sdk() if specs else None
        self._workers = {spec.image_key: _D405Worker(spec, rs) for spec in specs if rs is not None}

    def snapshot(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for key, worker in self._workers.items():
            frame = worker.snapshot()
            if frame is not None:
                images[key] = frame
        return images

    def hardware_status(self) -> dict[str, str]:
        return {key: worker.status() for key, worker in self._workers.items()}

    def close(self) -> None:
        for worker in self._workers.values():
            worker.close()
