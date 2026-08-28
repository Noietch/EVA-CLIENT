"""Intel RealSense D405 color-camera cache for the YAM execution node."""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CameraProfile:
    serial: str | None = None
    exposure_us: int | None = None
    auto_exposure_limit_us: int | None = None
    auto_gain_limit: int | None = None
    warmup_frames: int = 8


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
    auto_exposure_limit_us: int | None = None
    auto_gain_limit: int | None = None
    exposure_us: int | None = None
    warmup_frames: int = 8


def load_camera_profiles(path: str | Path) -> dict[str, CameraProfile]:
    """Load per-camera exposure profiles from JSON."""
    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load D405 camera profile {profile_path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("D405 camera profile must be an object with version=1")
    raw_cameras = raw.get("cameras")
    if not isinstance(raw_cameras, dict) or not raw_cameras:
        raise ValueError("D405 camera profile must contain a non-empty cameras object")

    allowed_fields = {
        "serial",
        "exposure_us",
        "auto_exposure_limit_us",
        "auto_gain_limit",
        "warmup_frames",
    }
    profiles: dict[str, CameraProfile] = {}
    for image_key, values in raw_cameras.items():
        if not isinstance(image_key, str) or not image_key or not isinstance(values, dict):
            raise ValueError("D405 camera profile entries must map image keys to objects")
        unknown_fields = set(values) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown D405 profile fields for {image_key}: {sorted(unknown_fields)}"
            )
        profile = CameraProfile(**values)
        if profile.exposure_us is not None and profile.exposure_us <= 0:
            raise ValueError(f"D405 {image_key} exposure_us must be positive")
        if profile.auto_exposure_limit_us is not None and profile.auto_exposure_limit_us <= 0:
            raise ValueError(f"D405 {image_key} auto_exposure_limit_us must be positive")
        if profile.auto_gain_limit is not None and profile.auto_gain_limit <= 0:
            raise ValueError(f"D405 {image_key} auto_gain_limit must be positive")
        if profile.warmup_frames < 0:
            raise ValueError(f"D405 {image_key} warmup_frames must be non-negative")
        profiles[image_key] = profile
    return profiles


def get_realsense_sdk() -> ModuleType:
    try:
        return import_module("pyrealsense2")
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is unavailable in the YAM environment; rerun "
            "examples/hardware/yam/setup_sdk.sh"
        ) from exc


def parse_camera_specs(
    values: list[str],
    *,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    timeout_ms: int = 1000,
    camera_profiles: dict[str, CameraProfile] | None = None,
    auto_exposure_limit_us: int | None = None,
    auto_gain_limit: int | None = None,
    exposure_us: int | None = None,
    warmup_frames: int = 8,
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
        profile = (camera_profiles or {}).get(image_key, CameraProfile())
        serial: str | None = selector
        device_index: int | None = None
        if selector.startswith("index:"):
            serial = None
            device_index = int(selector.removeprefix("index:"))
            if device_index < 0:
                raise ValueError("RealSense device index must be non-negative")
        if profile.serial is not None and serial != profile.serial:
            raise ValueError(
                f"D405 profile for {image_key} expects serial {profile.serial}, got {selector}"
            )
        specs.append(
            CameraSpec(
                image_key=image_key,
                serial=serial,
                device_index=device_index,
                width=width,
                height=height,
                fps=fps,
                timeout_ms=timeout_ms,
                auto_exposure_limit_us=(
                    profile.auto_exposure_limit_us
                    if auto_exposure_limit_us is None
                    else auto_exposure_limit_us
                ),
                auto_gain_limit=(
                    profile.auto_gain_limit if auto_gain_limit is None else auto_gain_limit
                ),
                exposure_us=(profile.exposure_us if exposure_us is None else exposure_us),
                warmup_frames=(
                    profile.warmup_frames if camera_profiles is not None else warmup_frames
                ),
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

    def _enable_auto_exposure(self, pipeline_profile: Any) -> None:
        """Configure exposure on the D405 imaging sensor on reconnect.

        D405 devices expose their color stream through the ``Stereo Module``.
        That sensor supports automatic exposure, but librealsense does not
        classify it as a color sensor, so ``first_color_sensor()`` raises
        ``Could not find requested sensor type!`` on this camera model.
        """
        option = self._rs.option.enable_auto_exposure
        device = pipeline_profile.get_device()
        sensors = tuple(device.query_sensors())
        compatible_sensors = [sensor for sensor in sensors if sensor.supports(option)]
        if not compatible_sensors:
            raise RuntimeError(
                f"RealSense D405 {self.spec.image_key} has no sensor that supports "
                "automatic exposure"
            )
        # Prefer the sensor that owns the requested color stream.  On a D405
        # the stereo module is often exposed as the color sensor; enabling AE
        # on every compatible sensor can also change the IR/depth exposure and
        # make the color image appear blown out after reconnects.
        color_sensors = [
            sensor for sensor in compatible_sensors if self._sensor_has_color_stream(sensor)
        ]
        selected_sensors = color_sensors or compatible_sensors
        exposure_us = getattr(self.spec, "exposure_us", None)
        auto_exposure_limit_us = getattr(self.spec, "auto_exposure_limit_us", None)
        auto_gain_limit = getattr(self.spec, "auto_gain_limit", None)
        warmup_frames = getattr(self.spec, "warmup_frames", 8)
        for sensor in selected_sensors:
            if exposure_us is not None:
                self._set_sensor_option(sensor, option, 0.0, "automatic exposure")
                self._set_sensor_option(
                    sensor,
                    self._rs.option.exposure,
                    float(exposure_us),
                    "manual exposure",
                    required=True,
                )
            else:
                if auto_exposure_limit_us is not None:
                    self._set_sensor_option(
                        sensor,
                        getattr(self._rs.option, "auto_exposure_limit_toggle", None),
                        1.0,
                        "automatic-exposure limit toggle",
                    )
                    self._set_sensor_option(
                        sensor,
                        getattr(self._rs.option, "auto_exposure_limit", None),
                        float(auto_exposure_limit_us),
                        "automatic-exposure limit",
                    )
                if auto_gain_limit is not None:
                    self._set_sensor_option(
                        sensor,
                        getattr(self._rs.option, "auto_gain_limit_toggle", None),
                        1.0,
                        "automatic-gain limit toggle",
                    )
                    self._set_sensor_option(
                        sensor,
                        getattr(self._rs.option, "auto_gain_limit", None),
                        float(auto_gain_limit),
                        "automatic-gain limit",
                    )
                self._set_sensor_option(sensor, option, 1.0, "automatic exposure")
        if exposure_us is None:
            logger.info(
                "Enabled automatic exposure for D405 camera %s on %d sensor(s), "
                "exposure_limit=%sus gain_limit=%s warmup_frames=%d",
                self.spec.image_key,
                len(selected_sensors),
                auto_exposure_limit_us,
                auto_gain_limit,
                warmup_frames,
            )
        else:
            logger.info(
                "Set fixed exposure for D405 camera %s on %d sensor(s): %dus",
                self.spec.image_key,
                len(selected_sensors),
                exposure_us,
            )

    def _set_sensor_option(
        self,
        sensor: Any,
        option: Any,
        value: float,
        label: str,
        *,
        required: bool = False,
    ) -> None:
        if option is None or not sensor.supports(option):
            message = f"D405 {self.spec.image_key} does not support {label}"
            if required:
                raise RuntimeError(message)
            logger.warning(message)
            return
        sensor.set_option(option, value)

    def _sensor_has_color_stream(self, sensor: Any) -> bool:
        """Return whether *sensor* advertises a color stream.

        ``get_stream_profiles`` is not present on a few older SDK wrappers;
        in that case we retain the historical behavior and let the caller use
        any sensor supporting automatic exposure.
        """
        get_profiles = getattr(sensor, "get_stream_profiles", None)
        if get_profiles is None:
            return True
        try:
            profiles = get_profiles()
            color_stream = self._rs.stream.color
            return any(profile.stream_type() == color_stream for profile in profiles)
        except Exception:
            logger.debug("Could not inspect D405 sensor stream profiles", exc_info=True)
            return True

    def _warm_up_auto_exposure(self, pipeline: Any) -> None:
        """Discard initial frames while the sensor converges its exposure."""
        for _ in range(self.spec.warmup_frames):
            if self._stop.is_set():
                return
            pipeline.wait_for_frames(self.spec.timeout_ms)

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
                pipeline_profile = pipeline.start(config)
                self._enable_auto_exposure(pipeline_profile)
                self._warm_up_auto_exposure(pipeline)
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
