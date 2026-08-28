"""Intel RealSense D405 color-camera cache for the ARX X5 hardware node."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

ARX_X5_CAMERA_KEYS: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ARX_X5_LIGHTING_PROFILES: tuple[str, ...] = ("day", "night")
DEFAULT_ARX_X5_D405_CAMERAS: dict[str, str] = {
    "cam_high": "409122271504",
    "cam_left_wrist": "352122272510",
    "cam_right_wrist": "352122271326",
}
DEFAULT_ARX_X5_D405_PROFILE_PATH = Path(__file__).with_name("d405_profile.yaml")


@dataclasses.dataclass(frozen=True)
class RealSenseColorProfile:
    enable_auto_exposure: bool
    exposure: float
    gain: float
    enable_auto_white_balance: bool
    white_balance: float
    bgr_gains: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class RealSenseCameraSpec:
    image_key: str
    serial: str | None = None
    device_index: int | None = None
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    timeout_ms: int = 1000
    retry_interval_s: float = 5.0
    color_profile: RealSenseColorProfile | None = None


def load_realsense_color_profiles(
    path: Path = DEFAULT_ARX_X5_D405_PROFILE_PATH,
    profile: str = "night",
) -> dict[str, RealSenseColorProfile]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"RealSense profile must be a mapping: {path}")

    if profile not in ARX_X5_LIGHTING_PROFILES:
        raise ValueError(
            f"Unknown RealSense lighting profile {profile!r}; "
            f"expected one of {ARX_X5_LIGHTING_PROFILES}"
        )

    # Accept the original camera-key-only file as a night profile so older
    # deployments can still be rolled back without rewriting their config.
    legacy_format = set(raw) == set(ARX_X5_CAMERA_KEYS)
    if legacy_format:
        if profile != "night":
            raise ValueError(
                f"RealSense profile file {path} only contains the legacy night profile"
            )
        selected = raw
    else:
        unknown = set(raw) - set(ARX_X5_LIGHTING_PROFILES)
        missing = set(ARX_X5_LIGHTING_PROFILES) - set(raw)
        if unknown or missing:
            raise ValueError(
                f"RealSense lighting profiles mismatch: missing={sorted(missing)} "
                f"unknown={sorted(unknown)}"
            )
        selected = raw[profile]
    if not isinstance(selected, dict):
        raise ValueError(f"RealSense profile {profile!r} must be a mapping")
    unknown = set(selected) - set(ARX_X5_CAMERA_KEYS)
    missing = set(ARX_X5_CAMERA_KEYS) - set(selected)
    if unknown or missing:
        raise ValueError(
            f"RealSense profile {profile!r} camera keys mismatch: "
            f"missing={sorted(missing)} unknown={sorted(unknown)}"
        )

    profiles: dict[str, RealSenseColorProfile] = {}
    for image_key in ARX_X5_CAMERA_KEYS:
        values = selected[image_key]
        if not isinstance(values, dict):
            raise ValueError(f"RealSense profile {image_key!r} must be a mapping")
        bgr_gains = tuple(float(value) for value in values["bgr_gains"])
        if len(bgr_gains) != 3 or any(value <= 0 for value in bgr_gains):
            raise ValueError(f"RealSense profile {image_key!r} bgr_gains must be 3 positive values")
        profiles[image_key] = RealSenseColorProfile(
            enable_auto_exposure=bool(values["enable_auto_exposure"]),
            exposure=float(values["exposure"]),
            gain=float(values["gain"]),
            enable_auto_white_balance=bool(values["enable_auto_white_balance"]),
            white_balance=float(values["white_balance"]),
            bgr_gains=bgr_gains,
        )
    return profiles


def parse_resolution(value: str) -> tuple[int, int]:
    width_text, height_text = value.strip().lower().split("x", maxsplit=1)
    width, height = int(width_text), int(height_text)
    if width <= 0 or height <= 0:
        raise ValueError("RealSense resolution dimensions must be positive")
    return width, height


def parse_realsense_camera_specs(
    specs: list[str],
    *,
    resolution: tuple[int, int],
    fps: int,
    timeout_ms: int,
    profile: str = "night",
) -> tuple[RealSenseCameraSpec, ...]:
    color_profiles = load_realsense_color_profiles(profile=profile)
    cameras: list[RealSenseCameraSpec] = []
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"Expected RealSense camera spec KEY=SERIAL_OR_INDEX, got {raw!r}")
        image_key, selector = (part.strip() for part in raw.split("=", 1))
        if image_key not in ARX_X5_CAMERA_KEYS:
            raise ValueError(
                f"Unknown ARX X5 camera key {image_key!r}; expected one of {ARX_X5_CAMERA_KEYS}"
            )
        if not selector:
            raise ValueError(f"Missing RealSense selector in {raw!r}")
        serial: str | None = selector
        device_index: int | None = None
        if selector.startswith("index:"):
            serial = None
            device_index = int(selector.removeprefix("index:"))
        cameras.append(
            RealSenseCameraSpec(
                image_key=image_key,
                serial=serial,
                device_index=device_index,
                resolution=resolution,
                fps=fps,
                timeout_ms=timeout_ms,
                color_profile=color_profiles[image_key],
            )
        )
    return tuple(cameras)


def default_realsense_camera_specs(
    *,
    resolution: tuple[int, int],
    fps: int,
    timeout_ms: int,
    profile: str = "night",
) -> tuple[RealSenseCameraSpec, ...]:
    color_profiles = load_realsense_color_profiles(profile=profile)
    return tuple(
        RealSenseCameraSpec(
            image_key=image_key,
            serial=serial,
            resolution=resolution,
            fps=fps,
            timeout_ms=timeout_ms,
            color_profile=color_profiles[image_key],
        )
        for image_key, serial in DEFAULT_ARX_X5_D405_CAMERAS.items()
    )


def merge_realsense_camera_specs(
    defaults: tuple[RealSenseCameraSpec, ...],
    overrides: tuple[RealSenseCameraSpec, ...],
    *,
    disabled_cameras: tuple[str, ...] = (),
) -> tuple[RealSenseCameraSpec, ...]:
    merged = {spec.image_key: spec for spec in defaults}
    merged.update({spec.image_key: spec for spec in overrides})
    disabled = set(disabled_cameras)
    return tuple(spec for key, spec in merged.items() if key not in disabled)


def list_realsense_devices() -> tuple[dict[str, str], ...]:
    import pyrealsense2 as rs

    devices = rs.context().query_devices()
    return tuple(
        {
            "index": str(index),
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
        }
        for index, device in enumerate(devices)
    )


class _RealSenseCameraWorker:
    def __init__(self, spec: RealSenseCameraSpec) -> None:
        self.spec = spec
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._latest: np.ndarray | None = None
        self._frame_seq = 0
        self._state = "starting"
        self._last_frame_time: float | None = None
        self._bgr_lut = self._build_bgr_lut(spec.color_profile)
        self._thread = threading.Thread(
            target=self._run,
            name=f"realsense-{spec.image_key}",
            daemon=True,
        )
        self._thread.start()

    def snapshot_versioned(self) -> tuple[int, np.ndarray] | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._frame_seq, self._latest.copy()

    def status(self) -> str:
        with self._lock:
            state = self._state
            last_frame_time = self._last_frame_time
        if last_frame_time is None:
            return state
        return f"{state}(age={time.monotonic() - last_frame_time:.1f}s)"

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    @staticmethod
    def _build_bgr_lut(color_profile: RealSenseColorProfile | None) -> np.ndarray | None:
        if color_profile is None:
            return None
        gains = np.asarray(color_profile.bgr_gains, dtype=np.float32)
        values = np.arange(256, dtype=np.float32)[:, np.newaxis] * gains[np.newaxis, :]
        return np.clip(values, 0, 255).astype(np.uint8).reshape(256, 1, 3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_camera_loop()
            except Exception as exc:
                self._set_state("retrying")
                logger.warning(
                    "RealSense camera %s stopped; retrying in %.1f s: %s",
                    self.spec.image_key,
                    self.spec.retry_interval_s,
                    exc,
                )
                self._stop.wait(self.spec.retry_interval_s)

    def _resolve_serial(self, rs: Any) -> str:
        if self.spec.serial is not None:
            return self.spec.serial
        assert self.spec.device_index is not None
        devices = rs.context().query_devices()
        if self.spec.device_index < 0 or self.spec.device_index >= len(devices):
            raise RuntimeError(f"RealSense camera index {self.spec.device_index} is out of range")
        return devices[self.spec.device_index].get_info(rs.camera_info.serial_number)

    def _run_camera_loop(self) -> None:
        import pyrealsense2 as rs

        self._set_state("connecting")
        serial = self._resolve_serial(rs)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        width, height = self.spec.resolution
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.spec.fps)
        pipeline_profile = pipeline.start(config)
        self._apply_color_profile(rs, pipeline_profile)
        self._set_state("online")
        logger.info("Started RealSense camera %s serial=%s", self.spec.image_key, serial)
        try:
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(self.spec.timeout_ms)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data()).copy()
                if self._bgr_lut is not None:
                    image = cv2.LUT(image, self._bgr_lut)
                with self._lock:
                    self._latest = image
                    self._frame_seq += 1
                    self._last_frame_time = time.monotonic()
        finally:
            pipeline.stop()

    def _apply_color_profile(self, rs: Any, pipeline_profile: Any) -> None:
        color_profile = self.spec.color_profile
        if color_profile is None:
            return

        sensor = next(
            (
                candidate
                for candidate in pipeline_profile.get_device().query_sensors()
                if candidate.supports(rs.option.exposure)
            ),
            None,
        )
        if sensor is None:
            raise RuntimeError(f"RealSense camera {self.spec.image_key} has no exposure sensor")

        options = (
            (rs.option.enable_auto_exposure, float(color_profile.enable_auto_exposure)),
            (rs.option.exposure, color_profile.exposure),
            (rs.option.gain, color_profile.gain),
            (
                rs.option.enable_auto_white_balance,
                float(color_profile.enable_auto_white_balance),
            ),
            (rs.option.white_balance, color_profile.white_balance),
        )
        for option, value in options:
            if sensor.supports(option):
                sensor.set_option(option, value)

        logger.info(
            "Loaded D405 profile %s: auto_exposure=%s exposure=%.0f gain=%.0f "
            "auto_white_balance=%s white_balance=%.0f bgr_gains=%s",
            self.spec.image_key,
            color_profile.enable_auto_exposure,
            color_profile.exposure,
            color_profile.gain,
            color_profile.enable_auto_white_balance,
            color_profile.white_balance,
            color_profile.bgr_gains,
        )


class RealSenseCameraCache:
    def __init__(self, camera_specs: tuple[RealSenseCameraSpec, ...]) -> None:
        self._workers = [_RealSenseCameraWorker(spec) for spec in camera_specs]
        logger.info("Started %d RealSense camera worker(s)", len(self._workers))

    def snapshot(self) -> dict[str, np.ndarray]:
        return self.snapshot_versioned()[1]

    def snapshot_versioned(self) -> tuple[dict[str, int], dict[str, np.ndarray]]:
        seqs: dict[str, int] = {}
        images: dict[str, np.ndarray] = {}
        for worker in self._workers:
            versioned = worker.snapshot_versioned()
            if versioned is not None:
                seqs[worker.spec.image_key], images[worker.spec.image_key] = versioned
        return seqs, images

    def hardware_status(self) -> dict[str, str]:
        return {worker.spec.image_key: worker.status() for worker in self._workers}

    def close(self) -> None:
        for worker in self._workers:
            worker.stop()
