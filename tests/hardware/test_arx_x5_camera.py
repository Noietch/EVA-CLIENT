from __future__ import annotations

import cv2
import numpy as np
import pytest

from examples.hardware.arx_x5.camera import (
    DEFAULT_ARX_X5_D405_CAMERAS,
    DEFAULT_ARX_X5_D405_PROFILE_PATH,
    ARX_X5_CAMERA_KEYS,
    RealSenseCameraSpec,
    _RealSenseCameraWorker,
    default_realsense_camera_specs,
    load_realsense_color_profiles,
    merge_realsense_camera_specs,
    parse_realsense_camera_specs,
    parse_resolution,
)


def test_default_d405_mapping_uses_stable_serials() -> None:
    specs = default_realsense_camera_specs(
        resolution=(640, 480),
        fps=30,
        timeout_ms=1000,
    )

    assert tuple(spec.image_key for spec in specs) == ARX_X5_CAMERA_KEYS
    assert tuple(spec.serial for spec in specs) == tuple(DEFAULT_ARX_X5_D405_CAMERAS.values())
    assert all(spec.color_profile is not None for spec in specs)
    assert all(spec.color_profile.enable_auto_exposure is False for spec in specs)
    assert [spec.color_profile.exposure for spec in specs] == [12000, 14000, 14000]
    assert all(spec.color_profile.gain == 16 for spec in specs)


def test_d405_profile_has_all_fixed_camera_keys() -> None:
    profiles = load_realsense_color_profiles(DEFAULT_ARX_X5_D405_PROFILE_PATH)

    assert tuple(profiles) == ARX_X5_CAMERA_KEYS
    assert all(not profile.enable_auto_white_balance for profile in profiles.values())
    assert [profile.white_balance for profile in profiles.values()] == [4390, 4450, 4410]
    assert all(len(profile.bgr_gains) == 3 for profile in profiles.values())


def test_d405_day_profile_uses_shorter_exposure_than_night() -> None:
    day = load_realsense_color_profiles(profile="day")
    night = load_realsense_color_profiles(profile="night")

    assert [profile.exposure for profile in day.values()] == [6000, 8000, 7000]
    assert all(day[key].exposure < night[key].exposure for key in ARX_X5_CAMERA_KEYS)
    assert [profile.white_balance for profile in day.values()] == [4390, 4450, 4410]


def test_worker_applies_d405_profile_options() -> None:
    class Options:
        enable_auto_exposure = "enable_auto_exposure"
        exposure = "exposure"
        gain = "gain"
        enable_auto_white_balance = "enable_auto_white_balance"
        white_balance = "white_balance"

    class Rs:
        option = Options()

    class Sensor:
        def __init__(self) -> None:
            self.values: dict[str, float] = {}

        def supports(self, _option: str) -> bool:
            return True

        def set_option(self, option: str, value: float) -> None:
            self.values[option] = value

    sensor = Sensor()

    class Device:
        def query_sensors(self) -> list[Sensor]:
            return [sensor]

    class PipelineProfile:
        def get_device(self) -> Device:
            return Device()

    worker = object.__new__(_RealSenseCameraWorker)
    worker.spec = RealSenseCameraSpec(
        image_key="cam_high",
        serial="serial",
        color_profile=load_realsense_color_profiles()["cam_high"],
    )

    worker._apply_color_profile(Rs(), PipelineProfile())

    assert sensor.values == {
        "enable_auto_exposure": 0.0,
        "exposure": 12000.0,
        "gain": 16.0,
        "enable_auto_white_balance": 0.0,
        "white_balance": 4390.0,
    }

    lut = worker._build_bgr_lut(worker.spec.color_profile)
    assert lut is not None
    corrected = cv2.LUT(np.asarray([[[100, 100, 100]]], dtype=np.uint8), lut)
    np.testing.assert_array_equal(corrected, [[[97, 104, 97]]])


def test_serial_override_and_disabled_camera() -> None:
    defaults = default_realsense_camera_specs(
        resolution=(640, 480),
        fps=30,
        timeout_ms=1000,
    )
    overrides = parse_realsense_camera_specs(
        ["cam_high=254723071149"],
        resolution=(1280, 720),
        fps=15,
        timeout_ms=2000,
    )

    specs = merge_realsense_camera_specs(
        defaults,
        overrides,
        disabled_cameras=("cam_right_wrist",),
    )

    assert [spec.image_key for spec in specs] == ["cam_high", "cam_left_wrist"]
    assert specs[0].serial == "254723071149"
    assert specs[0].resolution == (1280, 720)


def test_realsense_spec_validation() -> None:
    assert parse_resolution("640x480") == (640, 480)
    with pytest.raises(ValueError, match="Unknown ARX X5 camera key"):
        parse_realsense_camera_specs(
            ["unknown=index:0"],
            resolution=(640, 480),
            fps=30,
            timeout_ms=1000,
        )
