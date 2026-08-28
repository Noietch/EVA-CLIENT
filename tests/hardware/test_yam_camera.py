from types import SimpleNamespace

import pytest

from examples.hardware.yam.camera import (
    _D405Worker,
    load_camera_profiles,
    parse_camera_specs,
)


def test_d405_worker_enables_auto_exposure_on_supported_sensor() -> None:
    option = object()

    class _Sensor:
        def __init__(self) -> None:
            self.options: list[tuple[object, float]] = []

        def supports(self, candidate: object) -> bool:
            return candidate is option

        def set_option(self, candidate: object, value: float) -> None:
            self.options.append((candidate, value))

    sensor = _Sensor()
    profile = SimpleNamespace(
        get_device=lambda: SimpleNamespace(query_sensors=lambda: [sensor]),
    )
    worker = object.__new__(_D405Worker)
    worker.spec = SimpleNamespace(image_key="cam_high")
    worker._rs = SimpleNamespace(option=SimpleNamespace(enable_auto_exposure=option))

    worker._enable_auto_exposure(profile)

    assert sensor.options == [(option, 1.0)]


def test_d405_worker_ignores_sensor_without_auto_exposure() -> None:
    option = object()

    class _Sensor:
        def __init__(self, supported: bool) -> None:
            self.supported = supported
            self.options: list[tuple[object, float]] = []

        def supports(self, candidate: object) -> bool:
            return self.supported and candidate is option

        def set_option(self, candidate: object, value: float) -> None:
            self.options.append((candidate, value))

    unsupported_sensor = _Sensor(False)
    imaging_sensor = _Sensor(True)
    profile = SimpleNamespace(
        get_device=lambda: SimpleNamespace(
            query_sensors=lambda: [unsupported_sensor, imaging_sensor]
        ),
    )
    worker = object.__new__(_D405Worker)
    worker.spec = SimpleNamespace(image_key="cam_high")
    worker._rs = SimpleNamespace(option=SimpleNamespace(enable_auto_exposure=option))

    worker._enable_auto_exposure(profile)

    assert unsupported_sensor.options == []
    assert imaging_sensor.options == [(option, 1.0)]


def test_d405_worker_caps_auto_exposure_when_supported() -> None:
    auto_option = object()
    limit_option = object()
    limit_toggle_option = object()
    gain_limit_option = object()
    gain_toggle_option = object()

    class _Sensor:
        def __init__(self) -> None:
            self.options: list[tuple[object, float]] = []

        def supports(self, candidate: object) -> bool:
            return candidate in {
                auto_option,
                limit_option,
                limit_toggle_option,
                gain_limit_option,
                gain_toggle_option,
            }

        def set_option(self, candidate: object, value: float) -> None:
            self.options.append((candidate, value))

    sensor = _Sensor()
    profile = SimpleNamespace(get_device=lambda: SimpleNamespace(query_sensors=lambda: [sensor]))
    worker = object.__new__(_D405Worker)
    worker.spec = SimpleNamespace(
        image_key="cam_high",
        exposure_us=None,
        auto_exposure_limit_us=8000,
        auto_gain_limit=64,
        warmup_frames=90,
    )
    worker._rs = SimpleNamespace(
        option=SimpleNamespace(
            enable_auto_exposure=auto_option,
            auto_exposure_limit=limit_option,
            auto_exposure_limit_toggle=limit_toggle_option,
            auto_gain_limit=gain_limit_option,
            auto_gain_limit_toggle=gain_toggle_option,
        )
    )

    worker._enable_auto_exposure(profile)

    assert sensor.options == [
        (limit_toggle_option, 1.0),
        (limit_option, 8000.0),
        (gain_toggle_option, 1.0),
        (gain_limit_option, 64.0),
        (auto_option, 1.0),
    ]


def test_d405_worker_uses_fixed_exposure_when_requested() -> None:
    auto_option = object()
    exposure_option = object()

    class _Sensor:
        def __init__(self) -> None:
            self.options: list[tuple[object, float]] = []

        def supports(self, candidate: object) -> bool:
            return candidate in {auto_option, exposure_option}

        def set_option(self, candidate: object, value: float) -> None:
            self.options.append((candidate, value))

    sensor = _Sensor()
    profile = SimpleNamespace(get_device=lambda: SimpleNamespace(query_sensors=lambda: [sensor]))
    worker = object.__new__(_D405Worker)
    worker.spec = SimpleNamespace(
        image_key="cam_high",
        exposure_us=6000,
        auto_exposure_limit_us=None,
    )
    worker._rs = SimpleNamespace(
        option=SimpleNamespace(enable_auto_exposure=auto_option, exposure=exposure_option)
    )

    worker._enable_auto_exposure(profile)

    assert sensor.options == [(auto_option, 0.0), (exposure_option, 6000.0)]


def test_load_camera_profile_applies_to_matching_camera(tmp_path) -> None:
    profile_path = tmp_path / "d405.json"
    profile_path.write_text(
        """{
            "version": 1,
            "cameras": {
                "cam_high": {
                    "serial": "260422275306",
                    "auto_exposure_limit_us": 33000,
                    "auto_gain_limit": 64,
                    "warmup_frames": 90
                }
            }
        }"""
    )

    profiles = load_camera_profiles(profile_path)
    specs = parse_camera_specs(
        ["cam_high=260422275306"],
        camera_profiles=profiles,
    )

    assert specs[0].auto_exposure_limit_us == 33000
    assert specs[0].auto_gain_limit == 64
    assert specs[0].warmup_frames == 90
    assert specs[0].exposure_us is None


def test_camera_profile_rejects_wrong_serial(tmp_path) -> None:
    profile_path = tmp_path / "d405.json"
    profile_path.write_text('{"version": 1, "cameras": {"cam_high": {"serial": "expected"}}}')

    with pytest.raises(ValueError, match="expects serial expected"):
        parse_camera_specs(
            ["cam_high=actual"],
            camera_profiles=load_camera_profiles(profile_path),
        )
