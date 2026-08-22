from types import SimpleNamespace

from examples.hardware.i2rt.camera import _D405Worker


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
