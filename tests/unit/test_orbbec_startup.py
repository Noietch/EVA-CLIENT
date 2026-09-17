from types import SimpleNamespace

import pytest

from examples.hardware.yam import orbbec_camera as camera

pytestmark = pytest.mark.unit


def test_camera_start_waits_for_first_frame_before_opening_next(monkeypatch):
    states = [SimpleNamespace(value=0) for _ in range(3)]
    specs = [camera.OrbbecCameraSpec(f"cam_{i}", serial=str(i)) for i in range(3)]
    capture_args = tuple(
        (spec, None, None, None, None, state) for spec, state in zip(specs, states, strict=True)
    )
    started = []
    waits = []

    class Stop:
        def is_set(self):
            return False

        def wait(self, timeout):
            waits.append(timeout)
            if timeout == 0.05:
                states[len(started) - 1].value = camera._STATE_NAMES.index("online")
            # End the sidecar's idle loop after startup.
            return timeout == 0.2

    class Thread:
        def __init__(self, *, target, args, **kwargs):
            self.args = args

        def start(self):
            started.append(self.args[0].image_key)
            self.args[5].value = camera._STATE_NAMES.index("warming")

        def join(self, timeout):
            pass

    monkeypatch.setattr(camera.signal, "signal", lambda *args: None)
    monkeypatch.setattr(camera.cv2, "setNumThreads", lambda *args: None)
    monkeypatch.setattr(camera.os, "nice", lambda *args: None)
    monkeypatch.setattr(camera.threading, "Thread", Thread)
    monkeypatch.setattr(
        camera,
        "get_orbbec_sdk",
        lambda: SimpleNamespace(Context=lambda: SimpleNamespace(query_devices=lambda: [])),
    )
    monkeypatch.setattr(
        camera,
        "select_orbbec_device",
        lambda devices, spec: SimpleNamespace(
            get_device_info=lambda: SimpleNamespace(get_serial_number=lambda: spec.serial)
        ),
    )

    camera._capture_orbbec_cameras(capture_args, Stop())

    assert started == [spec.image_key for spec in specs]
    assert waits == [0.05, 0.05, 0.2]


def test_startup_reset_only_targets_enabled_devices():
    reboots = []
    devices = SimpleNamespace(
        get_device_by_serial_number=lambda serial: SimpleNamespace(
            reboot=lambda: reboots.append(serial)
        )
    )
    sdk = SimpleNamespace(Context=lambda: SimpleNamespace(query_devices=lambda: devices))
    specs = (
        camera.OrbbecCameraSpec("cam_left_wrist", serial="left", reset_on_start=True),
        camera.OrbbecCameraSpec("cam_right_wrist", serial="right"),
    )
    assert camera._reset_startup_devices(sdk, specs)
    assert reboots == ["left"]
    assert not camera._reset_startup_devices(sdk, specs[1:])
    assert reboots == ["left"]
