from types import SimpleNamespace

import pytest

from examples.hardware.yam import orbbec_camera as camera

pytestmark = pytest.mark.unit


def test_power_line_frequency_maps_50hz_to_sdk_mode():
    writes = []
    prop = object()
    sdk = SimpleNamespace(
        OBPropertyID=SimpleNamespace(OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT=prop),
        OBPermissionType=SimpleNamespace(PERMISSION_READ_WRITE="rw", PERMISSION_WRITE="write"),
        OBPowerLineFreqMode=SimpleNamespace(
            FREQUENCY_CLOSE=0,
            FREQUENCY_50HZ=1,
            FREQUENCY_60HZ=2,
        ),
    )
    device = SimpleNamespace(
        is_property_supported=lambda candidate, permission: (
            candidate is prop and permission == "rw"
        ),
        set_int_property=lambda candidate, value: writes.append((candidate, value)),
    )

    assert camera.set_color_power_line_frequency(device, sdk, 50) == 50
    assert writes == [(prop, 1)]


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
    monkeypatch.setattr(camera, "get_orbbec_sdk", lambda: SimpleNamespace(
        Context=lambda: SimpleNamespace(query_devices=lambda: [])
    ))
    monkeypatch.setattr(camera, "select_orbbec_device", lambda devices, spec: SimpleNamespace(
        get_device_info=lambda: SimpleNamespace(get_serial_number=lambda: spec.serial)
    ))

    camera._capture_orbbec_cameras(capture_args, Stop())

    assert started == [spec.image_key for spec in specs]
    assert waits == [0.05, 0.05, 0.2]


def test_camera_specs_support_per_camera_startup_timeout():
    specs = camera.parse_orbbec_camera_specs(
        ["cam_left_wrist=left", "cam_right_wrist=right"],
        startup_timeouts={"cam_left_wrist": 3.0},
    )

    assert specs[0].startup_timeout_s == 3.0
    assert specs[1].startup_timeout_s == 8.0


def test_camera_specs_reject_unknown_startup_timeout_key():
    with pytest.raises(ValueError, match="no matching camera"):
        camera.parse_orbbec_camera_specs(
            ["cam_left_wrist=left"],
            startup_timeouts={"cam_right_wrist": 3.0},
        )


def test_camera_uses_fast_retry_until_first_frame(monkeypatch):
    spec = camera.OrbbecCameraSpec("cam_left_wrist", serial="left")
    frame_count = SimpleNamespace(value=0)
    waits = []
    attempts = 0

    class Stop:
        def is_set(self):
            return attempts >= 2

        def wait(self, timeout):
            waits.append(timeout)
            return False

    def fail_start(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("no frame")

    monkeypatch.setattr(camera, "_run_orbbec_capture_loop", fail_start)
    camera._capture_orbbec_camera(
        spec,
        None,
        None,
        frame_count,
        None,
        SimpleNamespace(value=0),
        Stop(),
    )

    assert waits == [0.25, 0.25]
