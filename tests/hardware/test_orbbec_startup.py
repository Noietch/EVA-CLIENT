from types import SimpleNamespace

import pytest

from examples.hardware.yam import orbbec_camera as camera

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("state_name", ["warming", "online"])
def test_camera_start_overlaps_warmup(monkeypatch, state_name):
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
            # End the sidecar's idle loop after startup.
            return timeout == 0.2

    class Thread:
        def __init__(self, *, target, args, **kwargs):
            self.args = args

        def start(self):
            started.append(self.args[0].image_key)
            self.args[5].value = camera._STATE_NAMES.index(state_name)

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
    assert waits == [0.2]
