import threading
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from core.devices import camera
from examples.hardware.yam.camera import _D405Worker
from examples.hardware.yam.orbbec_camera import _OrbbecCameraWorker

pytestmark = pytest.mark.unit


def test_cached_frame_remains_available_after_one_second(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(camera.time, "monotonic", lambda: now[0])
    version = [1]
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    source = camera.CameraSource.__new__(camera.CameraSource)
    source.images, source.versions = {}, {}
    source.monitor = SimpleNamespace(poll=lambda _: False)
    source.socket = SimpleNamespace(
        poll=lambda _: True,
        recv=lambda: msgpack.packb(
            {"left": (version[0], image.shape, image.dtype.str, image.tobytes())}
        ),
    )
    assert "left" in source.snapshot()
    now[0] += 0.9
    assert "left" in source.snapshot()
    now[0] += 0.1
    np.testing.assert_array_equal(source.snapshot()["left"], image)
    version[0] += 1  # New capture with identical pixels is healthy.
    assert "left" in source.snapshot()


def test_yam_capture_versions_only_change_on_capture(monkeypatch):
    monkeypatch.setattr(camera.time, "monotonic", lambda: 10.0)
    worker = _D405Worker.__new__(_D405Worker)
    worker._lock = threading.Lock()
    worker._latest = np.zeros((2, 2, 3), dtype=np.uint8)
    worker._last_frame_time = 9.5
    assert worker.snapshot_versioned()[0] == worker.snapshot_versioned()[0] == 9.5
    worker._last_frame_time = 9.0
    assert worker.snapshot_versioned() is not None

    worker = _OrbbecCameraWorker.__new__(_OrbbecCameraWorker)
    worker._frame_lock = threading.Lock()
    worker._frame_buffer = bytearray(12)
    worker.spec = SimpleNamespace(height=2, width=2)
    worker._frame_count = SimpleNamespace(value=3)
    worker._last_frame_time = SimpleNamespace(value=9.5)
    assert worker.snapshot_versioned()[0] == worker.snapshot_versioned()[0] == 3
    worker._last_frame_time.value = 9.0
    assert worker.snapshot_versioned() is not None


def test_yam_node_keeps_latest_images_on_each_observation_tick():
    from examples.hardware.yam.node import YamZmqNode

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    frames = {"left": image}
    cache = SimpleNamespace(snapshot=lambda: dict(frames))
    node = SimpleNamespace(_camera_caches=(cache,))
    np.testing.assert_array_equal(YamZmqNode._camera_snapshot(node)["left"], image)
    # The camera has not captured again when the next robot tick arrives.
    np.testing.assert_array_equal(YamZmqNode._camera_snapshot(node)["left"], image)
    frames["left"] = image + 1
    np.testing.assert_array_equal(YamZmqNode._camera_snapshot(node)["left"], image + 1)
    frames.clear()  # A disconnected source must still remove the camera.
    assert YamZmqNode._camera_snapshot(node) == {}
