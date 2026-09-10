import threading
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from core.devices import camera
from examples.hardware.yam.camera import _D405Worker
from examples.hardware.yam.orbbec_camera import _OrbbecCameraWorker

pytestmark = pytest.mark.unit


def test_republished_cached_frame_does_not_refresh_liveness(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(camera.time, "monotonic", lambda: now[0])
    version = [1]
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    source = camera.CameraSource.__new__(camera.CameraSource)
    source.images, source.versions, source.received_at = {}, {}, {}
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
    assert source.snapshot() == {}
    version[0] += 1  # New capture with identical pixels is healthy.
    assert "left" in source.snapshot()


def test_preview_health_is_per_camera_and_recovers(monkeypatch):
    monkeypatch.setattr(camera.time, "monotonic", lambda: 10.0)
    preview = camera.CameraPreview.__new__(camera.CameraPreview)
    preview.lock = threading.Lock()
    preview.keys = ("left", "right", "front")
    preview.received_at = {"left": 9.0, "right": 9.9}
    health = preview.camera_health()
    assert health["left"]["stale"] and health["front"]["stale"]
    assert not health["right"]["stale"]
    preview.received_at["left"] = 10.0
    assert not preview.camera_health()["left"]["stale"]


def test_yam_capture_versions_only_change_on_capture(monkeypatch):
    monkeypatch.setattr(camera.time, "monotonic", lambda: 10.0)
    worker = _D405Worker.__new__(_D405Worker)
    worker._lock = threading.Lock()
    worker._latest = np.zeros((2, 2, 3), dtype=np.uint8)
    worker._last_frame_time = 9.5
    assert worker.snapshot_versioned()[0] == worker.snapshot_versioned()[0] == 9.5
    worker._last_frame_time = 9.0
    assert worker.snapshot_versioned() is None

    worker = _OrbbecCameraWorker.__new__(_OrbbecCameraWorker)
    worker._frame_lock = threading.Lock()
    worker._frame_buffer = bytearray(12)
    worker.spec = SimpleNamespace(height=2, width=2)
    worker._frame_count = SimpleNamespace(value=3)
    worker._last_frame_time = SimpleNamespace(value=9.5)
    assert worker.snapshot_versioned()[0] == worker.snapshot_versioned()[0] == 3
    worker._last_frame_time.value = 9.0
    assert worker.snapshot_versioned() is None


def test_yam_node_publishes_each_capture_once():
    from examples.hardware.yam.node import YamZmqNode

    version = [1]
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    cache = SimpleNamespace(snapshot_versioned=lambda: ({"left": version[0]}, {"left": image}))
    node = SimpleNamespace(_camera_caches=(cache,), _camera_versions={})
    assert "left" in YamZmqNode._camera_snapshot(node)
    assert YamZmqNode._camera_snapshot(node) == {}
    version[0] += 1
    assert "left" in YamZmqNode._camera_snapshot(node)
