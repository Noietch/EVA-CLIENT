from __future__ import annotations

import threading
import time
import types
from collections import deque
from typing import Any

import numpy as np
import pytest

import transport.utils as transport_utils
from core.config import ConfigDict
from core.types import CollectionRawImage, Observation
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot
from transport.base import _RosTransportBase
from transport.utils import StreamFreshness


class _FakeRosCollectionTransport(_RosTransportBase):
    def __init__(self, image: np.ndarray) -> None:
        self._image = image
        self._config = ConfigDict(
            collection=ConfigDict(
                schema=ConfigDict(
                    columns={"qpos": "observation.qpos"},
                    cameras={"cam_high": "observation.images.cam_high"},
                ),
            ),
        )
        self._robot = Robot(
            name="fake",
            actuator_groups=(ActuatorGroup("arm", 2, ("j0", "j1")),),
            initial_qpos=np.zeros(2, dtype=np.float32),
            observation_schema=ObservationSchema(
                cameras=(CameraSpec("front", "cam_high"),),
                state_composition=("arm",),
            ),
        )
        self._bridge = object()
        self._freshness = StreamFreshness()
        self._image_rate = transport_utils.ImageRateTracker()
        self._deque_lock = threading.RLock()
        self._camera_deques = {"front": deque([self._msg(1.0)])}
        self._collection_qpos_deques = {"arm": deque([self._msg(1.0, [1.0, 2.0])])}
        self._collection_eef_deques = {}
        self._collection_action_qpos_deques = {}
        self._collection_action_eef_deques = {}
        self._last_acquire_stamp = None
        self._stamp_calls = 0
        self._stamp_hook = None

    @staticmethod
    def _msg(t: float, position: list[float] | None = None) -> Any:
        return types.SimpleNamespace(t=t, position=[] if position is None else position)

    @property
    def _collection_cfg(self) -> Any:
        return ConfigDict(primary_camera="cam_high", max_frame_skew_sec=0.0)

    def _stamp_to_sec(self, msg: Any) -> float:
        self._stamp_calls += 1
        if self._stamp_hook is not None:
            self._stamp_hook(self, msg)
        return float(msg.t)

    def _decode_image_msg(self, camera_name: str, msg: Any) -> np.ndarray | None:
        _ = camera_name, msg
        return self._image

    def _eef_from_msg(self, group: Any, msg: Any, frame_time: float) -> np.ndarray:
        _ = group, msg, frame_time
        return np.zeros(0, dtype=np.float32)

    def get_frame(self) -> Observation | None:
        return None

    def publish_action(self, action: np.ndarray) -> None:
        _ = action

    def close(self) -> None:
        pass


def test_raw_collection_snapshot_has_no_decoded_frame_path():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    transport = _FakeRosCollectionTransport(image)

    snapshot = transport.acquire_collection_raw()
    assert snapshot is not None
    assert not hasattr(snapshot, "decode")


def test_raw_collection_snapshot_exposes_raw_stream_batch():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    transport = _FakeRosCollectionTransport(image)
    transport._camera_deques["front"].append(transport._msg(1.1))
    transport._collection_qpos_deques["arm"].append(transport._msg(1.1, [2.0, 3.0]))

    snapshot = transport.acquire_collection_raw()
    assert snapshot is not None
    batch = snapshot.decode_raw()

    assert [sample.timestamp for sample in batch.images["cam_high"]] == [1.0, 1.1]
    assert [sample.timestamp for sample in batch.vectors["state_qpos:arm"]] == [1.0, 1.1]
    np.testing.assert_allclose(batch.vectors["state_qpos:arm"][1].value, [2.0, 3.0])


def test_raw_collection_snapshot_defers_image_decode_until_alignment():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    transport = _FakeRosCollectionTransport(image)
    transport._camera_deques["front"].append(transport._msg(1.1))
    transport._collection_qpos_deques["arm"].append(transport._msg(1.1, [2.0, 3.0]))
    decoded = 0

    def decode(camera_name: str, msg: Any) -> np.ndarray:
        nonlocal decoded
        _ = camera_name, msg
        decoded += 1
        return image

    transport._decode_image_msg = decode

    snapshot = transport.acquire_collection_raw()
    assert snapshot is not None
    batch = snapshot.decode_raw()

    assert decoded == 0
    value = batch.images["cam_high"][0].value
    assert isinstance(value, CollectionRawImage)
    assert value.decode() is image
    assert decoded == 1
    assert value.decode() is image
    assert decoded == 1


def test_raw_collection_snapshot_defers_vector_conversion_until_save():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    transport = _FakeRosCollectionTransport(image)
    converted = 0
    convert = transport._collection_raw_batch_from_msgs

    def record_conversion(new_messages, pinned_streams):
        nonlocal converted
        converted += 1
        return convert(new_messages, pinned_streams)

    transport._collection_raw_batch_from_msgs = record_conversion

    snapshot = transport.acquire_collection_raw()

    assert snapshot is not None
    assert converted == 0
    first = snapshot.decode_raw()
    second = snapshot.decode_raw()
    assert converted == 1
    assert second is first


def test_acquire_collection_raw_scans_only_messages_after_cursor():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    transport = _FakeRosCollectionTransport(image)
    old_images = [transport._msg(float(i)) for i in range(2000)]
    old_qpos = [transport._msg(float(i), [float(i), float(i + 1)]) for i in range(2000)]
    transport._camera_deques["front"] = deque(
        [*old_images, transport._msg(2000.0), transport._msg(2000.1)]
    )
    transport._collection_qpos_deques["arm"] = deque(
        [
            *old_qpos,
            transport._msg(2000.0, [2000.0, 2001.0]),
            transport._msg(2000.1, [2000.1, 2001.1]),
        ]
    )
    transport._collection_raw_cursors().update(
        {
            "image:cam_high:front": 1999.0,
            "vector:state_qpos:arm": 1999.0,
        }
    )
    transport._stamp_calls = 0

    snapshot = transport.acquire_collection_raw()
    assert snapshot is not None
    batch = snapshot.decode_raw()

    assert transport._stamp_calls < 30
    assert [sample.timestamp for sample in batch.images["cam_high"]] == [2000.0, 2000.1]
    assert [sample.timestamp for sample in batch.vectors["state_qpos:arm"]] == [
        2000.0,
        2000.1,
    ]


def test_acquire_collection_raw_blocks_concurrent_deque_append_during_scan():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    transport = _FakeRosCollectionTransport(image)
    transport._camera_deques["front"] = deque(
        [transport._msg(float(i)) for i in range(10)]
    )
    transport._collection_qpos_deques["arm"] = deque(
        [transport._msg(float(i), [float(i), float(i + 1)]) for i in range(10)]
    )
    transport._collection_raw_cursors().update(
        {
            "image:cam_high:front": 5.0,
            "vector:state_qpos:arm": 5.0,
        }
    )
    started = threading.Event()
    appended = threading.Event()
    launched = False

    def append_during_scan() -> None:
        started.wait(timeout=1.0)
        transport._append_msg(transport._camera_deques["front"], transport._msg(10.0))
        appended.set()

    thread = threading.Thread(target=append_during_scan)
    thread.start()

    def stamp_hook(_transport: _FakeRosCollectionTransport, _msg: Any) -> None:
        nonlocal launched
        if not launched:
            launched = True
            started.set()
            time.sleep(0.05)

    transport._stamp_hook = stamp_hook
    snapshot = transport.acquire_collection_raw()
    thread.join(timeout=1.0)

    assert snapshot is not None
    assert appended.is_set()
    batch = snapshot.decode_raw()
    assert [sample.timestamp for sample in batch.images["cam_high"]] == [6.0, 7.0, 8.0, 9.0]


def test_image_rate_tracker_reports_minimum_camera_hz():
    assert hasattr(transport_utils, "ImageRateTracker")
    tracker = transport_utils.ImageRateTracker()

    tracker.mark("front", 0.0)
    tracker.mark("left", 0.0)
    assert tracker.min_hz(("front", "left")) is None

    tracker.mark("front", 0.05)
    tracker.mark("left", 0.10)

    assert tracker.min_hz(("front", "left")) == pytest.approx(10.0)


def test_ros_camera_append_updates_image_min_hz():
    transport = _FakeRosCollectionTransport(np.zeros((4, 4, 3), dtype=np.uint8))
    transport._camera_deques["left"] = deque()

    transport._append_camera_msg(
        "front", transport._camera_deques["front"], transport._msg(1.0), 0.0
    )
    transport._append_camera_msg(
        "left", transport._camera_deques["left"], transport._msg(1.0), 0.0
    )
    transport._append_camera_msg(
        "front", transport._camera_deques["front"], transport._msg(1.1), 0.05
    )
    transport._append_camera_msg(
        "left", transport._camera_deques["left"], transport._msg(1.2), 0.10
    )

    assert transport.image_min_hz() == pytest.approx(10.0)
