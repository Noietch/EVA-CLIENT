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
from core.types import Observation
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot
from transport.base import _RosTransportBase
from transport.utils import StreamFreshness

pytestmark = pytest.mark.unit


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


def test_acquire_collection_raw_blocks_concurrent_deque_append_during_scan():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    transport = _FakeRosCollectionTransport(image)
    transport._camera_deques["front"] = deque([transport._msg(float(i)) for i in range(10)])
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
