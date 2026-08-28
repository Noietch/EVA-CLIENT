from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from examples.hardware.franka.node import FrankaZmqNode
from transport.zmq import unpack_observation


def test_franka_publishes_state_at_full_rate_and_each_camera_frame_once() -> None:
    node = object.__new__(FrankaZmqNode)
    state = {
        "left_arm": np.arange(8, dtype=np.float32),
        "right_arm": np.arange(8, 16, dtype=np.float32),
    }
    frame = np.full((8, 12, 3), 127, dtype=np.uint8)
    node._robot = SimpleNamespace(read_state=lambda: state)
    node._cameras = SimpleNamespace(
        snapshot_versioned=lambda: ({"cam_high": 1}, {"cam_high": frame})
    )
    node._last_published_camera_seqs = {}
    node._published_observations = 0
    published: list[bytes] = []
    node._obs_pub = SimpleNamespace(send=published.append)

    node._publish_observation()
    node._publish_observation()

    first = unpack_observation(published[0])
    second = unpack_observation(published[1])
    np.testing.assert_array_equal(first.images["cam_high"], frame)
    np.testing.assert_array_equal(second.state["left_arm"], state["left_arm"])
    assert second.images == {}
    assert node._published_observations == 2
