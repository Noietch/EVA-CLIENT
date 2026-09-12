import pytest
import zmq

import robots  # noqa: F401
from core.config import ConfigDict
from core.registry import ROBOT_REGISTRY
from transport.zmq import WireObservation, _ObservationReader, pack_observation

pytestmark = pytest.mark.integration


def test_new_episode_discards_old_zmq_pipe(tmp_path):
    publisher = zmq.Context.instance().socket(zmq.XPUB)
    publisher.setsockopt(zmq.XPUB_VERBOSE, 1)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    robot = ROBOT_REGISTRY.build("agilex_piper")
    reader = _ObservationReader(
        ConfigDict(
            transport=ConfigDict(
                sub_endpoint=f"tcp://127.0.0.1:{port}", disabled_cameras=[], disabled_groups=[]
            )
        ),
        robot,
        zmq,
        preserve_collection_backlog=True,
    )

    def wait_for_subscription():
        while True:
            assert publisher.poll(2000), "subscriber did not connect"
            if publisher.recv()[0] == 1:
                return

    def publish(timestamp):
        publisher.send(pack_observation(WireObservation(t=timestamp, images={}, state={})))

    try:
        wait_for_subscription()
        for timestamp in range(64):
            publish(float(timestamp))
        assert reader._sub.poll(2000)
        old_socket = reader._sub
        assert reader.clear_collection_backlog() is None
        assert old_socket.closed
        reader.prepare_collection_capture(str(tmp_path))
        wait_for_subscription()
        publish(200.0)
        assert reader._sub.poll(2000)
        snapshot = reader.acquire_collection_raw()
        assert snapshot is not None
        assert snapshot.timestamp == 200.0
        assert reader.acquire_collection_raw() is None
    finally:
        reader.close()
        publisher.close(linger=0)


def test_camera_cached_frame_remains_available_after_one_second(monkeypatch):
    import numpy as np

    from transport import zmq as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    reader = _ObservationReader(
        ConfigDict(
            transport=ConfigDict(
                sub_endpoint="inproc://camera-health-test", disabled_cameras=[], disabled_groups=[]
            )
        ),
        ROBOT_REGISTRY.build("agilex_piper"),
        zmq,
    )
    try:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        observation = WireObservation(t=1, images={"cam_high": image}, state={})
        reader._latest = observation
        reader._cache_images(observation)
        now[0] = 11.0
        reader.get_camera_frame("cam_high")
        reader.get_camera_keys()
        np.testing.assert_array_equal(reader.get_camera_frame("cam_high"), image)
    finally:
        reader.close()
