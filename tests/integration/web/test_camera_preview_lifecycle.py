import time

import msgpack
import numpy as np
import pytest
import zmq

from core.devices import camera


@pytest.mark.integration
def test_camera_preview_clears_on_disconnect_and_recovers_on_restart():
    publisher = zmq.Context.instance().socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{port}"
    preview = camera.CameraPreview(endpoint, ["left"])
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    payload = msgpack.packb({"left": (1, image.shape, image.dtype.str, image.tobytes())})

    def wait_for(predicate, *, publish=False):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if publish:
                publisher.send(payload)
            if predicate():
                return
            time.sleep(0.02)
        pytest.fail("camera preview did not reach the expected state")

    try:
        wait_for(lambda: preview.get_camera_keys() == ["left"], publish=True)
        publisher.close(linger=0)
        wait_for(lambda: preview.get_camera_keys() == [])
        assert preview.get_camera_frame("left") is None
        publisher = zmq.Context.instance().socket(zmq.PUB)
        publisher.bind(endpoint)
        # Even a reused capture version must populate the cleared cache.
        wait_for(lambda: preview.get_camera_keys() == ["left"], publish=True)
        np.testing.assert_array_equal(preview.get_camera_frame("left"), image)
    finally:
        preview.close()
        publisher.close(linger=0)


