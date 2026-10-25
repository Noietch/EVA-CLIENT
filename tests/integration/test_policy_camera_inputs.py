"""Verify complete multi-camera requests over the policy clients' wire protocols."""

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

import msgpack_numpy
import numpy as np
import pytest
import zmq
from openpi_client import msgpack_numpy as ws_msgpack
from websockets.sync.server import serve

from core.config import ConfigDict
from policy_client.base import PolicyBuildContext
from policy_client.gr00t import Gr00tPolicyClient
from policy_client.openpi import OpenPiPolicyClient, RtcOpenPiPolicyClient
from policy_client.starvla import StarVlaPolicyClient

pytestmark = pytest.mark.integration


@pytest.fixture
def observation():
    return {
        "images": {
            name: np.full((8, 10, 3), index, dtype=np.uint8)
            for index, name in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist"))
        },
        "state": np.arange(14, dtype=np.float32),
        "prompt": "stack the bowls",
    }


@pytest.mark.parametrize(
    "client_type", [OpenPiPolicyClient, RtcOpenPiPolicyClient, StarVlaPolicyClient]
)
def test_websocket_clients_send_all_camera_images(client_type, observation):
    requests = []
    actions = np.zeros((2, 14), dtype=np.float32)

    def respond(connection):
        connection.send(ws_msgpack.packb({"action_chunk_size": 2}))
        for payload in connection:
            requests.append(ws_msgpack.unpackb(payload))
            response = {"actions": actions}
            if client_type is StarVlaPolicyClient:
                response = {"status": "ok", "data": {"actions": actions[None]}}
            connection.send(ws_msgpack.packb(response))

    with serve(respond, "127.0.0.1", 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = client_type.from_config(
                ConfigDict(
                    host="127.0.0.1",
                    port=server.socket.getsockname()[1],
                    backend_options={"unnorm_key": "dual_yam"},
                ),
                PolicyBuildContext(),
            )
            try:
                for _ in range(2):
                    np.testing.assert_array_equal(client.infer(observation)["actions"], actions)
            finally:
                client._ws.close()
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert len(requests) == 2
    for request in requests:
        if client_type is StarVlaPolicyClient:
            payload = request["payload"]
            assert payload["unnorm_key"] == "dual_yam"
            assert len(payload["examples"]) == 1
            example = payload["examples"][0]
            images = example["image"]
            assert example["lang"] == observation["prompt"]
        else:
            assert list(request["images"]) == list(observation["images"])
            images = list(request["images"].values())
        assert len(images) == 3
        for actual, expected in zip(images, observation["images"].values(), strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("camera_count", [0, 2, 3])
def test_gr00t_sends_all_images_or_rejects_missing_mapping(observation, camera_count):
    ports = Queue()
    video_keys = {name: f"video.{name}" for name in list(observation["images"])[:camera_count]}

    def respond():
        requests = []
        with zmq.Context() as context, context.socket(zmq.REP) as socket:
            socket.setsockopt(zmq.RCVTIMEO, 5000)
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            ports.put(port)
            for _ in range(2 if camera_count == 3 else 1):
                request = msgpack_numpy.unpackb(socket.recv())
                requests.append(request)
                response = {"video": list(video_keys.values())}
                if request["endpoint"] == "get_action":
                    response = {"action.qpos": np.zeros((2, 14), dtype=np.float32)}
                socket.send(msgpack_numpy.packb(response))
        return requests

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(respond)
        client = Gr00tPolicyClient(
            "127.0.0.1", ports.get(timeout=5), video_keys=video_keys, timeout_ms=5000
        )
        try:
            if camera_count == 3:
                assert client.infer(observation)["actions"].shape == (2, 14)
            else:
                missing_camera = list(observation["images"])[camera_count]
                with pytest.raises(KeyError, match=missing_camera):
                    client.infer(observation)
        finally:
            client._socket.close(linger=0)
            client._context.term()
        requests = future.result(timeout=5)

    if camera_count == 3:
        request = requests[-1]["data"]["observation"]
        assert {key for key in request if key.startswith("video.")} == set(video_keys.values())
        for name, image in observation["images"].items():
            np.testing.assert_array_equal(request[video_keys[name]], image[None, None])
    else:
        assert [request["endpoint"] for request in requests] == ["get_modality_config"]
