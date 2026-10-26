"""Verify complete multi-camera requests over the policy clients' wire protocols."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

import msgpack
import msgpack_numpy
import numpy as np
import pytest
import zmq
from openpi_client import msgpack_numpy as ws_msgpack
from websockets.sync.server import serve

from core.app.handlers.recording import build_policy_observation
from core.app.state import RuntimeState
from core.config import ConfigDict, load_config
from core.types import Observation
from policy_client.base import PolicyBuildContext
from policy_client.gr00t import Gr00tPolicyClient
from policy_client.openpi import OpenPiPolicyClient, RtcOpenPiPolicyClient
from policy_client.starvla import StarVlaPolicyClient
from policy_client.xpolicylab import XPolicyLabPolicyClient
from robots.zoo.dual_yam import DualYam

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


@pytest.mark.parametrize("client_type", [OpenPiPolicyClient, XPolicyLabPolicyClient])
@pytest.mark.parametrize("resize_pad", [False, True])
def test_raw_camera_pixels_reach_policy_after_observation_build(tmp_path, client_type, resize_pad):
    """Preserve native RGB sizes and pixels through preprocessing and real policy sockets."""
    base = Path(__file__).resolve().parents[2] / "configs/01_deploy/dual_yam/openpi_qpos.py"
    path = tmp_path / "raw_camera_config.py"
    path.write_text(
        f"_base_ = [{str(base)!r}]\n"
        "transport = dict(type='zmq', "
        f"resize_pad={resize_pad!r}, "
        "image_height=224, image_width=224)\n"
    )
    config = load_config(path)
    assert config.transport.resize is False
    assert config.transport.image_layout == "hwc"
    assert config.transport.convert_bgr_to_rgb is False
    robot = DualYam()
    runtime = RuntimeState(robot=robot, transport=None)
    assert config.robot.type == robot.name
    assert robot.total_action_dim == 14
    shapes = ((241, 319, 3), (127, 205, 3), (95, 143, 3))
    images = {
        camera.observation_key: (
            np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) + index * 73
        ).astype(np.uint8)
        for index, (camera, shape) in enumerate(
            zip(robot.observation_schema.cameras, shapes, strict=True)
        )
    }
    state = np.arange(14, dtype=np.float32)
    observation = build_policy_observation(
        Observation(
            images={name: image.copy() for name, image in images.items()}, state_qpos=state
        ),
        "stack the bowls",
        config,
        runtime,
    )
    actions = np.zeros((2, 14), dtype=np.float32)
    requests = []

    def respond(connection):
        if client_type is OpenPiPolicyClient:
            connection.send(ws_msgpack.packb({"action_chunk_size": 2}))
        for payload in connection:
            if client_type is OpenPiPolicyClient:
                request = ws_msgpack.unpackb(payload)
                received_images = request["images"]
                np.testing.assert_array_equal(request["state"], state)
                assert request["prompt"] == observation["prompt"]
                response = {"actions": actions}
            else:
                request = XPolicyLabPolicyClient._decode_frame(payload)
                if request["message_type"] == "hello":
                    connection.send(
                        msgpack.packb(
                            {
                                **request,
                                "message_type": "hello_ack",
                                "payload": {"ok": True, "server": "raw-camera-probe"},
                            },
                            use_bin_type=True,
                        )
                    )
                    continue
                assert request["message_type"] == "infer"
                wire_observation = request["payload"]["observation"]
                received_images = {
                    name: value["color"] for name, value in wire_observation["vision"].items()
                }
                wire_state = wire_observation["state"]
                assert list(wire_state) == [
                    "left_arm_joint_state", "left_ee_joint_state",
                    "right_arm_joint_state", "right_ee_joint_state",
                ]
                np.testing.assert_array_equal(np.concatenate(list(wire_state.values())), state)
                assert wire_observation["instruction"] == observation["prompt"]
                response = {
                    **request,
                    "message_type": "infer_result",
                    "payload": {"ok": True, "actions": actions},
                }
            assert list(received_images) == list(images)
            for name, expected in images.items():
                actual = received_images[name]
                assert actual.shape == expected.shape
                assert actual.dtype == np.uint8
                np.testing.assert_array_equal(actual, expected)
            requests.append(received_images)
            if client_type is OpenPiPolicyClient:
                connection.send(ws_msgpack.packb(response))
            else:
                connection.send(
                    msgpack.packb(response, default=msgpack_numpy.encode, use_bin_type=True)
                )

    with serve(respond, "127.0.0.1", 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = client_type.from_config(
                ConfigDict(
                    host="127.0.0.1",
                    port=server.socket.getsockname()[1],
                    backend_options=dict(action_type="joint", arm_dims=[6, 6], gripper_dims=[1, 1]),
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
