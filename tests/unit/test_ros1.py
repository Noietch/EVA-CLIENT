from __future__ import annotations

import types

import numpy as np
import pytest

import transport.ros1 as ros1
from core.config import ConfigDict
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot

pytestmark = pytest.mark.unit


class _FakeJointState:
    def __init__(self) -> None:
        self.header = types.SimpleNamespace(stamp=None)
        self.name = []
        self.position = []


class _FakeRospy:
    def __init__(self) -> None:
        self.subscribers = []
        self.publishers = {}
        self.Time = types.SimpleNamespace(now=lambda: 1.0)

    def Subscriber(self, topic, msg_type, callback, queue_size, tcp_nodelay):
        self.subscribers.append((topic, msg_type, callback, queue_size, tcp_nodelay))
        return object()

    def Publisher(self, topic, msg_type, queue_size):
        publisher = types.SimpleNamespace(messages=[])
        publisher.publish = publisher.messages.append
        self.publishers[topic] = publisher
        return publisher

    def Rate(self, hz):
        return types.SimpleNamespace(hz=hz)

    def is_shutdown(self):
        return False


def _build_ros1_transport(
    monkeypatch,
    config: ConfigDict,
    robot: Robot,
):
    fake_rospy = _FakeRospy()
    runtime = types.SimpleNamespace(
        rospy=fake_rospy,
        cv_bridge=object(),
        image_type=object,
        joint_state_type=_FakeJointState,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros1, "get_ros_runtime", lambda node_name: runtime)
    return fake_rospy, ros1.Ros1Transport(config, robot)


def _subscription_callback(rospy: _FakeRospy, topic: str):
    for sub_topic, _msg_type, callback, _queue_size, _tcp_nodelay in rospy.subscribers:
        if sub_topic == topic:
            return callback
    raise AssertionError(f"missing subscription for {topic}")


def test_ros1_collection_subscriptions_only_buffer_active_episode(monkeypatch):
    config = ConfigDict(
        transport=ConfigDict(
            node_name="test_ros1_collection",
            topics=ConfigDict(
                camera_topics={"front": "/cam/front"},
                group_topics={
                    "arm": {
                        "state_topic": "/joint_states",
                        "command_topic": "/joint_cmd",
                    }
                },
            ),
        ),
        inference_cfg=ConfigDict(obs_space=types.SimpleNamespace(is_eef=lambda: False)),
        collection=ConfigDict(
            schema=ConfigDict(columns={"qpos": {}}, cameras={"cam_front": {}}),
            transport=ConfigDict(
                ros1=ConfigDict(
                    primary_camera="cam_front",
                    max_frame_skew_sec=0.1,
                    groups=ConfigDict(
                        arm=ConfigDict(
                            qpos_topic="/collection/qpos",
                            eef_topic=None,
                            action_qpos_topic=None,
                            action_eef_topic=None,
                        )
                    ),
                )
            ),
        ),
    )
    robot = Robot(
        name="fake",
        actuator_groups=(ActuatorGroup("arm", 2, ("j0", "j1")),),
        initial_qpos=np.zeros(2, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_front"),),
            state_composition=("arm",),
        ),
    )
    fake_rospy, transport = _build_ros1_transport(monkeypatch, config, robot)
    camera_callbacks = [
        callback for topic, _, callback, _, _ in fake_rospy.subscribers if topic == "/cam/front"
    ]
    assert len(camera_callbacks) == 1
    camera_callback = camera_callbacks[0]
    qpos_callback = _subscription_callback(fake_rospy, "/collection/qpos")
    old_camera = types.SimpleNamespace(header=types.SimpleNamespace(stamp=1.0))
    old_qpos = types.SimpleNamespace(header=types.SimpleNamespace(stamp=1.0))

    camera_callback(old_camera)
    qpos_callback(old_qpos)
    assert not transport._collection_camera_deques["front"]
    assert not transport._collection_qpos_deques["arm"]

    transport.clear_collection_backlog()
    camera_callback(old_camera)
    qpos_callback(old_qpos)
    assert list(transport._collection_camera_deques["front"]) == [old_camera]
    assert list(transport._collection_qpos_deques["arm"]) == [old_qpos]

    transport.finish_collection_capture()
    camera_callback(old_camera)
    qpos_callback(old_qpos)
    assert not transport._collection_camera_deques["front"]
    assert not transport._collection_qpos_deques["arm"]


def test_ros1_hil_relative_relay_reorders_named_input(monkeypatch):
    config = ConfigDict(
        transport=ConfigDict(
            node_name="test_ros1_hil",
            topics=ConfigDict(
                camera_topics={},
                group_topics={
                    "arm": {
                        "state_topic": "/joint_states",
                        "command_topic": "/joint_cmd",
                        "hil_input_topic": "/hil_input",
                    }
                },
            ),
        ),
        inference_cfg=ConfigDict(obs_space=types.SimpleNamespace(is_eef=lambda: False)),
        collection=ConfigDict(
            schema=ConfigDict(columns={}),
            transport=ConfigDict(ros1=ConfigDict(groups={})),
        ),
    )
    robot = Robot(
        name="fake",
        actuator_groups=(ActuatorGroup("arm", 2, ("j0", "j1")),),
        initial_qpos=np.zeros(2, dtype=np.float32),
        observation_schema=ObservationSchema(cameras=(), state_composition=("arm",)),
    )
    fake_rospy, transport = _build_ros1_transport(monkeypatch, config, robot)
    transport._group_state_deques["arm"].append(types.SimpleNamespace(position=[10.0, 20.0]))
    assert transport.start_hil_control("relative").active is True
    callback = _subscription_callback(fake_rospy, "/hil_input")

    callback(types.SimpleNamespace(name=["j1", "j0"], position=[2.0, 1.0]))
    callback(types.SimpleNamespace(name=["j1", "j0"], position=[3.0, 1.5]))

    messages = fake_rospy.publishers["/joint_cmd"].messages
    np.testing.assert_allclose(messages[0].position, [10.0, 20.0])
    np.testing.assert_allclose(messages[1].position, [10.5, 21.0])
    assert transport.stop_hil_control().active is False
