from __future__ import annotations

import types

import numpy as np
import pytest

import transport.ros1 as ros1
import transport.utils as transport_utils
from core.config import ConfigDict
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot


class _FakeRospy:
    def __init__(self) -> None:
        self.subscribers = []

    def Subscriber(self, topic, msg_type, callback, queue_size, tcp_nodelay):
        self.subscribers.append((topic, msg_type, callback, queue_size, tcp_nodelay))
        return object()

    def Publisher(self, topic, msg_type, queue_size):
        return object()

    def Rate(self, hz):
        return types.SimpleNamespace(hz=hz)

    def is_shutdown(self):
        return False


def _subscription_callback(rospy: _FakeRospy, topic: str):
    for sub_topic, _msg_type, callback, _queue_size, _tcp_nodelay in rospy.subscribers:
        if sub_topic == topic:
            return callback
    raise AssertionError(f"missing subscription for {topic}")


def test_ros1_camera_subscriptions_report_minimum_image_hz(monkeypatch):
    fake_rospy = _FakeRospy()
    runtime = types.SimpleNamespace(
        rospy=fake_rospy,
        cv_bridge=object(),
        image_type=object,
        joint_state_type=object,
        pose_stamped_type=object,
    )
    monkeypatch.setattr(ros1, "get_ros_runtime", lambda node_name: runtime)
    now = [0.0]
    monkeypatch.setattr(transport_utils.time, "monotonic", lambda: now[0])
    config = ConfigDict(
        transport=ConfigDict(
            node_name="test_ros1",
            topics=ConfigDict(
                camera_topics={"front": "/cam/front", "left": "/cam/left"},
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
            schema=ConfigDict(columns={}),
            transport=ConfigDict(ros1=ConfigDict(groups={})),
        ),
    )
    robot = Robot(
        name="fake",
        actuator_groups=(ActuatorGroup("arm", 2, ("j0", "j1")),),
        initial_qpos=np.zeros(2, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_front"), CameraSpec("left", "cam_left")),
            state_composition=("arm",),
        ),
    )
    transport = ros1.Ros1Transport(config, robot)

    for topic in ("/cam/front", "/cam/left"):
        now[0] = 0.0
        _subscription_callback(fake_rospy, topic)(object())
    now[0] = 0.05
    _subscription_callback(fake_rospy, "/cam/front")(object())
    now[0] = 0.10
    _subscription_callback(fake_rospy, "/cam/left")(object())

    assert transport.image_min_hz() == pytest.approx(10.0)
