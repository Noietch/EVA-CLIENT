from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESET_TORSO_SCRIPT = REPO_ROOT / "examples" / "hardware" / "r1_lite" / "reset_torso.py"


class FakeJointState:
    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None)
        self.name: list[str] = []
        self.position: list[float] = []


class FakeNow:
    def __init__(self, stamp: str) -> None:
        self.stamp = stamp

    def to_msg(self) -> str:
        return self.stamp


class FakeClock:
    def __init__(self) -> None:
        self.count = 0

    def now(self) -> FakeNow:
        self.count += 1
        return FakeNow(f"stamp-{self.count}")


class FakePublisher:
    def __init__(self, msg_type: type[FakeJointState], topic: str, queue_size: int) -> None:
        self.msg_type = msg_type
        self.topic = topic
        self.queue_size = queue_size
        self.messages: list[FakeJointState] = []

    def publish(self, message: FakeJointState) -> None:
        self.messages.append(message)


class FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.clock = FakeClock()
        self.publisher: FakePublisher | None = None
        self.destroyed = False

    def create_publisher(
        self,
        msg_type: type[FakeJointState],
        topic: str,
        queue_size: int,
    ) -> FakePublisher:
        self.publisher = FakePublisher(msg_type, topic, queue_size)
        return self.publisher

    def create_rate(self, hz: float) -> None:
        raise AssertionError("reset_torso must not use rclpy Rate.sleep without an executor")

    def get_clock(self) -> FakeClock:
        return self.clock

    def destroy_node(self) -> None:
        self.destroyed = True


class FakeRclpy(ModuleType):
    def __init__(self) -> None:
        super().__init__("rclpy")
        self.init_count = 0
        self.shutdown_count = 0
        self.nodes: list[FakeNode] = []

    def init(self) -> None:
        self.init_count += 1

    def create_node(self, name: str) -> FakeNode:
        node = FakeNode(name)
        self.nodes.append(node)
        return node

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture
def reset_torso_module(monkeypatch: pytest.MonkeyPatch):
    fake_rclpy = FakeRclpy()
    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = FakeJointState

    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)

    spec = importlib.util.spec_from_file_location("r1lite_reset_torso", RESET_TORSO_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sleep_calls: list[float] = []
    monkeypatch.setattr(module.time, "sleep", sleep_calls.append)
    return module, fake_rclpy, sleep_calls


def test_reset_torso_publishes_fixed_pose_100_times(reset_torso_module) -> None:
    module, fake_rclpy, sleep_calls = reset_torso_module

    module.main()

    assert fake_rclpy.init_count == 1
    assert fake_rclpy.shutdown_count == 1
    assert len(fake_rclpy.nodes) == 1
    node = fake_rclpy.nodes[0]
    assert node.name == "reset_r1_lite_torso"
    assert node.publisher is not None
    assert node.publisher.topic == "/motion_target/target_joint_state_torso"
    assert node.publisher.msg_type is FakeJointState
    assert node.publisher.queue_size == 10
    assert sleep_calls == [1.0 / 30.0] * 100
    assert node.destroyed

    messages = node.publisher.messages
    assert len(messages) == 100
    for index, message in enumerate(messages, start=1):
        assert message.header.stamp == f"stamp-{index}"
        assert message.name == ["torso_joint1", "torso_joint2", "torso_joint3"]
        assert message.position == [-0.657, 1.516, 0.845]
