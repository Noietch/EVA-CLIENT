from __future__ import annotations

from pathlib import Path
import inspect
from types import SimpleNamespace

import numpy as np

from examples.hardware.r1_lite import fake_node

REPO_ROOT = Path(__file__).resolve().parents[2]
R1_LITE_DIR = REPO_ROOT / "examples" / "hardware" / "r1_lite"


def test_run_fake_node_launches_ros2_fake_without_zmq_endpoints():
    text = (R1_LITE_DIR / "run_fake_node.sh").read_text()

    assert "python examples/hardware/r1_lite/fake_node.py" in text
    assert "--role cameras" in text
    assert "--role robot" in text
    assert "--ui-host" in text
    assert "--ui-port" in text
    assert "--image-height" in text
    assert "--image-width" in text
    assert "--obs-endpoint" not in text
    assert "--action-endpoint" not in text
    assert "OBS_ENDPOINT" not in text
    assert "ACTION_ENDPOINT" not in text


def test_r1lite_ros2_fake_uses_real_r1lite_topics():
    assert fake_node.CAMERA_TOPICS == {
        "head": "/hdas/camera_head/right_raw/image_raw_color/compressed",
        "left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
        "right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
    }
    assert fake_node.ARM_TOPICS["left_arm"].feedback == "/hdas/feedback_arm_left"
    assert fake_node.ARM_TOPICS["left_arm"].command == "/motion_target/target_joint_state_arm_left"
    assert fake_node.ARM_TOPICS["left_arm"].hil == "/eva/hil/input_joint_state_arm_left"
    assert fake_node.ARM_TOPICS["right_arm"].gripper_command == (
        "/motion_target/target_position_gripper_right"
    )
    assert fake_node.OPERATOR_BUTTON_TOPIC == "/eva/operator_button"


def test_fake_state_tracks_feedback_and_hil_positions_separately():
    state = fake_node.R1LiteFakeState()

    state.apply_hil_delta("left_arm", joint_index=2, delta=0.25)
    state.apply_arm_command("left_arm", np.arange(6, dtype=np.float32))
    state.apply_gripper_command("left_arm", 0.0)
    snapshot = state.snapshot()

    assert snapshot["hil"]["left_arm"][2] == 0.25
    assert snapshot["feedback"]["left_arm"]["joints"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert snapshot["feedback"]["left_arm"]["gripper"] == 0.0
    assert snapshot["feedback"]["right_arm"]["gripper"] == fake_node.GRIPPER_OPEN


def test_fake_state_accepts_full_group_command_and_splits_gripper():
    state = fake_node.R1LiteFakeState()

    state.apply_arm_command("left_arm", np.array([0, 1, 2, 3, 4, 5, 12], dtype=np.float32))
    snapshot = state.snapshot()

    assert snapshot["feedback"]["left_arm"]["joints"] == [0, 1, 2, 3, 4, 5]
    assert snapshot["feedback"]["left_arm"]["gripper"] == 12.0


def test_control_api_publishes_operator_buttons_and_hil_joint_updates():
    state = fake_node.R1LiteFakeState()
    bridge = _FakeBridge()
    api = fake_node.R1LiteFakeControlApi(state, bridge)

    api.operator_button("x")
    api.gripper("right", "close")
    api.set_joint("left_arm", 1, 0.5)
    api.adjust_joint("left_arm", 1, -0.125)

    assert bridge.operator_buttons == ["x", "right_gripper_close"]
    assert [msg[0] for msg in bridge.hil_messages] == ["left_arm", "left_arm"]
    np.testing.assert_allclose(bridge.hil_messages[-1][1], [0.0, 0.375, 0.0, 0.0, 0.0, 0.0])


def test_control_api_can_publish_direct_motion_target_commands():
    state = fake_node.R1LiteFakeState()
    bridge = _FakeBridge()
    api = fake_node.R1LiteFakeControlApi(state, bridge)

    api.adjust_command("right_arm", 3, -0.2)
    api.adjust_command("right_arm", 3, -0.3)

    snapshot = state.snapshot()
    np.testing.assert_allclose(snapshot["feedback"]["right_arm"]["joints"], [0, 0, 0, -0.5, 0, 0])
    assert [msg[0] for msg in bridge.command_messages] == ["right_arm", "right_arm"]
    np.testing.assert_allclose(bridge.command_messages[-1][1], [0, 0, 0, -0.5, 0, 0])


def test_control_api_can_sync_hil_to_feedback_without_eva_web_commands():
    state = fake_node.R1LiteFakeState()
    bridge = _FakeBridge()
    api = fake_node.R1LiteFakeControlApi(state, bridge)
    state.apply_arm_command("left_arm", np.arange(6, dtype=np.float32))
    state.apply_arm_command("right_arm", np.arange(10, 16, dtype=np.float32))

    result = api.sync_hil_to_feedback()
    snapshot = state.snapshot()

    assert result == {"ok": "true"}
    np.testing.assert_allclose(snapshot["hil"]["left_arm"], [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(snapshot["hil"]["right_arm"], [10, 11, 12, 13, 14, 15])
    assert [msg[0] for msg in bridge.hil_messages] == ["left_arm", "right_arm"]
    np.testing.assert_allclose(bridge.hil_messages[0][1], [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(bridge.hil_messages[1][1], [10, 11, 12, 13, 14, 15])


def test_fake_node_ui_never_calls_eva_web_commands_directly():
    assert "127.0.0.1:8080" not in fake_node.UI_HTML
    assert "/api/halt" not in fake_node.UI_HTML
    assert "/api/run" not in fake_node.UI_HTML
    assert "/api/collect_start" not in fake_node.UI_HTML
    assert "/api/operator" in fake_node.UI_HTML
    assert "/api/joint_delta" in fake_node.UI_HTML


def test_ros2_fake_registers_publishers_subscribers_and_publishes_hil():
    node = _FakeNode()
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )
    live_qos = object()
    bridge = fake_node.R1LiteRos2Fake(
        node=node,
        message_types=types,
        state=fake_node.R1LiteFakeState(),
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=live_qos,
    )

    assert set(node.publishers) >= {
        "/hdas/camera_head/right_raw/image_raw_color/compressed",
        "/hdas/feedback_arm_left",
        "/hdas/feedback_gripper_left",
        "/relaxed_ik/motion_control/pose_ee_arm_right",
        "/eva/operator_button",
        "/eva/hil/input_joint_state_arm_right",
        "/motion_target/target_joint_state_arm_right",
    }
    assert set(node.subscriptions) >= {
        "/motion_target/target_joint_state_arm_left",
        "/motion_target/target_position_gripper_left",
        "/motion_target/target_joint_state_arm_right",
        "/motion_target/target_position_gripper_right",
    }
    assert node.publisher_qos["/eva/operator_button"] == 10
    assert node.publisher_qos["/eva/hil/input_joint_state_arm_right"] is live_qos
    assert set(node.subscription_callback_groups.values()) == {None}
    assert node.timer_callback_groups == [None]

    bridge.publish_operator_button("y")
    bridge.publish_hil("right_arm", np.arange(6, dtype=np.float32))
    bridge.publish_command("right_arm", np.arange(10, 16, dtype=np.float32))

    assert node.publishers["/eva/operator_button"].messages[-1].data == "y"
    hil_msg = node.publishers["/eva/hil/input_joint_state_arm_right"].messages[-1]
    assert hil_msg.name == list(fake_node.RIGHT_ARM_JOINTS)
    np.testing.assert_allclose(hil_msg.position, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    command_msg = node.publishers["/motion_target/target_joint_state_arm_right"].messages[-1]
    assert command_msg.name == list(fake_node.RIGHT_ARM_JOINTS)
    np.testing.assert_allclose(command_msg.position, [10, 11, 12, 13, 14, 15])


def test_ros2_fake_publish_once_publishes_current_motion_targets_for_both_arms():
    node = _FakeNode()
    state = fake_node.R1LiteFakeState()
    state.apply_arm_command("left_arm", np.arange(6, dtype=np.float32))
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )
    bridge = fake_node.R1LiteRos2Fake(
        node=node,
        message_types=types,
        state=state,
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=object(),
        create_timer=False,
    )

    bridge.publish_once()

    left_msg = node.publishers["/motion_target/target_joint_state_arm_left"].messages[-1]
    right_msg = node.publishers["/motion_target/target_joint_state_arm_right"].messages[-1]
    assert left_msg.name == list(fake_node.LEFT_ARM_JOINTS)
    assert right_msg.name == list(fake_node.RIGHT_ARM_JOINTS)
    np.testing.assert_allclose(left_msg.position, [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(right_msg.position, [0, 0, 0, 0, 0, 0])


def test_ros2_fake_uses_separate_camera_qos_from_robot_live_qos():
    node = _FakeNode()
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )
    live_qos = object()
    camera_qos = object()

    fake_node.R1LiteRos2Fake(
        node=node,
        message_types=types,
        state=fake_node.R1LiteFakeState(),
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=live_qos,
        camera_qos=camera_qos,
    )

    assert node.publisher_qos["/hdas/camera_head/right_raw/image_raw_color/compressed"] is (
        camera_qos
    )
    assert node.publisher_qos["/hdas/feedback_arm_left"] is live_qos
    assert node.publisher_qos["/eva/hil/input_joint_state_arm_right"] is live_qos


def test_ros2_fake_separates_command_subscriptions_from_sensor_timer_callback_group():
    node = _FakeNode()
    state = fake_node.R1LiteFakeState()
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )
    command_group = object()
    timer_group = object()

    fake_node.R1LiteRos2Fake(
        node=node,
        message_types=types,
        state=state,
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=object(),
        command_callback_group=command_group,
        timer_callback_group=timer_group,
    )

    assert node.subscription_callback_groups["/motion_target/target_joint_state_arm_left"] is (
        command_group
    )
    assert node.subscription_callback_groups["/motion_target/target_position_gripper_right"] is (
        command_group
    )
    assert node.timer_callback_groups == [timer_group]


def test_ros2_fake_can_skip_executor_timer_for_external_sensor_loop():
    node = _FakeNode()
    state = fake_node.R1LiteFakeState()
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )

    fake_node.R1LiteRos2Fake(
        node=node,
        message_types=types,
        state=state,
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=object(),
        create_timer=False,
    )

    assert node.timers == []


def test_ros2_fake_can_use_separate_control_node_for_command_subscriptions():
    sensor_node = _FakeNode()
    control_node = _FakeNode()
    state = fake_node.R1LiteFakeState()
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )

    fake_node.R1LiteRos2Fake(
        node=sensor_node,
        control_node=control_node,
        message_types=types,
        state=state,
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=object(),
    )

    assert "/hdas/feedback_arm_left" in sensor_node.publishers
    assert "/eva/operator_button" not in sensor_node.publishers
    assert "/motion_target/target_joint_state_arm_left" not in sensor_node.subscriptions
    assert "/eva/operator_button" in control_node.publishers
    assert "/eva/hil/input_joint_state_arm_left" in control_node.publishers
    assert "/motion_target/target_joint_state_arm_left" in control_node.subscriptions
    assert "/motion_target/target_position_gripper_right" in control_node.subscriptions


def test_ros2_fake_can_disable_camera_publishers_for_robot_process():
    node = _FakeNode()
    state = fake_node.R1LiteFakeState()
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )

    fake_node.R1LiteRos2Fake(
        node=node,
        message_types=types,
        state=state,
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=object(),
        publish_cameras=False,
    )

    assert "/hdas/camera_head/right_raw/image_raw_color/compressed" not in node.publishers
    assert "/hdas/feedback_arm_left" in node.publishers
    assert "/motion_target/target_joint_state_arm_left" in node.subscriptions


def test_ros2_fake_can_disable_robot_topics_for_camera_process():
    node = _FakeNode()
    state = fake_node.R1LiteFakeState()
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )

    bridge = fake_node.R1LiteRos2Fake(
        node=node,
        message_types=types,
        state=state,
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=object(),
        publish_robot=False,
        subscribe_commands=False,
    )
    bridge.publish_once()

    assert "/hdas/camera_head/right_raw/image_raw_color/compressed" in node.publishers
    assert "/hdas/feedback_arm_left" not in node.publishers
    assert "/motion_target/target_joint_state_arm_left" not in node.subscriptions
    assert len(node.publishers["/hdas/camera_head/right_raw/image_raw_color/compressed"].messages) == 1


def test_fake_node_main_uses_multithreaded_executor_for_ros_callbacks():
    source = inspect.getsource(fake_node.main)

    assert "MultiThreadedExecutor" in source
    assert 'Node("eva_r1lite_fake_sensor")' in source
    assert 'Node("eva_r1lite_fake_control")' in source
    assert "control_node=control_node" in source
    assert "executor.add_node(control_node)" in source
    assert "executor.spin()" in source
    assert "create_timer=False" in source
    assert "_run_sensor_publish_loop" in source


def test_ros2_fake_does_not_publish_idle_hil_that_overrides_policy_commands():
    node = _FakeNode()
    state = fake_node.R1LiteFakeState()
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )
    bridge = fake_node.R1LiteRos2Fake(
        node=node,
        message_types=types,
        state=state,
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=object(),
    )

    bridge.publish_once()
    state.apply_hil_delta("left_arm", joint_index=2, delta=0.25)
    bridge.publish_once()

    left_messages = node.publishers["/eva/hil/input_joint_state_arm_left"].messages
    right_messages = node.publishers["/eva/hil/input_joint_state_arm_right"].messages
    assert left_messages == []
    assert right_messages == []

    bridge.publish_hil("left_arm", np.asarray(state.snapshot()["hil"]["left_arm"], dtype=np.float32))
    assert len(left_messages) == 1
    np.testing.assert_allclose(left_messages[0].position, [0.0, 0.0, 0.25, 0.0, 0.0, 0.0])


def test_ros2_fake_reuses_precomputed_camera_frames_while_publishing(monkeypatch):
    calls = []

    def fake_jpeg(_color, _height, _width, frame_index):
        calls.append(frame_index)
        return bytes([frame_index % 256])

    monkeypatch.setattr(fake_node, "_jpeg_image", fake_jpeg)
    node = _FakeNode()
    state = fake_node.R1LiteFakeState()
    types = fake_node.Ros2MessageTypes(
        compressed_image=_CompressedImage,
        joint_state=_JointState,
        pose_stamped=_PoseStamped,
        string=_String,
    )
    bridge = fake_node.R1LiteRos2Fake(
        node=node,
        message_types=types,
        state=state,
        rate_hz=30.0,
        image_height=32,
        image_width=48,
        qos=object(),
    )
    precomputed_calls = len(calls)

    bridge.publish_once()
    bridge.publish_once()

    assert precomputed_calls == len(fake_node.CAMERA_TOPICS) * fake_node.CAMERA_FRAME_CACHE_SIZE
    assert len(calls) == precomputed_calls


class _FakeBridge:
    def __init__(self) -> None:
        self.operator_buttons: list[str] = []
        self.hil_messages: list[tuple[str, np.ndarray]] = []
        self.command_messages: list[tuple[str, np.ndarray]] = []

    def publish_operator_button(self, label: str) -> None:
        self.operator_buttons.append(label)

    def publish_hil(self, group_name: str, positions: np.ndarray) -> None:
        self.hil_messages.append((group_name, np.asarray(positions, dtype=np.float32)))

    def publish_command(self, group_name: str, positions: np.ndarray) -> None:
        self.command_messages.append((group_name, np.asarray(positions, dtype=np.float32)))


class _FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _FakeNode:
    def __init__(self) -> None:
        self.publishers: dict[str, _FakePublisher] = {}
        self.publisher_qos = {}
        self.subscriptions = {}
        self.subscription_callback_groups = {}
        self.timers = []
        self.timer_callback_groups = []
        self.clock = _FakeClock()

    def create_publisher(self, _msg_type, topic: str, qos) -> _FakePublisher:
        publisher = _FakePublisher()
        self.publishers[topic] = publisher
        self.publisher_qos[topic] = qos
        return publisher

    def create_subscription(self, _msg_type, topic: str, callback, _qos, callback_group=None):
        self.subscriptions[topic] = callback
        self.subscription_callback_groups[topic] = callback_group
        return callback

    def create_timer(self, period_s: float, callback, callback_group=None):
        self.timers.append((period_s, callback))
        self.timer_callback_groups.append(callback_group)
        return callback

    def get_clock(self):
        return self.clock

    def get_logger(self):
        return SimpleNamespace(info=lambda *_args, **_kwargs: None)


class _FakeClock:
    def now(self):
        return SimpleNamespace(to_msg=lambda: _Stamp())


class _Stamp:
    def __init__(self) -> None:
        self.sec = 1
        self.nanosec = 2


class _Header:
    def __init__(self) -> None:
        self.stamp = _Stamp()


class _CompressedImage:
    def __init__(self) -> None:
        self.header = _Header()
        self.format = ""
        self.data = b""


class _JointState:
    def __init__(self) -> None:
        self.header = _Header()
        self.name = []
        self.position = []


class _String:
    def __init__(self) -> None:
        self.data = ""


class _PoseStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        )
