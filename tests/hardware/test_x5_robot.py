from __future__ import annotations

from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from examples.hardware.x5 import sdk as x5_sdk
from examples.hardware.x5.robot import (
    X5_ARM_JOINT_LIMITS,
    X5_HOME_DURATION_S,
    X5_INITIAL_ARM_QPOS,
    X5DualArm,
    limit_arm_command,
    split_dual_action,
)
from robots.zoo.arx_x5 import ArxX5
from transport.zmq import WireAction


class _FakeArm:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.joint_commands: list[list[float]] = []
        self.joint_command_durations: list[float] = []
        self.gripper_commands: list[float] = []
        self.gripper_accepted = True
        self.positions = np.arange(6, dtype=np.float64)
        self.currents = np.zeros(6, dtype=np.float64)
        self.offline_joints: tuple[int, ...] = ()
        self.fault: object | None = None
        self.gripper_error = 0
        self.gripper_error_name = "no fault"
        self.protected = False
        self.disabled = False
        self.closed = False

    def get_joint_positions(self) -> list[float]:
        return self.positions.tolist()

    def set_joint_positions(self, positions: list[float], *, duration: float = 0.0) -> bool:
        self.joint_commands.append(list(positions))
        self.joint_command_durations.append(float(duration))
        return True

    def get_joint_currents(self) -> list[float]:
        return self.currents.tolist()

    def get_gripper_pos(self) -> float:
        return 0.0 if int(self.config.get("type", 0)) == 2 else 5.0

    def set_gripper_pos(self, position: float, *, duration: float | None = None) -> bool:
        self.gripper_commands.append(float(position))
        return self.gripper_accepted

    def protect_mode(self) -> None:
        self.protected = True

    def disable(self) -> bool:
        self.disabled = True
        return True

    def get_status(self) -> dict[str, object]:
        return {"offline_joints": self.offline_joints, "fault": self.fault}

    def close(self) -> None:
        self.closed = True


def _config(**overrides: object) -> SimpleNamespace:
    values = dict(
        can_ports={"left_arm": "can0", "right_arm": "can1"},
        arm_type=0,
        gripper_open_pos=0.0,
        gripper_close_pos=5.0,
        initial_gripper_scalar=None,
        start_at_zero=False,
        disabled_groups=(),
        publish_rate_hz=30.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_split_dual_action_clips_each_x5_gripper() -> None:
    action = np.asarray([*range(6), 2.0, *range(6, 12), -1.0], dtype=np.float32)

    parts = split_dual_action(action)

    np.testing.assert_allclose(parts["left_arm"], [0, 1, 2, 3, 4, 5, 1])
    np.testing.assert_allclose(parts["right_arm"], [6, 7, 8, 9, 10, 11, 0])


def test_limit_arm_command_clips_to_official_x5_software_limits() -> None:
    target = np.asarray([-3.0, -1.0, 4.0, 2.0, -2.0, 3.0])

    limited, was_limited = limit_arm_command(target)

    assert was_limited
    np.testing.assert_allclose(
        limited,
        [np.deg2rad(-150), 0.0, np.pi, np.pi / 2, -np.pi / 2, 2 * np.pi / 3],
    )


def test_x5_kinematics_urdf_uses_the_same_official_joint_limits() -> None:
    root = ElementTree.parse(ArxX5.URDF).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    actual = [
        [
            float(joints[name].find("limit").attrib["lower"]),
            float(joints[name].find("limit").attrib["upper"]),
        ]
        for name in ArxX5.ARM_JOINTS
    ]

    np.testing.assert_allclose(actual, X5_ARM_JOINT_LIMITS)


def test_x5_action_rejects_nonfinite_values_before_connecting() -> None:
    robot = X5DualArm(_config())
    action = np.zeros(14, dtype=np.float32)
    action[2] = np.nan

    robot.apply_action(WireAction(t=0.0, action=action, target="real"))

    assert robot.hardware_status() == {
        "left_arm": "connecting",
        "right_arm": "connecting",
    }


def test_x5_feedback_keeps_cached_gripper_when_sdk_returns_six_joints(
    monkeypatch,
) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config(initial_gripper_scalar=0.6))

    first = robot.read_state()
    second = robot.read_state()

    assert len(arms) == 2
    np.testing.assert_allclose(first["left_arm"][6], 0.6)
    np.testing.assert_allclose(second["right_arm"][6], 0.6)
    assert arms[0].config == {"can_port": "can0", "type": 0}
    assert arms[1].config == {"can_port": "can1", "type": 0}


def test_x5_status_reports_peak_joint_current(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config())
    robot.read_state()
    arms[0].currents = np.asarray([0.1, -4.2, 0.3, 0.0, 0.0, 0.0])
    arms[1].currents = np.asarray([0.1, 0.2, 0.3, 1.7, 0.0, 0.0])

    robot.read_state()

    assert robot.hardware_status() == {
        "left_arm": "online(current_max=4.20@J2)",
        "right_arm": "online(current_max=1.70@J4)",
    }


def test_x5_status_prioritizes_fault_and_offline_joints(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config())
    robot.read_state()
    arms[0].fault = "overcurrent"
    arms[1].offline_joints = (1,)

    assert robot.hardware_status() == {
        "left_arm": "fault(overcurrent)",
        "right_arm": "offline(J2)",
    }


def test_x5_status_reports_gripper_motor_fault(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config())
    robot.read_state()
    arms[0].gripper_error = 12
    arms[0].gripper_error_name = "coil overtemperature"

    assert robot.hardware_status()["left_arm"] == ("gripper_fault(12:coil overtemperature)")


def test_x5_does_not_send_gripper_target_while_motor_has_fault(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arm.gripper_error = 12
        arm.gripper_error_name = "coil overtemperature"
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config())
    action = np.zeros(14, dtype=np.float32)
    action[[6, 13]] = 1.0

    robot.apply_action(WireAction(t=0.0, action=action, target="real"))

    assert all(arm.gripper_commands == [] for arm in arms)


def test_x5_slow_home_refuses_arm_with_offline_joint(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        if len(arms) == 2:
            arm.offline_joints = (1,)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config())

    with pytest.raises(RuntimeError, match=r"right_arm: offline_joints=J2"):
        robot.slow_home()

    # Connecting each SDK arm re-issues its current position once, but the
    # synchronized home trajectory must not be sent after the health failure.
    assert all(len(arm.joint_commands) == 1 for arm in arms)
    assert all(arm.joint_command_durations == [0.0] for arm in arms)


def test_x5_action_maps_normalized_gripper_to_physical_open_close_direction(
    monkeypatch,
) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config())
    action = np.zeros(14, dtype=np.float32)
    action[6] = 0.5
    action[13] = 1.0

    robot.apply_action(WireAction(t=0.0, action=action, target="real"))

    np.testing.assert_allclose(arms[0].gripper_commands, [2.5])
    np.testing.assert_allclose(arms[1].gripper_commands, [0.0])


def test_x5_2025_action_maps_normalized_gripper_to_physical_range(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(
        _config(
            arm_type=2,
            gripper_open_pos=-3.4,
            gripper_close_pos=0.1,
        )
    )
    action = np.zeros(14, dtype=np.float32)
    action[6] = 0.0
    action[13] = 1.0

    robot.apply_action(WireAction(t=0.0, action=action, target="real"))

    np.testing.assert_allclose(arms[0].gripper_commands, [0.1])
    np.testing.assert_allclose(arms[1].gripper_commands, [-3.4])


def test_x5_retries_gripper_target_after_sdk_rejection(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arm.gripper_accepted = False
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config())
    action = np.zeros(14, dtype=np.float32)
    action[[6, 13]] = 1.0

    robot.apply_action(WireAction(t=0.0, action=action, target="real"))
    for arm in arms:
        arm.gripper_accepted = True
    robot.apply_action(WireAction(t=0.0, action=action, target="real"))

    assert all(arm.gripper_commands == [0.0, 0.0] for arm in arms)


def test_x5_action_clips_arm_qpos_before_calling_native_sdk(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config())
    action = np.asarray([*([-20.0] * 6), 1.0, *([20.0] * 6), 1.0], dtype=np.float32)

    robot.apply_action(WireAction(t=0.0, action=action, target="real"))

    np.testing.assert_allclose(arms[0].joint_commands[-1], X5_ARM_JOINT_LIMITS[:, 0])
    np.testing.assert_allclose(arms[1].joint_commands[-1], X5_ARM_JOINT_LIMITS[:, 1])


def test_x5_action_uses_one_collection_frame_as_official_sdk_duration(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config(publish_rate_hz=30.0))

    robot.apply_action(WireAction(t=0.0, action=np.zeros(14), target="real"))

    for arm in arms:
        assert arm.joint_command_durations[-1] == 1.0 / 30.0


def test_x5_slow_home_uses_official_synchronized_trajectory(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arm.positions = np.asarray([0.01, -0.007, 0.003, 0.0, 0.0, 0.0])
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config())

    assert robot.slow_home()

    assert len(arms) == 2
    for arm in arms:
        np.testing.assert_allclose(arm.joint_commands[-1], X5_INITIAL_ARM_QPOS, atol=1e-8)
        assert arm.joint_command_durations[-1] == pytest.approx(X5_HOME_DURATION_S)


def test_x5_slow_home_can_start_at_all_zero_joint_position(monkeypatch) -> None:
    arms: list[_FakeArm] = []

    def _single_arm(config: dict) -> _FakeArm:
        arm = _FakeArm(config)
        arms.append(arm)
        return arm

    monkeypatch.setattr(x5_sdk, "SingleArm", _single_arm)
    robot = X5DualArm(_config(start_at_zero=True))

    assert robot.slow_home()

    assert len(arms) == 2
    for arm in arms:
        np.testing.assert_allclose(arm.joint_commands[-1], np.zeros(6), atol=1e-8)
        assert arm.joint_command_durations[-1] == pytest.approx(X5_HOME_DURATION_S)
