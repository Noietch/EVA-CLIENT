from __future__ import annotations

import dataclasses
import fcntl
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from examples.hardware.yam.robot import YamFollowers
from examples.hardware.yam.wire import WireAction as YamWireAction

pytestmark = pytest.mark.integration


@dataclasses.dataclass(frozen=True)
class _Config:
    group_names: tuple[str, ...] = ("left_arm", "right_arm")
    follower_can_channels: dict[str, str] = dataclasses.field(
        default_factory=lambda: {"left_arm": "can0", "right_arm": "can1"}
    )
    leader_can_channels: dict[str, str] = dataclasses.field(default_factory=dict)
    leader_gripper_encoder_endpoints: dict[str, tuple[float, float]] = dataclasses.field(
        default_factory=dict
    )
    arm_type: str = "yam"
    gripper_type: str = "linear_4310"
    enable_auto_recovery: bool = False
    command_timeout_s: float = 0.5
    idle_mode: str = "gravity_comp"
    startup_position: str = "current"
    startup_duration_s: float = 5.0
    joint4_kp: float | None = None
    end_effector_mass: float | None = None
    gravity_comp_factor: tuple[float, ...] | None = None
    gripper_limits_override: tuple[float, float] | None = None
    gripper_kp: float = 5.0
    gripper_damping: float = 0.5
    gripper_close_torque_limit: float = 0.29
    gripper_open_torque_limit: float = 0.29
    gripper_max_speed: float = 0.0
    tracking_ki: float = 0.0
    tracking_trim_limit: float = 0.12
    tracking_deadband: float = 0.002
    tracking_settle_delay_s: float = 0.15
    startup_trim_duration_s: float = 3.0


class _FakeYam:
    def __init__(self) -> None:
        self.qpos = np.asarray([0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.idle_count = 0
        self.closed = False
        self.last_command: np.ndarray | None = None
        self.last_kp: np.ndarray | None = None
        self.last_kd: np.ndarray | None = None

    def num_dofs(self) -> int:
        return 7

    def get_joint_pos(self) -> np.ndarray:
        return self.qpos.copy()

    def command_joint_pos(self, qpos: np.ndarray) -> None:
        self.last_command = np.asarray(qpos, dtype=np.float64).copy()
        self.qpos = self.last_command.copy()

    def enter_gravity_comp_idle(self) -> None:
        self.idle_count += 1

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self.last_kp = np.asarray(kp, dtype=np.float64).copy()
        self.last_kd = np.asarray(kd, dtype=np.float64).copy()

    def close(self) -> None:
        self.closed = True


def test_follower_commands_both_arms_and_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    robots_by_channel: dict[str, _FakeYam] = {}
    factory_kwargs: dict[str, dict[str, object]] = {}

    def factory(**kwargs: object) -> _FakeYam:
        robot = _FakeYam()
        channel = str(kwargs["channel"])
        robots_by_channel[channel] = robot
        factory_kwargs[channel] = kwargs
        return robot

    clock = 10.0
    monkeypatch.setattr("examples.hardware.yam.robot.time.monotonic", lambda: clock)
    followers = YamFollowers(_Config(joint4_kp=40.0), factory=factory)
    target = np.asarray(
        [0.1, 0.4, 0.5, 0.1, 0.1, 0.1, 0.25] * 2,
        dtype=np.float32,
    )
    followers.apply_action(YamWireAction(t=clock, action=target, target="real"))

    assert robots_by_channel["can0"].qpos == pytest.approx(target[:7])
    assert robots_by_channel["can1"].qpos == pytest.approx(target[7:])
    assert robots_by_channel["can0"].last_kp == pytest.approx([80, 80, 80, 40, 10, 10, 5])
    assert robots_by_channel["can0"].last_kd == pytest.approx([5, 5, 5, 1.5, 1.5, 1.5, 0.5])
    for kwargs in factory_kwargs.values():
        assert kwargs["gripper_close_torque_limit"] == 0.29
        assert kwargs["gripper_open_torque_limit"] == 0.29
        assert kwargs["gripper_damping"] == 0.5
    assert factory_kwargs["can0"]["gripper_kp"] == 5.0
    assert factory_kwargs["can1"]["gripper_kp"] == 5.0

    clock = 10.6
    followers.watchdog_tick()
    assert robots_by_channel["can0"].idle_count == 1
    assert robots_by_channel["can1"].idle_count == 1


def test_run_hardware_rejects_a_second_instance_before_hardware_setup(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    fake_venv = tmp_path / "venv"
    fake_python = fake_venv / "bin/python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    lock_path = tmp_path / "hardware.lock"

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", "examples/hardware/yam/run_hardware.sh"],
            cwd=root,
            env={
                **os.environ,
                "YAM_VENV_DIR": str(fake_venv),
                "YAM_LOCK_FILE": str(lock_path),
            },
            check=False,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 1
    assert "YAM hardware is already running" in result.stderr
    assert "Resetting Orbbec" not in result.stdout
