from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from core.config import ConfigDict

logger = logging.getLogger(__name__)


def decode_teleop_gripper(
    config: ConfigDict,
    raw_value: float,
    previous_raw: float | None,
    previous_command: float,
) -> tuple[float, float]:
    """Decode a teleoperator gripper reading into a normalized command.

    Args:
        config: Gripper decoder settings.
        raw_value: Raw analog value or normalized command from the teleoperator.
        previous_raw: Previous raw teleoperator value, or None before the first read.
        previous_command: Previous normalized gripper command.

    Returns:
        A tuple of normalized command and current raw value.
    """
    raw = float(raw_value)
    if config.source == "command":
        return float(np.clip(raw, 0.0, 1.0)), raw

    if config.mode == "position":
        raw_open = float(config.raw_open)
        raw_close = float(config.raw_close)
        if raw_open == raw_close:
            return float(previous_command), raw
        value = (raw - raw_close) / (raw_open - raw_close)
        return float(np.clip(value, 0.0, 1.0)), raw

    command = float(previous_command)
    if previous_raw is not None and previous_raw > config.threshold and raw < config.threshold:
        open_value = float(config.open_value)
        close_value = float(config.close_value)
        if abs(command - open_value) <= abs(command - close_value):
            command = close_value
        else:
            command = open_value
    return command, raw


class AliciaTeleoperatorDevice:
    def __init__(self, port: str) -> None:
        self.port = port
        self.robot: Any | None = None
        self.prev_q = np.zeros(6, dtype=float)
        self.compensation = np.zeros(6, dtype=float)

    def connect(self) -> None:
        import alicia_d_sdk

        self.robot = alicia_d_sdk.create_robot(port=self.port)
        initial_q = self.get_joint_positions()
        if initial_q is not None:
            self.prev_q = initial_q[:6].copy()

    def disconnect(self) -> None:
        robot = self.robot
        if robot is None:
            return
        disconnect = getattr(robot, "disconnect", None)
        if disconnect is not None:
            disconnect()
        self.robot = None

    def get_joint_positions(self) -> np.ndarray | None:
        robot = self.robot
        if robot is None:
            return None
        q = robot.get_robot_state("joint")
        if q is None:
            return None
        return np.asarray(q, dtype=float)

    def get_smoothed_joint_positions(self) -> np.ndarray:
        current_q = self.get_joint_positions()
        if current_q is None:
            return self.prev_q + self.compensation

        current_q = current_q[:6]
        variation = current_q - self.prev_q
        self.compensation[variation > np.pi] -= 2.0 * np.pi
        self.compensation[variation < -np.pi] += 2.0 * np.pi
        self.prev_q = current_q.copy()
        return current_q + self.compensation

    def reset_smoothing(self) -> None:
        current_q = self.get_joint_positions()
        if current_q is None:
            return
        self.prev_q = current_q[:6].copy()
        self.compensation = np.zeros(6, dtype=float)

    def get_gripper_level(self) -> float | None:
        robot = self.robot
        if robot is None:
            return None
        level = robot.get_robot_state("gripper")
        if level is None:
            return None
        return float(level)


class Ur5eTeleopSource:
    def __init__(
        self,
        config: ConfigDict,
        device_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._config = config
        self._device_factory = device_factory or AliciaTeleoperatorDevice
        self._device: Any | None = None
        self._delta: np.ndarray | None = None
        self._gripper_command = config.gripper.open_value
        self._previous_gripper_raw: float | None = None
        joint_coef = np.asarray(config.joint_coef, dtype=float)
        self._joint_coef = joint_coef if joint_coef.shape == (6,) else np.ones(6, dtype=float)

    def connect(self) -> None:
        if self._device is not None:
            return
        self._device = self._device_factory(self._config.port)
        self._device.connect()
        initial_gripper = self._device.get_gripper_level()
        if initial_gripper is not None:
            self._previous_gripper_raw = initial_gripper

    def disconnect(self) -> None:
        device = self._device
        if device is None:
            return
        device.disconnect()
        self._device = None

    def calibrate_delta(self, robot_qpos: np.ndarray) -> None:
        device = self._device
        if device is None:
            raise RuntimeError("UR5e teleop source is not connected")
        teleop_qpos = device.get_joint_positions()
        if teleop_qpos is None:
            raise RuntimeError("UR5e teleop joints unavailable for calibration")
        self._delta = np.asarray(robot_qpos[:6], dtype=float) - self._joint_coef * np.asarray(
            teleop_qpos[:6], dtype=float
        )
        device.reset_smoothing()

    def read_action_qpos(self, robot_qpos: np.ndarray) -> np.ndarray:
        device = self._device
        if device is None:
            raise RuntimeError("UR5e teleop source is not connected")
        if self._delta is None:
            self.calibrate_delta(robot_qpos)

        teleop_qpos = device.get_smoothed_joint_positions()
        target_joint = self._joint_coef * np.asarray(teleop_qpos[:6], dtype=float) + self._delta
        raw_gripper = device.get_gripper_level()
        if raw_gripper is not None:
            self._gripper_command, self._previous_gripper_raw = decode_teleop_gripper(
                self._config.gripper,
                raw_value=raw_gripper,
                previous_raw=self._previous_gripper_raw,
                previous_command=self._gripper_command,
            )
        return np.asarray([*target_joint, self._gripper_command], dtype=np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read an Alicia-D teleop device once.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    device = AliciaTeleoperatorDevice(args.port)
    device.connect()
    try:
        logger.info(
            "joints=%s gripper=%s",
            device.get_joint_positions(),
            device.get_gripper_level(),
        )
    finally:
        device.disconnect()


if __name__ == "__main__":
    main()
