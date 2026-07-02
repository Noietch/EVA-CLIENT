from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np

from examples.hardware.ur5e.gripper import DahuanAG95Controller, ag95_to_raw


class Ur5eRobotConfig(Protocol):
    robot_ip: str
    gripper_port: str
    publish_rate_hz: float
    raw_open: int
    raw_close: int
    servo_speed: float
    servo_accel: float
    servo_lookahead_time: float
    servo_gain: int


def rtde_tcp_pose_to_eef(tcp_pose: np.ndarray, gripper: float) -> np.ndarray:
    """Convert RTDE TCP pose into EVA EEF layout.

    Args:
        tcp_pose: RTDE pose [x, y, z, rx, ry, rz] with rotation vector orientation.
        gripper: Normalized gripper value.

    Returns:
        EVA EEF vector [x, y, z, qw, qx, qy, qz, gripper].
    """
    pose = np.asarray(tcp_pose, dtype=np.float32)
    rotvec = pose[3:6].astype(float)
    angle = float(np.linalg.norm(rotvec))
    if angle == 0.0:
        quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = rotvec / angle
        half_angle = angle * 0.5
        quat = np.asarray(
            [np.cos(half_angle), *(axis * np.sin(half_angle))],
            dtype=np.float32,
        )
    return np.asarray([*pose[:3], *quat, float(gripper)], dtype=np.float32)


class Ur5eRobot:
    """UR RTDE arm and Dahuan AG95 gripper adapter."""

    def __init__(self, config: Ur5eRobotConfig) -> None:
        import rtde_control
        import rtde_receive

        self._config = config
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._rtde_r = rtde_receive.RTDEReceiveInterface(config.robot_ip)
        self._rtde_c = rtde_control.RTDEControlInterface(config.robot_ip)
        self._gripper = DahuanAG95Controller(
            config.gripper_port,
            raw_open=config.raw_open,
            raw_close=config.raw_close,
        )
        self._gripper.connect()
        initial_gripper = self._gripper.read_normalized()
        self._gripper_command = initial_gripper
        self._gripper_position = initial_gripper
        self._gripper_thread = threading.Thread(
            target=self._gripper_loop,
            name="ur5e-ag95",
            daemon=True,
        )
        self._gripper_thread.start()

    def _gripper_loop(self) -> None:
        with self._lock:
            last_command = ag95_to_raw(
                self._gripper_command,
                self._config.raw_open,
                self._config.raw_close,
            )
        while not self._stop.is_set():
            with self._lock:
                command = self._gripper_command
            raw_command = ag95_to_raw(command, self._config.raw_open, self._config.raw_close)
            if raw_command != last_command:
                self._gripper.set_normalized(command)
                last_command = raw_command
            with self._lock:
                self._gripper_position = self._gripper.read_normalized()
            time.sleep(0.02)

    def read_qpos(self) -> np.ndarray:
        """Read UR5e joint state in EVA qpos layout.

        Returns:
            qpos: [7] float32 vector containing six UR joints plus normalized gripper.
        """
        joints = self._rtde_r.getActualQ()
        with self._lock:
            gripper = self._gripper_position
        return np.asarray([float(value) for value in joints[:6]] + [gripper], dtype=np.float32)

    def read_eef(self, qpos: np.ndarray) -> np.ndarray:
        """Read UR5e TCP pose in EVA EEF layout.

        Args:
            qpos: [7] EVA qpos vector whose last value is normalized gripper state.

        Returns:
            eef: [8] vector [x, y, z, qw, qx, qy, qz, gripper].
        """
        return rtde_tcp_pose_to_eef(np.asarray(self._rtde_r.getActualTCPPose()), float(qpos[6]))

    def apply_action(self, action: np.ndarray) -> None:
        """Send one EVA qpos command to UR RTDE and AG95.

        Args:
            action: [7] command vector containing six UR joints plus normalized gripper.
        """
        vector = np.asarray(action, dtype=np.float32)
        self._rtde_c.servoJ(
            [float(value) for value in vector[:6]],
            self._config.servo_speed,
            self._config.servo_accel,
            1.0 / max(float(self._config.publish_rate_hz), 1e-6),
            self._config.servo_lookahead_time,
            self._config.servo_gain,
        )
        with self._lock:
            self._gripper_command = float(vector[6])

    def close(self) -> None:
        self._stop.set()
        self._gripper_thread.join(timeout=1.0)
        self._rtde_c.servoStop()
        self._rtde_c.stopScript()
        self._rtde_c.disconnect()
        self._rtde_r.disconnect()
        self._gripper.close()
