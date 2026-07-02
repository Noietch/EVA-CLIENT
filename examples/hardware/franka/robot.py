"""Franky-backed dual Franka arm adapter."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol

import numpy as np

from transport.zmq import WireAction

logger = logging.getLogger(__name__)

GROUP_NAMES: tuple[str, str] = ("left_arm", "right_arm")
GROUP_DOF = 8
TOTAL_DOF = len(GROUP_NAMES) * GROUP_DOF

FRANKA_ARM_LOWER = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
FRANKA_ARM_UPPER = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)
FRANKA_GRIPPER_MAX_WIDTH = 0.08
FRANKA_GRIPPER_SPEED = 0.05
FRANKA_GRIPPER_COMMAND_DELTA = 0.05
FRANKA_RELATIVE_DYNAMICS_VELOCITY = 0.08
FRANKA_RELATIVE_DYNAMICS_ACCELERATION = 0.08
FRANKA_RELATIVE_DYNAMICS_JERK = 0.04
FRANKA_RETRY_INTERVAL_SEC = 5.0

DEFAULT_LEFT_ROBOT_IP = "172.16.0.2"
DEFAULT_RIGHT_ROBOT_IP = "172.16.0.3"
DEFAULT_INITIAL_ARM_QPOS: tuple[float, ...] = (-1.82, -0.92, 1.3, -2.12, 0.0, 2.2, 0.892)
DEFAULT_INITIAL_GRIPPER = 1.0


class FrankaArmConfig(Protocol):
    robot_ips: dict[str, str]
    gripper_ips: dict[str, str]
    disabled_groups: tuple[str, ...]


def initial_qpos() -> np.ndarray:
    left = DEFAULT_INITIAL_ARM_QPOS + (DEFAULT_INITIAL_GRIPPER,)
    right = DEFAULT_INITIAL_ARM_QPOS + (DEFAULT_INITIAL_GRIPPER,)
    return np.asarray(left + right, dtype=np.float32)


def clip_group_action(action: np.ndarray) -> np.ndarray:
    """Clip one Franka 8D command to joint and normalized gripper limits.

    Args:
        action: [8] vector containing seven arm joints plus one gripper scalar.

    Returns:
        clipped: [8] float32 vector.
    """
    vector = np.asarray(action, dtype=np.float32)
    if vector.shape[0] < GROUP_DOF:
        raise ValueError(
            f"Expected Franka action with at least {GROUP_DOF} values, got {vector.shape}"
        )
    clipped = vector[:GROUP_DOF].copy()
    clipped[:7] = np.clip(clipped[:7], FRANKA_ARM_LOWER, FRANKA_ARM_UPPER)
    clipped[7] = float(np.clip(clipped[7], 0.0, 1.0))
    return clipped


def split_dual_action(action: np.ndarray) -> dict[str, np.ndarray]:
    """Split EVA's 16D Franka command into left and right 8D group commands.

    Args:
        action: [16] command vector ordered as left_arm then right_arm.

    Returns:
        parts: group name -> clipped [8] command.
    """
    vector = np.asarray(action, dtype=np.float32)
    parts = (vector[:GROUP_DOF], vector[GROUP_DOF:TOTAL_DOF])
    return {
        group_name: clip_group_action(part)
        for group_name, part in zip(GROUP_NAMES, parts, strict=True)
    }


def _format_group_action(action: np.ndarray) -> str:
    joints = np.array2string(
        action[:7],
        precision=4,
        separator=", ",
        suppress_small=False,
        max_line_width=1000,
    )
    return f"joints={joints} gripper={float(action[7]):.3f}"


def normalize_group_name(group_name: str) -> str:
    name = group_name.strip()
    aliases = {"left": "left_arm", "right": "right_arm"}
    normalized = aliases.get(name, name)
    if normalized not in GROUP_NAMES:
        raise ValueError(f"Unknown Franka arm {group_name!r}; expected one of {GROUP_NAMES}")
    return normalized


def parse_disabled_groups(group_names: list[str]) -> tuple[str, ...]:
    groups: list[str] = []
    for group_name in group_names:
        for item in group_name.split(","):
            if not item.strip():
                continue
            normalized = normalize_group_name(item)
            if normalized not in groups:
                groups.append(normalized)
    return tuple(groups)


class FrankyDualArm:
    """Dual Franka state reader and command publisher backed by franky."""

    def __init__(self, config: FrankaArmConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._qpos = initial_qpos()
        self._franky_robots: dict[str, Any] = {}
        self._franky_grippers: dict[str, Any] = {}
        self._offline_until: dict[str, float] = {}
        self._last_gripper_command: dict[str, float] = {}
        self._disabled_groups = set(config.disabled_groups)
        if self._disabled_groups:
            logger.info("Disabled Franka arms: %s", sorted(self._disabled_groups))

    def _connect(self, group_name: str) -> tuple[Any, Any] | None:
        if group_name in self._disabled_groups:
            return None
        if group_name in self._franky_robots and group_name in self._franky_grippers:
            return self._franky_robots[group_name], self._franky_grippers[group_name]

        now = time.monotonic()
        retry_at = self._offline_until[group_name] if group_name in self._offline_until else 0.0
        if now < retry_at:
            return None

        robot_ip = self._config.robot_ips[group_name]
        gripper_ip = self._config.gripper_ips[group_name]
        try:
            import franky

            robot = franky.Robot(robot_ip)
            gripper = franky.Gripper(gripper_ip)
            robot.recover_from_errors()
            robot.relative_dynamics_factor = franky.RelativeDynamicsFactor(
                velocity=FRANKA_RELATIVE_DYNAMICS_VELOCITY,
                acceleration=FRANKA_RELATIVE_DYNAMICS_ACCELERATION,
                jerk=FRANKA_RELATIVE_DYNAMICS_JERK,
            )
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return None

        self._franky_robots[group_name] = robot
        self._franky_grippers[group_name] = gripper
        logger.info(
            "Connected %s through franky: robot=%s gripper=%s",
            group_name,
            robot_ip,
            gripper_ip,
        )
        return robot, gripper

    def _mark_offline(self, group_name: str, exc: Exception) -> None:
        self._offline_until[group_name] = time.monotonic() + FRANKA_RETRY_INTERVAL_SEC
        self._franky_robots.pop(group_name, None)
        self._franky_grippers.pop(group_name, None)
        logger.warning(
            "Franka %s offline; using cached qpos for %.1f s: %s",
            group_name,
            FRANKA_RETRY_INTERVAL_SEC,
            exc,
        )

    def _offset(self, group_name: str) -> int:
        return GROUP_NAMES.index(group_name) * GROUP_DOF

    def _cached_group_qpos(self, group_name: str) -> np.ndarray:
        offset = self._offset(group_name)
        with self._lock:
            return self._qpos[offset : offset + GROUP_DOF].copy()

    def _read_group_qpos(self, group_name: str) -> np.ndarray:
        cached = self._cached_group_qpos(group_name)
        if group_name in self._disabled_groups:
            return cached
        handles = self._connect(group_name)
        if handles is None:
            return cached
        robot, gripper = handles
        try:
            joint_state = robot.current_joint_state
            arm_qpos = np.asarray(joint_state.position, dtype=np.float32)[:7]
            gripper_width = float(gripper.width)
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return cached

        gripper_scalar = gripper_width / FRANKA_GRIPPER_MAX_WIDTH
        return np.concatenate(
            [arm_qpos, [float(np.clip(gripper_scalar, 0.0, 1.0))]],
            axis=0,
        ).astype(np.float32)

    def read_state(self) -> dict[str, np.ndarray]:
        qpos = self._qpos.copy()
        for group_name in GROUP_NAMES:
            offset = self._offset(group_name)
            qpos[offset : offset + GROUP_DOF] = self._read_group_qpos(group_name)
        with self._lock:
            self._qpos = qpos.copy()
        return {
            "left_arm": qpos[:GROUP_DOF].copy(),
            "right_arm": qpos[GROUP_DOF:TOTAL_DOF].copy(),
        }

    def apply_action(self, wire_action: WireAction) -> None:
        parts = split_dual_action(wire_action.action)
        if wire_action.target != "real":
            qpos = self._qpos.copy()
            for group_name in GROUP_NAMES:
                part = parts[group_name]
                if group_name in self._disabled_groups:
                    logger.info(
                        "Franky action exec target=%s t=%.6f group=%s status=skipped "
                        "reason=disabled %s",
                        wire_action.target,
                        wire_action.t,
                        group_name,
                        _format_group_action(part),
                    )
                    continue
                offset = self._offset(group_name)
                qpos[offset : offset + GROUP_DOF] = part
                logger.info(
                    "Franky action exec target=%s t=%.6f group=%s status=cached %s",
                    wire_action.target,
                    wire_action.t,
                    group_name,
                    _format_group_action(part),
                )
            with self._lock:
                self._qpos = qpos.astype(np.float32)
            return

        qpos = self._qpos.copy()
        for group_name in GROUP_NAMES:
            part = parts[group_name]
            if group_name in self._disabled_groups:
                logger.info(
                    "Franky action exec target=real t=%.6f group=%s status=skipped "
                    "reason=disabled %s",
                    wire_action.t,
                    group_name,
                    _format_group_action(part),
                )
                continue
            handles = self._connect(group_name)
            if handles is None:
                logger.info(
                    "Franky action exec target=real t=%.6f group=%s status=skipped "
                    "reason=offline %s",
                    wire_action.t,
                    group_name,
                    _format_group_action(part),
                )
                continue
            robot, _gripper = handles
            try:
                import franky

                motion = franky.JointMotion(part[:7].astype(np.float64).tolist())
                robot.move(motion, asynchronous=True)
            except Exception as exc:
                logger.warning(
                    "Franky action exec target=real t=%.6f group=%s status=failed %s: %s",
                    wire_action.t,
                    group_name,
                    _format_group_action(part),
                    exc,
                )
                self._mark_offline(group_name, exc)
                continue

            offset = self._offset(group_name)
            qpos[offset : offset + GROUP_DOF] = part
            self._publish_gripper_if_needed(group_name, float(part[7]))
            logger.info(
                "Franky action exec target=real t=%.6f group=%s status=sent %s",
                wire_action.t,
                group_name,
                _format_group_action(part),
            )

        with self._lock:
            self._qpos = qpos.copy()

    def _publish_gripper_if_needed(self, group_name: str, gripper_scalar: float) -> None:
        if group_name in self._last_gripper_command:
            delta = abs(gripper_scalar - self._last_gripper_command[group_name])
            if delta < FRANKA_GRIPPER_COMMAND_DELTA:
                return

        handles = self._connect(group_name)
        if handles is None:
            return
        _robot, gripper = handles
        width = float(np.clip(gripper_scalar, 0.0, 1.0)) * FRANKA_GRIPPER_MAX_WIDTH
        try:
            gripper.move_async(width, FRANKA_GRIPPER_SPEED)
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return
        self._last_gripper_command[group_name] = float(gripper_scalar)

    def hardware_status(self) -> dict[str, str]:
        now = time.monotonic()
        status: dict[str, str] = {}
        for group_name in GROUP_NAMES:
            if group_name in self._disabled_groups:
                status[group_name] = "disabled"
            elif group_name in self._franky_robots and group_name in self._franky_grippers:
                status[group_name] = "online"
            elif group_name in self._offline_until and self._offline_until[group_name] > now:
                remaining_s = self._offline_until[group_name] - now
                status[group_name] = f"retrying({remaining_s:.1f}s)"
            else:
                status[group_name] = "connecting"
        return status

    def close(self) -> None:
        for group_name, robot in list(self._franky_robots.items()):
            try:
                robot.stop()
            except Exception as exc:
                logger.debug("Failed to stop Franka %s cleanly: %s", group_name, exc)
