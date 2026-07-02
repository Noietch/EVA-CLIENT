"""ARX R5 dual single-arm adapter backed by the ARX R5 Python SDK (bimanual.SingleArm)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol

import numpy as np

from transport.zmq import WireAction

logger = logging.getLogger(__name__)

GROUP_NAMES: tuple[str, str] = ("left_arm", "right_arm")
GROUP_DOF = 7  # joint1..6 + gripper scalar
ARM_DOF = 6
GRIPPER_INDEX = 6
TOTAL_DOF = len(GROUP_NAMES) * GROUP_DOF

# ARX set_arm_status: 5 = joint position control mode (see SDK single_arm.py).
ARM_STATUS_JOINT_CONTROL = 5
# Catch units commanded to set_catch(); the EVA gripper scalar 0..1 maps onto
# [0, ARX_GRIPPER_OPEN_POS]. Calibrate per gripper via --gripper-open-pos.
ARX_GRIPPER_OPEN_POS = 5.0
ARX_GRIPPER_COMMAND_DELTA = 0.02
ARX_RETRY_INTERVAL_SEC = 5.0

DEFAULT_LEFT_CAN_PORT = "can0"
DEFAULT_RIGHT_CAN_PORT = "can1"
DEFAULT_ARM_TYPE = 0


class ArxArmConfig(Protocol):
    can_ports: dict[str, str]
    arm_type: int
    gripper_open_pos: float
    initial_gripper_scalar: float | None
    disabled_groups: tuple[str, ...]


def initial_qpos(initial_gripper_scalar: float | None = None) -> np.ndarray:
    qpos = np.zeros(TOTAL_DOF, dtype=np.float32)
    if initial_gripper_scalar is not None:
        qpos[GRIPPER_INDEX::GROUP_DOF] = float(np.clip(initial_gripper_scalar, 0.0, 1.0))
    return qpos


def clip_group_action(action: np.ndarray) -> np.ndarray:
    """Clip one ARX 7D command to the normalized gripper range.

    Args:
        action: [7] vector containing six arm joints plus one gripper scalar.

    Returns:
        clipped: [7] float32 vector with gripper scalar in [0, 1].
    """
    vector = np.asarray(action, dtype=np.float32)
    if vector.shape[0] < GROUP_DOF:
        raise ValueError(
            f"Expected ARX action with at least {GROUP_DOF} values, got {vector.shape}"
        )
    clipped = vector[:GROUP_DOF].copy()
    clipped[GRIPPER_INDEX] = float(np.clip(clipped[GRIPPER_INDEX], 0.0, 1.0))
    return clipped


def split_dual_action(action: np.ndarray) -> dict[str, np.ndarray]:
    """Split EVA's 14D ARX command into left and right 7D group commands.

    Args:
        action: [14] command vector ordered as left_arm then right_arm.

    Returns:
        parts: group name -> clipped [7] command.
    """
    vector = np.asarray(action, dtype=np.float32)
    parts = (vector[:GROUP_DOF], vector[GROUP_DOF:TOTAL_DOF])
    return {
        group_name: clip_group_action(part)
        for group_name, part in zip(GROUP_NAMES, parts, strict=True)
    }


def normalize_group_name(group_name: str) -> str:
    name = group_name.strip()
    aliases = {"left": "left_arm", "right": "right_arm"}
    normalized = aliases.get(name, name)
    if normalized not in GROUP_NAMES:
        raise ValueError(f"Unknown ARX arm {group_name!r}; expected one of {GROUP_NAMES}")
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


class ArxDualArm:
    """Dual ARX R5 state reader and command publisher backed by bimanual.SingleArm."""

    def __init__(self, config: ArxArmConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._arms: dict[str, Any] = {}
        self._offline_until: dict[str, float] = {}
        self._last_gripper_command: dict[str, float] = {}
        self._gripper_open_pos = float(config.gripper_open_pos)
        initial_gripper_scalar = config.initial_gripper_scalar
        self._initial_gripper_scalar = (
            None
            if initial_gripper_scalar is None
            else float(np.clip(initial_gripper_scalar, 0.0, 1.0))
        )
        self._qpos = initial_qpos(self._initial_gripper_scalar)
        self._disabled_groups = set(config.disabled_groups)
        if self._disabled_groups:
            logger.info("Disabled ARX arms: %s", sorted(self._disabled_groups))

    def _connect(self, group_name: str) -> Any | None:
        if group_name in self._disabled_groups:
            return None
        if group_name in self._arms:
            return self._arms[group_name]

        now = time.monotonic()
        retry_at = self._offline_until[group_name] if group_name in self._offline_until else 0.0
        if now < retry_at:
            return None

        can_port = self._config.can_ports[group_name]
        try:
            from bimanual import SingleArm

            arm = SingleArm(
                {
                    "can_port": can_port,
                    "type": int(self._config.arm_type),
                    "num_joints": GROUP_DOF,
                }
            )
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return None

        try:
            positions = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            if positions.shape[0] >= ARM_DOF:
                arm.set_joint_positions(positions[:ARM_DOF].tolist())
            if self._initial_gripper_scalar is not None:
                catch_pos = self._initial_gripper_scalar * self._gripper_open_pos
                arm.set_catch_pos(catch_pos)
                self._last_gripper_command[group_name] = self._initial_gripper_scalar
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return None

        self._arms[group_name] = arm
        logger.info(
            "Connected and enabled %s through ARX SDK: can_port=%s",
            group_name,
            can_port,
        )
        return arm

    def _mark_offline(self, group_name: str, exc: Exception) -> None:
        self._offline_until[group_name] = time.monotonic() + ARX_RETRY_INTERVAL_SEC
        self._arms.pop(group_name, None)
        logger.warning(
            "ARX %s offline; using cached qpos for %.1f s: %s",
            group_name,
            ARX_RETRY_INTERVAL_SEC,
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
        arm = self._connect(group_name)
        if arm is None:
            return cached
        try:
            positions = np.asarray(arm.get_joint_positions(), dtype=np.float32)
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return cached

        arm_qpos = positions[:ARM_DOF]
        if arm_qpos.shape[0] < ARM_DOF:
            return cached
        gripper_raw = float(positions[GRIPPER_INDEX]) if positions.shape[0] > ARM_DOF else 0.0
        gripper_scalar = gripper_raw / self._gripper_open_pos if self._gripper_open_pos else 0.0
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
                if group_name in self._disabled_groups:
                    continue
                offset = self._offset(group_name)
                qpos[offset : offset + GROUP_DOF] = parts[group_name]
            with self._lock:
                self._qpos = qpos.astype(np.float32)
            return

        qpos = self._qpos.copy()
        for group_name in GROUP_NAMES:
            if group_name in self._disabled_groups:
                continue
            arm = self._connect(group_name)
            if arm is None:
                continue
            part = parts[group_name]
            try:
                arm.set_joint_positions(part[:ARM_DOF].astype(np.float64).tolist())
            except Exception as exc:
                self._mark_offline(group_name, exc)
                continue

            offset = self._offset(group_name)
            qpos[offset : offset + GROUP_DOF] = part
            self._publish_gripper_if_needed(group_name, float(part[GRIPPER_INDEX]))

        with self._lock:
            self._qpos = qpos.copy()

    def _publish_gripper_if_needed(self, group_name: str, gripper_scalar: float) -> None:
        if group_name in self._last_gripper_command:
            delta = abs(gripper_scalar - self._last_gripper_command[group_name])
            if delta < ARX_GRIPPER_COMMAND_DELTA:
                return

        arm = self._connect(group_name)
        if arm is None:
            return
        catch_pos = float(np.clip(gripper_scalar, 0.0, 1.0)) * self._gripper_open_pos
        try:
            arm.set_catch_pos(catch_pos)
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return
        self._last_gripper_command[group_name] = float(gripper_scalar)

    def go_home(self) -> dict[str, bool]:
        """Send each connected arm back to its home (zero) pose.

        Returns:
            results: group name -> True if the home command was issued.
        """
        results: dict[str, bool] = {}
        for group_name in GROUP_NAMES:
            if group_name in self._disabled_groups:
                continue
            arm = self._connect(group_name)
            if arm is None:
                results[group_name] = False
                continue
            try:
                arm.go_home()
            except Exception as exc:
                self._mark_offline(group_name, exc)
                results[group_name] = False
                continue
            offset = self._offset(group_name)
            with self._lock:
                self._qpos[offset : offset + GROUP_DOF] = 0.0
            results[group_name] = True
            logger.info("Sent %s to home pose", group_name)
        return results

    def hardware_status(self) -> dict[str, str]:
        now = time.monotonic()
        status: dict[str, str] = {}
        for group_name in GROUP_NAMES:
            if group_name in self._disabled_groups:
                status[group_name] = "disabled"
            elif group_name in self._arms:
                status[group_name] = "online"
            elif group_name in self._offline_until and self._offline_until[group_name] > now:
                remaining_s = self._offline_until[group_name] - now
                status[group_name] = f"retrying({remaining_s:.1f}s)"
            else:
                status[group_name] = "connecting"
        return status

    def close(self) -> None:
        for group_name, arm in list(self._arms.items()):
            try:
                arm.protect_mode()
            except Exception as exc:
                logger.debug("Failed to stop ARX %s cleanly: %s", group_name, exc)
