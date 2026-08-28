"""ARX X5 dual-arm adapter backed by the native ARX X5 motor interface.

Only the upstream motor binding is loaded here. FK, IK, and VR retargeting use
the repository's shared PyRoki solver with the X5A URDF.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from transport.zmq import WireAction

logger = logging.getLogger(__name__)

GROUP_NAMES: tuple[str, str] = ("left_arm", "right_arm")
GROUP_DOF = 7  # six arm joints plus one normalized gripper scalar
ARM_DOF = 6
GRIPPER_INDEX = 6
TOTAL_DOF = len(GROUP_NAMES) * GROUP_DOF

DEFAULT_LEFT_CAN_PORT = "can1"
DEFAULT_RIGHT_CAN_PORT = "can3"
DEFAULT_ARX_X5_TYPE = 2
DEFAULT_ARM_TYPE = DEFAULT_ARX_X5_TYPE

# The normalized EVA/R5 convention is 0.0 = closed and 1.0 = open. Keep the
# physical ARX X5 endpoints explicit because its SDK position increases toward closed.
ARX_X5_2023_GRIPPER_OPEN_POS = 0.0
ARX_X5_2023_GRIPPER_CLOSE_POS = 5.0
ARX_X5_2025_GRIPPER_OPEN_POS = -3.4
ARX_X5_2025_GRIPPER_CLOSE_POS = 0.1
DEFAULT_GRIPPER_OPEN_POS = ARX_X5_2025_GRIPPER_OPEN_POS
DEFAULT_GRIPPER_CLOSE_POS = ARX_X5_2025_GRIPPER_CLOSE_POS
ARX_X5_GRIPPER_COMMAND_DELTA = 0.02
ARX_X5_RETRY_INTERVAL_SEC = 5.0
ARX_X5_HOME_DURATION_S = 5.0
ARX_X5_INITIAL_ARM_QPOS = np.asarray((0.0, 1.58, 0.49, 1.15, 0.0, 0.0), dtype=np.float64)
ARX_X5_SAFETY_WARNING_INTERVAL_SEC = 2.0

# ARX X5 SDK-V2 physical limits, converted from the documented degree ranges.
ARX_X5_ARM_JOINT_LIMITS = np.asarray(
    [
        [np.deg2rad(-150.0), np.deg2rad(180.0)],
        [np.deg2rad(0.0), np.deg2rad(210.0)],
        [np.deg2rad(0.0), np.deg2rad(180.0)],
        [np.deg2rad(-90.0), np.deg2rad(90.0)],
        [np.deg2rad(-90.0), np.deg2rad(90.0)],
        [np.deg2rad(-120.0), np.deg2rad(120.0)],
    ],
    dtype=np.float64,
)


class ArxX5ArmConfig(Protocol):
    can_ports: dict[str, str]
    arm_type: int
    gripper_open_pos: float | None
    gripper_close_pos: float | None
    initial_gripper_scalar: float | None
    start_at_zero: bool
    disabled_groups: tuple[str, ...]
    publish_rate_hz: float


def initial_qpos(
    initial_gripper_scalar: float | None = None,
    arm_qpos: np.ndarray | None = None,
) -> np.ndarray:
    arm_target = (
        ARX_X5_INITIAL_ARM_QPOS if arm_qpos is None else np.asarray(arm_qpos, dtype=np.float64)
    )
    if arm_target.shape != (ARM_DOF,) or not np.all(np.isfinite(arm_target)):
        raise ValueError(f"Expected finite ARX X5 arm qpos shaped ({ARM_DOF},), got {arm_target}")
    qpos = np.zeros(TOTAL_DOF, dtype=np.float32)
    qpos[:ARM_DOF] = arm_target.astype(np.float32)
    qpos[GROUP_DOF : GROUP_DOF + ARM_DOF] = arm_target.astype(np.float32)
    if initial_gripper_scalar is not None:
        qpos[GRIPPER_INDEX::GROUP_DOF] = float(np.clip(initial_gripper_scalar, 0.0, 1.0))
    return qpos


def clip_group_action(action: np.ndarray) -> np.ndarray:
    """Clip one ARX X5 7D command to the normalized gripper range."""
    vector = np.asarray(action, dtype=np.float32)
    if vector.shape[0] < GROUP_DOF:
        raise ValueError(
            f"Expected ARX X5 action with at least {GROUP_DOF} values, got {vector.shape}"
        )
    clipped = vector[:GROUP_DOF].copy()
    if not np.all(np.isfinite(clipped)):
        raise ValueError("ARX X5 group action contains NaN or infinity")
    clipped[GRIPPER_INDEX] = float(np.clip(clipped[GRIPPER_INDEX], 0.0, 1.0))
    return clipped


def split_dual_action(action: np.ndarray) -> dict[str, np.ndarray]:
    """Split EVA's 14D command into left and right 7D ARX X5 commands."""
    vector = np.asarray(action, dtype=np.float32)
    parts = (vector[:GROUP_DOF], vector[GROUP_DOF:TOTAL_DOF])
    return {
        group_name: clip_group_action(part)
        for group_name, part in zip(GROUP_NAMES, parts, strict=True)
    }


def limit_arm_command(target_qpos: np.ndarray) -> tuple[np.ndarray, bool]:
    """Validate and clip one six-joint command to the official ARX X5 software limits."""
    target = np.asarray(target_qpos, dtype=np.float64)
    if target.shape != (ARM_DOF,):
        raise ValueError(f"Expected ARX X5 arm qpos shaped ({ARM_DOF},), got {target.shape}")
    if not np.all(np.isfinite(target)):
        raise ValueError("ARX X5 arm qpos contains NaN or infinity")
    limited = np.clip(
        target,
        ARX_X5_ARM_JOINT_LIMITS[:, 0],
        ARX_X5_ARM_JOINT_LIMITS[:, 1],
    )
    return limited, not np.array_equal(limited, target)


def normalize_group_name(group_name: str) -> str:
    name = group_name.strip()
    aliases = {"left": "left_arm", "right": "right_arm"}
    normalized = aliases.get(name, name)
    if normalized not in GROUP_NAMES:
        raise ValueError(f"Unknown ARX X5 arm {group_name!r}; expected one of {GROUP_NAMES}")
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


class ArxX5DualArm:
    """Dual ARX X5 state reader and command publisher backed by two SingleArm objects."""

    def __init__(self, config: ArxX5ArmConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._arms: dict[str, Any] = {}
        self._offline_until: dict[str, float] = {}
        self._last_gripper_command: dict[str, float] = {}
        self._last_safety_warning_time: dict[str, float] = {}
        self._last_joint_currents: dict[str, np.ndarray] = {}
        publish_rate_hz = float(getattr(config, "publish_rate_hz", 30.0))
        if not np.isfinite(publish_rate_hz) or publish_rate_hz <= 0:
            raise ValueError(f"Invalid ARX X5 publish rate: {publish_rate_hz}")
        self._action_duration_s = 1.0 / publish_rate_hz

        configured_type = getattr(config, "arm_type", None)
        if configured_type is None:
            configured_type = getattr(config, "arx_x5_type", DEFAULT_ARX_X5_TYPE)
        self._arx_x5_type = int(configured_type)
        if self._arx_x5_type == 2:
            default_open_pos = ARX_X5_2025_GRIPPER_OPEN_POS
            default_close_pos = ARX_X5_2025_GRIPPER_CLOSE_POS
        else:
            default_open_pos = ARX_X5_2023_GRIPPER_OPEN_POS
            default_close_pos = ARX_X5_2023_GRIPPER_CLOSE_POS
        configured_open_pos = getattr(config, "gripper_open_pos", None)
        configured_close_pos = getattr(config, "gripper_close_pos", None)
        self._gripper_open_pos = float(
            default_open_pos if configured_open_pos is None else configured_open_pos
        )
        self._gripper_close_pos = float(
            default_close_pos if configured_close_pos is None else configured_close_pos
        )

        initial_gripper_scalar = getattr(config, "initial_gripper_scalar", None)
        self._initial_gripper_scalar = (
            None
            if initial_gripper_scalar is None
            else float(np.clip(initial_gripper_scalar, 0.0, 1.0))
        )
        self._start_at_zero = bool(getattr(config, "start_at_zero", False))
        self._startup_arm_qpos = np.zeros(ARM_DOF, dtype=np.float64)
        if not self._start_at_zero:
            self._startup_arm_qpos[:] = ARX_X5_INITIAL_ARM_QPOS
        self._qpos = initial_qpos(self._initial_gripper_scalar, self._startup_arm_qpos)
        self._disabled_groups = set(getattr(config, "disabled_groups", ()))
        if self._disabled_groups:
            logger.info("Disabled ARX X5 arms: %s", sorted(self._disabled_groups))

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
            # Keep the native SDK import inside the retry boundary so an unbuilt
            # SDK is recoverable. EVA kinematics remain in robots.kinematics.pyroki.
            from examples.hardware.arx_x5.sdk import SingleArm

            arm = SingleArm(
                {
                    "can_port": can_port,
                    "type": self._arx_x5_type,
                }
            )
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return None

        try:
            positions = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            if positions.shape[0] >= ARM_DOF:
                # The SDK starts in home mode. Re-issue the current arm joints so
                # subsequent commands enter position control without a jump.
                arm.set_joint_positions(positions[:ARM_DOF].tolist(), duration=0.0)
            if self._initial_gripper_scalar is not None:
                gripper_pos = self._scalar_to_gripper_pos(self._initial_gripper_scalar)
                arm.set_gripper_pos(gripper_pos)
                self._last_gripper_command[group_name] = self._initial_gripper_scalar
                self._set_cached_gripper_scalar(group_name, self._initial_gripper_scalar)
            else:
                try:
                    gripper_pos = float(arm.get_gripper_pos())
                except Exception as exc:
                    logger.debug(
                        "Failed to read initial ARX X5 %s gripper position: %s", group_name, exc
                    )
                else:
                    gripper_scalar = self._gripper_pos_to_scalar(gripper_pos)
                    self._last_gripper_command[group_name] = gripper_scalar
                    self._set_cached_gripper_scalar(group_name, gripper_scalar)
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return None

        self._arms[group_name] = arm
        logger.info(
            "Connected and enabled %s through official ARX X5 SDK-V2: can_port=%s type=%d",
            group_name,
            can_port,
            self._arx_x5_type,
        )
        return arm

    def _mark_offline(self, group_name: str, exc: Exception) -> None:
        self._offline_until[group_name] = time.monotonic() + ARX_X5_RETRY_INTERVAL_SEC
        arm = self._arms.pop(group_name, None)
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass
        logger.warning(
            "ARX X5 %s offline; using cached qpos for %.1f s: %s",
            group_name,
            ARX_X5_RETRY_INTERVAL_SEC,
            exc,
        )

    def _offset(self, group_name: str) -> int:
        return GROUP_NAMES.index(group_name) * GROUP_DOF

    def _cached_group_qpos(self, group_name: str) -> np.ndarray:
        offset = self._offset(group_name)
        with self._lock:
            return self._qpos[offset : offset + GROUP_DOF].copy()

    def _set_cached_gripper_scalar(self, group_name: str, gripper_scalar: float) -> None:
        offset = self._offset(group_name)
        with self._lock:
            self._qpos[offset + GRIPPER_INDEX] = float(np.clip(gripper_scalar, 0.0, 1.0))

    def _scalar_to_gripper_pos(self, gripper_scalar: float) -> float:
        scalar = float(np.clip(gripper_scalar, 0.0, 1.0))
        return self._gripper_close_pos + scalar * (self._gripper_open_pos - self._gripper_close_pos)

    def _gripper_pos_to_scalar(self, gripper_pos: float) -> float:
        span = self._gripper_open_pos - self._gripper_close_pos
        if span == 0.0:
            return 0.0
        return float(np.clip((gripper_pos - self._gripper_close_pos) / span, 0.0, 1.0))

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
        if arm_qpos.shape != (ARM_DOF,) or not np.all(np.isfinite(arm_qpos)):
            self._warn_safety(group_name, f"invalid feedback qpos ignored: {arm_qpos}")
            return cached
        self._read_group_currents(group_name, arm)
        # The seventh EVA state dimension is the normalized gripper position.
        cached = self._cached_group_qpos(group_name)
        return np.concatenate([arm_qpos, [cached[GRIPPER_INDEX]]], axis=0).astype(np.float32)

    def _read_group_currents(self, group_name: str, arm: Any) -> None:
        try:
            currents = np.asarray(arm.get_joint_currents(), dtype=np.float64)[:ARM_DOF]
        except Exception as exc:
            logger.debug("Failed to read ARX X5 %s joint currents: %s", group_name, exc)
            return
        if currents.shape != (ARM_DOF,) or not np.all(np.isfinite(currents)):
            self._warn_safety(group_name, f"invalid joint-current feedback ignored: {currents}")
            return
        self._last_joint_currents[group_name] = currents.copy()

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

    def slow_home(self, should_stop: Any | None = None) -> bool:
        """Send a synchronized home trajectory through the SDK-V2 interpolator."""
        starts: dict[str, np.ndarray] = {}
        arms: dict[str, Any] = {}
        for group_name in GROUP_NAMES:
            if group_name in self._disabled_groups:
                continue
            arm = self._connect(group_name)
            if arm is None:
                raise RuntimeError(f"Cannot home offline ARX X5 arm: {group_name}")
            fault = getattr(arm, "fault", None)
            offline_joints = tuple(getattr(arm, "offline_joints", ()))
            if fault is not None or offline_joints:
                details = []
                if offline_joints:
                    details.append(f"offline_joints={self._format_joint_names(offline_joints)}")
                if fault is not None:
                    details.append(f"fault={fault}")
                raise RuntimeError(
                    f"Refusing to home unhealthy ARX X5 arm {group_name}: {', '.join(details)}"
                )
            positions = np.asarray(arm.get_joint_positions(), dtype=np.float64)[:ARM_DOF]
            if positions.shape != (ARM_DOF,) or not np.all(np.isfinite(positions)):
                raise RuntimeError(f"Invalid live ARX X5 qpos for {group_name}: {positions}")
            starts[group_name] = positions
            arms[group_name] = arm

        duration_s = ARX_X5_HOME_DURATION_S
        target_name = "zero" if self._start_at_zero else "home"
        logger.info(
            "Official ARX X5 %s trajectory starting: duration=%.2f s",
            target_name,
            duration_s,
        )

        for arm in arms.values():
            arm.set_joint_positions(self._startup_arm_qpos.tolist(), duration=duration_s)

        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                logger.info("Official ARX X5 %s trajectory interrupted", target_name)
                return False
            time.sleep(max(0.0, min(0.02, deadline - time.monotonic())))

        with self._lock:
            for group_name in starts:
                offset = self._offset(group_name)
                self._qpos[offset : offset + ARM_DOF] = self._startup_arm_qpos
        logger.info("Official ARX X5 %s trajectory complete", target_name)
        return True

    def apply_action(self, wire_action: WireAction) -> None:
        try:
            parts = split_dual_action(wire_action.action)
            limited_parts: dict[str, np.ndarray] = {}
            for group_name, part in parts.items():
                arm_command, was_limited = limit_arm_command(part[:ARM_DOF])
                limited_part = part.copy()
                limited_part[:ARM_DOF] = arm_command.astype(np.float32)
                limited_parts[group_name] = limited_part
                if was_limited:
                    self._warn_safety(
                        group_name,
                        f"target clipped to official software joint limits: {part[:ARM_DOF]}",
                    )
            parts = limited_parts
        except ValueError as exc:
            self._warn_safety("dual_arm", f"unsafe command ignored: {exc}")
            return
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
                accepted = arm.set_joint_positions(
                    part[:ARM_DOF].astype(np.float64).tolist(),
                    duration=self._action_duration_s,
                )
            except Exception as exc:
                self._mark_offline(group_name, exc)
                continue
            if not accepted:
                self._warn_safety(group_name, "official SDK rejected joint target")
                continue

            offset = self._offset(group_name)
            qpos[offset : offset + GROUP_DOF] = part
            self._publish_gripper_if_needed(group_name, float(part[GRIPPER_INDEX]))

        with self._lock:
            self._qpos = qpos.copy()

    def _warn_safety(self, group_name: str, message: str) -> None:
        now = time.monotonic()
        if (
            now - self._last_safety_warning_time.get(group_name, 0.0)
            < ARX_X5_SAFETY_WARNING_INTERVAL_SEC
        ):
            return
        self._last_safety_warning_time[group_name] = now
        logger.warning("ARX X5 %s safety limiter: %s", group_name, message)

    def _publish_gripper_if_needed(self, group_name: str, gripper_scalar: float) -> bool:
        if group_name in self._last_gripper_command:
            delta = abs(gripper_scalar - self._last_gripper_command[group_name])
            if delta < ARX_X5_GRIPPER_COMMAND_DELTA:
                return True

        arm = self._connect(group_name)
        if arm is None:
            return False
        gripper_error = int(getattr(arm, "gripper_error", 0))
        if gripper_error:
            error_name = str(getattr(arm, "gripper_error_name", "unknown"))
            self._warn_safety(
                group_name,
                f"gripper fault error={gripper_error} ({error_name})",
            )
            return False
        try:
            accepted = arm.set_gripper_pos(self._scalar_to_gripper_pos(gripper_scalar))
        except Exception as exc:
            self._mark_offline(group_name, exc)
            return False
        if not accepted:
            self._warn_safety(group_name, "official SDK rejected gripper target")
            return False
        self._last_gripper_command[group_name] = float(np.clip(gripper_scalar, 0.0, 1.0))
        return True

    def go_home(self) -> dict[str, bool]:
        """Send each connected ARX X5 arm back to its SDK home pose."""
        results: dict[str, bool] = {}
        for group_name in GROUP_NAMES:
            if group_name in self._disabled_groups:
                continue
            arm = self._connect(group_name)
            if arm is None:
                results[group_name] = False
                continue
            try:
                arm.set_joint_positions(ARX_X5_INITIAL_ARM_QPOS.tolist(), duration=2.0)
            except Exception as exc:
                self._mark_offline(group_name, exc)
                results[group_name] = False
                continue
            offset = self._offset(group_name)
            with self._lock:
                self._qpos[offset : offset + ARM_DOF] = ARX_X5_INITIAL_ARM_QPOS
            results[group_name] = True
            logger.info("Sent %s to ARX X5 home pose", group_name)
        return results

    def hardware_status(self) -> dict[str, str]:
        now = time.monotonic()
        status: dict[str, str] = {}
        for group_name in GROUP_NAMES:
            if group_name in self._disabled_groups:
                status[group_name] = "disabled"
            elif group_name in self._arms:
                arm = self._arms[group_name]
                fault = getattr(arm, "fault", None)
                offline_joints = tuple(getattr(arm, "offline_joints", ()))
                gripper_error = int(getattr(arm, "gripper_error", 0))
                if fault is not None:
                    status[group_name] = f"fault({fault})"
                    continue
                if offline_joints:
                    status[group_name] = f"offline({self._format_joint_names(offline_joints)})"
                    continue
                if gripper_error:
                    error_name = str(getattr(arm, "gripper_error_name", "unknown"))
                    status[group_name] = f"gripper_fault({gripper_error}:{error_name})"
                    continue
                currents = self._last_joint_currents.get(group_name)
                if currents is None:
                    status[group_name] = "online"
                else:
                    peak_index = int(np.argmax(np.abs(currents)))
                    status[group_name] = (
                        f"online(current_max={abs(currents[peak_index]):.2f}@J{peak_index + 1})"
                    )
            elif group_name in self._offline_until and self._offline_until[group_name] > now:
                remaining_s = self._offline_until[group_name] - now
                status[group_name] = f"retrying({remaining_s:.1f}s)"
            else:
                status[group_name] = "connecting"
        return status

    @staticmethod
    def _format_joint_names(joints: tuple[Any, ...]) -> str:
        names = []
        for joint in joints:
            text = str(joint)
            if text.isdigit():
                index = int(text)
                text = f"J{index + 1}" if 0 <= index < ARM_DOF else f"J{index}"
            names.append(text)
        return ",".join(names)

    def close(self) -> None:
        """Put every connected arm into protect mode and stop its SDK worker."""
        for group_name, arm in list(self._arms.items()):
            try:
                arm.protect_mode()
            except Exception as exc:
                logger.debug("Failed to stop ARX X5 %s cleanly: %s", group_name, exc)
            finally:
                try:
                    arm.close()
                except Exception as exc:
                    logger.debug("Failed to close ARX X5 %s SDK worker: %s", group_name, exc)
        self._arms.clear()


__all__ = [
    "ARM_DOF",
    "DEFAULT_ARM_TYPE",
    "DEFAULT_GRIPPER_CLOSE_POS",
    "DEFAULT_GRIPPER_OPEN_POS",
    "DEFAULT_LEFT_CAN_PORT",
    "DEFAULT_RIGHT_CAN_PORT",
    "DEFAULT_ARX_X5_TYPE",
    "GROUP_DOF",
    "GROUP_NAMES",
    "GRIPPER_INDEX",
    "TOTAL_DOF",
    "ARX_X5_2023_GRIPPER_CLOSE_POS",
    "ARX_X5_2023_GRIPPER_OPEN_POS",
    "ARX_X5_2025_GRIPPER_CLOSE_POS",
    "ARX_X5_2025_GRIPPER_OPEN_POS",
    "ARX_X5_ARM_JOINT_LIMITS",
    "ARX_X5_HOME_DURATION_S",
    "ARX_X5_INITIAL_ARM_QPOS",
    "ArxX5ArmConfig",
    "ArxX5DualArm",
    "clip_group_action",
    "initial_qpos",
    "limit_arm_command",
    "normalize_group_name",
    "parse_disabled_groups",
    "split_dual_action",
]
