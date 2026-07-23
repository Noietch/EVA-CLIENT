"""I2RT YAM follower and teaching-handle adapters for the EVA execution node."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from examples.hardware.i2rt.wire import WireAction

logger = logging.getLogger(__name__)

GROUP_DOF = 7
ARM_DOF = 6
GRIPPER_INDEX = 6

YAM_ARM_LOWER = np.asarray(
    [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944],
    dtype=np.float32,
)
YAM_ARM_UPPER = np.asarray(
    [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944],
    dtype=np.float32,
)
# The official SDK expands each model limit by 0.15 rad before applying its
# final motor-command clip. Logical EVA targets stay inside the model limits;
# bounded tracking trim may use that existing headroom to remove steady-state
# error when a target lies exactly on a model boundary (notably q2/q3 at zero).
SDK_LIMIT_HEADROOM_RAD = 0.15
TRACKING_ARM_LOWER = YAM_ARM_LOWER.astype(np.float64) - SDK_LIMIT_HEADROOM_RAD
TRACKING_ARM_UPPER = YAM_ARM_UPPER.astype(np.float64) + SDK_LIMIT_HEADROOM_RAD
DEFAULT_ARM_QPOS = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
RETRY_INTERVAL_SEC = 5.0
MIN_SERVO_PERIOD_S = 0.002


class I2RTArmConfig(Protocol):
    @property
    def group_names(self) -> tuple[str, ...]: ...

    @property
    def follower_can_channels(self) -> dict[str, str]: ...

    @property
    def leader_can_channels(self) -> dict[str, str]: ...

    @property
    def disabled_groups(self) -> tuple[str, ...]: ...

    @property
    def arm_type(self) -> str: ...

    @property
    def gripper_type(self) -> str: ...

    @property
    def sim(self) -> bool: ...

    @property
    def enable_auto_recovery(self) -> bool: ...

    @property
    def command_timeout_s(self) -> float: ...

    @property
    def idle_mode(self) -> str: ...

    @property
    def startup_position(self) -> str: ...

    @property
    def startup_duration_s(self) -> float: ...

    @property
    def joint4_kp(self) -> float | None: ...

    @property
    def end_effector_mass(self) -> float | None: ...

    @property
    def gravity_comp_factor(self) -> tuple[float, ...] | None: ...

    @property
    def gripper_limits_override(self) -> tuple[float, float] | None: ...

    @property
    def tracking_ki(self) -> float: ...

    @property
    def tracking_trim_limit(self) -> float: ...

    @property
    def tracking_deadband(self) -> float: ...

    @property
    def tracking_settle_delay_s(self) -> float: ...

    @property
    def startup_trim_duration_s(self) -> float: ...


RobotFactory = Callable[..., Any]


def clip_group_action(action: np.ndarray) -> np.ndarray:
    """Validate and clip one YAM 7-D arm + normalized-gripper command."""
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != (GROUP_DOF,):
        raise ValueError(f"Expected a {GROUP_DOF}-D YAM command, got {vector.shape}")
    clipped = vector.copy()
    clipped[:ARM_DOF] = np.clip(clipped[:ARM_DOF], YAM_ARM_LOWER, YAM_ARM_UPPER)
    clipped[GRIPPER_INDEX] = float(np.clip(clipped[GRIPPER_INDEX], 0.0, 1.0))
    return clipped


def split_action(action: np.ndarray, group_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Split EVA's flat action into one clipped 7-D command per YAM group."""
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = len(group_names) * GROUP_DOF
    if vector.shape != (expected,):
        raise ValueError(f"Expected a {expected}-D I2RT action, got {vector.shape}")
    return {
        name: clip_group_action(vector[index * GROUP_DOF : (index + 1) * GROUP_DOF])
        for index, name in enumerate(group_names)
    }


def flatten_state(state: dict[str, np.ndarray], group_names: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([state[name] for name in group_names]).astype(np.float32)


def _sdk_factory(
    *,
    channel: str,
    arm_type: str,
    gripper_type: str,
    sim: bool,
    enable_auto_recovery: bool,
    zero_gravity_mode: bool = True,
    ee_mass: float | None = None,
    gravity_comp_factor: np.ndarray | None = None,
    gripper_limits_override: np.ndarray | None = None,
) -> Any:
    try:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType
    except ImportError as exc:
        raise RuntimeError(
            "I2RT SDK is unavailable; run examples/hardware/i2rt/setup_sdk.sh"
        ) from exc
    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name(arm_type),
        gripper_type=GripperType.from_string_name(gripper_type),
        zero_gravity_mode=zero_gravity_mode,
        ee_mass=ee_mass,
        gravity_comp_factor=gravity_comp_factor,
        gripper_limits_override=gripper_limits_override,
        sim=sim,
        enable_auto_recovery=enable_auto_recovery,
    )


class I2RTYamFollowers:
    """One or two YAM followers with retry, state caching, and a command watchdog."""

    def __init__(self, config: I2RTArmConfig, factory: RobotFactory | None = None) -> None:
        self._config = config
        self._factory = factory or _sdk_factory
        self._lock = threading.RLock()
        self._robots: dict[str, Any] = {}
        self._offline_until: dict[str, float] = {}
        self._disabled_groups = set(config.disabled_groups)
        self._qpos = np.concatenate([DEFAULT_ARM_QPOS.copy() for _ in config.group_names]).astype(
            np.float32
        )
        self._last_command_time: float | None = None
        self._last_servo_time: float | None = None
        self._control_active = False
        self._active_targets: dict[str, np.ndarray] = {}
        self._tracking_trim = {
            name: np.zeros(ARM_DOF, dtype=np.float64) for name in config.group_names
        }
        self._last_target = {
            name: DEFAULT_ARM_QPOS.copy().astype(np.float64) for name in config.group_names
        }
        now = time.monotonic()
        self._last_target_time = dict.fromkeys(config.group_names, now)
        self._stationary_since = dict.fromkeys(config.group_names, now)

    def _offset(self, group_name: str) -> int:
        return self._config.group_names.index(group_name) * GROUP_DOF

    def _read_group_qpos(self, robot: Any) -> np.ndarray:
        raw = np.asarray(robot.get_joint_pos(), dtype=np.float32).reshape(-1)
        if raw.shape != (GROUP_DOF,):
            raise ValueError(f"Expected a {GROUP_DOF}-D YAM + gripper state, got {raw.shape}")
        state = raw.copy()
        state[GRIPPER_INDEX] = float(np.clip(state[GRIPPER_INDEX], 0.0, 1.0))
        return state

    def _integrate_tracking_trim(
        self,
        group_name: str,
        target: np.ndarray,
        measured: np.ndarray,
        dt: float,
    ) -> None:
        if self._config.tracking_ki <= 0 or self._config.tracking_trim_limit <= 0:
            return
        error = target[:ARM_DOF].astype(np.float64) - measured[:ARM_DOF].astype(np.float64)
        error[np.abs(error) <= self._config.tracking_deadband] = 0.0
        # Never integrate farther into a model limit. q2/q3 zero is a real
        # kinematic boundary on the current cell; treating its small positive
        # residual as free-space sag can learn a large negative trim that then
        # corrupts every raised-arm target.
        at_lower = target[:ARM_DOF] <= YAM_ARM_LOWER + self._config.tracking_deadband
        at_upper = target[:ARM_DOF] >= YAM_ARM_UPPER - self._config.tracking_deadband
        error[at_lower & (error < 0)] = 0.0
        error[at_upper & (error > 0)] = 0.0
        trim = self._tracking_trim[group_name]
        trim += self._config.tracking_ki * error * max(dt, 0.0)
        np.clip(
            trim,
            -self._config.tracking_trim_limit,
            self._config.tracking_trim_limit,
            out=trim,
        )

    def _tracking_command(self, group_name: str, target: np.ndarray) -> np.ndarray:
        command = target.astype(np.float64, copy=True)
        command[:ARM_DOF] += self._tracking_trim[group_name]
        command[:ARM_DOF] = np.clip(command[:ARM_DOF], TRACKING_ARM_LOWER, TRACKING_ARM_UPPER)
        return command

    def _update_runtime_tracking_trim(
        self,
        group_name: str,
        target: np.ndarray,
        measured: np.ndarray,
        now: float,
    ) -> None:
        previous_time = self._last_target_time[group_name]
        dt = min(max(now - previous_time, 0.0), 0.05)
        previous = self._last_target[group_name]
        speed = float(np.max(np.abs(target[:ARM_DOF] - previous[:ARM_DOF])) / max(dt, 1e-6))
        if speed > 0.05:
            self._stationary_since[group_name] = now
            # Static feed-forward learned at one pose is not transferable to a
            # different gravity configuration. Relearn only after the new target
            # settles instead of dragging a zero-pose trim through the workspace.
            self._tracking_trim[group_name].fill(0.0)
        elif now - self._stationary_since[group_name] >= self._config.tracking_settle_delay_s:
            self._integrate_tracking_trim(group_name, target, measured, dt)
        self._last_target[group_name] = target.astype(np.float64, copy=True)
        self._last_target_time[group_name] = now

    def _mark_offline(self, group_name: str, exc: Exception) -> None:
        robot = self._robots.pop(group_name, None)
        if robot is not None:
            try:
                robot.close()
            except Exception:
                logger.debug("Failed to close offline I2RT %s", group_name, exc_info=True)
        self._offline_until[group_name] = time.monotonic() + RETRY_INTERVAL_SEC
        logger.warning(
            "I2RT follower %s offline; retrying in %.1f s: %s",
            group_name,
            RETRY_INTERVAL_SEC,
            exc,
        )

    def _connect(self, group_name: str) -> Any | None:
        if group_name in self._disabled_groups:
            return None
        robot = self._robots.get(group_name)
        if robot is not None:
            return robot
        if time.monotonic() < self._offline_until.get(group_name, 0.0):
            return None
        robot = None
        try:
            robot = self._factory(
                channel=self._config.follower_can_channels[group_name],
                arm_type=self._config.arm_type,
                gripper_type=self._config.gripper_type,
                sim=self._config.sim,
                enable_auto_recovery=self._config.enable_auto_recovery,
                zero_gravity_mode=self._config.idle_mode == "gravity_comp",
                ee_mass=self._config.end_effector_mass,
                gravity_comp_factor=(
                    None
                    if self._config.gravity_comp_factor is None
                    else np.asarray(self._config.gravity_comp_factor, dtype=np.float64)
                ),
                gripper_limits_override=(
                    None
                    if self._config.gripper_limits_override is None
                    else np.asarray(self._config.gripper_limits_override, dtype=np.float64)
                ),
            )
            if int(robot.num_dofs()) != GROUP_DOF:
                raise ValueError(
                    f"{group_name} has {robot.num_dofs()} DoF; expected {GROUP_DOF} "
                    f"for arm + {self._config.gripper_type} gripper."
                )
            if self._config.joint4_kp is not None:
                update_kp_kd = getattr(robot, "update_kp_kd", None)
                if not callable(update_kp_kd):
                    if not self._config.sim:
                        raise AttributeError(
                            f"{type(robot).__name__} does not support position gain updates"
                        )
                else:
                    kp = np.asarray([80.0, 80.0, 80.0, self._config.joint4_kp, 10.0, 10.0])
                    kd = np.asarray([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
                    gripper_gains = {
                        "linear_3507": (10.0, 0.3),
                        "crank_4310": (20.0, 0.5),
                        "linear_4310": (20.0, 0.5),
                        "flexible_4310": (20.0, 0.5),
                    }
                    gripper_kp, gripper_kd = gripper_gains[self._config.gripper_type]
                    kp = np.append(kp, gripper_kp)
                    kd = np.append(kd, gripper_kd)
                    update_kp_kd(kp, kd)
                    logger.info(
                        "Set I2RT follower %s joint4 kp to %.1f",
                        group_name,
                        self._config.joint4_kp,
                    )
            if self._config.end_effector_mass is not None:
                logger.info(
                    "Set I2RT follower %s end-effector model mass to %.3f kg",
                    group_name,
                    self._config.end_effector_mass,
                )
            if self._config.gravity_comp_factor is not None:
                logger.info(
                    "Set I2RT follower %s gravity compensation factors to %s",
                    group_name,
                    self._config.gravity_comp_factor,
                )
            current = self._read_group_qpos(robot)
        except Exception as exc:
            if robot is not None:
                try:
                    robot.close()
                except Exception:
                    logger.debug(
                        "Failed to close rejected I2RT %s",
                        group_name,
                        exc_info=True,
                    )
            self._mark_offline(group_name, exc)
            return None

        offset = self._offset(group_name)
        with self._lock:
            self._qpos[offset : offset + GROUP_DOF] = current
        self._robots[group_name] = robot
        self._offline_until.pop(group_name, None)
        logger.info(
            "Connected I2RT follower %s on %s",
            group_name,
            self._config.follower_can_channels[group_name],
        )
        return robot

    def read_state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        for group_name in self._config.group_names:
            offset = self._offset(group_name)
            with self._lock:
                current = self._qpos[offset : offset + GROUP_DOF].copy()
            robot = self._connect(group_name)
            if robot is not None:
                try:
                    current = self._read_group_qpos(robot)
                except Exception as exc:
                    self._mark_offline(group_name, exc)
            with self._lock:
                self._qpos[offset : offset + GROUP_DOF] = current
            state[group_name] = current
        return state

    def move_to_startup_position(self) -> None:
        """Connect all followers and optionally interpolate their arm joints to zero."""
        initial = self.read_state()
        if self._config.startup_position != "zero":
            return
        duration_s = self._config.startup_duration_s
        if duration_s <= 0:
            raise ValueError("I2RT startup duration must be positive")
        missing = [
            name
            for name in self._config.group_names
            if name not in self._disabled_groups and name not in self._robots
        ]
        if missing:
            raise RuntimeError(f"Cannot move offline I2RT followers to zero: {missing}")

        targets: dict[str, np.ndarray] = {}
        for group_name, current in initial.items():
            target = current.astype(np.float64, copy=True)
            target[:ARM_DOF] = 0.0
            targets[group_name] = target

        control_hz = 50.0
        steps = max(int(round(duration_s * control_hz)), 1)
        logger.info(
            "Moving I2RT followers to all-zero arm position over %.1f s",
            duration_s,
        )
        for step in range(1, steps + 1):
            unit = step / steps
            alpha = unit * unit * (3.0 - 2.0 * unit)
            for group_name, robot in tuple(self._robots.items()):
                command = initial[group_name] + alpha * (targets[group_name] - initial[group_name])
                try:
                    robot.command_joint_pos(command.astype(np.float64, copy=True))
                except Exception as exc:
                    self._mark_offline(group_name, exc)
                    raise RuntimeError(
                        f"I2RT follower {group_name} failed while moving to zero"
                    ) from exc
            if step < steps:
                time.sleep(duration_s / steps)

        with self._lock:
            for group_name, target in targets.items():
                offset = self._offset(group_name)
                self._qpos[offset : offset + GROUP_DOF] = target.astype(np.float32)

        if self._config.tracking_ki > 0 and self._config.startup_trim_duration_s > 0:
            trim_period_s = 1.0 / control_hz
            trim_steps = max(int(round(self._config.startup_trim_duration_s * control_hz)), 1)
            settled_steps = 0
            logger.info(
                "Settling I2RT all-zero target with integral trim: "
                "ki=%.2f limit=%.3f rad deadband=%.4f rad duration=%.1f s",
                self._config.tracking_ki,
                self._config.tracking_trim_limit,
                self._config.tracking_deadband,
                self._config.startup_trim_duration_s,
            )
            for step in range(trim_steps):
                max_error = 0.0
                for group_name, robot in tuple(self._robots.items()):
                    try:
                        current = self._read_group_qpos(robot)
                        error = targets[group_name][:ARM_DOF] - current[:ARM_DOF]
                        max_error = max(max_error, float(np.max(np.abs(error))))
                        self._integrate_tracking_trim(
                            group_name,
                            targets[group_name],
                            current,
                            trim_period_s,
                        )
                        command = self._tracking_command(group_name, targets[group_name])
                        robot.command_joint_pos(command.copy())
                        offset = self._offset(group_name)
                        with self._lock:
                            self._qpos[offset : offset + GROUP_DOF] = current
                    except Exception as exc:
                        self._mark_offline(group_name, exc)
                        raise RuntimeError(
                            f"I2RT follower {group_name} failed while trimming zero"
                        ) from exc
                if max_error <= self._config.tracking_deadband:
                    settled_steps += 1
                    if settled_steps >= 5:
                        break
                else:
                    settled_steps = 0
                if step + 1 < trim_steps:
                    time.sleep(trim_period_s)

        time.sleep(0.5)
        measured = self.read_state()
        residuals = {
            name: np.round(qpos[:ARM_DOF].astype(np.float64), 5).tolist()
            for name, qpos in measured.items()
            if name not in self._disabled_groups
        }
        max_error = max(
            (float(np.max(np.abs(qpos[:ARM_DOF]))) for qpos in measured.values()),
            default=0.0,
        )
        log = logger.info if max_error <= 0.03 else logger.warning
        log(
            "I2RT all-zero move settled: residuals_rad=%s max_error=%.5f trim_rad=%s",
            residuals,
            max_error,
            self.tracking_trim(),
        )

    def _servo_active_targets(self, now: float) -> None:
        qpos = self._qpos.copy()
        for group_name, part in self._active_targets.items():
            if group_name in self._disabled_groups:
                continue
            offset = self._offset(group_name)
            robot = self._connect(group_name)
            if robot is None:
                continue
            try:
                measured = self._read_group_qpos(robot)
                qpos[offset : offset + GROUP_DOF] = measured
                self._update_runtime_tracking_trim(group_name, part, measured, now)
                command = self._tracking_command(group_name, part)
                robot.command_joint_pos(command.astype(np.float64, copy=True))
            except Exception as exc:
                self._mark_offline(group_name, exc)
                continue

        with self._lock:
            self._qpos = qpos.astype(np.float32)
        self._last_servo_time = now

    def apply_action(self, wire_action: WireAction) -> None:
        parts = split_action(wire_action.action, self._config.group_names)
        if wire_action.target != "real":
            qpos = self._qpos.copy()
            for group_name, part in parts.items():
                if group_name in self._disabled_groups:
                    continue
                offset = self._offset(group_name)
                qpos[offset : offset + GROUP_DOF] = part
            with self._lock:
                self._qpos = qpos.astype(np.float32)
            return

        now = time.monotonic()
        for group_name, part in parts.items():
            if group_name not in self._disabled_groups:
                self._active_targets[group_name] = part.astype(np.float64, copy=True)
        self._servo_active_targets(now)
        if wire_action.target == "real":
            self._last_command_time = now
            self._control_active = True

    def tracking_trim(self) -> dict[str, list[float]]:
        return {name: np.round(trim, 5).tolist() for name, trim in self._tracking_trim.items()}

    def watchdog_tick(self) -> None:
        """Return all connected arms to the configured safe idle after command loss."""
        if not self._control_active or self._last_command_time is None:
            return
        now = time.monotonic()
        if self._config.command_timeout_s <= 0 or (
            now - self._last_command_time <= self._config.command_timeout_s
        ):
            if self._last_servo_time is None or now - self._last_servo_time >= MIN_SERVO_PERIOD_S:
                self._servo_active_targets(now)
            return
        self.enter_safe_idle()
        logger.warning(
            "I2RT command watchdog expired after %.3f s; followers entered %s idle",
            self._config.command_timeout_s,
            self._config.idle_mode,
        )

    def enter_safe_idle(self) -> None:
        """Enter gravity compensation or hold the measured position, as configured."""
        for group_name, robot in tuple(self._robots.items()):
            try:
                if self._config.idle_mode == "hold_position":
                    current = np.asarray(robot.get_joint_pos(), dtype=np.float64).reshape(-1)
                    robot.command_joint_pos(current.copy())
                    continue
                enter_idle = getattr(robot, "enter_gravity_comp_idle", None)
                if callable(enter_idle):
                    enter_idle()
                    continue
                enable_gravity_comp = getattr(robot, "enable_gravity_comp", None)
                if self._config.sim and callable(enable_gravity_comp):
                    enable_gravity_comp()
                    continue
                raise AttributeError(
                    f"{type(robot).__name__} has no gravity-compensation idle method"
                )
            except Exception as exc:
                self._mark_offline(group_name, exc)
        self._control_active = False
        self._last_command_time = None
        self._last_servo_time = None
        self._active_targets.clear()

    def hardware_status(self) -> dict[str, str]:
        now = time.monotonic()
        result: dict[str, str] = {}
        for name in self._config.group_names:
            if name in self._disabled_groups:
                result[name] = "disabled"
            elif name in self._robots:
                result[name] = "online"
            elif self._offline_until.get(name, 0.0) > now:
                result[name] = "retrying"
            else:
                result[name] = "connecting"
        return result

    def close(self) -> None:
        for group_name, robot in tuple(self._robots.items()):
            try:
                robot.close()
            except Exception:
                logger.exception("Failed to close I2RT follower %s", group_name)
        self._robots.clear()


class I2RTYamLeaders:
    """One or more teaching-handle YAMs used for collection and HIL."""

    def __init__(self, config: I2RTArmConfig, factory: RobotFactory | None = None) -> None:
        self._config = config
        self._factory = factory or _sdk_factory
        self._robots: dict[str, Any] = {}

    def connect(self) -> None:
        configured = [
            name
            for name in self._config.group_names
            if name in self._config.leader_can_channels and name not in self._config.disabled_groups
        ]
        if not configured:
            raise RuntimeError("No I2RT leader CAN channel is configured")
        try:
            for name in configured:
                if name in self._robots:
                    continue
                robot = self._factory(
                    channel=self._config.leader_can_channels[name],
                    arm_type=self._config.arm_type,
                    gripper_type="yam_teaching_handle",
                    sim=self._config.sim,
                    enable_auto_recovery=self._config.enable_auto_recovery,
                )
                if int(robot.num_dofs()) != ARM_DOF:
                    raise ValueError(
                        f"Leader {name} has {robot.num_dofs()} DoF; expected {ARM_DOF}"
                    )
                self._robots[name] = robot
                logger.info(
                    "Connected I2RT leader %s on %s",
                    name,
                    self._config.leader_can_channels[name],
                )
        except Exception:
            self.close()
            raise

    def _read_gripper(self, robot: Any) -> float:
        if self._config.sim:
            return 1.0
        encoder_states = robot.motor_chain.get_same_bus_device_states()
        if not encoder_states:
            raise RuntimeError("I2RT teaching-handle encoder state is not ready")
        return float(np.clip(1.0 - encoder_states[0].position, 0.0, 1.0))

    def read_buttons(self) -> dict[str, tuple[bool, ...]]:
        """Return cached teaching-handle digital inputs for connected leaders.

        The official SDK reports the two inputs as ``(SYNC, RECORD)``. This read
        does not issue another CAN request; it consumes the snapshot refreshed by
        the SDK's existing motor-control thread.
        """
        buttons: dict[str, tuple[bool, ...]] = {}
        for name, robot in self._robots.items():
            if self._config.sim:
                buttons[name] = (False, False)
                continue
            encoder_states = robot.motor_chain.get_same_bus_device_states()
            if not encoder_states:
                continue
            buttons[name] = tuple(bool(value) for value in encoder_states[0].io_inputs)
        return buttons

    def read_action(self, unled_follower_anchor: np.ndarray) -> np.ndarray:
        """Read configured leaders and hold unled groups at their supplied anchor."""
        anchor_parts = split_action(unled_follower_anchor, self._config.group_names)
        self.connect()
        parts: list[np.ndarray] = []
        for name in self._config.group_names:
            if name not in self._robots:
                parts.append(anchor_parts[name])
                continue
            robot = self._robots[name]
            arm = np.asarray(robot.get_joint_pos(), dtype=np.float32).reshape(-1)
            if arm.shape != (ARM_DOF,):
                raise ValueError(f"I2RT leader {name} returned shape {arm.shape}")
            parts.append(clip_group_action(np.concatenate([arm, [self._read_gripper(robot)]])))
        return np.concatenate(parts).astype(np.float32)

    def close(self) -> None:
        for name, robot in tuple(self._robots.items()):
            try:
                robot.close()
            except Exception:
                logger.exception("Failed to close I2RT leader %s", name)
        self._robots.clear()


class I2RTYamKinematics:
    """Official I2RT MuJoCo FK exposed in EVA's xyz + wxyz + gripper layout."""

    def __init__(self, config: I2RTArmConfig) -> None:
        try:
            import mujoco
            from i2rt.robots.kinematics import Kinematics
            from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
        except ImportError as exc:
            raise RuntimeError(
                "I2RT kinematics is unavailable; run examples/hardware/i2rt/setup_sdk.sh"
            ) from exc
        xml_path = combine_arm_and_gripper_xml(
            ArmType.from_string_name(config.arm_type),
            GripperType.NO_GRIPPER,
        )
        self._kinematics = Kinematics(xml_path, "grasp_site")
        self._mujoco = mujoco
        self._group_names = config.group_names

    def group_eef(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for group_name in self._group_names:
            qpos = clip_group_action(state[group_name])
            pose = self._kinematics.fk(qpos[:ARM_DOF].astype(np.float64))
            quat_wxyz = np.empty(4, dtype=np.float64)
            self._mujoco.mju_mat2Quat(quat_wxyz, pose[:3, :3].reshape(-1))
            result[group_name] = np.concatenate(
                [pose[:3, 3], quat_wxyz, [qpos[GRIPPER_INDEX]]]
            ).astype(np.float32)
        return result

    def flat_eef(self, qpos: np.ndarray) -> np.ndarray:
        state = split_action(qpos, self._group_names)
        eef = self.group_eef(state)
        return np.concatenate([eef[name] for name in self._group_names]).astype(np.float32)
