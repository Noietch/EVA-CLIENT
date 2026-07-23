#!/usr/bin/env python3
"""I2RT YAM execution-layer node for EVA's ZMQ transport."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import threading
import time

import numpy as np
import zmq

from examples.hardware.i2rt.camera import (
    CameraSpec,
    RealSenseCameraCache,
    list_realsense_devices,
    parse_camera_specs,
)
from examples.hardware.i2rt.robot import (
    I2RTYamFollowers,
    I2RTYamKinematics,
    I2RTYamLeaders,
    flatten_state,
    split_action,
)
from examples.hardware.i2rt.wire import (
    COLLECTION_START_TARGET,
    COLLECTION_STOP_TARGET,
    HIL_START_TARGET,
    HIL_STOP_TARGET,
    WireAction,
    WireObservation,
    pack_observation,
    unpack_action,
)

logger = logging.getLogger(__name__)

ROBOT_GROUPS: dict[str, tuple[str, ...]] = {
    "i2rt_yam": ("arm",),
    "i2rt_dual_yam": ("left_arm", "right_arm"),
}
ROBOT_CAMERA_KEYS: dict[str, tuple[str, ...]] = {
    "i2rt_yam": ("cam_high", "cam_wrist"),
    "i2rt_dual_yam": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
}
LEADER_RECORD_BUTTON_INDEX = 1
LEADER_BUTTON_DEBOUNCE_S = 0.08


class _RisingEdgeDebouncer:
    """Debounce a button and emit only a stable released-to-pressed edge."""

    def __init__(self, debounce_s: float) -> None:
        self._debounce_s = debounce_s
        self._raw: bool | None = None
        self._stable: bool | None = None
        self._changed_at = 0.0

    def update(self, pressed: bool, now: float) -> bool:
        if self._raw is None:
            # A button already held while the leader connects is a baseline, not
            # an operator command. It must be released before it can trigger.
            self._raw = pressed
            self._stable = pressed
            self._changed_at = now
            return False
        if pressed != self._raw:
            self._raw = pressed
            self._changed_at = now
            return False
        if pressed == self._stable or now - self._changed_at < self._debounce_s:
            return False
        self._stable = pressed
        return pressed


@dataclasses.dataclass(frozen=True)
class I2RTZmqConfig:
    observation_endpoint: str
    action_endpoint: str
    robot_name: str
    group_names: tuple[str, ...]
    follower_can_channels: dict[str, str]
    leader_can_channels: dict[str, str]
    disabled_groups: tuple[str, ...]
    arm_type: str
    gripper_type: str
    sim: bool
    enable_auto_recovery: bool
    command_timeout_s: float
    idle_mode: str
    startup_position: str
    startup_duration_s: float
    joint4_kp: float | None
    end_effector_mass: float | None
    gravity_comp_factor: tuple[float, ...] | None
    gripper_limits_override: tuple[float, float] | None
    allow_gripper_calibration: bool
    tracking_ki: float
    tracking_trim_limit: float
    tracking_deadband: float
    tracking_settle_delay_s: float
    startup_trim_duration_s: float
    cameras: tuple[CameraSpec, ...]
    control_rate_hz: float
    publish_rate_hz: float
    status_log_interval_s: float


def _parse_group_map(
    values: list[str],
    group_names: tuple[str, ...],
    defaults: dict[str, str] | None = None,
) -> dict[str, str]:
    result = dict(defaults or {})
    for value in values:
        if "=" not in value:
            if len(group_names) != 1:
                raise ValueError(f"Expected GROUP=CAN_INTERFACE, got {value!r}")
            group_name, interface = group_names[0], value.strip()
        else:
            group_name, interface = (part.strip() for part in value.split("=", 1))
        if group_name not in group_names:
            raise ValueError(f"Unknown I2RT group {group_name!r}; expected one of {group_names}")
        if not interface:
            raise ValueError(f"Missing CAN interface in {value!r}")
        result[group_name] = interface
    return result


def _parse_name_list(values: list[str], allowed: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        for raw_name in value.split(","):
            name = raw_name.strip()
            if not name:
                continue
            if name not in allowed:
                raise ValueError(f"Unknown I2RT group {name!r}; expected one of {allowed}")
            if name not in result:
                result.append(name)
    return tuple(result)


class I2RTZmqNode:
    """Bridge EVA wire messages to I2RT follower and leader YAM arms."""

    def __init__(self, config: I2RTZmqConfig) -> None:
        self._config = config
        self._stop = threading.Event()
        self._ctx = zmq.Context.instance()
        self._obs_pub = self._ctx.socket(zmq.PUB)
        self._obs_pub.bind(config.observation_endpoint)
        self._action_sub = self._ctx.socket(zmq.SUB)
        self._action_sub.bind(config.action_endpoint)
        self._action_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._action_sub.setsockopt(zmq.RCVTIMEO, 0)

        self._followers = I2RTYamFollowers(config)
        self._leaders = I2RTYamLeaders(config)
        self._kinematics = I2RTYamKinematics(config)
        self._cameras = RealSenseCameraCache(config.cameras)
        self._collection_active = False
        self._hil_active = False
        self._hil_mode = "relative"
        self._hil_error = ""
        self._leader_anchor: np.ndarray | None = None
        self._follower_anchor: np.ndarray | None = None
        self._latest_leader_action: np.ndarray | None = None
        self._tracking_error: dict[str, list[float]] = {}
        self._received_actions = 0
        self._leader_control_updates = 0
        self._published_observations = 0
        self._record_button = _RisingEdgeDebouncer(LEADER_BUTTON_DEBOUNCE_S)
        self._operator_event = ""
        self._operator_event_id = 0
        now = time.monotonic()
        self._last_status_time = now
        self._next_status_time = now + max(config.status_log_interval_s, 0.0)
        self._last_status_action_count = 0
        self._last_status_control_count = 0
        self._last_status_observation_count = 0
        logger.info(
            "I2RT node ready: robot=%s followers=%s leaders=%s sim=%s",
            config.robot_name,
            config.follower_can_channels,
            config.leader_can_channels,
            config.sim,
        )

    def stop(self) -> None:
        self._stop.set()

    def _start_collection(self) -> None:
        if self._collection_active:
            return
        follower = flatten_state(self._followers.read_state(), self._config.group_names)
        leader = self._leaders.read_action(follower)
        self._leader_anchor = leader
        self._follower_anchor = follower
        self._collection_active = True
        logger.info(
            "I2RT leader-follower collection started: leader_groups=%s unled_groups_hold_anchor=%s",
            tuple(self._config.leader_can_channels),
            tuple(
                name
                for name in self._config.group_names
                if name not in self._config.leader_can_channels
            ),
        )

    def _stop_collection(self) -> None:
        self._collection_active = False
        if not self._hil_active:
            self._leader_anchor = None
            self._follower_anchor = None
            self._latest_leader_action = None
            self._leaders.close()
            self._followers.enter_safe_idle()
        logger.info("I2RT leader-follower collection stopped")

    def _start_hil(self, mode: str) -> None:
        if mode not in {"absolute", "relative"}:
            self._hil_error = f"Unsupported HIL mode: {mode}"
            return
        try:
            follower = flatten_state(self._followers.read_state(), self._config.group_names)
            leader = self._leaders.read_action(follower)
        except Exception as exc:
            self._hil_error = str(exc)
            self._hil_active = False
            logger.exception("Could not start I2RT HIL")
            return
        self._leader_anchor = leader
        self._follower_anchor = follower
        self._hil_mode = mode
        self._hil_error = ""
        self._hil_active = True
        logger.info("I2RT HIL started: mode=%s", mode)

    def _stop_hil(self) -> None:
        self._hil_active = False
        self._leader_anchor = None
        self._follower_anchor = None
        self._latest_leader_action = None
        self._hil_error = ""
        if not self._collection_active:
            self._leaders.close()
            self._followers.enter_safe_idle()
        logger.info("I2RT HIL stopped")

    def _drain_actions(self) -> None:
        while True:
            try:
                payload = self._action_sub.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                action = unpack_action(payload)
            except Exception as exc:
                logger.warning("Dropped malformed I2RT action: %s", exc)
                continue
            if action.target == COLLECTION_START_TARGET:
                try:
                    self._start_collection()
                except Exception as exc:
                    logger.exception("Could not start I2RT collection")
                    self._hil_error = str(exc)
                continue
            if action.target == COLLECTION_STOP_TARGET:
                self._stop_collection()
                continue
            if action.target == HIL_START_TARGET:
                self._start_hil(action.mode or "relative")
                continue
            if action.target == HIL_STOP_TARGET:
                self._stop_hil()
                continue
            if action.target == "sim" or self._hil_active or self._collection_active:
                continue
            try:
                self._followers.apply_action(action)
                self._received_actions += 1
            except ValueError as exc:
                logger.warning("Dropped invalid I2RT action: %s", exc)

    def _leader_action(self) -> np.ndarray | None:
        if not self._collection_active and not self._hil_active:
            return None
        try:
            assert self._follower_anchor is not None
            assert self._leader_anchor is not None
            leader = self._leaders.read_action(self._follower_anchor)
            button_states = self._leaders.read_buttons()
            record_pressed = any(
                len(buttons) > LEADER_RECORD_BUTTON_INDEX
                and buttons[LEADER_RECORD_BUTTON_INDEX]
                for buttons in button_states.values()
            )
            if self._record_button.update(record_pressed, time.monotonic()):
                self._operator_event_id += 1
                self._operator_event = "collection_record_toggle"
                logger.info(
                    "I2RT leader RECORD button: collection toggle event=%d",
                    self._operator_event_id,
                )
            action = leader
            # Collection is anchored at takeover time for the same reason as
            # relative HIL: leader and follower motor zeros are never perfectly
            # identical.  Absolute collection previously discarded the anchors
            # captured in _start_collection(), causing an immediate pose jump and
            # a constant tracking offset.
            relative_control = (
                self._hil_mode == "relative" if self._hil_active else self._collection_active
            )
            if relative_control:
                action = self._follower_anchor + (leader - self._leader_anchor)
            action = flatten_state(
                split_action(action, self._config.group_names),
                self._config.group_names,
            )
            self._followers.apply_action(
                WireAction(t=time.monotonic(), action=action, target="real")
            )
            self._leader_control_updates += 1
            self._hil_error = ""
            return action.astype(np.float32)
        except Exception as exc:
            self._hil_error = str(exc)
            logger.warning("I2RT leader read/control failed: %s", exc)
            return None

    def _publish_observation(self) -> None:
        action = self._latest_leader_action
        state = self._followers.read_state()
        self._tracking_error = {}
        if action is not None:
            action_parts = split_action(action, self._config.group_names)
            for group_name in self._config.group_names:
                error = action_parts[group_name][:6] - state[group_name][:6]
                self._tracking_error[group_name] = np.round(error, 5).tolist()
        eef = self._kinematics.group_eef(state)
        action_eef = None if action is None else self._kinematics.flat_eef(action)
        observation = WireObservation(
            t=time.monotonic(),
            images=self._cameras.snapshot(),
            state=state,
            action=action,
            eef=eef,
            action_eef=action_eef,
            hil_supported=bool(self._config.leader_can_channels),
            hil_active=self._hil_active,
            hil_error=self._hil_error,
            operator_event=self._operator_event,
            operator_event_id=self._operator_event_id,
        )
        self._obs_pub.send(pack_observation(observation))
        self._published_observations += 1

    def _log_status_if_due(self) -> None:
        if self._config.status_log_interval_s <= 0:
            return
        now = time.monotonic()
        if now < self._next_status_time:
            return
        elapsed = max(now - self._last_status_time, 1e-6)
        obs_hz = (self._published_observations - self._last_status_observation_count) / elapsed
        action_hz = (self._received_actions - self._last_status_action_count) / elapsed
        control_hz = (self._leader_control_updates - self._last_status_control_count) / elapsed
        logger.info(
            "I2RT status: arms=%s cameras=%s obs=%.1fHz actions=%.1fHz "
            "leader_control=%.1fHz tracking_error_rad=%s tracking_trim_rad=%s",
            self._followers.hardware_status(),
            self._cameras.hardware_status(),
            obs_hz,
            action_hz,
            control_hz,
            self._tracking_error,
            self._followers.tracking_trim(),
        )
        self._last_status_time = now
        self._next_status_time = now + self._config.status_log_interval_s
        self._last_status_action_count = self._received_actions
        self._last_status_control_count = self._leader_control_updates
        self._last_status_observation_count = self._published_observations

    def serve_forever(self) -> None:
        self._followers.move_to_startup_position()
        control_period = 1.0 / self._config.control_rate_hz
        publish_period = 1.0 / self._config.publish_rate_hz
        next_control = time.monotonic()
        next_publish = next_control
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_control:
                self._drain_actions()
                self._latest_leader_action = self._leader_action()
                # A fresh policy/leader action already performs one follower
                # servo update. With a one-shot MANUAL target, watchdog_tick()
                # keeps closing the follower loop until the safety timeout.
                self._followers.watchdog_tick()
                next_control += control_period
                if next_control <= now:
                    next_control = now + control_period

            now = time.monotonic()
            if now >= next_publish:
                self._publish_observation()
                self._log_status_if_due()
                next_publish += publish_period
                if next_publish <= now:
                    next_publish = now + publish_period

            sleep_s = min(next_control, next_publish) - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def close(self) -> None:
        self._stop.set()
        self._collection_active = False
        self._hil_active = False
        self._leaders.close()
        self._followers.close()
        self._cameras.close()
        self._action_sub.close(linger=0)
        self._obs_pub.close(linger=0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--robot", choices=tuple(ROBOT_GROUPS), default="i2rt_dual_yam")
    parser.add_argument(
        "--follower-can",
        action="append",
        default=[],
        metavar="GROUP=INTERFACE",
        help="Follower CAN mapping; repeat once per arm. Single-arm accepts INTERFACE.",
    )
    parser.add_argument(
        "--leader-can",
        action="append",
        default=[],
        metavar="GROUP=INTERFACE",
        help="Optional teaching-handle CAN mapping for collection/HIL.",
    )
    parser.add_argument(
        "--disabled-arm",
        action="append",
        default=[],
        metavar="GROUP",
        help="Intentionally disable one follower group; repeatable or comma-separated.",
    )
    parser.add_argument(
        "--arm-type",
        # EVA's robot layout, limits, URDF, and FK are intentionally the base
        # YAM model. Add a separate registered robot before enabling variants.
        choices=("yam",),
        default="yam",
    )
    parser.add_argument(
        "--gripper-type",
        choices=(
            "crank_4310",
            "linear_3507",
            "linear_4310",
            "flexible_4310",
        ),
        default="linear_4310",
    )
    parser.add_argument("--sim", action="store_true", help="Use I2RT SimRobot; no CAN needed.")
    parser.add_argument("--enable-auto-recovery", action="store_true")
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=0.5,
        help="Seconds without an EVA/leader command before safe idle; <=0 disables.",
    )
    parser.add_argument(
        "--idle-mode",
        choices=("hold_position", "gravity_comp"),
        default="gravity_comp",
        help="Follower behavior before/after active control.",
    )
    parser.add_argument(
        "--startup-position",
        choices=("current", "zero"),
        default="current",
        help="Initial follower target after connection.",
    )
    parser.add_argument(
        "--startup-duration",
        type=float,
        default=5.0,
        help="Seconds used to interpolate to the configured startup position.",
    )
    parser.add_argument(
        "--joint4-kp",
        type=float,
        default=None,
        help="Optional position gain override for each arm's fourth joint.",
    )
    parser.add_argument(
        "--end-effector-mass",
        type=float,
        default=None,
        help=(
            "Optional effective end-effector mass in kg used by the gravity model; "
            "use only for a measured payload override."
        ),
    )
    parser.add_argument(
        "--gravity-comp-factor",
        type=float,
        nargs=6,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Optional six official SDK gravity-compensation multipliers.",
    )
    parser.add_argument(
        "--gripper-limits-override",
        type=float,
        nargs=2,
        default=None,
        metavar=("CLOSED", "OPEN"),
        help="Calibrated raw gripper motor limits; skips automatic limit detection.",
    )
    parser.add_argument(
        "--allow-gripper-calibration",
        action="store_true",
        help=(
            "Explicitly allow the SDK to drive a motorized gripper to both hard stops "
            "when no calibrated limit override is supplied."
        ),
    )
    parser.add_argument(
        "--tracking-ki",
        type=float,
        default=0.0,
        help="Outer-loop integral gain used to remove stationary joint error; 0 disables.",
    )
    parser.add_argument(
        "--tracking-trim-limit",
        type=float,
        default=0.12,
        help="Maximum absolute per-joint integral command trim in radians.",
    )
    parser.add_argument(
        "--tracking-deadband",
        type=float,
        default=0.002,
        help="Tracking errors at or below this magnitude are not integrated.",
    )
    parser.add_argument(
        "--tracking-settle-delay",
        type=float,
        default=0.15,
        help="Seconds a target must stay nearly stationary before trim integration.",
    )
    parser.add_argument(
        "--startup-trim-duration",
        type=float,
        default=3.0,
        help="Maximum seconds spent learning trim after an all-zero startup move.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="IMAGE_KEY=SERIAL_OR_INDEX",
        help="RealSense D405 mapping, e.g. cam_high=255... or cam_high=index:0; repeatable.",
    )
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List connected RealSense devices and exit.",
    )
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument(
        "--control-rate",
        type=float,
        default=200.0,
        help="Leader/follower control-loop rate; observations remain at --rate.",
    )
    parser.add_argument("--status-log-interval", type=float, default=5.0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def build_config(args: argparse.Namespace) -> I2RTZmqConfig:
    group_names = ROBOT_GROUPS[args.robot]
    if len(group_names) == 1:
        follower_defaults = {"arm": "can0"}
    else:
        follower_defaults = {"left_arm": "can0", "right_arm": "can1"}
    follower_can = _parse_group_map(args.follower_can, group_names, follower_defaults)
    leader_can = _parse_group_map(args.leader_can, group_names)
    disabled_groups = _parse_name_list(args.disabled_arm, group_names)
    cameras = parse_camera_specs(
        args.camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout_ms,
    )
    invalid_cameras = {
        camera.image_key
        for camera in cameras
        if camera.image_key not in ROBOT_CAMERA_KEYS[args.robot]
    }
    if invalid_cameras:
        allowed = ", ".join(ROBOT_CAMERA_KEYS[args.robot])
        raise ValueError(
            f"Unknown camera keys for {args.robot}: {sorted(invalid_cameras)}; expected {allowed}"
        )
    if args.joint4_kp is not None and not 0.0 < args.joint4_kp <= 500.0:
        raise ValueError("--joint4-kp must be in (0, 500]")
    if args.end_effector_mass is not None and not 0.0 <= args.end_effector_mass <= 5.0:
        raise ValueError("--end-effector-mass must be in [0, 5]")
    if args.gravity_comp_factor is not None and not all(
        0.0 <= value <= 5.0 for value in args.gravity_comp_factor
    ):
        raise ValueError("--gravity-comp-factor values must be in [0, 5]")
    if args.gripper_limits_override is not None:
        closed, opened = args.gripper_limits_override
        if closed == opened or not all(-20.0 <= value <= 20.0 for value in (closed, opened)):
            raise ValueError(
                "--gripper-limits-override values must be distinct and within [-20, 20]"
            )
    if args.gripper_limits_override is None and not args.allow_gripper_calibration:
        raise ValueError(
            "motorized gripper startup requires calibrated --gripper-limits-override "
            "or explicit --allow-gripper-calibration"
        )
    if args.tracking_ki < 0 or args.tracking_ki > 10:
        raise ValueError("--tracking-ki must be in [0, 10]")
    if args.tracking_trim_limit < 0 or args.tracking_trim_limit > 0.3:
        raise ValueError("--tracking-trim-limit must be in [0, 0.3]")
    if args.tracking_deadband < 0 or args.tracking_deadband > 0.05:
        raise ValueError("--tracking-deadband must be in [0, 0.05]")
    if args.tracking_settle_delay < 0:
        raise ValueError("--tracking-settle-delay must be non-negative")
    if args.startup_trim_duration < 0:
        raise ValueError("--startup-trim-duration must be non-negative")
    if args.control_rate <= 0:
        raise ValueError("--control-rate must be positive")
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    return I2RTZmqConfig(
        observation_endpoint=args.obs_endpoint,
        action_endpoint=args.action_endpoint,
        robot_name=args.robot,
        group_names=group_names,
        follower_can_channels=follower_can,
        leader_can_channels=leader_can,
        disabled_groups=disabled_groups,
        arm_type=args.arm_type,
        gripper_type=args.gripper_type,
        sim=bool(args.sim),
        enable_auto_recovery=bool(args.enable_auto_recovery),
        command_timeout_s=float(args.command_timeout),
        idle_mode=str(args.idle_mode),
        startup_position=str(args.startup_position),
        startup_duration_s=float(args.startup_duration),
        joint4_kp=None if args.joint4_kp is None else float(args.joint4_kp),
        end_effector_mass=(
            None if args.end_effector_mass is None else float(args.end_effector_mass)
        ),
        gravity_comp_factor=(
            None
            if args.gravity_comp_factor is None
            else tuple(float(value) for value in args.gravity_comp_factor)
        ),
        gripper_limits_override=(
            None
            if args.gripper_limits_override is None
            else tuple(float(value) for value in args.gripper_limits_override)
        ),
        allow_gripper_calibration=bool(args.allow_gripper_calibration),
        tracking_ki=float(args.tracking_ki),
        tracking_trim_limit=float(args.tracking_trim_limit),
        tracking_deadband=float(args.tracking_deadband),
        tracking_settle_delay_s=float(args.tracking_settle_delay),
        startup_trim_duration_s=float(args.startup_trim_duration),
        cameras=cameras,
        control_rate_hz=float(args.control_rate),
        publish_rate_hz=float(args.rate),
        status_log_interval_s=float(args.status_log_interval),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.list_cameras:
        for device in list_realsense_devices():
            print(
                f"{device['name']} serial={device['serial']} product_line={device['product_line']}"
            )
        return
    try:
        config = build_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    node = I2RTZmqNode(config)

    def stop(*_args: object) -> None:
        node.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        node.serve_forever()
    finally:
        node.close()


if __name__ == "__main__":
    main()
