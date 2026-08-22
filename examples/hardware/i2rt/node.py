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

import robots  # noqa: F401  # register EVA robot definitions before registry lookup
from core.registry import ROBOT_REGISTRY
from examples.hardware.i2rt.camera import (
    CameraSpec,
    RealSenseCameraCache,
    list_realsense_devices,
    load_camera_profiles,
    parse_camera_specs,
)
from examples.hardware.i2rt.orbbec_camera import (
    OrbbecCameraCache,
    OrbbecCameraSpec,
    list_orbbec_devices,
    parse_orbbec_camera_specs,
)
from examples.hardware.i2rt.robot import (
    GRIPPER_INDEX,
    GROUP_DOF,
    I2RTYamFollowers,
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

ROBOT_NAME = "i2rt_dual_yam"
GROUP_NAMES = ("left_arm", "right_arm")
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LEADER_CANCEL_BUTTON_INDEX = 0
LEADER_RECORD_BUTTON_INDEX = 1
LEADER_BUTTON_DEBOUNCE_S = 0.08
LEADER_CONNECT_RETRY_S = 5.0
COLLECTION_RECORD_TOGGLE_EVENT = "collection_record_toggle"
COLLECTION_CANCEL_EVENT = "collection_cancel"
EEF_DOF = 8


def split_group_eef(
    eef: np.ndarray,
    group_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Split flat EVA xyz + wxyz + gripper poses into actuator groups."""
    vector = np.asarray(eef, dtype=np.float32).reshape(-1)
    expected = len(group_names) * EEF_DOF
    if vector.shape != (expected,):
        raise ValueError(f"Expected a {expected}-D I2RT EEF vector, got {vector.shape}")
    return {
        name: vector[index * EEF_DOF : (index + 1) * EEF_DOF].copy()
        for index, name in enumerate(group_names)
    }


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
    leader_gripper_encoder_endpoints: dict[str, tuple[float, float]]
    direct_leader_control: bool
    arm_type: str
    gripper_type: str
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
    orbbec_cameras: tuple[OrbbecCameraSpec, ...]
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


def _parse_leader_gripper_endpoints(
    values: list[str],
    group_names: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected GROUP=OPEN_RAD,CLOSED_RAD, got {value!r}")
        group_name, encoded = (part.strip() for part in value.split("=", 1))
        if group_name not in group_names:
            raise ValueError(f"Unknown I2RT group {group_name!r}; expected one of {group_names}")
        if group_name in result:
            raise ValueError(f"Duplicate leader gripper endpoints for {group_name!r}")
        try:
            opened_text, closed_text = (part.strip() for part in encoded.split(",", 1))
            opened, closed = float(opened_text), float(closed_text)
        except ValueError as exc:
            raise ValueError(f"Expected GROUP=OPEN_RAD,CLOSED_RAD, got {value!r}") from exc
        if not np.isfinite([opened, closed]).all() or opened == closed:
            raise ValueError(f"Leader gripper endpoints must be finite and distinct: {value!r}")
        if not all(-2 * np.pi <= endpoint <= 2 * np.pi for endpoint in (opened, closed)):
            raise ValueError(f"Leader gripper endpoints must be within [-2pi, 2pi]: {value!r}")
        result[group_name] = (opened, closed)
    return result


class I2RTZmqNode:
    """Bridge EVA wire messages to I2RT follower and leader YAM arms."""

    def __init__(self, config: I2RTZmqConfig) -> None:
        self._config = config
        self._stop = threading.Event()
        self._ctx = zmq.Context.instance()
        self._obs_pub: zmq.Socket | None = None
        self._publisher_thread: threading.Thread | None = None
        self._publisher_ready = threading.Event()
        self._publisher_error: BaseException | None = None
        self._action_sub = self._ctx.socket(zmq.SUB)
        self._action_sub.bind(config.action_endpoint)
        self._action_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._action_sub.setsockopt(zmq.RCVTIMEO, 0)

        self._followers = I2RTYamFollowers(config)
        self._leaders = I2RTYamLeaders(config)
        robot = ROBOT_REGISTRY.build(config.robot_name)
        self._fk_solver = robot.build_kinematics(initial_qpos_groups=robot.initial_qpos_by_group())
        self._camera_caches = (RealSenseCameraCache(config.cameras),)
        self._orbbec_camera_cache: OrbbecCameraCache | None = None
        self._collection_active = False
        self._hil_active = False
        self._direct_leader_control = config.direct_leader_control
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
        self._cancel_button = _RisingEdgeDebouncer(LEADER_BUTTON_DEBOUNCE_S)
        self._operator_event = ""
        self._operator_event_id = 0
        self._next_leader_connect_time = 0.0
        now = time.monotonic()
        self._last_status_time = now
        self._next_status_time = now + max(config.status_log_interval_s, 0.0)
        self._last_status_action_count = 0
        self._last_status_control_count = 0
        self._last_status_observation_count = 0
        logger.info(
            "I2RT node ready: robot=%s followers=%s leaders=%s",
            config.robot_name,
            config.follower_can_channels,
            config.leader_can_channels,
        )

    def stop(self) -> None:
        self._stop.set()

    def _capture_leader_anchors(self) -> None:
        follower = flatten_state(self._followers.read_state(), self._config.group_names)
        leader = self._leaders.read_action()
        self._leader_anchor = leader
        self._follower_anchor = follower

    def _ensure_direct_leader_control(self) -> bool:
        if not self._direct_leader_control:
            return False
        if self._leader_anchor is not None and self._follower_anchor is not None:
            return True
        now = time.monotonic()
        if now < self._next_leader_connect_time:
            return False
        try:
            self._capture_leader_anchors()
        except Exception as exc:
            self._hil_error = str(exc)
            self._next_leader_connect_time = now + LEADER_CONNECT_RETRY_S
            logger.warning(
                "I2RT direct leader control is not ready; retrying in %.1f s: %s",
                LEADER_CONNECT_RETRY_S,
                exc,
            )
            return False
        self._hil_error = ""
        logger.info("I2RT direct dual-leader control started with relative anchors")
        return True

    def _start_collection(self) -> None:
        if self._collection_active:
            return
        self._capture_leader_anchors()
        self._collection_active = True
        logger.info("I2RT dual-leader collection started")

    def _stop_collection(self) -> None:
        self._collection_active = False
        if not self._hil_active and not self._direct_leader_control:
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
            self._capture_leader_anchors()
        except Exception as exc:
            self._hil_error = str(exc)
            self._hil_active = False
            logger.exception("Could not start I2RT HIL")
            return
        self._hil_mode = mode
        self._hil_error = ""
        self._hil_active = True
        logger.info("I2RT HIL started: mode=%s", mode)

    def _stop_hil(self) -> None:
        self._hil_active = False
        self._hil_error = ""
        if not self._collection_active and not self._direct_leader_control:
            self._leader_anchor = None
            self._follower_anchor = None
            self._latest_leader_action = None
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
            if (
                action.target == "sim"
                or self._direct_leader_control
                or self._hil_active
                or self._collection_active
            ):
                continue
            try:
                self._followers.apply_action(action)
                self._received_actions += 1
            except ValueError as exc:
                logger.warning("Dropped invalid I2RT action: %s", exc)

    def _leader_action(self) -> np.ndarray | None:
        if not self._direct_leader_control and not self._collection_active and not self._hil_active:
            return None
        if not self._ensure_direct_leader_control() and (
            self._leader_anchor is None or self._follower_anchor is None
        ):
            return None
        try:
            assert self._follower_anchor is not None
            assert self._leader_anchor is not None
            leader = self._leaders.read_action()
            button_states = self._leaders.read_buttons()
            self._update_operator_buttons(button_states, time.monotonic())
            action = leader
            # Collection is anchored at takeover time for the same reason as
            # relative HIL: leader and follower motor zeros are never perfectly
            # identical.  Absolute collection previously discarded the anchors
            # captured in _start_collection(), causing an immediate pose jump and
            # a constant tracking offset.
            relative_control = self._hil_mode == "relative" if self._hil_active else True
            if relative_control:
                action = self._follower_anchor + (leader - self._leader_anchor)
                # Arm motor zeros need relative anchoring. Calibrated grippers
                # already share the same [0, 1] space and must retain their
                # absolute endpoints so full leader travel reaches both stops.
                for index in range(len(self._config.group_names)):
                    gripper_index = index * GROUP_DOF + GRIPPER_INDEX
                    action[gripper_index] = leader[gripper_index]
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
            # A crashed SDK CAN loop keeps returning its last cached pose. Drop
            # both leader objects on any read failure (including collection/HIL)
            # so the next control tick can reconnect instead of replaying stale
            # commands forever.
            self._leader_anchor = None
            self._follower_anchor = None
            self._leaders.close()
            self._next_leader_connect_time = time.monotonic() + LEADER_CONNECT_RETRY_S
            return None

    def _update_operator_buttons(
        self,
        button_states: dict[str, tuple[bool, ...]],
        now: float,
    ) -> None:
        """Emit debounced RECORD/save or SYNC/cancel events from either leader."""
        record_pressed = any(
            len(buttons) > LEADER_RECORD_BUTTON_INDEX and buttons[LEADER_RECORD_BUTTON_INDEX]
            for buttons in button_states.values()
        )
        cancel_pressed = any(
            len(buttons) > LEADER_CANCEL_BUTTON_INDEX and buttons[LEADER_CANCEL_BUTTON_INDEX]
            for buttons in button_states.values()
        )
        cancel_edge = self._cancel_button.update(cancel_pressed, now)
        record_edge = self._record_button.update(record_pressed, now)
        if cancel_edge:
            event = COLLECTION_CANCEL_EVENT
            button_name = "SYNC/CANCEL"
        elif record_edge:
            event = COLLECTION_RECORD_TOGGLE_EVENT
            button_name = "RECORD"
        else:
            return
        self._operator_event_id += 1
        self._operator_event = event
        logger.info(
            "I2RT leader %s button: event=%s id=%d",
            button_name,
            event,
            self._operator_event_id,
        )

    def _publish_observation(self) -> None:
        action = self._latest_leader_action
        state = self._followers.snapshot_state()
        self._tracking_error = {}
        if action is not None:
            action_parts = split_action(action, self._config.group_names)
            for group_name in self._config.group_names:
                error = action_parts[group_name][:6] - state[group_name][:6]
                self._tracking_error[group_name] = np.round(error, 5).tolist()
        eef = split_group_eef(
            self._fk(flatten_state(state, self._config.group_names)),
            self._config.group_names,
        )
        action_eef = None if action is None else self._fk(action)
        observation = WireObservation(
            t=time.monotonic(),
            images=self._camera_snapshot(),
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
        publisher = self._obs_pub
        if publisher is None:
            raise RuntimeError("I2RT observation publisher is not ready")
        publisher.send(pack_observation(observation))
        self._published_observations += 1

    def _fk(self, qpos: np.ndarray) -> np.ndarray:
        """Run EVA's registered FK so hardware and client share one pose convention."""
        return self._fk_solver.fk_chunk(np.asarray(qpos, dtype=np.float32)[np.newaxis, :])[0]

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
            self._camera_status(),
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

    def _publish_loop(self) -> None:
        publish_period = 1.0 / self._config.publish_rate_hz
        publisher: zmq.Socket | None = None
        try:
            publisher = self._ctx.socket(zmq.PUB)
            publisher.bind(self._config.observation_endpoint)
            self._obs_pub = publisher
            self._publisher_ready.set()
            next_publish = time.monotonic()
            while not self._stop.is_set():
                now = time.monotonic()
                if now < next_publish:
                    self._stop.wait(next_publish - now)
                    continue
                self._publish_observation()
                self._log_status_if_due()
                next_publish += publish_period
                finished = time.monotonic()
                if next_publish <= finished:
                    next_publish = finished + publish_period
        except BaseException as exc:
            self._publisher_error = exc
            self._stop.set()
            logger.exception("I2RT observation publisher failed")
        finally:
            self._publisher_ready.set()
            self._obs_pub = None
            if publisher is not None:
                publisher.close(linger=0)

    def serve_forever(self) -> None:
        if self._config.orbbec_cameras:
            self._orbbec_camera_cache = OrbbecCameraCache(self._config.orbbec_cameras)
            self._camera_caches += (self._orbbec_camera_cache,)
            if not self._orbbec_camera_cache.wait_until_online(timeout_s=30.0):
                raise RuntimeError(
                    "Orbbec cameras did not all become ready: "
                    f"{self._orbbec_camera_cache.hardware_status()}"
                )
            logger.info(
                "All Orbbec cameras online before motor bring-up: %s",
                self._orbbec_camera_cache.hardware_status(),
            )
        self._followers.move_to_startup_position()
        if self._config.leader_can_channels:
            self._leaders.move_to_startup_position()
        self._ensure_direct_leader_control()
        self._publisher_thread = threading.Thread(
            target=self._publish_loop,
            name="i2rt-observation-publisher",
            daemon=True,
        )
        self._publisher_thread.start()
        if not self._publisher_ready.wait(timeout=2.0):
            raise RuntimeError("I2RT observation publisher did not start")
        if self._publisher_error is not None:
            raise RuntimeError(
                "I2RT observation publisher failed to start"
            ) from self._publisher_error

        control_period = 1.0 / self._config.control_rate_hz
        next_control = time.monotonic()
        while not self._stop.is_set():
            self._drain_actions()
            self._latest_leader_action = self._leader_action()
            # A fresh policy/leader action already performs one follower
            # servo update. With a one-shot MANUAL target, watchdog_tick()
            # keeps closing the follower loop until the safety timeout.
            self._followers.watchdog_tick()
            next_control += control_period
            now = time.monotonic()
            if next_control <= now:
                next_control = now + control_period
            sleep_s = next_control - time.monotonic()
            if sleep_s > 0:
                self._stop.wait(sleep_s)
        if self._publisher_error is not None:
            raise RuntimeError(
                "I2RT observation publisher stopped unexpectedly"
            ) from self._publisher_error

    def close(self) -> None:
        self._stop.set()
        self._collection_active = False
        self._hil_active = False
        self._direct_leader_control = False
        if (
            self._publisher_thread is not None
            and self._publisher_thread.is_alive()
            and self._publisher_thread is not threading.current_thread()
        ):
            self._publisher_thread.join(timeout=5.0)
        self._leaders.close()
        self._followers.close()
        self._fk_solver.close()
        for camera_cache in self._camera_caches:
            camera_cache.close()
        self._action_sub.close(linger=0)

    def _camera_snapshot(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for camera_cache in self._camera_caches:
            images.update(camera_cache.snapshot())
        return images

    def _camera_status(self) -> dict[str, str]:
        status: dict[str, str] = {}
        for camera_cache in self._camera_caches:
            status.update(camera_cache.hardware_status())
        return status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument(
        "--follower-can",
        action="append",
        default=[],
        metavar="GROUP=INTERFACE",
        help="Follower CAN mapping; repeat once per arm.",
    )
    parser.add_argument(
        "--leader-cans",
        nargs=2,
        default=None,
        metavar=("LEFT_INTERFACE", "RIGHT_INTERFACE"),
        help="Dual teaching-handle CAN interfaces for collection/HIL.",
    )
    parser.add_argument(
        "--leader-gripper-endpoint",
        action="append",
        default=[],
        metavar="GROUP=OPEN_RAD,CLOSED_RAD",
        help="Raw teaching-handle encoder endpoints; repeat once per leader.",
    )
    parser.add_argument(
        "--direct-leader-control",
        action="store_true",
        help="Start relative dual-leader follower control immediately.",
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
        "--camera-profile",
        default=None,
        help="JSON file containing per-camera D405 exposure profiles.",
    )
    parser.add_argument(
        "--camera-auto-exposure-limit-us",
        type=int,
        default=None,
        help="Maximum D405 auto-exposure time in microseconds; 0 disables the limit.",
    )
    parser.add_argument(
        "--camera-exposure-us",
        type=int,
        default=None,
        help="Fixed D405 exposure time in microseconds; overrides automatic exposure.",
    )
    parser.add_argument("--camera-auto-gain-limit", type=int, default=None)
    parser.add_argument("--camera-warmup-frames", type=int, default=None)
    parser.add_argument(
        "--orbbec-camera",
        action="append",
        default=[],
        metavar="IMAGE_KEY=SERIAL_OR_INDEX",
        help="Orbbec mapping, e.g. cam_left_wrist=CV2L...; repeatable.",
    )
    parser.add_argument("--orbbec-width", type=int, default=640)
    parser.add_argument("--orbbec-height", type=int, default=480)
    parser.add_argument("--orbbec-fps", type=int, default=30)
    parser.add_argument("--orbbec-color-format", default="MJPG")
    parser.add_argument("--orbbec-timeout-ms", type=int, default=1000)
    parser.add_argument("--orbbec-warmup-frames", type=int, default=30)
    parser.add_argument("--orbbec-brightness", type=int, default=5)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List connected RealSense and Orbbec devices and exit.",
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
    group_names = GROUP_NAMES
    follower_defaults = {"left_arm": "can0", "right_arm": "can1"}
    follower_can = _parse_group_map(args.follower_can, group_names, follower_defaults)
    leader_gripper_endpoints = _parse_leader_gripper_endpoints(
        args.leader_gripper_endpoint,
        group_names,
    )
    leader_can: dict[str, str] = {}
    if args.leader_cans is not None:
        left_interface, right_interface = (value.strip() for value in args.leader_cans)
        if not left_interface or not right_interface:
            raise ValueError("--leader-cans requires two non-empty CAN interfaces")
        if left_interface == right_interface:
            raise ValueError("--leader-cans requires two distinct CAN interfaces")
        leader_can = {
            "left_arm": left_interface,
            "right_arm": right_interface,
        }
        reused_interfaces = set(leader_can.values()) & set(follower_can.values())
        if reused_interfaces:
            raise ValueError(
                f"leader and follower CAN interfaces must be distinct: {sorted(reused_interfaces)}"
            )
    if args.direct_leader_control and not leader_can:
        raise ValueError("--direct-leader-control requires --leader-cans")
    if leader_gripper_endpoints and set(leader_gripper_endpoints) != set(leader_can):
        raise ValueError(
            "--leader-gripper-endpoint must be provided once for each configured leader"
        )
    camera_profiles = load_camera_profiles(args.camera_profile) if args.camera_profile else None
    cameras = parse_camera_specs(
        args.camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout_ms,
        camera_profiles=camera_profiles,
    )
    if args.camera_auto_exposure_limit_us is not None:
        cameras = tuple(
            dataclasses.replace(
                camera,
                auto_exposure_limit_us=(
                    None
                    if args.camera_auto_exposure_limit_us == 0
                    else args.camera_auto_exposure_limit_us
                ),
            )
            for camera in cameras
        )
    if args.camera_auto_gain_limit is not None:
        cameras = tuple(
            dataclasses.replace(camera, auto_gain_limit=args.camera_auto_gain_limit)
            for camera in cameras
        )
    if args.camera_exposure_us is not None:
        cameras = tuple(
            dataclasses.replace(camera, exposure_us=args.camera_exposure_us) for camera in cameras
        )
    if args.camera_warmup_frames is not None:
        cameras = tuple(
            dataclasses.replace(camera, warmup_frames=args.camera_warmup_frames)
            for camera in cameras
        )
    orbbec_cameras = parse_orbbec_camera_specs(
        args.orbbec_camera,
        width=args.orbbec_width,
        height=args.orbbec_height,
        fps=args.orbbec_fps,
        color_format=args.orbbec_color_format,
        timeout_ms=args.orbbec_timeout_ms,
        warmup_frames=args.orbbec_warmup_frames,
        brightness=args.orbbec_brightness,
    )
    invalid_cameras = {
        camera.image_key
        for camera in cameras + orbbec_cameras
        if camera.image_key not in CAMERA_KEYS
    }
    if invalid_cameras:
        allowed = ", ".join(CAMERA_KEYS)
        raise ValueError(f"Unknown camera keys: {sorted(invalid_cameras)}; expected {allowed}")
    duplicate_camera_keys = {camera.image_key for camera in cameras} & {
        camera.image_key for camera in orbbec_cameras
    }
    if duplicate_camera_keys:
        raise ValueError(
            "Camera keys cannot use both RealSense and Orbbec backends: "
            f"{sorted(duplicate_camera_keys)}"
        )
    if args.camera_auto_exposure_limit_us is not None and args.camera_auto_exposure_limit_us < 0:
        raise ValueError("--camera-auto-exposure-limit-us must be non-negative")
    if args.camera_exposure_us is not None and args.camera_exposure_us <= 0:
        raise ValueError("--camera-exposure-us must be positive")
    if args.camera_auto_gain_limit is not None and args.camera_auto_gain_limit <= 0:
        raise ValueError("--camera-auto-gain-limit must be positive")
    if args.camera_warmup_frames is not None and args.camera_warmup_frames < 0:
        raise ValueError("--camera-warmup-frames must be non-negative")
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
        robot_name=ROBOT_NAME,
        group_names=group_names,
        follower_can_channels=follower_can,
        leader_can_channels=leader_can,
        leader_gripper_encoder_endpoints=leader_gripper_endpoints,
        direct_leader_control=bool(args.direct_leader_control),
        arm_type=args.arm_type,
        gripper_type=args.gripper_type,
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
        orbbec_cameras=orbbec_cameras,
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
                "RealSense "
                f"{device['name']} serial={device['serial']} product_line={device['product_line']}"
            )
        for device in list_orbbec_devices():
            print(
                f"Orbbec {device['name']} serial={device['serial']} usb={device['connection_type']}"
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
