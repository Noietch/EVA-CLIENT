#!/usr/bin/env python3
"""ARX R5 execution-layer node for EVA's ZMQ transport."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robots  # noqa: F401, E402  # registers every zoo robot under the registries on import
from core.config import load_config  # noqa: E402
from core.registry import ROBOT_REGISTRY  # noqa: E402
from examples.hardware.arx.camera import (  # noqa: E402
    OrbbecCameraCache,
    OrbbecCameraSpec,
    apply_orbbec_white_balances,
    default_arx_orbbec_camera_specs,
    load_orbbec_white_balances,
    merge_orbbec_camera_specs,
    parse_orbbec_camera_specs,
    parse_resolution,
)
from examples.hardware.arx.robot import (  # noqa: E402
    ARM_DOF,
    ARX_GRIPPER_OPEN_POS,
    DEFAULT_ARM_TYPE,
    DEFAULT_LEFT_CAN_PORT,
    DEFAULT_RIGHT_CAN_PORT,
    GROUP_DOF,
    GROUP_NAMES,
    ArxDualArm,
    parse_disabled_groups,
)
from examples.hardware.arx.teleop import (  # noqa: E402
    ARX_R5_JOINT_LIMITS,
    DEFAULT_ALICIA_PORT,
    DEFAULT_ALICIA_READER,
    DEFAULT_LEFT_ALICIA_PORT,
    DEFAULT_RIGHT_ALICIA_PORT,
    calibrate_arx_joint_offset,
    create_alicia_robot,
    map_alicia_to_arx_joints,
    read_alicia_state,
    resolve_alicia_group_ports,
)
from transport.zmq import (  # noqa: E402
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


@dataclasses.dataclass(frozen=True)
class ArxZmqConfig:
    """Runtime config for the ARX R5 ZMQ execution node.

    Args:
        observation_endpoint: ZMQ PUB bind endpoint for WireObservation frames.
        action_endpoint: ZMQ SUB bind endpoint for WireAction commands.
        can_ports: arm group name -> ARX CAN interface (e.g. can0).
        arm_type: ARX SDK arm type code (0 = X5lite, 1 = R5 master).
        gripper_open_pos: ARX catch position mapped to gripper scalar 1.0.
        initial_gripper_scalar: gripper scalar commanded when an ARX arm first connects.
        disabled_groups: arm groups intentionally absent from this deployment.
        orbbec_cameras: Orbbec SDK cameras mapped to EVA image keys.
        orbbec_white_balances: configured manual white-balance values by image key.
        publish_rate_hz: observation publish rate.
        left_alicia_port: Alicia-D leader port for the left ARX arm.
        right_alicia_port: Alicia-D leader port for the right ARX arm.
        alicia_port: Legacy Alicia-D leader port alias for right_alicia_port.
        alicia_reader: Alicia reader backend, "sdk" or "duo".
        debug_alicia: Enable Alicia-D debug logs.
        passive_collection: Only observe ARX/cameras during collection. Default false:
            Alicia-D drives the follower arms and supplies action_qpos.
        status_log_interval_s: hardware status log interval; non-positive disables it.
    """

    observation_endpoint: str
    action_endpoint: str
    can_ports: dict[str, str]
    arm_type: int
    gripper_open_pos: float
    initial_gripper_scalar: float | None
    disabled_groups: tuple[str, ...]
    orbbec_cameras: tuple[OrbbecCameraSpec, ...]
    publish_rate_hz: float
    orbbec_white_balances: dict[str, int] = dataclasses.field(default_factory=dict)
    left_alicia_port: str = DEFAULT_LEFT_ALICIA_PORT
    right_alicia_port: str = DEFAULT_RIGHT_ALICIA_PORT
    alicia_port: str = DEFAULT_ALICIA_PORT
    alicia_reader: str = DEFAULT_ALICIA_READER
    debug_alicia: bool = False
    passive_collection: bool = False
    status_log_interval_s: float = 5.0


class EmptyCameraCache:
    def snapshot(self) -> dict[str, np.ndarray]:
        return {}

    def snapshot_versioned(self) -> tuple[dict[str, int], dict[str, np.ndarray]]:
        return {}, {}

    def hardware_status(self) -> dict[str, str]:
        return {}

    def close(self) -> None:
        return


def parse_name_list(values: list[str]) -> tuple[str, ...]:
    names: list[str] = []
    for value in values:
        for item in value.split(","):
            name = item.strip()
            if name and name not in names:
                names.append(name)
    return tuple(names)


def load_disabled_cameras(config_path: str) -> tuple[str, ...]:
    if not config_path:
        return ()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"EVA config not found: {path}")
    cfg = load_config(path)
    transport = cfg.get("transport", {})
    if not isinstance(transport, dict):
        return ()
    disabled = transport.get("disabled_cameras", ())
    if disabled is None:
        return ()
    return tuple(str(name) for name in disabled)


def format_status(status: dict[str, str]) -> str:
    if not status:
        return "none"
    return " ".join(f"{name}={state}" for name, state in status.items())


def flatten_group_state(state: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(state[group_name], dtype=np.float32) for group_name in GROUP_NAMES],
        axis=0,
    )


def split_group_eef(eef: np.ndarray) -> dict[str, np.ndarray]:
    vector = np.asarray(eef, dtype=np.float32)
    return {
        "left_arm": vector[:8].copy(),
        "right_arm": vector[8:16].copy(),
    }


class ArxTeleopSource:
    def __init__(self, config: ArxZmqConfig) -> None:
        self._config = config
        self._robots: dict[str, Any] = {}
        self._joint_offsets: dict[str, np.ndarray] = {}
        self._previous_joints: dict[str, np.ndarray] = {}
        self._joint_compensation: dict[str, np.ndarray] = {}
        self._last_gripper: dict[str, float] = dict.fromkeys(GROUP_NAMES, 0.0)
        self._disabled_groups = set(config.disabled_groups)

    def connect(self) -> None:
        if self._robots:
            return
        ports = resolve_alicia_group_ports(
            self._config.left_alicia_port,
            self._config.right_alicia_port,
            self._config.alicia_port,
        )
        for group_name in GROUP_NAMES:
            if group_name in self._disabled_groups:
                continue
            self._robots[group_name] = create_alicia_robot(
                ports[group_name],
                self._config.debug_alicia,
                self._config.alicia_reader,
            )

    def disconnect(self) -> None:
        for robot in self._robots.values():
            disconnect = getattr(robot, "disconnect", None)
            if disconnect is not None:
                disconnect()
        self._robots = {}
        self._joint_offsets = {}
        self._previous_joints = {}
        self._joint_compensation = {}

    def calibrate_delta(self, robot_qpos: np.ndarray) -> None:
        for group_name in GROUP_NAMES:
            if group_name in self._disabled_groups:
                continue
            joints, gripper = self._read_group_state(group_name)
            offset = GROUP_NAMES.index(group_name) * GROUP_DOF
            self._joint_offsets[group_name] = calibrate_arx_joint_offset(
                joints,
                robot_qpos[offset : offset + ARM_DOF],
            )
            self._previous_joints[group_name] = joints[:ARM_DOF].copy()
            self._joint_compensation[group_name] = np.zeros(ARM_DOF, dtype=float)
            self._last_gripper[group_name] = self._normalize_gripper(gripper)

    def read_action_qpos(self, robot_qpos: np.ndarray) -> np.ndarray:
        if not self._joint_offsets:
            self.calibrate_delta(robot_qpos)

        parts: list[np.ndarray] = []
        for group_name in GROUP_NAMES:
            offset = GROUP_NAMES.index(group_name) * GROUP_DOF
            if group_name in self._disabled_groups:
                parts.append(robot_qpos[offset : offset + GROUP_DOF].copy())
                continue

            joints, gripper = self._read_group_state(group_name)
            joints = self._unwrap_joints(group_name, joints)
            arm_joints = np.clip(
                map_alicia_to_arx_joints(joints) + self._joint_offsets[group_name],
                ARX_R5_JOINT_LIMITS[:, 0],
                ARX_R5_JOINT_LIMITS[:, 1],
            )
            gripper_scalar = self._normalize_gripper(gripper)
            self._last_gripper[group_name] = gripper_scalar
            parts.append(np.asarray([*arm_joints, gripper_scalar], dtype=np.float32))

        return np.concatenate(parts, axis=0).astype(np.float32)

    def _read_group_state(self, group_name: str) -> tuple[np.ndarray, float | None]:
        robot = self._robots.get(group_name)
        if robot is None:
            raise RuntimeError(f"{group_name} Alicia-D teleop source is not connected")
        return read_alicia_state(robot, group_name)

    def _unwrap_joints(self, group_name: str, joints: np.ndarray) -> np.ndarray:
        current = np.asarray(joints, dtype=float)[:ARM_DOF]
        previous = self._previous_joints[group_name]
        compensation = self._joint_compensation[group_name]
        variation = current - previous
        compensation[variation > np.pi] -= 2.0 * np.pi
        compensation[variation < -np.pi] += 2.0 * np.pi
        self._previous_joints[group_name] = current.copy()
        return current + compensation

    def _normalize_gripper(self, raw_gripper: float | None) -> float:
        if raw_gripper is None:
            return 0.0
        return float(np.clip(float(raw_gripper) / 1000.0, 0.0, 1.0))


class ArxZmqNode:
    """Bridge EVA ZMQ wire messages to an ARX-SDK-controlled dual R5 pair."""

    def __init__(self, config: ArxZmqConfig) -> None:
        import zmq

        self._config = config
        self._stop = threading.Event()
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._obs_pub = self._ctx.socket(zmq.PUB)
        self._obs_pub.bind(config.observation_endpoint)
        self._action_sub = self._ctx.socket(zmq.SUB)
        self._action_sub.bind(config.action_endpoint)
        self._action_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._action_sub.setsockopt(zmq.RCVTIMEO, 0)
        self._robot = ArxDualArm(config)
        self._cameras = self._build_camera_cache(config)
        self._teleop_source = ArxTeleopSource(config)
        self._collection_active = False
        self._hil_active = False
        self._hil_error = ""
        self._fk_solver: Any = None
        self._received_actions = 0
        self._published_observations = 0
        self._last_published_seqs: dict[str, int] = {}
        now = time.monotonic()
        self._last_status_log_time = now
        self._next_status_log_time = now + max(config.status_log_interval_s, 0.0)
        self._last_status_action_count = 0
        self._last_status_observation_count = 0
        logger.info(
            "ARX ZMQ node ready: obs_pub=%s action_sub=%s can_ports=%s "
            "orbbec_cameras=%d status_log_interval=%.1fs",
            config.observation_endpoint,
            config.action_endpoint,
            config.can_ports,
            len(config.orbbec_cameras),
            config.status_log_interval_s,
        )

    def _build_camera_cache(self, config: ArxZmqConfig) -> Any:
        if not config.orbbec_cameras:
            logger.info("Orbbec camera backend disabled: no cameras configured")
            return EmptyCameraCache()
        return OrbbecCameraCache(config.orbbec_cameras)

    def stop(self) -> None:
        self._stop.set()

    def get_latest_qpos(self) -> np.ndarray:
        return flatten_group_state(self._robot.read_state())

    def start_collection(self) -> None:
        if self._collection_active:
            return
        self._last_published_seqs = {}
        if self._config.passive_collection:
            self._collection_active = True
            return
        try:
            self._teleop_source.connect()
            self._teleop_source.calibrate_delta(self.get_latest_qpos())
            self._collection_active = True
        except Exception:
            self._collection_active = False
            self._teleop_source.disconnect()
            raise

    def stop_collection(self) -> None:
        self._collection_active = False
        self._teleop_source.disconnect()

    def start_hil(self, mode: str) -> None:
        if self._config.passive_collection:
            self._hil_error = "ARX passive collection mode cannot control HIL"
            return
        if mode not in {"absolute", "relative"}:
            self._hil_error = f"Unsupported HIL control mode: {mode}"
            return
        try:
            self.start_collection()
        except Exception as exc:
            self._hil_error = str(exc)
            return
        self._hil_error = ""
        self._hil_active = True

    def stop_hil(self) -> None:
        self.stop_collection()
        self._hil_active = False
        self._hil_error = ""

    def _drain_actions(self) -> None:
        while True:
            try:
                payload = self._action_sub.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                return
            try:
                action = unpack_action(payload)
            except Exception as exc:
                preview = bytes(payload[:16]).hex()
                logger.warning(
                    "Dropped malformed action frame: bytes=%d head=%s error=%s",
                    len(payload),
                    preview,
                    exc,
                )
                continue
            if action.target == COLLECTION_START_TARGET:
                self.start_collection()
                continue
            if action.target == COLLECTION_STOP_TARGET:
                self.stop_collection()
                continue
            if action.target == HIL_START_TARGET:
                self.start_hil(action.mode or "relative")
                continue
            if action.target == HIL_STOP_TARGET:
                self.stop_hil()
                continue
            if action.target == "sim":
                continue
            if self._collection_active and self._config.passive_collection:
                continue
            self._robot.apply_action(action)
            self._received_actions += 1

    def _publish_observation(self) -> None:
        state = self._robot.read_state()
        seqs, images = self._cameras.snapshot_versioned()
        eef = None
        action_qpos = None
        action_eef = None
        if self._collection_active:
            qpos = flatten_group_state(state)
            eef_flat = self._fk(qpos)
            eef = split_group_eef(eef_flat)
            if self._config.passive_collection:
                action_qpos = qpos.copy()
                action_eef = eef_flat.copy()
            else:
                action_qpos = self._teleop_source.read_action_qpos(qpos)
                action_eef = self._fk(action_qpos)
                self._robot.apply_action(
                    WireAction(t=time.monotonic(), action=action_qpos, target="real")
                )
            # Gate collection publishes on fresh camera frames: the loop runs at
            # publish_rate_hz (> camera fps), so without this the same cached frame
            # would be republished and recorded as a static video. Teleop control
            # above still runs every tick; only the recorded observation waits for
            # every camera to advance, which also keeps the views mutually aligned.
            if seqs and seqs == self._last_published_seqs:
                return
            self._last_published_seqs = seqs
        obs = WireObservation(
            t=time.monotonic(),
            images=images,
            state=state,
            eef=eef,
            action=action_qpos,
            action_eef=action_eef,
            hil_supported=not self._config.passive_collection,
            hil_active=self._hil_active,
            hil_error=self._hil_error,
        )
        self._obs_pub.send(pack_observation(obs))
        self._published_observations += 1

    def _fk(self, qpos: np.ndarray) -> np.ndarray:
        if self._fk_solver is None:
            groups = [np.zeros(GROUP_DOF, dtype=np.float32) for _ in GROUP_NAMES]
            self._fk_solver = ROBOT_REGISTRY.build("arx_r5").build_kinematics(
                initial_qpos_groups=groups
            )
        return self._fk_solver.fk_chunk(qpos[np.newaxis, :])[0]

    def _log_hardware_status_if_due(self) -> None:
        if self._config.status_log_interval_s <= 0:
            return
        now = time.monotonic()
        if now < self._next_status_log_time:
            return

        elapsed_s = max(now - self._last_status_log_time, 1e-6)
        observation_hz = (
            self._published_observations - self._last_status_observation_count
        ) / elapsed_s
        action_hz = (self._received_actions - self._last_status_action_count) / elapsed_s
        logger.info(
            "Hardware status: arms=[%s] cameras=[%s] rates=[obs=%.1fHz actions=%.1fHz]",
            format_status(self._robot.hardware_status()),
            format_status(self._cameras.hardware_status()),
            observation_hz,
            action_hz,
        )
        self._last_status_log_time = now
        self._next_status_log_time = now + self._config.status_log_interval_s
        self._last_status_observation_count = self._published_observations
        self._last_status_action_count = self._received_actions

    def serve_forever(self) -> None:
        period = 1.0 / max(self._config.publish_rate_hz, 1e-6)
        next_tick = time.monotonic()
        while not self._stop.is_set():
            self._drain_actions()
            self._publish_observation()
            self._log_hardware_status_if_due()
            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def close(self) -> None:
        self._stop.set()
        self.stop_collection()
        if self._fk_solver is not None:
            self._fk_solver.close()
        self._robot.close()
        self._cameras.close()
        self._action_sub.close(linger=0)
        self._obs_pub.close(linger=0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--left-can-port", default=DEFAULT_LEFT_CAN_PORT)
    parser.add_argument("--right-can-port", default=DEFAULT_RIGHT_CAN_PORT)
    parser.add_argument(
        "--alicia-port",
        default=DEFAULT_ALICIA_PORT,
        help="Legacy alias for --right-alicia-port.",
    )
    parser.add_argument("--left-alicia-port", default=DEFAULT_LEFT_ALICIA_PORT)
    parser.add_argument("--right-alicia-port", default=DEFAULT_RIGHT_ALICIA_PORT)
    parser.add_argument(
        "--alicia-reader",
        choices=("duo", "sdk"),
        default=DEFAULT_ALICIA_READER,
    )
    parser.add_argument("--debug-alicia", action="store_true")
    parser.add_argument(
        "--collection-teleop-control",
        dest="passive_collection",
        action="store_false",
        help="Drive ARX arms from Alicia-D during collection; this is the default.",
    )
    parser.add_argument(
        "--passive-collection",
        dest="passive_collection",
        action="store_true",
        help="Only observe ARX arms/cameras during collection; do not drive ARX from Alicia-D.",
    )
    parser.set_defaults(passive_collection=False)
    parser.add_argument(
        "--arm-type",
        type=int,
        default=DEFAULT_ARM_TYPE,
        help="ARX SDK arm type code: 0 = X5lite URDF, 1 = R5 master URDF.",
    )
    parser.add_argument(
        "--gripper-open-pos",
        type=float,
        default=ARX_GRIPPER_OPEN_POS,
        help="ARX catch position commanded for a fully open gripper (scalar 1.0).",
    )
    parser.add_argument(
        "--initial-gripper-scalar",
        type=float,
        default=None,
        help="Initial gripper scalar to command on first arm connection; 1.0 is fully open.",
    )
    parser.add_argument(
        "--disabled-arm",
        action="append",
        default=[],
        metavar="left_arm|right_arm",
        help="Intentionally skip one ARX arm; repeatable.",
    )
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument(
        "--eva-config",
        default="",
        help="Optional EVA config; transport.disabled_cameras disables missing camera keys.",
    )
    parser.add_argument(
        "--disabled-camera",
        action="append",
        default=[],
        metavar="KEY",
        help="Disable an EVA image key in the hardware node; repeatable or comma-separated.",
    )
    parser.add_argument(
        "--orbbec-camera",
        action="append",
        default=[],
        metavar="KEY=SERIAL_OR_INDEX",
        help="Orbbec camera mapping, e.g. cam_high=CP053... or cam_high=index:0; repeatable.",
    )
    parser.add_argument(
        "--orbbec-resolution",
        default="640x480",
        help="Requested Orbbec color resolution, e.g. 640x480; use default for the SDK profile.",
    )
    parser.add_argument("--orbbec-fps", type=int, default=30)
    parser.add_argument("--orbbec-color-format", default="MJPG")
    parser.add_argument("--orbbec-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--orbbec-white-balance-file",
        default="",
        help=(
            "YAML generated by utils/calibrate_white_balance.py; empty leaves camera WB unchanged."
        ),
    )
    parser.add_argument(
        "--status-log-interval",
        type=float,
        default=5.0,
        help="Seconds between hardware status logs; set <=0 to disable.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def build_config(args: argparse.Namespace) -> ArxZmqConfig:
    can_ports = {
        "left_arm": args.left_can_port,
        "right_arm": args.right_can_port,
    }
    configured_disabled = load_disabled_cameras(args.eva_config)
    cli_disabled = parse_name_list(args.disabled_camera)
    disabled_cameras = tuple(dict.fromkeys(configured_disabled + cli_disabled))
    resolution = parse_resolution(args.orbbec_resolution)
    default_cameras = default_arx_orbbec_camera_specs(
        resolution=resolution,
        fps=int(args.orbbec_fps),
        color_format=str(args.orbbec_color_format),
        timeout_ms=int(args.orbbec_timeout_ms),
    )
    override_cameras = parse_orbbec_camera_specs(
        args.orbbec_camera,
        resolution=resolution,
        fps=int(args.orbbec_fps),
        color_format=str(args.orbbec_color_format),
        timeout_ms=int(args.orbbec_timeout_ms),
    )
    white_balances = load_orbbec_white_balances(args.orbbec_white_balance_file)
    cameras = merge_orbbec_camera_specs(
        default_cameras,
        override_cameras,
        disabled_cameras=disabled_cameras,
    )
    cameras = apply_orbbec_white_balances(cameras, white_balances)
    return ArxZmqConfig(
        observation_endpoint=args.obs_endpoint,
        action_endpoint=args.action_endpoint,
        can_ports=can_ports,
        arm_type=int(args.arm_type),
        gripper_open_pos=float(args.gripper_open_pos),
        initial_gripper_scalar=args.initial_gripper_scalar,
        disabled_groups=parse_disabled_groups(args.disabled_arm),
        orbbec_cameras=cameras,
        orbbec_white_balances=white_balances,
        publish_rate_hz=args.rate,
        left_alicia_port=args.left_alicia_port,
        right_alicia_port=args.right_alicia_port,
        alicia_port=args.alicia_port,
        alicia_reader=args.alicia_reader,
        debug_alicia=bool(args.debug_alicia),
        passive_collection=bool(args.passive_collection),
        status_log_interval_s=args.status_log_interval,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = build_config(args)
    node = ArxZmqNode(config)

    def _stop(_signum: int, _frame: Any) -> None:
        node.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        node.serve_forever()
    finally:
        node.close()


if __name__ == "__main__":
    main()
