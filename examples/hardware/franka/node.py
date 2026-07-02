#!/usr/bin/env python3
"""Franka execution-layer node for EVA's ZMQ transport."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.config import load_config
from examples.hardware.franka.camera import (
    OrbbecCameraCache,
    OrbbecCameraSpec,
    default_franka_orbbec_camera_specs,
    merge_orbbec_camera_specs,
    parse_orbbec_camera_specs,
    parse_resolution,
)
from examples.hardware.franka.robot import (
    DEFAULT_LEFT_ROBOT_IP,
    DEFAULT_RIGHT_ROBOT_IP,
    FrankyDualArm,
    parse_disabled_groups,
)
from transport.zmq import WireObservation, pack_observation, unpack_action

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FrankaZmqConfig:
    """Runtime config for the Franka ZMQ execution node.

    Args:
        observation_endpoint: ZMQ PUB bind endpoint for WireObservation frames.
        action_endpoint: ZMQ SUB bind endpoint for WireAction commands.
        robot_ips: arm group name -> Franka robot IP.
        gripper_ips: arm group name -> Franka gripper IP.
        disabled_groups: arm groups intentionally absent from this deployment.
        orbbec_cameras: Orbbec SDK cameras mapped to EVA image keys.
        publish_rate_hz: observation publish rate.
        status_log_interval_s: hardware status log interval; non-positive disables it.
    """

    observation_endpoint: str
    action_endpoint: str
    robot_ips: dict[str, str]
    gripper_ips: dict[str, str]
    disabled_groups: tuple[str, ...]
    orbbec_cameras: tuple[OrbbecCameraSpec, ...]
    publish_rate_hz: float
    status_log_interval_s: float = 5.0


class EmptyCameraCache:
    def snapshot(self) -> dict[str, np.ndarray]:
        return {}

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


def _configure_logging(log_level: str, log_file: str) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


class FrankaZmqNode:
    """Bridge EVA ZMQ wire messages to a Franky-controlled dual Franka pair."""

    def __init__(self, config: FrankaZmqConfig) -> None:
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
        self._robot = FrankyDualArm(config)
        self._cameras = self._build_camera_cache(config)
        self._received_actions = 0
        self._published_observations = 0
        now = time.monotonic()
        self._last_status_log_time = now
        self._next_status_log_time = now + max(config.status_log_interval_s, 0.0)
        self._last_status_action_count = 0
        self._last_status_observation_count = 0
        logger.info(
            "Franka ZMQ node ready: obs_pub=%s action_sub=%s robot_ips=%s "
            "orbbec_cameras=%d status_log_interval=%.1fs",
            config.observation_endpoint,
            config.action_endpoint,
            config.robot_ips,
            len(config.orbbec_cameras),
            config.status_log_interval_s,
        )

    def _build_camera_cache(self, config: FrankaZmqConfig) -> Any:
        if not config.orbbec_cameras:
            logger.info("Orbbec camera backend disabled: no cameras configured")
            return EmptyCameraCache()
        return OrbbecCameraCache(config.orbbec_cameras)

    def stop(self) -> None:
        self._stop.set()

    def _drain_actions(self) -> None:
        while True:
            try:
                payload = self._action_sub.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                return
            action = unpack_action(payload)
            self._robot.apply_action(action)
            self._received_actions += 1

    def _publish_observation(self) -> None:
        state = self._robot.read_state()
        images = self._cameras.snapshot()
        obs = WireObservation(t=time.monotonic(), images=images, state=state)
        self._obs_pub.send(pack_observation(obs))
        self._published_observations += 1

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
        self._robot.close()
        self._cameras.close()
        self._action_sub.close(linger=0)
        self._obs_pub.close(linger=0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--left-robot-ip", default=DEFAULT_LEFT_ROBOT_IP)
    parser.add_argument("--right-robot-ip", default=DEFAULT_RIGHT_ROBOT_IP)
    parser.add_argument("--left-gripper-ip", default="")
    parser.add_argument("--right-gripper-ip", default="")
    parser.add_argument(
        "--disabled-arm",
        action="append",
        default=[],
        metavar="left_arm|right_arm",
        help="Intentionally skip one Franka arm; repeatable.",
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
        default="default",
        help="Requested Orbbec color resolution, e.g. 1280x720; default uses SDK profile.",
    )
    parser.add_argument("--orbbec-fps", type=int, default=30)
    parser.add_argument("--orbbec-color-format", default="RGB")
    parser.add_argument("--orbbec-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--status-log-interval",
        type=float,
        default=5.0,
        help="Seconds between hardware status logs; set <=0 to disable.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file",
        default="Log/franka_node.log",
        help="Write Franky, Orbbec camera, and node Python logs to one file; empty disables it.",
    )
    return parser


def build_config(args: argparse.Namespace) -> FrankaZmqConfig:
    robot_ips = {
        "left_arm": args.left_robot_ip,
        "right_arm": args.right_robot_ip,
    }
    gripper_ips = {
        "left_arm": args.left_gripper_ip or args.left_robot_ip,
        "right_arm": args.right_gripper_ip or args.right_robot_ip,
    }
    configured_disabled = load_disabled_cameras(args.eva_config)
    cli_disabled = parse_name_list(args.disabled_camera)
    disabled_cameras = tuple(dict.fromkeys(configured_disabled + cli_disabled))
    resolution = parse_resolution(args.orbbec_resolution)
    default_cameras = default_franka_orbbec_camera_specs(
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
    return FrankaZmqConfig(
        observation_endpoint=args.obs_endpoint,
        action_endpoint=args.action_endpoint,
        robot_ips=robot_ips,
        gripper_ips=gripper_ips,
        disabled_groups=parse_disabled_groups(args.disabled_arm),
        orbbec_cameras=merge_orbbec_camera_specs(
            default_cameras,
            override_cameras,
            disabled_cameras=disabled_cameras,
        ),
        publish_rate_hz=args.rate,
        status_log_interval_s=args.status_log_interval,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level, args.log_file)
    config = build_config(args)
    node = FrankaZmqNode(config)

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
