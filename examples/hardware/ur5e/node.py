#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import threading
import time

import numpy as np

import robots  # noqa: F401  # registers every zoo robot under the registries on import
from core.config import ConfigDict
from core.registry import ROBOT_REGISTRY
from examples.hardware.ur5e.camera import CameraSpec, open_cameras, read_camera_images
from examples.hardware.ur5e.robot import Ur5eRobot
from examples.hardware.ur5e.teleop import Ur5eTeleopSource
from transport.zmq import (
    COLLECTION_START_TARGET,
    COLLECTION_STOP_TARGET,
    WireObservation,
    pack_observation,
    unpack_action,
)

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_SPECS = [
    "wrist_image:2:1280:720:25:0",
    "exterior_image:0:1280:720:25:0",
]


def _format_diag_vector(value: np.ndarray, max_dims: int = 8) -> str:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return np.array2string(vector[:max_dims], precision=4, separator=", ")


@dataclasses.dataclass(frozen=True)
class Ur5eHardwareConfig:
    observation_endpoint: str
    action_endpoint: str
    robot_ip: str
    gripper_port: str
    cameras: tuple[CameraSpec, ...]
    publish_rate_hz: float
    raw_open: int = 1000
    raw_close: int = 0
    servo_speed: float = 0.1
    servo_accel: float = 0.1
    servo_lookahead_time: float = 0.1
    servo_gain: int = 500
    teleop: ConfigDict | None = None


class Ur5eZmqNode:
    def __init__(self, config: Ur5eHardwareConfig) -> None:
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
        self._robot = Ur5eRobot(config)
        robot = ROBOT_REGISTRY.build("ur5e")
        self._fk_solver = robot.build_kinematics(
            initial_qpos_groups=robot.initial_qpos_by_group()
        )
        self._captures = open_cameras(config.cameras)
        self._collection_active = False
        self._applied_action_count = 0
        self._last_obs_log_time = 0.0
        self._teleop_source = (
            Ur5eTeleopSource(config.teleop)
            if config.teleop is not None and config.teleop.type == "ur5e_master_arm"
            else None
        )

    def _read_qpos(self) -> np.ndarray:
        return self._robot.read_qpos()

    def _read_images(self) -> dict[str, np.ndarray]:
        return read_camera_images(self._captures, self._config.cameras)

    def get_latest_qpos(self) -> np.ndarray | None:
        return self._read_qpos()

    def start_collection(self) -> None:
        if self._collection_active:
            return
        if self._teleop_source is None:
            return
        self._teleop_source.connect()
        try:
            qpos = self.get_latest_qpos()
            if qpos is None:
                raise RuntimeError("UR5e qpos unavailable for teleop calibration")
            self._teleop_source.calibrate_delta(qpos)
            self._collection_active = True
        except Exception:
            self._collection_active = False
            self._teleop_source.disconnect()
            raise

    def stop_collection(self) -> None:
        self._collection_active = False
        if self._teleop_source is not None:
            self._teleop_source.disconnect()

    def _drain_actions(self) -> None:
        while True:
            try:
                payload = self._action_sub.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                return
            action = unpack_action(payload)
            if action.target == COLLECTION_START_TARGET:
                self.start_collection()
                continue
            if action.target == COLLECTION_STOP_TARGET:
                self.stop_collection()
                continue
            if action.target == "sim":
                continue
            self._apply_action(action.action)

    def _apply_action(self, action: np.ndarray) -> None:
        self._applied_action_count = getattr(self, "_applied_action_count", 0) + 1
        start = time.monotonic()
        vector = np.asarray(action, dtype=np.float32)
        grip = float(vector[6]) if len(vector) > 6 else float("nan")
        logger.info(
            "[UR5E_APPLY_BEGIN] index=%d action=%s grip=%.4f",
            self._applied_action_count,
            _format_diag_vector(vector),
            grip,
        )
        self._robot.apply_action(action)
        logger.info(
            "[UR5E_APPLY_END] index=%d elapsed_ms=%.1f",
            self._applied_action_count,
            1000.0 * (time.monotonic() - start),
        )

    def _publish_observation(self) -> None:
        qpos = self._read_qpos()
        now = time.monotonic()
        if now - getattr(self, "_last_obs_log_time", 0.0) >= 1.0:
            grip = float(qpos[6]) if len(qpos) > 6 else float("nan")
            logger.info(
                "[UR5E_OBS] qpos=%s grip=%.4f collection=%s",
                _format_diag_vector(qpos),
                grip,
                self._collection_active,
            )
            self._last_obs_log_time = now
        action_qpos = None
        action_eef = None
        if self._collection_active and self._teleop_source is not None:
            action_qpos = self._teleop_source.read_action_qpos(qpos)
            action_eef = self._fk_solver.fk_chunk(action_qpos[np.newaxis, :])[0]
            self._apply_action(action_qpos)
        eef = self._robot.read_eef(qpos)
        obs = WireObservation(
            t=time.monotonic(),
            images=self._read_images(),
            state={"arm": qpos},
            eef={"arm": eef},
            action=action_qpos,
            action_eef=action_eef,
        )
        self._obs_pub.send(pack_observation(obs))

    def serve_forever(self) -> None:
        period = 1.0 / max(self._config.publish_rate_hz, 1e-6)
        next_tick = time.monotonic()
        while not self._stop.is_set():
            self._drain_actions()
            self._publish_observation()
            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def close(self) -> None:
        self._stop.set()
        self.stop_collection()
        self._fk_solver.close()
        self._robot.close()
        for capture in self._captures.values():
            capture.release()
        self._action_sub.close(linger=0)
        self._obs_pub.close(linger=0)


def _parse_camera_specs(raw_specs: list[str]) -> tuple[CameraSpec, ...]:
    specs = []
    for raw in raw_specs:
        name, source, width, height, fps, rotation = raw.split(":")
        obs_key = "cam_wrist" if name == "wrist_image" else "cam_high"
        specs.append(
            CameraSpec(
                name=name,
                observation_key=obs_key,
                source=source,
                width=int(width),
                height=int(height),
                fps=int(fps),
                rotation=int(rotation),
            )
        )
    return tuple(specs)


def _parse_joint_coef(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _build_teleop_config(args: argparse.Namespace) -> ConfigDict | None:
    if args.teleop_port == "":
        return None
    return ConfigDict(
        type="ur5e_master_arm",
        port=args.teleop_port,
        joint_coef=_parse_joint_coef(args.teleop_joint_coef),
        gripper=ConfigDict(source="analog"),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UR5e hardware-side ZMQ node for EVA.")
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--robot-ip", default="192.168.31.123")
    parser.add_argument("--gripper-port", default="/dev/ttyUSB1")
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--teleop-port", default="")
    parser.add_argument("--teleop-joint-coef", default="1,-1,-1,-1,-1,-1")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    camera_specs = args.camera if args.camera is not None else DEFAULT_CAMERA_SPECS
    node = Ur5eZmqNode(
        Ur5eHardwareConfig(
            observation_endpoint=args.obs_endpoint,
            action_endpoint=args.action_endpoint,
            robot_ip=args.robot_ip,
            gripper_port=args.gripper_port,
            cameras=_parse_camera_specs(camera_specs),
            publish_rate_hz=args.rate,
            teleop=_build_teleop_config(args),
        )
    )
    signal.signal(signal.SIGINT, lambda *_args: node.close())
    signal.signal(signal.SIGTERM, lambda *_args: node.close())
    try:
        node.serve_forever()
    finally:
        node.close()


if __name__ == "__main__":
    main()
