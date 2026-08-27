#!/usr/bin/env python3
"""Generic configurable-response ZMQ fake hardware node for EVA Client development.

The node builds any registered robot, applies full-robot joint-position targets either
directly or through independent second-order qpos systems, and publishes the resulting
state with deterministic animated camera frames over EVA's ZMQ wire protocol.

The plant runs for the lifetime of this process. EVA collection, HIL, and simulator
control messages do not affect it. In an interactive terminal, enter ``r`` to reset the
plant to the robot's initial qpos or ``q`` to stop the process.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

import numpy as np

import robots  # noqa: F401  # registers every zoo robot on import
from core.registry import ROBOT_REGISTRY
from transport.zmq import WireObservation, pack_observation, unpack_action

logger = logging.getLogger(__name__)

_EEF_DOF = 8
_DEFAULT_PUBLISH_RATE_HZ = 30.0
_DEFAULT_DYNAMICS_RATE_HZ = 200.0
_DEFAULT_NATURAL_FREQUENCY_HZ = 2.0
_DEFAULT_DAMPING_RATIO = 1.0
_DEFAULT_DYNAMICS_MODE = "second-order"
_DYNAMICS_MODES = ("second-order", "direct")
_MAX_DYNAMICS_CATCHUP_STEPS = 20


def _positive(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return parsed


def _synth_image(
    height: int,
    width: int,
    elapsed_s: float,
    frame_index: int,
    camera_index: int,
) -> np.ndarray:
    """Create a camera-distinct RGB frame with periodic colors and moving bands."""
    camera_phase = camera_index * (2.0 * np.pi / 3.0)
    phase = 2.0 * np.pi * elapsed_s / 4.0 + camera_phase
    channels = 112.0 + 88.0 * np.sin(
        phase + np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    )
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = np.clip(channels, 0.0, 255.0).astype(np.uint8)

    row_width = max(1, height // 24)
    row = int((frame_index * (camera_index + 2)) % height)
    image[row : min(row + row_width, height), :, :] = 255

    column_width = max(1, width // 32)
    column = int((frame_index * (camera_index + 3)) % width)
    image[:, column : min(column + column_width, width), camera_index % 3] = 0
    return image


class FakeRobotNode:
    """Autonomous second-order plant behind EVA's ZMQ action/state protocol."""

    def __init__(
        self,
        robot_name: str,
        observation_endpoint: str,
        action_endpoint: str,
        publish_rate_hz: float = _DEFAULT_PUBLISH_RATE_HZ,
        image_height: int = 224,
        image_width: int = 224,
        dynamics_rate_hz: float = _DEFAULT_DYNAMICS_RATE_HZ,
        natural_frequency_hz: float = _DEFAULT_NATURAL_FREQUENCY_HZ,
        damping_ratio: float = _DEFAULT_DAMPING_RATIO,
        dynamics_mode: str = _DEFAULT_DYNAMICS_MODE,
    ) -> None:
        import zmq

        self._dynamics_mode = str(dynamics_mode).strip().lower()
        if self._dynamics_mode not in _DYNAMICS_MODES:
            raise ValueError(
                f"dynamics_mode must be one of {_DYNAMICS_MODES}, got {dynamics_mode!r}"
            )
        self._publish_rate_hz = _positive(publish_rate_hz, "publish_rate_hz")
        self._dynamics_rate_hz = _positive(dynamics_rate_hz, "dynamics_rate_hz")
        self._natural_frequency_hz = _positive(natural_frequency_hz, "natural_frequency_hz")
        self._damping_ratio = _positive(damping_ratio, "damping_ratio")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image dimensions must be positive")

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._obs_pub = self._ctx.socket(zmq.PUB)
        self._obs_pub.bind(observation_endpoint)
        self._action_sub = self._ctx.socket(zmq.SUB)
        self._action_sub.bind(action_endpoint)
        self._action_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._action_sub.setsockopt(zmq.RCVTIMEO, 0)

        self._robot = ROBOT_REGISTRY.build(robot_name)
        self._action_dim = self._robot.total_action_dim
        self._groups = self._robot.actuator_groups
        self._cameras = self._robot.observation_schema.cameras
        self._image_height = int(image_height)
        self._image_width = int(image_width)

        self._initial_qpos = np.asarray(self._robot.initial_qpos, dtype=np.float64).copy()
        if self._initial_qpos.shape != (self._action_dim,):
            raise ValueError(
                f"{robot_name} initial_qpos must have shape ({self._action_dim},), "
                f"got {self._initial_qpos.shape}"
            )
        self._qpos = self._initial_qpos.copy()
        self._qvel = np.zeros(self._action_dim, dtype=np.float64)
        self._target_qpos = self._initial_qpos.copy()

        self._frame_index = 0
        self._started_at = time.monotonic()
        self._reset_requested = threading.Event()
        self._stop = threading.Event()
        self._closed = False

    @property
    def qpos(self) -> np.ndarray:
        return self._qpos.astype(np.float32, copy=True)

    @property
    def qvel(self) -> np.ndarray:
        return self._qvel.astype(np.float32, copy=True)

    @property
    def target_qpos(self) -> np.ndarray:
        return self._target_qpos.astype(np.float32, copy=True)

    @property
    def is_stopped(self) -> bool:
        return self._stop.is_set()

    def set_target_qpos(self, action: np.ndarray) -> bool:
        """Accept one finite full-robot position target and apply the configured response."""
        target = np.asarray(action, dtype=np.float64).reshape(-1)
        if target.shape != (self._action_dim,) or not np.all(np.isfinite(target)):
            logger.warning(
                "[FAKE] ignored invalid action: expected finite shape (%d,), got %s",
                self._action_dim,
                target.shape,
            )
            return False
        self._apply_target_qpos(target)
        return True

    def _apply_target_qpos(self, target: np.ndarray) -> None:
        self._target_qpos = target.copy()
        if self._dynamics_mode == "direct":
            self._qpos = target.copy()
            self._qvel.fill(0.0)

    def reset(self) -> None:
        """Reset position, velocity, and target to the registered initial qpos."""
        self._qpos = self._initial_qpos.copy()
        self._qvel.fill(0.0)
        self._target_qpos = self._initial_qpos.copy()
        self._reset_requested.clear()
        logger.info("[FAKE] plant reset to initial qpos")

    def request_reset(self) -> None:
        """Ask the simulation thread to reset at its next dynamics tick."""
        self._reset_requested.set()

    def stop(self) -> None:
        self._stop.set()

    def _split_by_group(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        parts: dict[str, np.ndarray] = {}
        offset = 0
        for group in self._groups:
            parts[group.name] = np.asarray(
                vector[offset : offset + group.dof], dtype=np.float32
            ).copy()
            offset += group.dof
        return parts

    def _zero_eef_by_group(self) -> dict[str, np.ndarray]:
        return {
            group.name: np.zeros(_EEF_DOF, dtype=np.float32) for group in self._robot.arm_groups
        }

    def _read_images(self, timestamp: float) -> dict[str, np.ndarray]:
        elapsed_s = max(0.0, timestamp - self._started_at)
        return {
            camera.observation_key: _synth_image(
                self._image_height,
                self._image_width,
                elapsed_s,
                self._frame_index,
                camera_index,
            )
            for camera_index, camera in enumerate(self._cameras)
        }

    def _drain_actions(self, *, apply: bool = True) -> None:
        """Drain the socket and apply only the newest valid real-robot target."""
        newest: np.ndarray | None = None
        while True:
            try:
                payload = self._action_sub.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            try:
                action = unpack_action(payload)
            except Exception as error:
                logger.warning("[FAKE] ignored malformed action payload: %s", error)
                continue
            if action.target != "real":
                continue
            candidate = np.asarray(action.action, dtype=np.float64).reshape(-1)
            if candidate.shape != (self._action_dim,) or not np.all(np.isfinite(candidate)):
                logger.warning(
                    "[FAKE] ignored invalid action: expected finite shape (%d,), got %s",
                    self._action_dim,
                    candidate.shape,
                )
                continue
            newest = candidate
        if apply and newest is not None:
            self._apply_target_qpos(newest)

    def step_dynamics(self, dt: float | None = None) -> None:
        """Advance second-order qpos systems by one semi-implicit Euler step when enabled."""
        step = 1.0 / self._dynamics_rate_hz if dt is None else _positive(dt, "dt")
        if self._dynamics_mode == "direct":
            return
        omega = 2.0 * np.pi * self._natural_frequency_hz
        acceleration = omega * omega * (self._target_qpos - self._qpos)
        acceleration -= 2.0 * self._damping_ratio * omega * self._qvel
        self._qvel += step * acceleration
        self._qpos += step * self._qvel

    def _publish_observation(self) -> None:
        timestamp = time.monotonic()
        observation = WireObservation(
            t=timestamp,
            images=self._read_images(timestamp),
            state=self._split_by_group(self._qpos),
            eef=self._zero_eef_by_group(),
        )
        self._obs_pub.send(pack_observation(observation))
        self._frame_index += 1

    def serve_forever(self) -> None:
        dynamics_period = 1.0 / self._dynamics_rate_hz
        publish_period = 1.0 / self._publish_rate_hz
        next_dynamics = time.monotonic()
        next_publish = next_dynamics
        logger.info(
            "[FAKE] serving robot=%s action_dim=%d groups=%s cameras=%s "
            "dynamics_mode=%s dynamics=%.1fHz publish=%.1fHz "
            "natural_frequency=%.2fHz damping_ratio=%.2f",
            self._robot.name,
            self._action_dim,
            [group.name for group in self._groups],
            [camera.observation_key for camera in self._cameras],
            self._dynamics_mode,
            self._dynamics_rate_hz,
            self._publish_rate_hz,
            self._natural_frequency_hz,
            self._damping_ratio,
        )
        while not self._stop.is_set():
            if self._reset_requested.is_set():
                self._drain_actions(apply=False)
                self.reset()
                next_dynamics = time.monotonic()
            else:
                self._drain_actions()

            now = time.monotonic()
            steps = 0
            while now >= next_dynamics and steps < _MAX_DYNAMICS_CATCHUP_STEPS:
                self.step_dynamics(dynamics_period)
                next_dynamics += dynamics_period
                steps += 1
            if steps == _MAX_DYNAMICS_CATCHUP_STEPS and now >= next_dynamics:
                logger.warning("[FAKE] dynamics loop fell behind; dropping accumulated lag")
                next_dynamics = now + dynamics_period

            if now >= next_publish:
                self._publish_observation()
                missed = int((now - next_publish) // publish_period)
                next_publish += (missed + 1) * publish_period

            wait_s = max(0.0, min(next_dynamics, next_publish) - time.monotonic())
            self._stop.wait(min(wait_s, 0.01))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._action_sub.close(linger=0)
        self._obs_pub.close(linger=0)


def build_arg_parser(robot_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"ZMQ fake {robot_name} node for EVA debugging.")
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--rate", type=float, default=_DEFAULT_PUBLISH_RATE_HZ)
    parser.add_argument(
        "--dynamics-mode",
        choices=_DYNAMICS_MODES,
        default=_DEFAULT_DYNAMICS_MODE,
        help="apply actions through a second-order plant or copy them directly to state",
    )
    parser.add_argument("--dynamics-rate", type=float, default=_DEFAULT_DYNAMICS_RATE_HZ)
    parser.add_argument("--natural-frequency-hz", type=float, default=_DEFAULT_NATURAL_FREQUENCY_HZ)
    parser.add_argument("--damping-ratio", type=float, default=_DEFAULT_DAMPING_RATIO)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    return parser


def _console_input(node: FakeRobotNode) -> None:
    logger.info("[FAKE] terminal controls: r/reset + Enter, q/quit + Enter")
    while not node.is_stopped:
        line = sys.stdin.readline()
        if line == "":
            return
        command = line.strip().lower()
        if command in {"r", "reset"}:
            node.request_reset()
        elif command in {"q", "quit", "exit"}:
            node.stop()
            return
        elif command:
            logger.warning("[FAKE] unknown terminal command: %s", command)


def main(
    robot_name: str,
    node_type: type[FakeRobotNode] = FakeRobotNode,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser(robot_name).parse_args()
    node = node_type(
        robot_name=robot_name,
        observation_endpoint=args.obs_endpoint,
        action_endpoint=args.action_endpoint,
        publish_rate_hz=args.rate,
        image_height=args.image_height,
        image_width=args.image_width,
        dynamics_rate_hz=args.dynamics_rate,
        natural_frequency_hz=args.natural_frequency_hz,
        damping_ratio=args.damping_ratio,
        dynamics_mode=args.dynamics_mode,
    )
    signal.signal(signal.SIGINT, lambda *_args: node.stop())
    signal.signal(signal.SIGTERM, lambda *_args: node.stop())
    if sys.stdin.isatty():
        threading.Thread(target=_console_input, args=(node,), daemon=True).start()
    try:
        node.serve_forever()
    finally:
        node.close()


__all__ = ["FakeRobotNode", "build_arg_parser", "main"]
