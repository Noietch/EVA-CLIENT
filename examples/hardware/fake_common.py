#!/usr/bin/env python3
"""Generic fake hardware node for debugging the EVA client without real hardware.

Speaks the same ZMQ protocol as every real execution-layer node — PUBlishes
WireObservation frames and SUBscribes to WireAction commands — but synthesizes
everything in software for whichever robot is named:

  * state starts at the robot's initial_qpos and is fed back from the last action
    received (closed-loop follow), so RUN/DEBUG inference visibly moves the arms;
  * images are solid-color frames stamped with a moving band so the console shows
    a live, changing picture for each declared camera;
  * COLLECT teleop is synthesized as a slow sine wobble around the current qpos,
    so action_qpos is never None and the full collection/save pipeline can be
    exercised end to end;
  * when idle (no live policy/teleop), each ARM group's shoulder-pitch joint
    free-runs a small in-phase wobble so the arms always visibly move without
    swinging toward the centerline or self-colliding.

The robot is built from its bundled zoo config via ROBOT_REGISTRY, so a single
node generalizes over single-arm, dual-arm, and multi-group robots: state/eef are
keyed per actuator group exactly like the real node. No real-hardware SDK is
imported, so it runs anywhere.

Usage (per-robot wrappers call ``main`` with the registry name):
  python examples/hardware/<robot>/fake_node.py
  # then, in another shell:
  eva --config configs/01_deploy/<robot>/openpi_qpos.py
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time

import numpy as np

import robots  # noqa: F401  # registers every zoo robot under ROBOT_REGISTRY on import
from core.registry import ROBOT_REGISTRY
from transport.zmq import (
    COLLECTION_START_TARGET,
    COLLECTION_STOP_TARGET,
    WireObservation,
    pack_observation,
    unpack_action,
)

logger = logging.getLogger(__name__)

# Per-arm EEF pose width: xyz + quat wxyz + gripper.
_EEF_DOF = 8

# No real action within this window => free-run idle wobble so the arms always move.
_IDLE_ACTION_TIMEOUT_S = 0.5

# Idle wobble drives only the shoulder-pitch joint (arm-local index 1) of each arm,
# in phase and small-amplitude, so dual arms mirror each other without swinging toward
# the centerline or folding a single arm onto itself — no self-collision.
_IDLE_WOBBLE_JOINT = 1
_IDLE_WOBBLE_AMP = 0.15

# Distinct base colors per camera so the console tiles are easy to tell apart.
_CAMERA_COLORS = {
    "cam_high": (160, 70, 40),
    "cam_wrist": (40, 90, 160),
    "cam_left_wrist": (40, 90, 160),
    "cam_right_wrist": (40, 160, 90),
}
_DEFAULT_COLOR = (80, 80, 80)


def _synth_image(
    color: tuple[int, int, int], height: int, width: int, frame_index: int
) -> np.ndarray:
    """Build one solid-color RGB frame whose brightness pulses with the frame index.

    Args:
        color: base (r, g, b) for this camera.
        height: image height in pixels.
        width: image width in pixels.
        frame_index: running frame counter, drives a slow brightness pulse.

    Returns:
        img: [height, width, 3] uint8 RGB image.
    """
    pulse = 0.5 + 0.5 * float(np.sin(frame_index * 0.05))
    rgb = np.asarray(color, dtype=np.float32) * (0.4 + 0.6 * pulse)
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = rgb.astype(np.uint8)
    # A moving bright band gives an unmistakable "this is live" cue in the UI.
    band = int((frame_index * 4) % height)
    img[band : min(band + max(height // 20, 2), height), :, :] = 255
    return img


class FakeRobotNode:
    """Software-only stand-in for any zoo robot: synthesizes obs, echoes actions."""

    def __init__(
        self,
        robot_name: str,
        observation_endpoint: str,
        action_endpoint: str,
        publish_rate_hz: float,
        image_height: int,
        image_width: int,
    ) -> None:
        import zmq

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
        self._publish_rate_hz = publish_rate_hz
        self._image_height = image_height
        self._image_width = image_width

        self._qpos = np.asarray(self._robot.initial_qpos, dtype=np.float32).copy()
        self._initial_qpos = self._qpos.copy()
        self._collection_active = False
        self._frame_index = 0
        self._started_at = time.monotonic()
        self._last_action_time = 0.0
        self._stop = threading.Event()

    def _split_by_group(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        """Slice a flat full-robot vector into per-actuator-group chunks.

        Args:
            vector: [action_dim] flat vector ordered by actuator group.

        Returns:
            group name -> [group.dof] slice (copies).
        """
        parts: dict[str, np.ndarray] = {}
        offset = 0
        for group in self._groups:
            parts[group.name] = vector[offset : offset + group.dof].copy()
            offset += group.dof
        return parts

    def _zero_eef_by_group(self) -> dict[str, np.ndarray]:
        return {group.name: np.zeros(_EEF_DOF, dtype=np.float32) for group in self._robot.arm_groups}

    def _read_images(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for cam in self._cameras:
            color = _CAMERA_COLORS.get(cam.observation_key, _DEFAULT_COLOR)
            images[cam.observation_key] = _synth_image(
                color, self._image_height, self._image_width, self._frame_index
            )
        return images

    def start_collection(self) -> None:
        if self._collection_active:
            return
        self._collection_active = True
        logger.info("[FAKE] collection started (synthetic teleop wobble)")

    def stop_collection(self) -> None:
        if not self._collection_active:
            return
        self._collection_active = False
        logger.info("[FAKE] collection stopped")

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
            # Closed-loop: the commanded joint vector becomes the next reported state.
            vector = np.asarray(action.action, dtype=np.float32).reshape(-1)
            if vector.shape == (self._action_dim,):
                self._qpos = vector.copy()
                self._last_action_time = time.monotonic()

    def _synth_idle_qpos(self) -> np.ndarray:
        """Free-run wobble so the arms keep moving when idle, without self-collision.

        Only each ARM group's shoulder-pitch joint moves, small-amplitude and in
        phase, so dual arms mirror each other and never swing toward the centerline
        or fold; all other joints and the grippers stay at initial_qpos.

        Returns:
            qpos: [action_dim] float32 state.
        """
        t = time.monotonic() - self._started_at
        qpos = self._initial_qpos.copy()
        offset = 0
        for group in self._groups:
            if group.name.endswith("arm") and group.dof > _IDLE_WOBBLE_JOINT:
                qpos[offset + _IDLE_WOBBLE_JOINT] += _IDLE_WOBBLE_AMP * float(np.sin(t * 1.0))
            offset += group.dof
        return qpos.astype(np.float32)

    def _publish_observation(self) -> None:
        action_qpos = None
        action_eef = None
        eef = self._zero_eef_by_group()
        # Idle and COLLECT share one wobble so the visible motion never changes when
        # recording starts; COLLECT only additionally reports it as action_qpos.
        if self._collection_active or (
            time.monotonic() - self._last_action_time > _IDLE_ACTION_TIMEOUT_S
        ):
            self._qpos = self._synth_idle_qpos()
        if self._collection_active:
            action_qpos = self._qpos.copy()
            action_eef = np.zeros(_EEF_DOF * len(self._robot.arm_groups), dtype=np.float32)
        obs = WireObservation(
            t=time.monotonic(),
            images=self._read_images(),
            state=self._split_by_group(self._qpos),
            eef=eef,
            action=action_qpos,
            action_eef=action_eef,
        )
        self._obs_pub.send(pack_observation(obs))
        self._frame_index += 1

    def serve_forever(self) -> None:
        period = 1.0 / max(self._publish_rate_hz, 1e-6)
        next_tick = time.monotonic()
        logger.info(
            "[FAKE] serving: action_dim=%d groups=%s cameras=%s rate=%.1fHz",
            self._action_dim,
            [group.name for group in self._groups],
            [cam.observation_key for cam in self._cameras],
            self._publish_rate_hz,
        )
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
        self._action_sub.close(linger=0)
        self._obs_pub.close(linger=0)


def build_arg_parser(robot_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Fake {robot_name} hardware node for EVA debugging."
    )
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    return parser


def main(robot_name: str) -> None:
    """Run a fake node for the named zoo robot until SIGINT/SIGTERM.

    Args:
        robot_name: ROBOT_REGISTRY key (zoo robot ``name``, e.g. "ur5e").
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser(robot_name).parse_args()
    node = FakeRobotNode(
        robot_name=robot_name,
        observation_endpoint=args.obs_endpoint,
        action_endpoint=args.action_endpoint,
        publish_rate_hz=args.rate,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    signal.signal(signal.SIGINT, lambda *_args: node.close())
    signal.signal(signal.SIGTERM, lambda *_args: node.close())
    try:
        node.serve_forever()
    finally:
        node.close()
