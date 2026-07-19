#!/usr/bin/env python3
"""In-sim teleoperation node for ARX X5: drive eva_sim from Alicia-D leaders.

This is the ARX X5 analogue of ``examples/hardware/arx/node.py``, but the ARX
follower is **simulated by eva_sim** instead of driven over CAN. eva_sim is the
execution layer: it binds the observation (PUB) and action (SUB) ZMQ endpoints,
applies the absolute QPos we publish, renders, and — when EVA has armed a
collection episode — bundles our teleop action back into its observation stream.
EVA (``eva --config configs/02_collection/arx_x5.py``) subscribes to eva_sim and
records that ``action_qpos`` exactly as it records the real ARX node.

Data path:  Alicia-D leaders --(this node)--> eva_sim --(obs)--> EVA (records)

The Alicia leader reading and Alicia->ARX joint mapping are **reused verbatim**
from the real ARX hardware example (``examples.hardware.arx``); the only ARX-
specific hardware this node needs is the leader arms on serial, so it defaults to
the pure-serial ``duo`` reader (no ARX SDK, no CAN).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the real ARX Alicia-D leader reader + Alicia->ARX mapping verbatim.
from examples.hardware.arx.node import ArxTeleopSource  # noqa: E402
from examples.hardware.arx.robot import GROUP_NAMES, TOTAL_DOF  # noqa: E402
from examples.hardware.arx.teleop import (  # noqa: E402
    DEFAULT_ALICIA_PORT,
    DEFAULT_LEFT_ALICIA_PORT,
    DEFAULT_RIGHT_ALICIA_PORT,
)
from transport.zmq import (  # noqa: E402
    WireAction,
    pack_action,
    unpack_observation,
)

logger = logging.getLogger(__name__)

GROUP_DOF = TOTAL_DOF // len(GROUP_NAMES)


def _teleop_source_config(args: argparse.Namespace) -> Any:
    """Minimal duck-typed config for ``ArxTeleopSource`` (leader side only)."""
    return types.SimpleNamespace(
        left_alicia_port=args.left_alicia_port,
        right_alicia_port=args.right_alicia_port,
        alicia_port=args.alicia_port,
        alicia_reader=args.alicia_reader,
        debug_alicia=bool(args.debug_alicia),
        disabled_groups=(),
    )


def _read_sim_qpos(obs_sub: Any, zmq_mod: Any, timeout_s: float) -> np.ndarray | None:
    """Return eva_sim's latest 14-D follower QPos from its obs stream, or None."""
    deadline = time.monotonic() + timeout_s
    latest: np.ndarray | None = None
    while True:
        try:
            payload = obs_sub.recv(zmq_mod.NOBLOCK)
        except zmq_mod.Again:
            if latest is not None or time.monotonic() >= deadline:
                return latest
            time.sleep(0.005)
            continue
        obs = unpack_observation(payload)
        state = obs.state or {}
        if "left_arm" in state and "right_arm" in state:
            latest = np.concatenate(
                [
                    np.asarray(state["left_arm"], dtype=np.float32),
                    np.asarray(state["right_arm"], dtype=np.float32),
                ]
            )


def run(args: argparse.Namespace) -> None:
    import zmq

    ctx = zmq.Context.instance()
    action_pub = ctx.socket(zmq.PUB)
    action_pub.connect(args.action_endpoint)
    obs_sub = ctx.socket(zmq.SUB)
    obs_sub.setsockopt(zmq.SUBSCRIBE, b"")
    obs_sub.connect(args.obs_endpoint)

    # PUB/SUB peers need a moment to finish connecting before the first send.
    time.sleep(0.3)
    logger.info(
        "ARX X5 sim-teleop node ready: action_pub=%s obs_sub=%s reader=%s rate=%.1fHz",
        args.action_endpoint,
        args.obs_endpoint,
        args.alicia_reader,
        args.rate,
    )

    source = ArxTeleopSource(_teleop_source_config(args))
    source.connect()

    # Calibrate the Alicia->follower offset against eva_sim's current QPos so
    # teleop starts without a jump.
    robot_qpos = _read_sim_qpos(obs_sub, zmq, timeout_s=5.0)
    if robot_qpos is None:
        logger.warning("no observation from eva_sim yet; calibrating against zero QPos")
        robot_qpos = np.zeros(TOTAL_DOF, dtype=np.float32)
    source.calibrate_delta(robot_qpos)
    logger.info("calibrated against follower QPos; driving eva_sim")

    stop = {"flag": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    period = 1.0 / max(args.rate, 1e-6)
    next_tick = time.monotonic()
    try:
        while not stop["flag"]:
            latest = _read_sim_qpos(obs_sub, zmq, timeout_s=0.0)
            if latest is not None:
                robot_qpos = latest
            action_qpos = source.read_action_qpos(robot_qpos)
            action_pub.send(
                pack_action(WireAction(t=time.monotonic(), action=action_qpos, target="sim"))
            )
            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    finally:
        source.disconnect()
        action_pub.close(linger=0)
        obs_sub.close(linger=0)
        logger.info("ARX X5 sim-teleop node stopped")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--left-alicia-port", default=DEFAULT_LEFT_ALICIA_PORT)
    parser.add_argument("--right-alicia-port", default=DEFAULT_RIGHT_ALICIA_PORT)
    parser.add_argument(
        "--alicia-port", default=DEFAULT_ALICIA_PORT, help="Legacy alias for --right-alicia-port."
    )
    parser.add_argument(
        "--alicia-reader",
        choices=("duo", "sdk"),
        default="duo",
        help="Alicia state reader: 'duo' is pure serial (no ARX SDK); 'sdk' uses alicia_d_sdk.",
    )
    parser.add_argument("--debug-alicia", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
