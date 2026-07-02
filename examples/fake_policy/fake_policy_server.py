#!/usr/bin/env python3
"""Fake policy server for debugging the EVA client without a real model server.

Speaks the same msgpack-over-WebSocket protocol as the real OpenPI and starVLA
policy servers, so it drops in behind any ``policy.type`` of ``openpi``,
``openpi_rtc`` or ``starvla``:

  * on connect it sends one handshake metadata frame (chunk_size, action_mode, …);
  * for each inference request it returns a smooth synthetic action chunk shaped
    ``[chunk_size, action_dim]`` so the 3D canvas shows continuous motion.

The request envelope is auto-detected: OpenPI sends a bare observation dict
(``{"state", "images", "prompt", ...}``) and expects a bare ``{"actions": ...}``
reply; starVLA wraps requests as ``{"type": "infer", "payload": {"examples": [...]}}``
and expects ``{"status": "ok", "data": {"actions": ndarray[B, T, D]}}``. The
action dimension is inferred from the request's state vector, with a CLI fallback.

Usage:
  bash examples/fake_policy/run_fake_policy.sh            # POLICY_PORT/CHUNK_SIZE env-overridable
  python examples/fake_policy/fake_policy_server.py --port 9000
  # then, in another shell:
  eva --config configs/01_deploy/ur5e/openpi_qpos.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import websockets.exceptions
import websockets.sync.server
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


def _infer_state(request: dict, action_dim: int) -> np.ndarray:
    """Pull the current state vector from a request, else fall back to zeros.

    Args:
        request: Decoded request dict (OpenPI observation or starVLA envelope).
        action_dim: Dimension to use when no state is present in the request.

    Returns:
        state: [action_dim] float32 current pose to anchor the action chunk on.
    """
    state = request.get("state")
    if state is None:
        payload = request.get("payload") or {}
        examples = payload.get("examples") or [{}]
        state = examples[0].get("state")
    if state is None:
        return np.zeros(action_dim, dtype=np.float32)
    return np.asarray(state, dtype=np.float32).reshape(-1)


class FakePolicy:
    """Generates smooth action chunks anchored on the observed state.

    Each chunk starts at the current pose (zero increment on the first step) and
    drifts by a small per-dimension sinusoid, so plugging in mid-episode never
    jumps the arm. With a closed-loop fake node feeding the last action back as
    the next state, successive chunks re-anchor on the new pose and the motion
    reads as a real policy predicting a short trajectory from where it is now.
    """

    def __init__(self, chunk_size: int) -> None:
        self._chunk_size = chunk_size
        self._call_index = 0
        rng = np.random.default_rng(42)
        # Per-dimension drift shape, pre-drawn for any plausible action_dim.
        # cycles: how many sine periods span one chunk; amps: drift magnitude (rad).
        self._cycles = rng.uniform(0.5, 1.5, size=64).astype(np.float32)
        self._amps = rng.uniform(0.04, 0.10, size=64).astype(np.float32)

    def chunk(self, state: np.ndarray) -> np.ndarray:
        """Synthesize one action chunk anchored on the current state.

        Args:
            state: [action_dim] float32 current pose.

        Returns:
            actions: [chunk_size, action_dim] float32 chunk; first step equals
                ``state``, later steps drift smoothly around it.
        """
        action_dim = state.shape[0]
        t_local = np.linspace(0.0, 1.0, self._chunk_size, dtype=np.float32)
        actions = np.tile(state, (self._chunk_size, 1))

        # Per-call phase so successive chunks differ; sin(phase)-subtraction
        # pins the first step's increment to exactly 0 (no jump at chunk start).
        call_phase = self._call_index * 0.3
        for d in range(action_dim):
            i = d % self._amps.shape[0]
            angle = 2 * np.pi * self._cycles[i] * t_local + call_phase
            actions[:, d] += self._amps[i] * (np.sin(angle) - np.sin(call_phase))

        self._call_index += 1
        return actions.astype(np.float32)


def serve(host: str, port: int, chunk_size: int, action_dim: int, action_mode: str) -> None:
    """Run the fake policy server until interrupted.

    Args:
        host, port: Bind address for the WebSocket server.
        chunk_size: Number of action steps per returned chunk.
        action_dim: Fallback action dimension when a request omits its state.
        action_mode: action_mode advertised in the handshake ("qpos" | "eef").
    """
    handshake = {
        "server_name": "eva-fake-policy",
        "action_mode": action_mode,
        "chunk_size": chunk_size,
        "action_dim": action_dim,
    }
    packer = msgpack_numpy.Packer()

    def handler(conn: websockets.sync.server.ServerConnection) -> None:
        policy = FakePolicy(chunk_size)
        conn.send(packer.pack(handshake))
        logger.info("[FAKE-POLICY] client connected: %s", conn.remote_address)
        try:
            for raw in conn:
                request = msgpack_numpy.unpackb(raw)
                state = _infer_state(request, action_dim)
                actions = policy.chunk(state)
                if request.get("type") == "infer":
                    # starVLA envelope: actions are [B, T, D].
                    reply = {"status": "ok", "data": {"actions": actions[None]}}
                else:
                    reply = {"actions": actions, "action_mode": action_mode}
                conn.send(packer.pack(reply))
        except websockets.exceptions.ConnectionClosed:
            pass
        logger.info("[FAKE-POLICY] client disconnected")

    logger.info(
        "[FAKE-POLICY] serving ws://%s:%d chunk_size=%d action_dim=%d mode=%s",
        host,
        port,
        chunk_size,
        action_dim,
        action_mode,
    )
    with websockets.sync.server.serve(
        handler, host, port, compression=None, max_size=None
    ) as server:
        server.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fake policy server for EVA debugging.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--action-dim",
        type=int,
        default=14,
        help="Fallback action dim; normally inferred from the request's state vector.",
    )
    parser.add_argument("--action-mode", default="qpos", choices=["qpos", "eef"])
    args = parser.parse_args()
    try:
        serve(args.host, args.port, args.chunk_size, args.action_dim, args.action_mode)
    except KeyboardInterrupt:
        logger.info("[FAKE-POLICY] shutting down")


if __name__ == "__main__":
    main()
