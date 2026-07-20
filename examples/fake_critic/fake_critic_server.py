#!/usr/bin/env python3
"""Serve a deterministic scalar Critic over EVA's msgpack WebSocket contract."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import websockets.exceptions
import websockets.sync.server
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


def critic_value(observation: dict, actions: np.ndarray) -> float:
    """Score an action chunk by its mean squared distance from the observed state.

    Args:
        observation: Policy-style observation containing a flat ``state`` vector.
        actions: Candidate action chunk ``[T, D]``.

    Returns:
        Scalar in ``(0, 1]``; actions closer to the current state score higher.
    """
    state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
    chunk = np.asarray(actions, dtype=np.float32).reshape(-1, actions.shape[-1])
    target = chunk.mean(axis=0)
    dim = min(state.shape[0], target.shape[0])
    distance = float(np.mean(np.square(target[:dim] - state[:dim])))
    return 1.0 / (1.0 + distance)


def serve(host: str, port: int, delay_ms: float) -> None:
    """Serve Critic requests until interrupted.

    Args:
        host: Bind interface.
        port: WebSocket port.
        delay_ms: Deliberate per-request latency used to exercise coalescing.
    """
    metadata = {
        "server_name": "eva-fake-critic",
        "output": "scalar_value",
        "delay_ms": float(delay_ms),
    }
    packer = msgpack_numpy.Packer()

    def handler(connection: websockets.sync.server.ServerConnection) -> None:
        connection.send(packer.pack(metadata))
        logger.info("[FAKE-CRITIC] client connected: %s", connection.remote_address)
        try:
            for raw in connection:
                request = msgpack_numpy.unpackb(raw)
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)
                value = critic_value(request["observation"], request["actions"])
                connection.send(packer.pack({"value": value}))
        except websockets.exceptions.ConnectionClosed:
            pass
        logger.info("[FAKE-CRITIC] client disconnected")

    logger.info("[FAKE-CRITIC] serving ws://%s:%d delay_ms=%.1f", host, port, delay_ms)
    with websockets.sync.server.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
    ) as server:
        server.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--delay-ms", type=float, default=8.0)
    args = parser.parse_args()
    try:
        serve(args.host, args.port, args.delay_ms)
    except KeyboardInterrupt:
        logger.info("[FAKE-CRITIC] shutting down")


if __name__ == "__main__":
    main()
