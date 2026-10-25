#!/usr/bin/env python3
"""Bridge Dream-1 to EVA's OpenPI-compatible WebSocket policy protocol.

Run in the example's separate environment:
    bash examples/policy/dreamlabs/serve.sh --robot yam
Credentials come from DREAM_API_KEY or ~/.dreamlabs/credentials.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging

import dream
import numpy as np
import websockets.exceptions
import websockets.sync.server
from dream.client_ik import ARMS
from openpi_client import msgpack_numpy
from PIL import Image

logger = logging.getLogger(__name__)
VIDEO_KEYS = ("video.overhead", "video.wrist_l", "video.wrist_r")


class DreamPolicy:
    """One episode session, with WebSocket dispatch and Dream API conversion."""

    def __init__(
        self,
        config: dict,
        arm_dims: tuple[int, int] = (6, 6),
        gripper_key: str = "gripper",
    ) -> None:
        self._arm_dims = arm_dims
        self._state_slices = {}
        offset = 0
        for side, dim in zip(("left", "right"), arm_dims, strict=True):
            self._state_slices[f"state.{side}_arm"] = slice(offset, offset + dim)
            self._state_slices[f"state.{side}_{gripper_key}"] = slice(
                offset + dim, offset + dim + 1
            )
            offset += dim + 1
        self._action_dim = offset
        self._config = {**config, "_record": False}
        self._initialized = False
        self._client = dream.DreamClient()

    def infer(self, observation: dict) -> dict:
        """Convert an EVA observation to Dream and return left-then-right joint targets.

        Input state is [D], images are HWC RGB, and prompt is a string.
        Output actions is float32 [T, D], with each gripper
        following its arm's joints (0 closed, 1 open).
        """
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (self._action_dim,) or not np.isfinite(state).all():
            raise ValueError(f"Dream state must contain {self._action_dim} finite values")
        frames = {}
        for name, key in zip(
            ("cam_high", "cam_left_wrist", "cam_right_wrist"), VIDEO_KEYS, strict=True
        ):
            image = np.asarray(observation["images"][name])
            with io.BytesIO() as buffer:
                Image.fromarray(image).resize((320, 176)).save(buffer, format="JPEG", quality=85)
                frames[key] = base64.b64encode(buffer.getvalue()).decode("ascii")
        request = {key: state[span] for key, span in self._state_slices.items()}
        request["prompt"] = observation["prompt"]
        if not self._initialized:
            # Bind before model lookup and IK compilation; metadata needs no API key.
            self._client.policy(self._config)
            self._initialized = True
        self._client.set_observation(frames, request)
        response = self._client.get_action()
        parts = [np.asarray(response[key], dtype=np.float32) for key in ("q_l_chunk", "q_r_chunk")]
        for key, part, dim in zip(("q_l_chunk", "q_r_chunk"), parts, self._arm_dims, strict=True):
            if part.ndim != 2 or part.shape[1] != dim + 1:
                raise ValueError(f"Dream {key} must have shape [T, {dim + 1}]")
        actions = np.concatenate(parts, axis=1)
        if actions.shape[0] == 0 or not np.isfinite(actions).all():
            raise ValueError("Dream actions must contain nonempty finite rows")
        return {"actions": actions}

    @property
    def metadata(self) -> dict:
        """Advertise joint actions and the episode-reset capability on connection."""
        return {
            "server_name": "dreamlabs",
            "action_mode": "qpos",
            "action_dim": self._action_dim,
            "chunk_size": 48,
            "supports_reset": True,
        }

    def reset(self) -> None:
        """End the previous episode and clear its history when a model is open."""
        if self._initialized:
            self._client.reset_episode()

    def handle_connection(self, connection: websockets.sync.server.ServerConnection) -> None:
        """Send the handshake, then handle inference and reset messages for one client."""
        packer = msgpack_numpy.Packer()
        try:
            connection.send(packer.pack(self.metadata))
            for raw in connection:
                try:
                    request = msgpack_numpy.unpackb(raw)
                    if request.get("type") == "reset":
                        self.reset()
                        reply = {}
                    else:
                        reply = self.infer(request)
                    connection.send(packer.pack(reply))
                except (dream.DreamError, KeyError, TypeError, ValueError) as error:
                    connection.send(str(error))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.reset()

    def close(self) -> None:
        """End the API session and close its HTTP connection."""
        self._client.close()


def serve(policy: DreamPolicy, host: str, port: int) -> None:
    """Serve one EVA-CLIENT over WebSocket with OpenPI msgpack-numpy payloads."""
    with websockets.sync.server.serve(
        policy.handle_connection, host, port, compression=None, max_size=None
    ) as server:
        logger.info("Dream policy serving at ws://%s:%d", host, port)
        server.serve_forever()


def main() -> None:
    """Start the Dream-1 bridge for the selected robot; credentials stay on this server."""
    from dream_ik.specs import EMBODIMENTS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--checkpoint", default="dreamlabs/dream-1")
    parser.add_argument("--robot", choices=ARMS, default="yam")
    parser.add_argument("--head", choices=("joint", "eef"))
    args = parser.parse_args()
    head = args.head or ("joint" if args.robot == "yam" else "eef")
    if args.robot != "yam" and head == "joint":
        parser.error("Dream-1 on another robot requires --head eef")
    spec = EMBODIMENTS[args.robot]
    arm_dims = tuple(
        len(spec.side_arm(side).joints) - len(spec.side_arm(side).held_joints)
        for side in ("left", "right")
    )
    gripper_key = "gripper" if args.robot == "yam" else "grip"
    state_shapes = {
        key: {"shape": [width]}
        for side, dim in zip(("left", "right"), arm_dims, strict=True)
        for key, width in ((f"{side}_arm", dim), (f"{side}_{gripper_key}", 1))
    }
    policy = DreamPolicy(
        {
            "checkpoint": args.checkpoint,
            "ik": {"arm": args.robot, "head": head},
            "robot": "yam" if args.robot == "yam" else "other_embodiment",
            "embodiment": {
                "state": state_shapes,
                "video": {key.removeprefix("video."): {} for key in VIDEO_KEYS},
            },
            "action_hz": 20,
            "timeout_ms": 30000,
            "obs_window": 9,
            "obs_fps": 5,
        },
        arm_dims=arm_dims,
        gripper_key=gripper_key,
    )
    try:
        serve(policy, args.host, args.port)
    except KeyboardInterrupt:
        logger.info("Dream policy shutting down")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
