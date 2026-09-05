"""Policy-agnostic client for the XPolicyLab websocket protocol."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import msgpack
import msgpack_numpy
import numpy as np
import websockets.exceptions
import websockets.sync.client

from core.config import ConfigDict
from core.registry import POLICY_REGISTRY
from policy_client.base import (
    PolicyBuildContext,
    PolicyClient,
    PolicyConnectionError,
    PolicyRequestError,
)

logger = logging.getLogger(__name__)
_CONNECT_ERRORS = (
    OSError,
    TimeoutError,
    websockets.exceptions.WebSocketException,
    PolicyRequestError,
)


class _Connection(Protocol):
    def send(self, message: bytes) -> None: ...

    def recv(self, timeout: float | None = None) -> bytes | str: ...

    def close(self) -> None: ...


def _load_robot_codec(xpolicylab_root: str | None):
    if xpolicylab_root is not None:
        root = Path(xpolicylab_root).expanduser().resolve()
        if not (root / "XPolicyLab.py").is_file():
            raise ValueError(f"invalid XPolicyLab repository root: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from XPolicyLab.utils.process_data import (
            get_robot_action_dim_info,
            pack_robot_state,
            unpack_robot_state,
        )
    except ModuleNotFoundError as error:
        if error.name != "XPolicyLab" and not str(error.name).startswith("XPolicyLab."):
            raise
        raise PolicyConnectionError(
            "XPolicyLab must be installed or provided through xpolicylab_root"
        ) from error
    return get_robot_action_dim_info, pack_robot_state, unpack_robot_state


@POLICY_REGISTRY.register_client("xpolicylab")
class XPolicyLabPolicyClient(PolicyClient):
    """Synchronous client for XPolicyLab's standard websocket protocol."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        env_cfg_type: str,
        action_type: str,
        xpolicylab_root: str | None = None,
        evaluation_id: str | None = None,
        request_timeout_s: float = 120.0,
        connect_timeout_s: float = 30.0,
        handshake_timeout_s: float = 60.0,
        retry_until_connected: bool = True,
        max_retries: int = 0,
        client: _Connection | None = None,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._env_cfg_type = env_cfg_type
        self._action_type = action_type
        get_dim_info, self._pack_robot_state, self._unpack_robot_state = _load_robot_codec(
            xpolicylab_root
        )
        self._robot_action_dim_info = get_dim_info(env_cfg_type)
        self._action_dim = sum(self._robot_action_dim_info["arm_dim"]) + sum(
            self._robot_action_dim_info["ee_dim"]
        )
        self._evaluation_id = evaluation_id or f"eva-{uuid4()}"
        self._trial_id = f"trial-{uuid4()}"
        self._request_timeout_s = float(request_timeout_s)
        self._connect_timeout_s = float(connect_timeout_s)
        self._handshake_timeout_s = float(handshake_timeout_s)
        self._request_lock = threading.Lock()
        self._step = 0

        if min(self._request_timeout_s, self._connect_timeout_s, self._handshake_timeout_s) <= 0:
            raise ValueError("XPolicyLab timeouts must be positive")

        if client is None:
            self._ws, handshake = self._connect(retry_until_connected, max_retries)
        else:
            self._ws = client
            handshake = self._request("hello", {}, "hello_ack", self._handshake_timeout_s)

        self._server_metadata = dict(handshake)
        self._server_metadata.setdefault(
            "server_name", self._server_metadata.get("server", "xpolicylab")
        )
        self._server_metadata.update(
            action_mode=self._action_type,
            action_dim=self._action_dim,
        )

    @classmethod
    def from_config(cls, config: ConfigDict, ctx: PolicyBuildContext) -> XPolicyLabPolicyClient:
        opts = config.backend_options
        return cls(
            config.host,
            config.port,
            env_cfg_type=str(opts["env_cfg_type"]),
            action_type=str(opts["action_type"]),
            xpolicylab_root=opts.get("xpolicylab_root"),
            evaluation_id=opts.get("evaluation_id"),
            request_timeout_s=float(opts.get("request_timeout_s", 120.0)),
            connect_timeout_s=float(opts.get("connect_timeout_s", 30.0)),
            handshake_timeout_s=float(opts.get("handshake_timeout_s", 60.0)),
            retry_until_connected=ctx.retry_until_connected,
            max_retries=ctx.max_retries,
        )

    @property
    def metadata(self) -> dict:
        return self._server_metadata

    @staticmethod
    def _decode_frame(data: bytes) -> dict[str, Any]:
        decoded = msgpack.unpackb(
            data,
            raw=False,
            strict_map_key=False,
            object_hook=msgpack_numpy.decode,
        )
        if not isinstance(decoded, dict):
            raise PolicyRequestError("XPolicyLab response must be a msgpack map")
        return decoded

    def _connect(
        self, retry_until_connected: bool, max_retries: int
    ) -> tuple[_Connection, dict[str, Any]]:
        attempts = 0
        while True:
            attempts += 1
            connection: _Connection | None = None
            try:
                connection = websockets.sync.client.connect(
                    f"ws://{self._host}:{self._port}",
                    compression=None,
                    max_size=None,
                    open_timeout=self._connect_timeout_s,
                )
                self._ws = connection
                handshake = self._request("hello", {}, "hello_ack", self._handshake_timeout_s)
                return connection, handshake
            except _CONNECT_ERRORS as error:
                if connection is not None:
                    connection.close()
                if not retry_until_connected or (max_retries and attempts >= max_retries):
                    raise PolicyConnectionError(
                        f"Failed to connect to XPolicyLab server {self._host}:{self._port}"
                    ) from error
                logger.info(
                    "XPolicyLab server %s:%s is not ready; retrying (attempt %s%s)",
                    self._host,
                    self._port,
                    attempts,
                    f"/{max_retries}" if max_retries else "",
                )
                time.sleep(1.0)

    def _request(
        self,
        message_type: str,
        payload: dict[str, Any],
        expected_type: str,
        timeout_s: float | None = None,
        *,
        trial_id: str | None = None,
        step: int = 0,
    ) -> dict[str, Any]:
        request_id = str(uuid4())
        frame: dict[str, Any] = {
            "message_type": message_type,
            "message_id": request_id,
            "evaluation_id": self._evaluation_id,
            "step": step,
            "payload": payload,
        }
        if trial_id is not None:
            frame["trial_id"] = trial_id

        try:
            with self._request_lock:
                self._ws.send(msgpack.packb(frame, default=msgpack_numpy.encode, use_bin_type=True))
                raw = self._ws.recv(timeout=timeout_s or self._request_timeout_s)
            if isinstance(raw, str):
                raise PolicyRequestError(raw)
            response = self._decode_frame(raw)
        except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as error:
            raise PolicyRequestError(f"XPolicyLab {message_type} request failed") from error

        response_payload = response.get("payload")
        if response.get("message_id") != request_id:
            raise PolicyRequestError("XPolicyLab response message_id does not match the request")
        if not isinstance(response_payload, dict):
            raise PolicyRequestError("XPolicyLab response payload must be a map")
        if response.get("message_type") == "error":
            raise PolicyRequestError(
                "XPolicyLab server error "
                f"[{response_payload.get('code', 'internal')}]: "
                f"{response_payload.get('message', 'unknown error')}"
            )
        if response.get("message_type") != expected_type:
            raise PolicyRequestError(
                f"Expected XPolicyLab {expected_type}, got {response.get('message_type')!r}"
            )
        return response_payload

    def _build_observation(self, observation: dict) -> dict[str, Any]:
        return {
            "instruction": str(observation["prompt"]),
            "vision": {
                name: {"color": np.asarray(image)} for name, image in observation["images"].items()
            },
            "state": self._unpack_robot_state(
                np.asarray(observation["state"], dtype=np.float32),
                self._action_type,
                self._robot_action_dim_info,
                source_type="obs",
            ),
        }

    def _flatten_action(self, action: Mapping[str, Any]) -> np.ndarray:
        try:
            return self._pack_robot_state(
                {"state": action},
                self._action_type,
                self._robot_action_dim_info,
                source_type="obs",
            )
        except (KeyError, ValueError) as error:
            raise PolicyRequestError(f"invalid XPolicyLab action: {error}") from error

    def _flatten_actions(self, actions: object) -> np.ndarray:
        if isinstance(actions, Mapping):
            chunk = self._flatten_action(actions)[None]
        elif isinstance(actions, Sequence) and not isinstance(actions, (str, bytes, np.ndarray)):
            chunk = (
                np.stack([self._flatten_action(action) for action in actions])
                if actions and isinstance(actions[0], Mapping)
                else np.asarray(actions, dtype=np.float32)
            )
        else:
            chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim == 1:
            chunk = chunk[None]
        if chunk.ndim != 2 or chunk.shape[1] != self._action_dim:
            raise PolicyRequestError(
                f"XPolicyLab actions must have shape [T, {self._action_dim}], got {chunk.shape}"
            )
        chunk = np.asarray(chunk, dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise PolicyRequestError("XPolicyLab returned NaN or Inf actions")
        return chunk

    def infer(
        self,
        observation: dict,
        prev_action: np.ndarray | None = None,
        rtc_params: dict | None = None,
    ) -> dict:
        """Send one observation and return a finite ``[T, D]`` action chunk."""
        response = self._request(
            "infer",
            {"observation": self._build_observation(observation)},
            "infer_result",
            trial_id=self._trial_id,
            step=self._step,
        )
        if "actions" not in response:
            raise PolicyRequestError("XPolicyLab infer response does not contain actions")
        actions = self._flatten_actions(response["actions"])
        self._step += 1
        self._server_metadata["chunk_size"] = actions.shape[0]
        result: dict[str, Any] = {"actions": actions, "action_mode": self._action_type}
        if "latency_ms" in response:
            result["latency_ms"] = response["latency_ms"]
        return result

    def reset(self) -> None:
        """Reset the remote model and restart the inference-step counter."""
        trial_id = f"trial-{uuid4()}"
        self._request(
            "reset",
            {"trial_id": trial_id},
            "reset_result",
            trial_id=trial_id,
        )
        self._trial_id = trial_id
        self._step = 0

    def close(self) -> None:
        self._ws.close()

    def __enter__(self) -> XPolicyLabPolicyClient:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
