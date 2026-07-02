"""OpenPI backend — WebSocket policy clients for an OpenPI policy server.

One backend, one file. This module contains:
  * OpenPiPolicyClient      stateless WebSocket client (msgpack observations)
  * RtcOpenPiPolicyClient   RTC variant with prev_action feedback

Each client registers itself under its own policy ``type`` via
``@POLICY_REGISTRY.register_client``; ``build(type, config, ctx)`` invokes the
class's ``from_config`` classmethod.
"""

from __future__ import annotations

import copy
import logging
import threading

import numpy as np
import websockets.exceptions
import websockets.sync.client
from openpi_client import msgpack_numpy

from core.config import ConfigDict
from core.registry import POLICY_REGISTRY
from policy_client._ws import WebSocketPolicyClient
from policy_client.base import (
    PolicyBuildContext,
    PolicyRequestError,
)

logger = logging.getLogger(__name__)
_RETRY_ERRORS = (ConnectionRefusedError, OSError, websockets.exceptions.InvalidMessage)


@POLICY_REGISTRY.register_client("openpi")
class OpenPiPolicyClient(WebSocketPolicyClient):
    """Stateless OpenPI policy client over WebSocket.

    Connects to an OpenPI-compatible policy server, sends observations serialized
    via msgpack (including images, joint state, and prompt), and receives action
    chunks. Supports retry-until-connected semantics for robust startup in
    containerized deployments.
    """

    def __init__(
        self,
        host: str,
        port: int,
        retry_until_connected: bool = True,
        client: tuple[websockets.sync.client.ClientConnection, dict] | None = None,
        max_retries: int = 0,
    ) -> None:
        super().__init__(
            host,
            port,
            retry_until_connected=retry_until_connected,
            max_retries=max_retries,
            server_label="policy server",
            retry_errors=_RETRY_ERRORS,
            client=client,
        )

    @classmethod
    def from_config(cls, config: ConfigDict, ctx: PolicyBuildContext) -> OpenPiPolicyClient:
        """Build a stateless OpenPI client from the policy config."""
        return cls(
            config.host,
            config.port,
            retry_until_connected=ctx.retry_until_connected,
            max_retries=ctx.max_retries,
        )

    def infer(
        self,
        observation: dict,
        prev_action: np.ndarray | None = None,
        rtc_params: dict | None = None,
    ) -> dict:
        """Send the observation over WebSocket and return the server's action chunk.

        prev_action (``[T, action_dim] float32``) and rtc_params, when given, are
        attached to the request for server-side conditioning. Returns the decoded
        response dict, typically ``{"actions": ndarray[T, action_dim] float32}``.
        """
        try:
            data_to_send = dict(observation)
            if prev_action is not None:
                data_to_send["prev_action"] = prev_action
            if rtc_params is not None:
                data_to_send["rtc_params"] = rtc_params
            self._ws.send(self._packer.pack(data_to_send))
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(response)
            return msgpack_numpy.unpackb(response)
        except Exception as error:
            raise PolicyRequestError("Policy request failed") from error


@POLICY_REGISTRY.register_client("openpi_rtc")
class RtcOpenPiPolicyClient(OpenPiPolicyClient):
    """RTC-aware OpenPI policy client with automatic prev_action feedback.

    Each infer() call sends the previous response's raw actions and
    origin_actions back to the server along with the latency shift parameter.
    The server handles all normalization and conditioning.

    Thread-safe: uses separate locks for response state (_lock) and network I/O
    (_request_lock) so the background inference thread and main loop do not block
    each other. A generation counter invalidates stale responses after reset().
    """

    def __init__(
        self,
        host: str,
        port: int,
        latency_k: int,
        retry_until_connected: bool = True,
        client: tuple[websockets.sync.client.ClientConnection, dict] | None = None,
    ) -> None:
        super().__init__(
            host, port, retry_until_connected=retry_until_connected, client=client
        )
        self._latency_k = latency_k
        self._active_response: dict | None = None
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._generation = 0

    @classmethod
    def from_config(cls, config: ConfigDict, ctx: PolicyBuildContext) -> RtcOpenPiPolicyClient:
        """Build an RTC client; latency shift ``s`` comes from backend_options.latency_k."""
        return cls(
            config.host,
            config.port,
            latency_k=int(config.backend_options.get("latency_k", 1)),
            retry_until_connected=ctx.retry_until_connected,
        )

    def reset(self) -> None:
        """Drop the cached response and bump the generation counter so stale chunks are ignored."""
        with self._lock:
            self._generation += 1
            self._active_response = None

    def infer(
        self,
        observation: dict,
        prev_action: np.ndarray | None = None,
        rtc_params: dict | None = None,
    ) -> dict:
        """Run RTC inference, feeding back the previous chunk shifted by ``latency_k``.

        The cached previous response's actions are shifted forward by
        ``latency_k`` steps (tail held at the last pose) and sent as prev_action
        with ``rtc_params={"s": latency_k}``; the caller's prev_action and
        rtc_params arguments are ignored. The fresh response is cached for the
        next call. Returns ``{"actions": ndarray[T, action_dim] float32}``.
        """
        _ = prev_action, rtc_params
        with self._lock:
            active_response = copy.deepcopy(self._active_response)
            latency_k = self._latency_k

        prev_action_arg: np.ndarray | None = None
        rtc_params_arg: dict | None = None
        if active_response is not None:
            actions = np.asarray(active_response["actions"], dtype=np.float32)
            if actions.ndim >= 2 and 0 < latency_k < actions.shape[0]:
                shifted = np.empty_like(actions)
                shifted[: actions.shape[0] - latency_k] = actions[latency_k:]
                shifted[actions.shape[0] - latency_k :] = actions[-1]
                actions = shifted
            prev_action_arg = actions
            rtc_params_arg = {"s": latency_k}

        with self._request_lock:
            response = super().infer(
                observation, prev_action=prev_action_arg, rtc_params=rtc_params_arg
            )

        with self._lock:
            self._active_response = copy.deepcopy(response)
        return response
