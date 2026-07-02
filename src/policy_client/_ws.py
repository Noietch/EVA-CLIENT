"""Shared WebSocket plumbing for the msgpack policy backends (OpenPI, starVLA).

``connect_ws`` performs the connect-handshake-retry dance; ``WebSocketPolicyClient``
holds the connection lifecycle (handshake metadata, msgpack packer, no-op reset).
Per-backend request/response shaping stays in each subclass's ``infer``.
"""

from __future__ import annotations

import logging
import time

import websockets.exceptions
import websockets.sync.client
from openpi_client import msgpack_numpy

from policy_client.base import PolicyClient, PolicyConnectionError

_CONNECT_RETRY_INTERVAL_SECONDS = 1.0
logger = logging.getLogger(__name__)


def connect_ws(
    host: str,
    port: int,
    retry_until_connected: bool,
    max_retries: int,
    server_label: str,
    retry_errors: tuple[type[BaseException], ...],
) -> tuple[websockets.sync.client.ClientConnection, dict]:
    """Open a WebSocket to a policy server and read its handshake metadata frame.

    Args:
        host, port: Policy server address.
        retry_until_connected: When True, keep retrying transient connect errors
            (those in ``retry_errors``) instead of failing on the first attempt.
        max_retries: Stop retrying after this many attempts (0 means unbounded).
        server_label: Human-readable server name used in error/log messages.
        retry_errors: Exception types treated as transient and retried.

    Returns:
        ``(connection, metadata)`` — the live connection and the decoded
        handshake descriptor sent by the server as its first frame.
    """
    logger.info("Waiting for %s at %s:%s", server_label, host, port)
    attempts = 0
    while True:
        try:
            connection = websockets.sync.client.connect(
                f"ws://{host}:{port}",
                compression=None,
                max_size=None,
            )
            metadata = msgpack_numpy.unpackb(connection.recv())
            return connection, metadata
        except retry_errors as error:
            if not retry_until_connected:
                raise PolicyConnectionError(
                    f"Failed to connect to {server_label} {host}:{port}"
                ) from error
            attempts += 1
            if max_retries > 0 and attempts >= max_retries:
                raise PolicyConnectionError(
                    f"Failed to connect to {server_label} {host}:{port} after {attempts} attempts"
                ) from error
            logger.info(
                "%s %s:%s is not ready; retrying in %.1fs (attempt %s%s)",
                server_label,
                host,
                port,
                _CONNECT_RETRY_INTERVAL_SECONDS,
                attempts,
                f"/{max_retries}" if max_retries > 0 else "",
            )
            time.sleep(_CONNECT_RETRY_INTERVAL_SECONDS)


class WebSocketPolicyClient(PolicyClient):
    """Base for msgpack-over-WebSocket policy clients.

    Establishes the connection (or adopts a pre-connected one), exposes the
    server handshake metadata, and provides the msgpack packer. Subclasses
    implement ``infer`` to shape the per-backend request/response payload.
    """

    def __init__(
        self,
        host: str,
        port: int,
        retry_until_connected: bool,
        max_retries: int,
        server_label: str,
        retry_errors: tuple[type[BaseException], ...],
        client: tuple[websockets.sync.client.ClientConnection, dict] | None = None,
    ) -> None:
        self._ws, self._server_metadata = client or connect_ws(
            host,
            port,
            retry_until_connected=retry_until_connected,
            max_retries=max_retries,
            server_label=server_label,
            retry_errors=retry_errors,
        )
        self._packer = msgpack_numpy.Packer()

    @property
    def metadata(self) -> dict:
        """Server capability descriptor received on the handshake frame at connect."""
        return self._server_metadata

    def reset(self) -> None:
        """No-op; the stateless client holds no per-episode state."""
        return
