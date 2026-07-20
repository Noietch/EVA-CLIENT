"""Msgpack-over-WebSocket scalar critic client."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import websockets.exceptions
import websockets.sync.client
from openpi_client import msgpack_numpy

from core.config import ConfigDict
from core.registry import CRITIC_REGISTRY
from critic_client.base import (
    CriticBuildContext,
    CriticClient,
    CriticConnectionError,
    CriticRequestError,
)

_CONNECT_RETRY_INTERVAL_SECONDS = 1.0
_RETRY_ERRORS = (ConnectionRefusedError, OSError, websockets.exceptions.InvalidMessage)


@CRITIC_REGISTRY.register_client("websocket")
class WebSocketCriticClient(CriticClient):
    """Send observation + actions and require a scalar ``value`` response."""

    def __init__(
        self,
        host: str,
        port: int,
        retry_until_connected: bool = False,
        max_retries: int = 0,
        client: tuple[Any, dict] | None = None,
    ) -> None:
        if client is None:
            self._ws, self._metadata = self._connect(
                host, port, retry_until_connected, max_retries
            )
        else:
            self._ws, self._metadata = client
        self._packer = msgpack_numpy.Packer()

    @classmethod
    def from_config(
        cls, config: ConfigDict, ctx: CriticBuildContext
    ) -> WebSocketCriticClient:
        return cls(
            str(config.host),
            int(config.port),
            retry_until_connected=ctx.retry_until_connected,
            max_retries=ctx.max_retries,
        )

    @staticmethod
    def _connect(
        host: str,
        port: int,
        retry_until_connected: bool,
        max_retries: int,
    ) -> tuple[Any, dict]:
        attempts = 0
        while True:
            try:
                connection = websockets.sync.client.connect(
                    f"ws://{host}:{port}", compression=None, max_size=None
                )
                metadata = msgpack_numpy.unpackb(connection.recv())
                return connection, metadata
            except _RETRY_ERRORS as error:
                if not retry_until_connected:
                    raise CriticConnectionError(
                        f"Failed to connect to critic {host}:{port}"
                    ) from error
                attempts += 1
                if max_retries > 0 and attempts >= max_retries:
                    raise CriticConnectionError(
                        f"Failed to connect to critic {host}:{port} after {attempts} attempts"
                    ) from error
                time.sleep(_CONNECT_RETRY_INTERVAL_SECONDS)

    def evaluate(self, observation: dict, actions: np.ndarray) -> float:
        try:
            self._ws.send(
                self._packer.pack(
                    {
                        "observation": observation,
                        "actions": np.asarray(actions, dtype=np.float32),
                    }
                )
            )
            response = msgpack_numpy.unpackb(self._ws.recv())
            value = np.asarray(response["value"])
            if value.ndim != 0:
                raise ValueError(f"critic value must be scalar, got shape={value.shape}")
            return float(value)
        except Exception as error:
            raise CriticRequestError("Critic request failed") from error

    @property
    def metadata(self) -> dict:
        return self._metadata

    def close(self) -> None:
        self._ws.close()
