"""starVLA backend — WebSocket policy client for a starVLA model server.

One backend, one file. The starVLA server speaks msgpack over WebSocket with a
typed envelope ``{"type": "infer", "payload": {...}}`` and returns
``{"status": "ok", "data": {"actions": ndarray[B, T, D]}}``. Its handshake frame
carries ``action_chunk_size`` and the un-normalization keys.

The EVA observation (``{"state", "images": {cam: HWC}, "prompt"}``) is adapted
into the starVLA ``examples=[{"image", "lang", "state"}]`` request shape. Tune
the camera pick and un-normalization key through ``policy.backend_options``::

    policy:
      type: starvla
      host: 127.0.0.1
      port: 10093
      backend_options:
        camera_key: cam_high      # which EVA camera feeds starVLA's single image
        unnorm_key: agilex        # dataset key for server-side un-normalization
"""

from __future__ import annotations

import logging

import numpy as np
from openpi_client import msgpack_numpy

from core.config import ConfigDict
from core.registry import POLICY_REGISTRY
from policy_client._ws import WebSocketPolicyClient
from policy_client.base import (
    PolicyBuildContext,
    PolicyRequestError,
)

logger = logging.getLogger(__name__)
_RETRY_ERRORS = (ConnectionRefusedError, OSError)


@POLICY_REGISTRY.register_client("starvla")
class StarVlaPolicyClient(WebSocketPolicyClient):
    """WebSocket client for a starVLA policy server.

    Sends one ``examples`` entry per inference (single image + language + state)
    and returns the un-normalized action chunk the server produced, with the
    leading batch axis dropped to match the (T, D) chunk shape EVA expects.
    """

    def __init__(
        self,
        host: str,
        port: int,
        camera_key: str | None = None,
        unnorm_key: str | None = None,
        retry_until_connected: bool = True,
        max_retries: int = 0,
    ) -> None:
        self._camera_key = camera_key
        self._unnorm_key = unnorm_key
        super().__init__(
            host,
            port,
            retry_until_connected=retry_until_connected,
            max_retries=max_retries,
            server_label="starVLA server",
            retry_errors=_RETRY_ERRORS,
        )

    @classmethod
    def from_config(cls, config: ConfigDict, ctx: PolicyBuildContext) -> StarVlaPolicyClient:
        """Build a starVLA client; camera/unnorm keys come from backend_options."""
        opts = config.backend_options
        return cls(
            config.host,
            config.port,
            camera_key=opts.get("camera_key"),
            unnorm_key=opts.get("unnorm_key"),
            retry_until_connected=ctx.retry_until_connected,
            max_retries=ctx.max_retries,
        )

    def _pick_image(self, images: dict[str, np.ndarray]) -> np.ndarray:
        # starVLA consumes a single image; use the configured camera or the first one.
        if self._camera_key is not None:
            return images[self._camera_key]
        return next(iter(images.values()))

    def infer(
        self,
        observation: dict,
        prev_action: np.ndarray | None = None,
        rtc_params: dict | None = None,
    ) -> dict:
        """Send one ``examples`` entry and return the server's un-normalized action chunk.

        prev_action and rtc_params are ignored (starVLA is stateless per call).
        The server returns actions shaped ``[B, T, D]``; the leading batch axis is
        dropped. Returns ``{"actions": ndarray[T, action_dim] float32}``.
        """
        _ = prev_action, rtc_params
        example = {
            "image": self._pick_image(observation["images"]),
            "lang": observation["prompt"],
            "state": np.asarray(observation["state"], dtype=np.float32),
        }
        payload: dict = {"examples": [example]}
        if self._unnorm_key is not None:
            payload["unnorm_key"] = self._unnorm_key
        request = {"type": "infer", "request_id": "eva", "payload": payload}
        try:
            self._ws.send(self._packer.pack(request))
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(response)
            decoded = msgpack_numpy.unpackb(response)
        except Exception as error:
            raise PolicyRequestError("starVLA request failed") from error
        if decoded.get("status") != "ok":
            raise PolicyRequestError(f"starVLA server error: {decoded.get('error')}")
        actions = np.asarray(decoded["data"]["actions"], dtype=np.float32)
        # Server returns (B, T, D); EVA's strategy layer expects (T, D).
        if actions.ndim == 3:
            actions = actions[0]
        return {"actions": actions}
