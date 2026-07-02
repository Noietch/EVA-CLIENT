"""Isaac-GR00T backend — ZeroMQ policy client for an Isaac-GR00T policy server.

One backend, one file. Unlike the WebSocket backends, GR00T uses a ZeroMQ
REQ/REP socket with an endpoint envelope ``{"endpoint": "get_action", "data":
{...}}`` and msgpack-numpy framing. Observations and actions are *nested by
modality key* (``video.<cam>``, ``state.<key>``, ``annotation.<key>``), each
shaped ``(B=1, T=1, ...)``, and the action response comes back keyed the same
way.

Because those keys are embodiment-specific, the EVA→GR00T mapping is supplied
through ``policy.backend_options``::

    policy:
      type: gr00t
      host: 127.0.0.1
      port: 5555
      backend_options:
        video_keys:                       # EVA camera -> GR00T video modality key
          cam_high: video.exterior_image_1
          cam_left_wrist: video.wrist_image
        state_key: state.qpos             # GR00T state modality key for EVA's qpos
        language_key: annotation.human.task_description
        action_keys: [action.qpos]        # response keys to concatenate into the chunk
        timeout_ms: 15000

``zmq`` is imported lazily so the dependency is only required when this backend
is actually selected.
"""

from __future__ import annotations

import logging

import numpy as np

from core.config import ConfigDict
from core.registry import POLICY_REGISTRY
from policy_client.base import (
    PolicyBuildContext,
    PolicyClient,
    PolicyConnectionError,
    PolicyRequestError,
)

logger = logging.getLogger(__name__)


@POLICY_REGISTRY.register_client("gr00t")
class Gr00tPolicyClient(PolicyClient):
    """ZeroMQ client for an Isaac-GR00T policy server.

    Maps EVA's flat observation onto GR00T's nested modality dict, calls the
    ``get_action`` endpoint, and concatenates the configured action keys into the
    (T, D) chunk EVA executes.
    """

    def __init__(
        self,
        host: str,
        port: int,
        api_token: str | None = None,
        video_keys: dict[str, str] | None = None,
        state_key: str = "state.qpos",
        language_key: str = "annotation.human.task_description",
        action_keys: list[str] | None = None,
        timeout_ms: int = 15000,
    ) -> None:
        import msgpack_numpy as mnp
        import zmq

        self._zmq = zmq
        self._mnp = mnp
        self._host = host
        self._port = port
        self._api_token = api_token
        self._video_keys = video_keys or {}
        self._state_key = state_key
        self._language_key = language_key
        self._action_keys = action_keys or ["action.qpos"]
        self._timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._init_socket()
        self._server_metadata = self._fetch_metadata()

    @classmethod
    def from_config(cls, config: ConfigDict, ctx: PolicyBuildContext) -> Gr00tPolicyClient:
        """Build a GR00T client; all GR00T-specific keys come from backend_options."""
        _ = ctx
        opts = config.backend_options
        return cls(
            config.host,
            config.port,
            api_token=opts.get("api_token"),
            video_keys=opts.get("video_keys"),
            state_key=opts.get("state_key", "state.qpos"),
            language_key=opts.get("language_key", "annotation.human.task_description"),
            action_keys=opts.get("action_keys"),
            timeout_ms=int(opts.get("timeout_ms", 15000)),
        )

    def _init_socket(self) -> None:
        self._socket = self._context.socket(self._zmq.REQ)
        self._socket.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self._api_token:
            request["api_token"] = self._api_token
        try:
            self._socket.send(self._mnp.packb(request))
            message = self._socket.recv()
        except self._zmq.error.Again as error:
            self._init_socket()
            raise PolicyRequestError(
                f"GR00T request timed out at {self._host}:{self._port}"
            ) from error
        response = self._mnp.unpackb(message)
        if isinstance(response, dict) and "error" in response:
            raise PolicyRequestError(f"GR00T server error: {response['error']}")
        return response

    def _fetch_metadata(self) -> dict:
        try:
            cfg = self._call("get_modality_config", requires_input=False)
        except Exception as error:
            raise PolicyConnectionError(
                f"Failed to reach GR00T server {self._host}:{self._port}"
            ) from error
        return {"server_name": "isaac-gr00t", "action_mode": "qpos", "modality_config": cfg}

    @property
    def metadata(self) -> dict:
        """Server capability descriptor fetched at connect.

        Keys: ``server_name``, ``action_mode``, ``modality_config``.
        """
        return self._server_metadata

    def _build_observation(self, observation: dict) -> dict:
        # GR00T expects nested modality keys, each (B=1, T=1, ...).
        state = np.asarray(observation["state"], dtype=np.float32)[None, None, ...]
        request: dict = {self._state_key: state, self._language_key: [[observation["prompt"]]]}
        for cam_key, image in observation["images"].items():
            gr00t_key = self._video_keys.get(cam_key)
            if gr00t_key is not None:
                request[gr00t_key] = np.asarray(image)[None, None, ...]
        return request

    def infer(
        self,
        observation: dict,
        prev_action: np.ndarray | None = None,
        rtc_params: dict | None = None,
    ) -> dict:
        """Map the observation to GR00T modality keys, call ``get_action``, concat the result.

        prev_action and rtc_params are ignored (GR00T is stateless per call).
        Returns ``{"actions": ndarray[T, action_dim] float32}`` formed by
        concatenating the configured ``action_keys`` along the last axis.
        """
        _ = prev_action, rtc_params
        request_data = self._build_observation(observation)
        response = self._call("get_action", {"observation": request_data, "options": None})
        # Server returns (action_dict, info); action arrays are (T, D_part) per key.
        action_dict = response[0] if isinstance(response, (list, tuple)) else response
        parts = [np.asarray(action_dict[k], dtype=np.float32) for k in self._action_keys]
        chunk = np.concatenate(parts, axis=-1)
        return {"actions": chunk}

    def reset(self) -> None:
        """Call the server's ``reset`` endpoint to clear its per-episode state."""
        self._call("reset", {"options": None})


