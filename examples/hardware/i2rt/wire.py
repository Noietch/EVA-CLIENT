"""Small, dependency-light copy of EVA's ZMQ wire codec for the I2RT venv.

The I2RT SDK pins NumPy 2 while EVA pins NumPy 1, so the hardware process runs
in an isolated environment. This module keeps its messages byte-compatible with
``src/transport/zmq.py`` without importing the rest of EVA in that environment.
"""

from __future__ import annotations

import dataclasses
import functools

import msgpack
import numpy as np

COLLECTION_START_TARGET = "collect_start"
COLLECTION_STOP_TARGET = "collect_stop"
HIL_START_TARGET = "hil_start"
HIL_STOP_TARGET = "hil_stop"


def _pack_array(obj: object) -> object:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict) -> object:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_PACKER = functools.partial(msgpack.Packer, default=_pack_array)()
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


@dataclasses.dataclass
class WireObservation:
    t: float
    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    action: np.ndarray | None = None
    eef: dict[str, np.ndarray] | None = None
    action_eef: np.ndarray | None = None
    hil_supported: bool = False
    hil_active: bool = False
    hil_error: str = ""
    operator_event: str = ""
    operator_event_id: int = 0


@dataclasses.dataclass
class WireAction:
    t: float
    action: np.ndarray
    target: str = "real"
    mode: str | None = None


def pack_observation(obs: WireObservation) -> bytes:
    payload: dict = {
        "t": float(obs.t),
        "images": obs.images,
        "state": {key: np.asarray(value, dtype=np.float32) for key, value in obs.state.items()},
        "hil_supported": bool(obs.hil_supported),
        "hil_active": bool(obs.hil_active),
    }
    if obs.action is not None:
        payload["action"] = np.asarray(obs.action, dtype=np.float32)
    if obs.eef is not None:
        payload["eef"] = {
            key: np.asarray(value, dtype=np.float32) for key, value in obs.eef.items()
        }
    if obs.action_eef is not None:
        payload["action_eef"] = np.asarray(obs.action_eef, dtype=np.float32)
    if obs.hil_error:
        payload["hil_error"] = str(obs.hil_error)
    if obs.operator_event:
        payload["operator_event"] = str(obs.operator_event)
        payload["operator_event_id"] = int(obs.operator_event_id)
    return _PACKER.pack(payload)


def unpack_action(payload: bytes) -> WireAction:
    raw = _unpackb(payload)
    return WireAction(
        t=float(raw["t"]),
        action=np.asarray(raw["action"], dtype=np.float32),
        target=str(raw.get("target", "real")),
        mode=None if raw.get("mode") is None else str(raw["mode"]),
    )
