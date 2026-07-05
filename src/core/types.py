"""Core type definitions — ControlMode enum and the unified Observation dataclass
used across all transport, recorder, and policy layers.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable

import numpy as np


class ControlMode(str, enum.Enum):
    """Robot control mode — JOINT sends raw joint angles, EEF sends Cartesian
    end-effector poses that require an IK solver to convert to joint space."""

    JOINT = "joint"
    EEF = "eef"


@dataclasses.dataclass
class Observation:
    """Unified robot frame across transport, recorder, and policy layers.

    state_qpos is the raw joint-space state the transport produces. state_eef is
    the end-effector state (8D per arm: xyz + quat wxyz + gripper), either read
    from an external source or derived from state_qpos via forward kinematics;
    None when not available. action_qpos / action_eef carry the teleop (collection)
    or executed (eval/exec) action in joint / EEF space. timestamp is the source
    capture time (collection/recording); None for pure inference frames.
    eef_uv is per-camera projected end-effector pixel coordinates keyed by camera
    observation_key; each value is a (n_arms, 2) float32 array with NaN entries
    for arms that fall behind the camera. Only populated when the loop was told to
    project (calibration + kinematics available); None otherwise.
    """

    images: dict[str, np.ndarray]
    state_qpos: np.ndarray
    state_eef: np.ndarray | None = None
    action_qpos: np.ndarray | None = None
    action_eef: np.ndarray | None = None
    timestamp: float | None = None
    eef_uv: dict[str, np.ndarray] | None = None


@dataclasses.dataclass
class RawCollectionSnapshot:
    """Pre-decode collection snapshot crossing the main->ingest thread boundary.

    Filled by the main loop in O(1): the source capture time plus raw source
    references (zmq raw bytes / pinned ROS msgs), WITHOUT decoding images,
    copying, or color-converting. The ingest thread later calls decode() to
    produce an Observation, keeping all heavy work off the capture loop.

    Args:
        timestamp: source capture time used for alignment and dedup
            (zmq wire ``t`` / ROS ``header.stamp``). Stamped on the main thread
            so ingest backpressure can never shift it.
        decode: thread-safe closure the ingest thread runs to turn the pinned
            raw references into a fully decoded Observation.
    """

    timestamp: float
    decode: Callable[[], Observation]
