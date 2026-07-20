"""Core type definitions — ControlMode enum and the unified Observation dataclass
used across all transport, recorder, and policy layers.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable
from typing import Any

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
    """

    images: dict[str, np.ndarray]
    state_qpos: np.ndarray
    state_eef: np.ndarray | None = None
    action_qpos: np.ndarray | None = None
    action_eef: np.ndarray | None = None
    timestamp: float | None = None


@dataclasses.dataclass
class CollectionRawSample:
    """One timestamped raw collection value before fixed-grid alignment."""

    timestamp: float
    value: Any


@dataclasses.dataclass
class CollectionRawImage:
    """A lazily decoded raw collection image sample."""

    decoder: Callable[[], np.ndarray | None]
    _decoded: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _decoded_once: bool = dataclasses.field(default=False, init=False, repr=False)

    def decode(self) -> np.ndarray:
        """Decode the raw image message once and return a contiguous HWC array."""
        if not self._decoded_once:
            image = self.decoder()
            if image is None:
                raise ValueError("collection raw image decode returned None")
            self._decoded = np.ascontiguousarray(image)
            self._decoded_once = True
        assert self._decoded is not None
        return self._decoded

    def release_decoded(self) -> None:
        """Release the decoded pixels while retaining the raw decoder."""
        self._decoded = None
        self._decoded_once = False


@dataclasses.dataclass
class CollectionRawBatch:
    """A batch of raw collection stream samples captured since the previous tick."""

    images: dict[str, list[CollectionRawSample]] = dataclasses.field(default_factory=dict)
    vectors: dict[str, list[CollectionRawSample]] = dataclasses.field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None


@dataclasses.dataclass
class RawCollectionSnapshot:
    """Pre-decode collection snapshot crossing the main->ingest thread boundary.

    Filled by the main loop in O(1): the source capture time plus raw source
    references (zmq raw bytes / pinned ROS msgs), WITHOUT decoding images,
    copying, or color-converting. The ingest thread stores timestamped raw stream
    samples; the save worker later aligns the whole episode to a fixed clock.

    Args:
        timestamp: source capture time used for alignment and dedup
            (zmq wire ``t`` / ROS ``header.stamp``). Stamped on the main thread
            so ingest backpressure can never shift it.
        decode_raw: Thread-safe closure returning timestamped raw stream samples.
    """

    timestamp: float
    decode_raw: Callable[[], CollectionRawBatch]


@dataclasses.dataclass
class RolloutInterventionSegment:
    """One in-memory teleop intervention segment inside a normal rollout."""

    segment_index: int
    start_policy_frame_index: int
    pre_intervention_qpos: np.ndarray
    start_time: float | None = None
    frames: list[Observation] = dataclasses.field(default_factory=list)
    resume_policy_frame_index: int | None = None
    invalid_reason: str = ""
