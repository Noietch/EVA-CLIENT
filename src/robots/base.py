"""Robot description layer — the Robot base class plus the small structs it holds
(actuator groups, camera/observation schema).

A Robot is a single self-contained class: declarative physical data (joints,
cameras, state composition, visualization) plus a ``build_kinematics`` hook that
constructs its FK/IK solver. Transport-specific details like ROS topics live in the
transport backend, so the same robot drives any backend. Each concrete robot
subclasses Robot and registers under ROBOT_REGISTRY. The 3D-vis structs
(``VisPart`` / ``RobotVisConfig``) live in ``robots.utils``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from robots.utils import (  # noqa: F401  (re-exported as the robot API surface)
    RobotVisConfig,
    VisPart,
)


@dataclasses.dataclass(frozen=True)
class ActuatorGroup:
    """One logical group of actuators (e.g. "left_arm").

    Fixed DOF count and ordered joint names matching the URDF. The action vector is
    split into groups in declaration order — dual-arm Piper: [left_arm(7), right_arm(7)].
    """

    name: str
    dof: int
    joint_names: tuple[str, ...]
    gripper_index: int | None = None


@dataclasses.dataclass(frozen=True)
class CameraSpec:
    """Maps a logical camera name (e.g. "front") to its policy-input observation_key.

    The transport backend resolves the logical name to a concrete source (ROS topic,
    dataset column, ...).
    """

    name: str
    observation_key: str


@dataclasses.dataclass(frozen=True)
class ObservationSchema:
    """The observation structure expected by the policy.

    cameras: which images to capture. state_composition: actuator-group names whose
    qpos are concatenated, in order, into the state vector.
    """

    cameras: tuple[CameraSpec, ...]
    state_composition: tuple[str, ...]


class Robot:
    """A single self-contained robot: declarative structure + its FK/IK solver.

    Holds the robot's actuator groups, initial joint configuration, observation
    schema, and optional 3D-vis config, and exposes ``build_kinematics`` to construct
    its FK/IK solver (None when the robot has none). All other subsystems interact
    with the robot through this class, making EVA Client robot-agnostic. Concrete
    robots subclass Robot and register under ROBOT_REGISTRY.
    """

    def __init__(
        self,
        name: str,
        actuator_groups: tuple[ActuatorGroup, ...],
        initial_qpos: np.ndarray,
        observation_schema: ObservationSchema,
        vis_config: RobotVisConfig | None = None,
        supported_reference_frames: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.actuator_groups = tuple(actuator_groups)
        self.initial_qpos = np.asarray(initial_qpos, dtype=np.float32)
        self.observation_schema = observation_schema
        self.vis_config = vis_config
        self.supported_reference_frames = tuple(supported_reference_frames)

    def build_kinematics(self, **kwargs: Any) -> Any:
        """Construct this robot's FK/IK solver, or None if it has no kinematics.

        Args:
            **kwargs: runtime solver parameters — ``initial_qpos_groups`` (always),
                ``dt`` (control period), and optionally ``reference_frame`` (only when
                the robot declares ``supported_reference_frames``).

        Returns:
            A solver exposing ``solve_chunk``/``fk_chunk``/``reset``/``default_seed``/
            ``close``, or None when the robot has no kinematics.
        """
        return None

    @property
    def total_action_dim(self) -> int:
        """Total action dimensionality, summed over all actuator groups."""
        return sum(g.dof for g in self.actuator_groups)

    def initial_qpos_by_group(self) -> list[np.ndarray]:
        """Split initial_qpos into per-group slices in actuator-group order."""
        groups: list[np.ndarray] = []
        offset = 0
        for group in self.actuator_groups:
            groups.append(np.asarray(self.initial_qpos[offset : offset + group.dof]))
            offset += group.dof
        return groups

    @property
    def arm_groups(self) -> tuple[ActuatorGroup, ...]:
        """Actuator groups that carry an end-effector (those declaring a gripper).

        EEF vectors (xyz + quat wxyz + gripper, 8D per arm) are produced only for
        these groups; non-arm groups like a head or torso have no end-effector.
        """
        return tuple(g for g in self.actuator_groups if g.gripper_index is not None)

    @property
    def gripper_indices(self) -> tuple[int, ...]:
        """Absolute action-vector index of each group that declares a gripper, in order."""
        indices: list[int] = []
        offset = 0
        for group in self.actuator_groups:
            if group.gripper_index is not None:
                indices.append(offset + group.gripper_index)
            offset += group.dof
        return tuple(indices)

    @property
    def gripper_mask(self) -> tuple[bool, ...]:
        """Boolean mask over the flat action vector, True at gripper indices."""
        mask = [False] * self.total_action_dim
        for idx in self.gripper_indices:
            mask[idx] = True
        return tuple(mask)

    def build_observation(
        self, images: dict[str, np.ndarray], state: np.ndarray, prompt: str
    ) -> dict:
        """Assemble the observation dict ({state, images, prompt}) sent to the policy."""
        return {
            "state": np.asarray(state, dtype=np.float32),
            "images": images,
            "prompt": prompt,
        }

    def set_gripper_value(self, action: np.ndarray, group_name: str, value: float) -> None:
        """Set the gripper command for one named group, in-place (no-op if it has none)."""
        offset = 0
        for group in self.actuator_groups:
            if group.name == group_name and group.gripper_index is not None:
                index = offset + group.gripper_index
                if index < len(action):
                    action[index] = value
                return
            offset += group.dof

    def split_action(self, action: np.ndarray) -> tuple[np.ndarray, ...]:
        """Split a flat joint action into per-group [group.dof] segments, in order."""
        vector = np.asarray(action, dtype=np.float32)
        parts: list[np.ndarray] = []
        offset = 0
        for group in self.actuator_groups:
            parts.append(vector[offset : offset + group.dof].copy())
            offset += group.dof
        return tuple(parts)

    def snap_grippers(
        self,
        action: np.ndarray,
        threshold: float | None,
        open_value: float = 1.0,
        close_value: float = 0.0,
    ) -> np.ndarray:
        """Binarize this robot's gripper dimensions of an action, in-place.

        No-op (returns ``action`` unchanged) when ``threshold`` is None or the robot
        has no gripper dimensions. Handles both a single action vector [D] and a
        chunk [T, D] by snapping each row.

        Args:
            action: action vector [D] or chunk [T, D]; gripper dims set to
                open_value/close_value.
            threshold: cutoff applied to each gripper value; None disables snapping.
            open_value: command value for an open gripper.
            close_value: command value for a closed gripper.

        Returns:
            The same ``action`` array with gripper dims snapped.
        """
        mask = self.gripper_mask
        if threshold is None or not any(mask):
            return action
        rows = action if action.ndim > 1 else action[None]
        for row in rows:
            for idx, is_gripper in enumerate(mask):
                if is_gripper and idx < len(row):
                    row[idx] = open_value if row[idx] >= threshold else close_value
        return action
