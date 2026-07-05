"""Agilex Piper dual-arm robot package.

Two identical 6-DoF Piper arms reusing one URDF, each with a single-scalar gripper.
Kinematics is the shared PyRoki backend (both arms share the same chain).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.registry import ROBOT_REGISTRY
from robots.base import (
    ActuatorGroup,
    CameraSpec,
    ObservationSchema,
    Robot,
    RobotVisConfig,
    VisPart,
)
from robots.kinematics.pyroki import pyroki_arms


@ROBOT_REGISTRY.register("agilex_piper")
class AgilexPiper(Robot):
    """Agilex Piper: dual 6-DoF arm + gripper, 14-D action."""

    URDF = (
        Path(__file__).resolve().parent
        / "assets"
        / "piper_description"
        / "urdf"
        / "piper_description.urdf"
    )
    ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    JOINT_NAMES = ("joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    GRIPPER_SEGMENTS = [
        {"copy": [0, 6]},
        {"gripper": 6, "range": [0.0, 0.035], "stroke": 0.035, "fingers": [1, -1]},
    ]

    def __init__(self) -> None:
        super().__init__(
            name="agilex_piper",
            actuator_groups=(
                ActuatorGroup("left_arm", 7, self.JOINT_NAMES, gripper_index=6),
                ActuatorGroup("right_arm", 7, self.JOINT_NAMES, gripper_index=6),
            ),
            initial_qpos=np.zeros(14, dtype=np.float32),
            observation_schema=ObservationSchema(
                cameras=(
                    CameraSpec("front", "cam_high"),
                    CameraSpec("left_wrist", "cam_left_wrist"),
                    CameraSpec("right_wrist", "cam_right_wrist"),
                ),
                state_composition=("left_arm", "right_arm"),
            ),
            vis_config=RobotVisConfig(
                parts=(
                    VisPart.from_segments(
                        "left_arm", self.URDF, (-0.25, 0.0, 0.0), (0.7071068, 0.0, 0.0, 0.7071068),
                        0, 7, self.GRIPPER_SEGMENTS,
                    ),
                    VisPart.from_segments(
                        "right_arm", self.URDF, (0.25, 0.0, 0.0), (0.7071068, 0.0, 0.0, 0.7071068),
                        7, 7, self.GRIPPER_SEGMENTS,
                    ),
                )
            ),
        )

    def build_kinematics(self, **kwargs: Any) -> Any:
        return pyroki_arms(
            self.URDF, [{"joints": self.ARM_JOINTS, "eef_link": "gripper_base"}] * 2, **kwargs
        )

    def default_calibration_poses(self) -> list[np.ndarray]:
        """8 preset poses covering wide angle coverage for wrist-camera hand-eye.

        Layout: [left_arm(7), right_arm(7)] = 14. Each qpos varies the "look
        angle" of the wrist by rotating joint4 (wrist pitch) and joint5 (wrist
        yaw) while keeping the arm forward-reachable and both grippers open.
        """
        base_l = [0.0, 0.6, -0.8, 0.0, 0.5, 0.0, 0.0]
        base_r = [0.0, -0.6, 0.8, 0.0, -0.5, 0.0, 0.0]
        variants = [
            (0.0, 0.0), (0.4, 0.0), (-0.4, 0.0), (0.0, 0.4),
            (0.0, -0.4), (0.4, 0.4), (-0.4, -0.4), (0.4, -0.4),
        ]
        poses: list[np.ndarray] = []
        for dpitch, dyaw in variants:
            left = base_l.copy(); left[3] += dpitch; left[4] += dyaw
            right = base_r.copy(); right[3] -= dpitch; right[4] -= dyaw
            poses.append(np.array(left + right, dtype=np.float32))
        return poses
