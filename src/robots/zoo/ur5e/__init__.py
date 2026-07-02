"""UR5e single-arm robot package.

One 6-DoF UR5e arm with a single-scalar gripper, shared PyRoki backend.
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


@ROBOT_REGISTRY.register("ur5e")
class UR5e(Robot):
    """UR5e: single 6-DoF arm + gripper, 7-D action."""

    URDF = Path(__file__).resolve().parent / "assets" / "ur5e_description" / "urdf" / "ur5e.urdf"
    ARM_JOINTS = (
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    )
    INITIAL_QPOS = (
        -3.3475797812091272, -1.2882714432528992, 1.4286816755877894,
        1.5116821962543945, 1.6023021936416626, -0.21513206163515264, 1.0,
    )

    def __init__(self) -> None:
        super().__init__(
            name="ur5e",
            actuator_groups=(
                ActuatorGroup("arm", 7, (*self.ARM_JOINTS, "gripper_position"), gripper_index=6),
            ),
            initial_qpos=np.asarray(self.INITIAL_QPOS, dtype=np.float32),
            observation_schema=ObservationSchema(
                cameras=(
                    CameraSpec("wrist_image", "cam_wrist"),
                    CameraSpec("exterior_image", "cam_high"),
                ),
                state_composition=("arm",),
            ),
            vis_config=RobotVisConfig(
                parts=(
                    VisPart.from_segments(
                        "arm", self.URDF, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 0, 7,
                        [
                            {"copy": [0, 6]},
                            {
                                "gripper": 6,
                                "range": [0.0, 1.0],
                                "stroke": 0.93,
                                "invert": True,
                                "fingers": [1],
                            },
                        ],
                    ),
                )
            ),
        )

    def build_kinematics(self, **kwargs: Any) -> Any:
        return pyroki_arms(
            self.URDF,
            [{"joints": self.ARM_JOINTS, "eef_link": "grasp_link", "reference_frame": "base_link"}],
            **kwargs,
        )
