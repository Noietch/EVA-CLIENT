"""ARX X5 dual-arm robot package.

Two identical 6-DoF X5 arms reusing one URDF, each with a single-scalar gripper.
Shared PyRoki backend (both arms share the same chain). Structurally identical to
the ARX R5 zoo entry — same joint/link topology — but backed by the X5A URDF so
the X5 embodiment matches the ARX X5 USD used inside eva_sim.
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


@ROBOT_REGISTRY.register("arx_x5")
class ArxX5(Robot):
    """ARX X5: dual 6-DoF arm + gripper, 14-D action."""

    URDF = Path(__file__).resolve().parent / "assets" / "X5a" / "urdf" / "X5a.urdf"
    ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    JOINT_NAMES = (*ARM_JOINTS, "gripper")
    ARM_QPOS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    # X5a finger joints have opposite axes (0 1 0 / 0 -1 0), so both take +f.
    GRIPPER_SEGMENTS = [
        {"copy": [0, 6]},
        {"gripper": 6, "range": [0.0, 0.044], "stroke": 0.044, "fingers": [1, 1]},
    ]

    def __init__(self) -> None:
        super().__init__(
            name="arx_x5",
            actuator_groups=(
                ActuatorGroup("left_arm", 7, self.JOINT_NAMES, gripper_index=6),
                ActuatorGroup("right_arm", 7, self.JOINT_NAMES, gripper_index=6),
            ),
            initial_qpos=np.asarray(self.ARM_QPOS + self.ARM_QPOS, dtype=np.float32),
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
            self.URDF,
            [{"joints": self.ARM_JOINTS, "eef_link": "link6", "reference_frame": "base_link"}] * 2,
            **kwargs,
        )
