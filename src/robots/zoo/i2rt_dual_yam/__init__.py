"""I2RT dual YAM robot description for a YAM Cell-style follower pair."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.registry import ROBOT_REGISTRY
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot, RobotVisConfig
from robots.zoo.i2rt_yam import (
    ARM_JOINTS,
    ARM_QPOS,
    JOINT_NAMES,
    find_i2rt_yam_urdf,
    make_vis_part,
)

LEFT_BASE_POSITION = (0.0, 0.25, 0.0)
RIGHT_BASE_POSITION = (0.0, -0.25, 0.0)

try:
    from robots.kinematics.pyroki import pyroki_arms
except ImportError as exc:
    _PYROKI_IMPORT_ERROR: ImportError | None = exc
    pyroki_arms = None
else:
    _PYROKI_IMPORT_ERROR = None


@ROBOT_REGISTRY.register("i2rt_dual_yam")
class I2RTDualYam(Robot):
    """Two I2RT YAM followers: 2 x (6 arm joints + gripper), 14-D action."""

    def __init__(self) -> None:
        urdf = find_i2rt_yam_urdf()
        vis_config = None
        if urdf is not None:
            vis_config = RobotVisConfig(
                parts=(
                    make_vis_part(
                        "left_arm",
                        urdf,
                        LEFT_BASE_POSITION,
                        (1.0, 0.0, 0.0, 0.0),
                        0,
                    ),
                    make_vis_part(
                        "right_arm",
                        urdf,
                        RIGHT_BASE_POSITION,
                        (1.0, 0.0, 0.0, 0.0),
                        7,
                    ),
                )
            )
        super().__init__(
            name="i2rt_dual_yam",
            actuator_groups=(
                ActuatorGroup("left_arm", 7, JOINT_NAMES, gripper_index=6),
                ActuatorGroup("right_arm", 7, JOINT_NAMES, gripper_index=6),
            ),
            initial_qpos=np.asarray(ARM_QPOS + ARM_QPOS, dtype=np.float32),
            observation_schema=ObservationSchema(
                cameras=(
                    CameraSpec("front", "cam_high"),
                    CameraSpec("left_wrist", "cam_left_wrist"),
                    CameraSpec("right_wrist", "cam_right_wrist"),
                ),
                state_composition=("left_arm", "right_arm"),
            ),
            vis_config=vis_config,
        )
        self.urdf = urdf

    def build_kinematics(self, **kwargs: Any) -> Any:
        if self.urdf is None:
            raise FileNotFoundError(
                "I2RT YAM URDF is unavailable; run examples/hardware/i2rt/setup_sdk.sh"
            )
        if _PYROKI_IMPORT_ERROR is not None:
            raise ImportError("I2RT dual YAM kinematics requires PyRoki/JAX") from (
                _PYROKI_IMPORT_ERROR
            )
        assert pyroki_arms is not None
        return pyroki_arms(
            self.urdf,
            [{"joints": ARM_JOINTS, "eef_link": "gripper"}] * 2,
            **kwargs,
        )
