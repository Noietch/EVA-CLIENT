"""YAM dual YAM robot description for a YAM Cell-style follower pair."""

from __future__ import annotations

from importlib.util import find_spec
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

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
JOINT_NAMES = (*ARM_JOINTS, "gripper")
ARM_QPOS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
GRIPPER_SEGMENTS = [
    {"copy": [0, 6]},
    {
        "gripper": 6,
        "range": [0.0, 1.0],
        "stroke": 0.04695,
        "fingers": [-1, -1],
    },
]
LEFT_BASE_POSITION = (0.0, 0.25, 0.0)
RIGHT_BASE_POSITION = (0.0, -0.25, 0.0)

try:
    from robots.kinematics.pyroki import pyroki_arms
except ImportError as exc:
    _PYROKI_IMPORT_ERROR: ImportError | None = exc
    pyroki_arms = None
else:
    _PYROKI_IMPORT_ERROR = None


def find_yam_urdf() -> Path | None:
    """Locate the official YAM URDF in the in-tree SDK checkout or installation."""
    project_root = Path(__file__).resolve().parents[4]
    fallback = Path(__file__).resolve().parent / "assets" / "yam.urdf"
    checkout = (
        project_root
        / "examples"
        / "hardware"
        / "yam"
        / "SDK"
        / "i2rt"
        / "i2rt"
        / "robot_models"
        / "arm"
        / "yam"
        / "yam.urdf"
    )
    if checkout.is_file():
        return checkout
    if fallback.is_file():
        return fallback

    spec = find_spec("i2rt")
    if spec is None:
        return None
    roots = spec.submodule_search_locations or ()
    for root in roots:
        installed = Path(root) / "robot_models" / "arm" / "yam" / "yam.urdf"
        if installed.is_file():
            return installed
    return None


def make_vis_part(
    name: str,
    urdf: Path,
    base_position: tuple[float, float, float],
    base_wxyz: tuple[float, float, float, float],
    qpos_offset: int,
) -> VisPart:
    return VisPart.from_segments(
        name,
        urdf,
        base_position,
        base_wxyz,
        qpos_offset,
        7,
        GRIPPER_SEGMENTS,
    )


@ROBOT_REGISTRY.register("dual_yam")
class DualYam(Robot):
    """Two YAM YAM followers: 2 x (6 arm joints + gripper), 14-D action."""

    def __init__(self) -> None:
        urdf = find_yam_urdf()
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
            name="dual_yam",
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
                "YAM YAM URDF is unavailable; run examples/hardware/yam/setup_sdk.sh"
            )
        if _PYROKI_IMPORT_ERROR is not None:
            raise ImportError("YAM dual YAM kinematics requires PyRoki/JAX") from (
                _PYROKI_IMPORT_ERROR
            )
        assert pyroki_arms is not None
        return pyroki_arms(
            self.urdf,
            [{"joints": ARM_JOINTS, "eef_link": "gripper"}] * 2,
            **kwargs,
        )
