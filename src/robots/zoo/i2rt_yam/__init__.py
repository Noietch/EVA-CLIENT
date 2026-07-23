"""I2RT YAM single-arm robot description.

The hardware SDK stays in ``examples/hardware/i2rt`` and talks to EVA over ZMQ.
When that SDK checkout exists, its official URDF is also reused for browser
visualization and PyRoki FK/IK; joint-space deployment remains available without
the checkout.
"""

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

try:
    from robots.kinematics.pyroki import pyroki_arms
except ImportError as exc:
    _PYROKI_IMPORT_ERROR: ImportError | None = exc
    pyroki_arms = None
else:
    _PYROKI_IMPORT_ERROR = None


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

def find_i2rt_yam_urdf() -> Path | None:
    """Locate the official YAM URDF in the in-tree SDK checkout or installation."""
    project_root = Path(__file__).resolve().parents[4]
    checkout = (
        project_root
        / "examples"
        / "hardware"
        / "i2rt"
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


@ROBOT_REGISTRY.register("i2rt_yam")
class I2RTYam(Robot):
    """I2RT YAM: one 6-DoF arm and normalized gripper, 7-D action."""

    def __init__(self) -> None:
        urdf = find_i2rt_yam_urdf()
        vis_config = None
        if urdf is not None:
            vis_config = RobotVisConfig(
                parts=(
                    make_vis_part(
                        "arm",
                        urdf,
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0, 0.0),
                        0,
                    ),
                )
            )
        super().__init__(
            name="i2rt_yam",
            actuator_groups=(ActuatorGroup("arm", 7, JOINT_NAMES, gripper_index=6),),
            initial_qpos=np.asarray(ARM_QPOS, dtype=np.float32),
            observation_schema=ObservationSchema(
                cameras=(
                    CameraSpec("front", "cam_high"),
                    CameraSpec("wrist", "cam_wrist"),
                ),
                state_composition=("arm",),
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
            raise ImportError("I2RT YAM kinematics requires PyRoki/JAX") from (_PYROKI_IMPORT_ERROR)
        assert pyroki_arms is not None
        return pyroki_arms(
            self.urdf,
            [{"joints": ARM_JOINTS, "eef_link": "gripper"}],
            **kwargs,
        )
