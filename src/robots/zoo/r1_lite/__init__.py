"""R1 Lite dual-arm robot package.

Two 6-DoF arms branch from a torso held at a fixed non-zero pose; the whole body
renders as a single URDF part driven by the full 14-D qpos. PyRoki IK/FK: each arm
is a PyrokiSingleArm (torso pinned, all other actuated joints frozen at seed) wrapped
by PyrokiDualArm. EEF poses default to ``base_link``, runtime-switchable to ``torso_link3``.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from robots.kinematics.pyroki import PyrokiDualArm, PyrokiSingleArm


class R1LiteKinematicsSolver(PyrokiDualArm):
    """Dual-arm R1 Lite whole-chunk kinematics (IK + FK), PyRoki-backed.

    Two PyrokiSingleArm solvers (6 arm joints each, torso pinned via ``fixed_joints``,
    all other actuated joints frozen at seed) driven through the dual-arm chunk
    orchestration. EEF poses are expressed in ``reference_frame``.
    """

    def __init__(
        self,
        initial_qpos_groups: Sequence[Sequence[float]],
        *,
        urdf: Path,
        arm_joints: dict[int, list[str]],
        eef_links: dict[int, str],
        fixed_joints: dict[str, float],
        reference_frame: str,
        dt: float = 1.0 / 30.0,
        position_tolerance: float = 5e-3,
        orientation_tolerance: float = 5e-2,
    ) -> None:
        def make_arm(arm: int) -> PyrokiSingleArm:
            return PyrokiSingleArm(
                urdf, arm_joints[arm], eef_links[arm],
                reference_frame=reference_frame, fixed_joints=fixed_joints,
            )

        super().__init__(
            initial_qpos_groups,
            single_arm_factory=make_arm,
            arm_width=7,
            dt=dt,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )


@ROBOT_REGISTRY.register("r1_lite")
class R1Lite(Robot):
    """R1 Lite: dual 6-DoF arm + gripper on a fixed torso, 14-D action."""

    URDF = (
        Path(__file__).resolve().parent
        / "assets"
        / "r1lite_description"
        / "urdf"
        / "r1lite.urdf"
    )
    ARM_JOINTS = {
        0: [f"left_arm_joint{i}" for i in range(1, 7)],
        1: [f"right_arm_joint{i}" for i in range(1, 7)],
    }
    EEF_LINKS = {0: "left_gripper_link", 1: "right_gripper_link"}
    FIXED_JOINTS = {"torso_joint1": -0.657, "torso_joint2": 1.516, "torso_joint3": 0.845}
    LEFT_JOINTS = (*(f"left_arm_joint{i}" for i in range(1, 7)), "left_gripper_joint")
    RIGHT_JOINTS = (*(f"right_arm_joint{i}" for i in range(1, 7)), "right_gripper_joint")
    DEFAULT_FRAME = "base_link"
    SUPPORTED_FRAMES = ("base_link", "torso_link3")
    # 14-D qpos -> 25-D URDF cfg in joint order; whole body as one part.
    VIS_SEGMENTS = [
        {"fixed": [0, 0, 0, 0, 0, 0]},                # wheels / steer
        {"fixed": [-0.657, 1.516, 0.845]},           # torso (held fixed)
        {"copy": [0, 6]},                            # left arm joint1..6
        {"gripper": 6, "range": [0.0, 100.0], "stroke": 0.035, "fingers": [1, -1]},
        {"copy": [7, 13]},                           # right arm joint1..6
        {"gripper": 13, "range": [0.0, 100.0], "stroke": 0.035, "fingers": [1, -1]},
    ]

    def __init__(self) -> None:
        super().__init__(
            name="r1_lite",
            actuator_groups=(
                ActuatorGroup("left_arm", 7, self.LEFT_JOINTS, gripper_index=6),
                ActuatorGroup("right_arm", 7, self.RIGHT_JOINTS, gripper_index=6),
            ),
            initial_qpos=np.zeros(14, dtype=np.float32),
            observation_schema=ObservationSchema(
                cameras=(
                    CameraSpec("head", "cam_high"),
                    CameraSpec("left_wrist", "cam_left_wrist"),
                    CameraSpec("right_wrist", "cam_right_wrist"),
                ),
                state_composition=("left_arm", "right_arm"),
            ),
            vis_config=RobotVisConfig(
                parts=(
                    VisPart.from_segments(
                        "body",
                        self.URDF,
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0, 0.0),
                        0,
                        14,
                        self.VIS_SEGMENTS,
                    ),
                )
            ),
        )

    def build_kinematics(self, *, reference_frame: str | None = None, **kwargs: Any) -> Any:
        frame = reference_frame or self.DEFAULT_FRAME
        if frame not in self.SUPPORTED_FRAMES:
            allowed = ", ".join(self.SUPPORTED_FRAMES)
            raise ValueError(f"Unsupported R1 Lite reference frame {frame!r}; expected: {allowed}")
        return R1LiteKinematicsSolver(
            urdf=self.URDF, arm_joints=self.ARM_JOINTS, eef_links=self.EEF_LINKS,
            fixed_joints=self.FIXED_JOINTS, reference_frame=frame, **kwargs,
        )
