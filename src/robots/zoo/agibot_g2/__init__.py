"""Agibot G2 humanoid robot package.

Dual-arm humanoid with a 24-DoF body (2x 8-DoF arms + 3-DoF head + 5-DoF torso).
PyRoki IK/FK operates relative to the chest frame (body_link5): it keeps the full
24-DoF body state (head/body frozen at their reference pose) while actively solving
only the 14 arm joints, via PyrokiDualArm with ``carry_full_state``. Gripper finger
coupling is not declarable, so the vis qpos->cfg mapping is a custom callable.
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
from robots.kinematics.solver import IkError, build_initial_seed

try:
    from robots.kinematics.pyroki import PyrokiDualArm, PyrokiSingleArm
except ImportError as exc:
    _PYROKI_IMPORT_ERROR: ImportError | None = exc

    class PyrokiDualArm:  # type: ignore[no-redef]
        pass

    PyrokiSingleArm = None  # type: ignore[assignment]
else:
    _PYROKI_IMPORT_ERROR = None


class AgibotKinematicsSolver(PyrokiDualArm):
    """Dual-arm Agibot G2 kinematics (IK + FK) in the chest frame, PyRoki-backed.

    Keeps the full 24-DoF body state (head/body frozen at their reference pose from
    the seed) while actively solving the 14 arm joints, via PyrokiDualArm with
    ``carry_full_state``. EEF poses are expressed relative to ``frame_name``.
    """

    def __init__(
        self,
        initial_qpos_groups: Sequence[Sequence[float]],
        *,
        urdf: Path,
        arm_joints: Sequence[Sequence[str]],
        eef_links: Sequence[str],
        head_joints: Sequence[str],
        body_joints: Sequence[str],
        frame_name: str = "body_link5",
        dt: float = 1.0 / 30.0,
        position_tolerance: float = 5e-3,
        orientation_tolerance: float = 5e-2,
    ) -> None:
        if not urdf.exists():
            raise IkError(f"Could not find URDF at {urdf}")
        seed = build_initial_seed(initial_qpos_groups)  # full 24 DoF
        fixed = {name: float(seed[16 + i]) for i, name in enumerate(head_joints)}
        fixed.update({name: float(seed[19 + i]) for i, name in enumerate(body_joints)})

        def make_arm(arm_index: int) -> PyrokiSingleArm:
            return PyrokiSingleArm(
                urdf, arm_joints[arm_index], eef_links[arm_index],
                reference_frame=frame_name, fixed_joints=fixed,
            )

        super().__init__(
            initial_qpos_groups,
            make_arm,
            arm_width=8,
            carry_full_state=True,
            dt=dt,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )


@ROBOT_REGISTRY.register("agibot_g2")
class AgibotG2(Robot):
    """Agibot G2 humanoid: dual 7-DoF arm + gripper, 3-DoF head, 5-DoF torso, 24-D action."""

    URDF = (
        Path(__file__).resolve().parent
        / "assets"
        / "g2_description"
        / "urdf"
        / "G2_rokae_ominipicker.urdf"
    )
    LEFT_ARM = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
    RIGHT_ARM = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
    HEAD = ("idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3")
    BODY = (
        "idx01_body_joint1",
        "idx02_body_joint2",
        "idx03_body_joint3",
        "idx04_body_joint4",
        "idx05_body_joint5",
    )
    LEFT_GROUP = (*LEFT_ARM, "idx41_gripper_l_outer_joint1")
    RIGHT_GROUP = (*RIGHT_ARM, "idx81_gripper_r_outer_joint1")
    EEF_LINKS = ("gripper_l_center_link", "gripper_r_center_link")
    INITIAL_QPOS = (
        0.739033, -0.717023, -1.524419, -1.537612, 0.27811, -0.925845, -0.839257, -0.785,
        -0.739033, -0.717023, 1.524419, -1.537612, -0.27811, -0.925845, 0.839257, -0.785,
        0.0, 0.0, 0.11464,
        -0.83423, 1.2172, 0.10025, 0.0, 0.0,
    )
    # One gripper scalar -> its 8 driven finger joints (visualization only; IK freezes grippers).
    GRIPPER_COUPLING = (1.0, 0.1, 0.25, -0.7, -1.0, 0.1, -0.25, 0.7)
    LEFT_GRIPPER_JOINTS = (
        "idx31_gripper_l_inner_joint1",
        "idx32_gripper_l_inner_joint3",
        "idx33_gripper_l_inner_joint4",
        "idx39_gripper_l_inner_joint0",
        "idx41_gripper_l_outer_joint1",
        "idx42_gripper_l_outer_joint3",
        "idx43_gripper_l_outer_joint4",
        "idx49_gripper_l_outer_joint0",
    )
    RIGHT_GRIPPER_JOINTS = (
        "idx71_gripper_r_inner_joint1",
        "idx72_gripper_r_inner_joint3",
        "idx73_gripper_r_inner_joint4",
        "idx79_gripper_r_inner_joint0",
        "idx81_gripper_r_outer_joint1",
        "idx82_gripper_r_outer_joint3",
        "idx83_gripper_r_outer_joint4",
        "idx89_gripper_r_outer_joint0",
    )

    def _vis_qpos_to_cfg(self, qpos: np.ndarray) -> dict[str, float]:
        """Map the 24-DoF policy qpos to the URDF actuated-joint cfg dict for visualization."""
        q = np.asarray(qpos, dtype=np.float64)
        left_arm, left_gripper = q[0:7], q[7]
        right_arm, right_gripper = q[8:15], q[15]
        head, body = q[16:19], q[19:24]

        cfg: dict[str, float] = {}
        for i, name in enumerate(self.LEFT_ARM):
            cfg[name] = left_arm[i]
        for i, name in enumerate(self.RIGHT_ARM):
            cfg[name] = right_arm[i]
        for i, name in enumerate(self.HEAD):
            cfg[name] = head[i]
        for i, name in enumerate(self.BODY):
            cfg[name] = body[i]
        for name, coeff in zip(self.LEFT_GRIPPER_JOINTS, self.GRIPPER_COUPLING, strict=True):
            cfg[name] = left_gripper * coeff
        for name, coeff in zip(self.RIGHT_GRIPPER_JOINTS, self.GRIPPER_COUPLING, strict=True):
            cfg[name] = right_gripper * coeff
        return cfg

    def __init__(self) -> None:
        super().__init__(
            name="agibot_g2",
            actuator_groups=(
                ActuatorGroup("left_arm", 8, self.LEFT_GROUP, gripper_index=7),
                ActuatorGroup("right_arm", 8, self.RIGHT_GROUP, gripper_index=7),
                ActuatorGroup("head", 3, self.HEAD),
                ActuatorGroup("body", 5, self.BODY),
            ),
            initial_qpos=np.asarray(self.INITIAL_QPOS, dtype=np.float32),
            observation_schema=ObservationSchema(
                cameras=(
                    CameraSpec("front", "top_head"),
                    CameraSpec("left_wrist", "hand_left"),
                    CameraSpec("right_wrist", "hand_right"),
                ),
                state_composition=("left_arm", "right_arm", "head", "body"),
            ),
            vis_config=RobotVisConfig(
                parts=(
                    VisPart(
                        name="whole_body",
                        urdf_path=self.URDF,
                        base_position=(0.0, 0.0, 0.0),
                        base_wxyz=(1.0, 0.0, 0.0, 0.0),
                        qpos_offset=0,
                        n_joints=24,
                        qpos_to_cfg=self._vis_qpos_to_cfg,
                        render_exclude=(
                            "base_link.STL",
                            "chassis_lwheel_front_link1.STL",
                            "chassis_lwheel_front_link2.STL",
                            "chassis_rwheel_front_link1.STL",
                            "chassis_rwheel_front_link2.STL",
                            "chassis_lwheel_rear_link1.STL",
                            "chassis_lwheel_rear_link2.STL",
                            "chassis_rwheel_rear_link1.STL",
                            "chassis_rwheel_rear_link2.STL",
                        ),
                    ),
                )
            ),
        )

    def build_kinematics(self, **kwargs: Any) -> Any:
        if _PYROKI_IMPORT_ERROR is not None:
            raise ImportError("Agibot G2 kinematics requires the PyRoki/JAX dependencies") from (
                _PYROKI_IMPORT_ERROR
            )
        return AgibotKinematicsSolver(
            urdf=self.URDF,
            arm_joints=(self.LEFT_ARM, self.RIGHT_ARM),
            eef_links=self.EEF_LINKS,
            head_joints=self.HEAD,
            body_joints=self.BODY,
            **kwargs,
        )
