"""Dual-arm Franka Panda robot package.

Two identical 8-DoF Panda arms (7 revolute + 1 gripper) reusing one URDF and the
``ee_link`` end-effector, distinguished by a per-arm mount SE3. PyRoki (JAX + jaxls)
IK/FK: each arm is a PyrokiSingleArm (finger joints frozen by the mask) wrapped by
PyrokiDualArm chunk orchestration. Two reference frames: ``workcell`` applies the
per-arm mount, ``base_link`` uses identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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
    from robots.kinematics.pyroki import PyrokiDualArm, PyrokiSingleArm
except ImportError as exc:
    _PYROKI_IMPORT_ERROR: ImportError | None = exc

    class PyrokiDualArm:  # type: ignore[no-redef]
        pass

    PyrokiSingleArm = None  # type: ignore[assignment]
else:
    _PYROKI_IMPORT_ERROR = None


def _unit_quat(quat_wxyz: Iterable[float]) -> tuple[float, float, float, float]:
    """Normalize a (w, x, y, z) quaternion to unit length."""
    v = np.asarray(tuple(quat_wxyz), dtype=np.float64)
    v = v / float(np.linalg.norm(v))
    return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


class FrankaKinematicsSolver(PyrokiDualArm):
    """Dual-arm Franka whole-chunk kinematics (IK + FK), PyRoki-backed.

    Two 7-DoF arms share one URDF and the ``eef_link`` frame; the per-arm ``mounts``
    SE3 distinguishes left/right and maps arm-local <-> workcell (None = identity).
    8D-per-arm EEF.
    """

    def __init__(
        self,
        initial_qpos_groups: Sequence[Sequence[float]],
        *,
        urdf: Path,
        arm_joints: Sequence[str],
        eef_link: str,
        mounts: Sequence[tuple[Sequence[float], Sequence[float]] | None],
        dt: float = 1.0 / 30.0,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-3,
    ) -> None:
        def make_arm(arm_index: int) -> PyrokiSingleArm:
            return PyrokiSingleArm(
                urdf, arm_joints, eef_link, reference_frame=None, mount=mounts[arm_index]
            )

        super().__init__(
            initial_qpos_groups,
            single_arm_factory=make_arm,
            arm_width=8,
            carry_full_state=False,
            dt=dt,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )


@ROBOT_REGISTRY.register("dual_franka")
class DualFranka(Robot):
    """Dual-arm Franka Panda: two 7-DoF arms + gripper, 16-D action."""

    URDF = Path(__file__).resolve().parent / "assets" / "franka_description" / "franka_panda.urdf"
    EEF_LINK = "ee_link"
    ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
    JOINT_NAMES = (*ARM_JOINTS, "panda_finger_joint1")
    ARM_QPOS = (
        -1.88733017,
        -0.98627802,
        1.34066187,
        -1.77472402,
        0.08241637,
        1.80246260,
        0.79725709,
        1.0,
    )
    GRIPPER_SEGMENTS = [
        {"copy": [0, 7]},
        {"gripper": 7, "range": [0.0, 1.0], "stroke": 0.04, "fingers": [1, 1]},
    ]
    DEFAULT_FRAME = "workcell"
    SUPPORTED_FRAMES = ("workcell", "base_link")
    # Per-arm rigid mount (pos, wxyz) relative to the workcell reference frame, by arm index.
    MOUNTS = (
        (
            (0.036395, 0.050681, 0.0508858),
            _unit_quat((0.865806672, -0.436878319, 0.022287915, -0.242939065)),
        ),
        (
            (0.036395, -0.050681, 0.0508858),
            _unit_quat((0.865806672, 0.436878319, 0.022287915, 0.242939065)),
        ),
    )

    def __init__(self) -> None:
        super().__init__(
            name="dual_franka",
            actuator_groups=(
                ActuatorGroup("left_arm", 8, self.JOINT_NAMES, gripper_index=7),
                ActuatorGroup("right_arm", 8, self.JOINT_NAMES, gripper_index=7),
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
                        "left_arm", self.URDF, (0.036395, 0.050681, 0.0508858),
                        (0.865806672, -0.436878319, 0.022287915, -0.242939065),
                        0, 8, self.GRIPPER_SEGMENTS,
                    ),
                    VisPart.from_segments(
                        "right_arm", self.URDF, (0.036395, -0.050681, 0.0508858),
                        (0.865806672, 0.436878319, 0.022287915, 0.242939065),
                        8, 8, self.GRIPPER_SEGMENTS,
                    ),
                )
            ),
        )

    def build_kinematics(self, *, reference_frame: str | None = None, **kwargs: Any) -> Any:
        if _PYROKI_IMPORT_ERROR is not None:
            raise ImportError("Franka kinematics requires the PyRoki/JAX dependencies") from (
                _PYROKI_IMPORT_ERROR
            )
        frame = reference_frame or self.DEFAULT_FRAME
        if frame not in self.SUPPORTED_FRAMES:
            allowed = ", ".join(self.SUPPORTED_FRAMES)
            raise ValueError(f"Unsupported Franka reference frame {frame!r}; expected: {allowed}")
        mounts = self.MOUNTS if frame == self.DEFAULT_FRAME else (None, None)
        return FrankaKinematicsSolver(
            urdf=self.URDF,
            arm_joints=self.ARM_JOINTS,
            eef_link=self.EEF_LINK,
            mounts=mounts,
            **kwargs,
        )
