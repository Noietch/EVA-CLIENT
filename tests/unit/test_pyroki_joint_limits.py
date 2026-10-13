"""IK output must honor the robot limits even when optimization stops outside them."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from robots.kinematics.pyroki import PyrokiSingleArm

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("direction", [-1, 1])
def test_ik_clamps_output_and_the_next_frames_seed(monkeypatch, direction):
    root = Path(__file__).resolve().parents[2]
    solver = PyrokiSingleArm(
        root / "src/robots/zoo/dual_yam/assets/yam.urdf",
        [f"joint{index}" for index in range(1, 7)],
        "gripper",
    )
    start = np.array([0.0, 1.58, 0.49, 1.15, 0.0, 0.0])
    pose = solver.fk_arm(np.append(start, 1.0))
    result = solver._full_config(start)
    bound = solver._arm_upper[-1] if direction > 0 else solver._arm_lower[-1]
    result[solver._arm_cols[-1]] = bound + direction * 0.0163
    solve_frame = Mock(return_value=result)
    monkeypatch.setattr("robots.kinematics.pyroki._solve_frame_jax", solve_frame)

    trajectory = solver.solve_arm_chunk(np.tile(pose, (2, 1)), start)

    np.testing.assert_allclose(trajectory[:, :-1], np.tile(start[:-1], (2, 1)))
    np.testing.assert_allclose(trajectory[:, -1], bound)
    next_seed = np.asarray(solve_frame.call_args_list[1].args[6])
    assert next_seed[solver._arm_cols[-1]] == bound
