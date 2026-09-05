"""Tests for the Piper IK/FK solver, built via the shared PyRoki backend.

Marked ``slow``: requires the JAX/pyroki stack + the Piper URDF assets.
Run the fast suite with ``pytest -m "not slow"`` to skip these.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyroki")

import robots  # noqa: F401  (registers robots under ROBOT_REGISTRY)
from core.registry import ROBOT_REGISTRY

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_PIPER_URDF = (
    Path(robots.__file__).resolve().parent
    / "zoo"
    / "agilex_piper"
    / "assets"
    / "piper_description"
    / "urdf"
    / "piper_description.urdf"
)


@pytest.fixture(scope="module")
def solver():
    if not _PIPER_URDF.exists():
        pytest.skip("Piper URDF assets not found")
    return ROBOT_REGISTRY.build("agilex_piper").build_kinematics(
        initial_qpos_groups=[[0.0] * 7, [0.0] * 7]
    )


@pytest.fixture(scope="module")
def single_arm(solver):
    """The left per-arm PyRoki solver underlying the dual-arm registry build."""
    return solver._left_solver


def test_fk_ik_fk_round_trip(single_arm):
    """A reachable pose from FK must be recoverable by whole-chunk IK."""
    q_true = np.array([0.2, 0.3, -0.4, 0.1, 0.5, -0.2])
    eef_target = single_arm.fk_arm(np.concatenate([q_true, [0.0]]))

    eef_chunk = np.tile(eef_target, (3, 1))
    solved = single_arm.solve_arm_chunk(eef_chunk, q_true)
    eef_back = single_arm.fk_arm(np.concatenate([solved[-1], [0.0]]))
    np.testing.assert_allclose(eef_back[:3], eef_target[:3], atol=5e-3)


def test_solve_arm_chunk_anchors_first_frame(single_arm):
    """Frame 0 of the solved trajectory equals the measured start state."""
    start = np.array([0.1, 0.5, -0.3, 0.2, 0.4, -0.1])
    eef_target = single_arm.fk_arm(np.concatenate([start, [0.0]]))
    eef_chunk = np.tile(eef_target, (5, 1))
    solved = single_arm.solve_arm_chunk(eef_chunk, start)
    np.testing.assert_allclose(solved[0], start, atol=1e-4)
