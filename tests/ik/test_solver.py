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
from robots.kinematics.solver import IkError, build_initial_seed

pytestmark = pytest.mark.slow

_PIPER_URDF = (
    Path(robots.__file__).resolve().parent
    / "zoo"
    / "agilex_piper"
    / "assets"
    / "piper_description"
    / "urdf"
    / "piper_description.urdf"
)


def _build_piper_solver():
    """Build the Piper dual-arm PyRoki solver via the robot, zero initial qpos."""
    return ROBOT_REGISTRY.build("agilex_piper").build_kinematics(
        initial_qpos_groups=[[0.0] * 7, [0.0] * 7]
    )


@pytest.fixture(scope="module")
def solver():
    if not _PIPER_URDF.exists():
        pytest.skip("Piper URDF assets not found")
    return _build_piper_solver()


@pytest.fixture(scope="module")
def single_arm(solver):
    """The left per-arm PyRoki solver underlying the dual-arm registry build."""
    return solver._left_solver


# --- FK ---


def test_fk_arm_output_layout(single_arm):
    eef = single_arm.fk_arm(np.zeros(7))
    assert eef.shape == (8,)
    quat = eef[3:7]
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-5)


def test_fk_arm_wrong_dim_raises(single_arm):
    with pytest.raises(IkError):
        single_arm.fk_arm(np.zeros(6))


def test_fk_carries_gripper_through(single_arm):
    eef = single_arm.fk_arm([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42])
    assert eef[7] == pytest.approx(0.42, abs=1e-6)


# --- IK round-trip ---


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


# --- build_initial_seed ---


def test_build_initial_seed_concatenates():
    seed = build_initial_seed([list(range(7)), list(range(7, 14))])
    assert seed.shape == (14,)
    np.testing.assert_array_equal(seed, np.arange(14.0))


# --- ForwardKinematicsSolver (dual arm) ---


def test_fk_chunk_shape(solver):
    chunk = np.zeros((3, 14), dtype=np.float32)
    out = solver.fk_chunk(chunk)
    assert out.shape == (3, 16)  # 8D per arm x 2 arms


def test_fk_chunk_1d_promoted(solver):
    out = solver.fk_chunk(np.zeros(14))
    assert out.shape == (1, 16)


def test_fk_chunk_insufficient_dim_raises(solver):
    with pytest.raises(IkError):
        solver.fk_chunk(np.zeros((2, 10)))


# --- dual-arm IK ---


def test_solve_chunk_shape(solver):
    """Round-trip a zero-pose qpos chunk through FK then IK."""
    qpos_chunk = np.zeros((2, 14), dtype=np.float32)
    eef_chunk = solver.fk_chunk(qpos_chunk)  # (2, 16)
    solver.reset([np.zeros(7), np.zeros(7)])
    solved = solver.solve_chunk(eef_chunk)
    assert solved.shape == (2, 14)


def test_solve_chunk_insufficient_dim_raises(solver):
    with pytest.raises(IkError):
        solver.solve_chunk(np.zeros((2, 10)))


def test_solve_chunk_passes_tail_through(solver):
    qpos_chunk = np.zeros((2, 14), dtype=np.float32)
    eef_chunk = solver.fk_chunk(qpos_chunk)  # (2, 16)
    tail = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    eef_with_tail = np.concatenate([eef_chunk, tail], axis=1)  # (2, 18)
    solver.reset([np.zeros(7), np.zeros(7)])
    solved = solver.solve_chunk(eef_with_tail)
    assert solved.shape == (2, 16)
    np.testing.assert_allclose(solved[:, 14:], tail, atol=1e-5)


def test_default_seed_returns_copy(solver):
    s1 = solver.default_seed()
    s1[0] = 999.0
    s2 = solver.default_seed()
    assert s2[0] != 999.0
