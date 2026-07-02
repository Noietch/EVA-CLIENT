"""Tests for rotation/quaternion math utilities (core.utils.math)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.utils.math import (
    build_eef_pose_from_xyzw,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    quaternion_xyzw_to_wxyz,
)


def _euler_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Helper: build a wxyz quaternion from xyz Euler radians for test inputs."""
    x, y, z, w = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def test_quaternion_xyzw_to_wxyz_reorders():
    out = quaternion_xyzw_to_wxyz(1.0, 2.0, 3.0, 4.0)
    np.testing.assert_array_equal(out, np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float32))


def test_build_eef_pose_layout():
    pose = build_eef_pose_from_xyzw(0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 4.0, 0.5)
    assert pose.shape == (8,)
    np.testing.assert_allclose(pose[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(pose[3:7], [4.0, 1.0, 2.0, 3.0], atol=1e-6)  # wxyz
    assert pose[7] == pytest.approx(0.5)


def test_identity_matrix_round_trip():
    q = matrix_to_quaternion_wxyz(np.eye(3))
    np.testing.assert_allclose(np.abs(q), [1.0, 0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize(
    "roll,pitch,yaw",
    [
        (0.0, 0.0, 0.0),
        (math.pi / 2, 0.0, 0.0),
        (0.0, math.pi / 2, 0.0),
        (0.0, 0.0, math.pi / 2),
        (0.3, -0.7, 1.2),
        (-2.5, 1.3, -0.4),
        (math.pi, 0.0, 0.0),
    ],
)
def test_quat_matrix_quat_round_trip(roll, pitch, yaw):
    """quat -> matrix -> quat must recover the same rotation (up to sign)."""
    q = _euler_wxyz(roll, pitch, yaw)
    matrix = quaternion_wxyz_to_matrix(q[0], q[1], q[2], q[3])
    q_back = matrix_to_quaternion_wxyz(matrix)
    # A quaternion and its negation represent the same rotation.
    if np.dot(q, q_back) < 0:
        q_back = -q_back
    np.testing.assert_allclose(q_back, q, atol=1e-5)


@pytest.mark.parametrize(
    "roll,pitch,yaw",
    [
        (0.3, -0.7, 1.2),
        (-2.5, 1.3, -0.4),
        (math.pi / 3, math.pi / 4, -math.pi / 6),
    ],
)
def test_rotation_matrix_is_orthonormal(roll, pitch, yaw):
    q = _euler_wxyz(roll, pitch, yaw)
    matrix = quaternion_wxyz_to_matrix(q[0], q[1], q[2], q[3])
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-6)


def test_quaternion_wxyz_to_matrix_normalizes_input():
    """Unnormalized quaternion should still yield a proper rotation matrix."""
    matrix = quaternion_wxyz_to_matrix(2.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-6)


def test_quaternion_wxyz_to_matrix_zero_norm_raises():
    with pytest.raises(ValueError):
        quaternion_wxyz_to_matrix(0.0, 0.0, 0.0, 0.0)


def test_matrix_to_quaternion_branches():
    """Cover the trace<=0 branches by rotating 180 deg about each axis."""
    # 180 deg about x: diag(1, -1, -1)
    rx = np.diag([1.0, -1.0, -1.0])
    qx = matrix_to_quaternion_wxyz(rx)
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(qx[0], qx[1], qx[2], qx[3]), rx, atol=1e-6)
    # 180 deg about y
    ry = np.diag([-1.0, 1.0, -1.0])
    qy = matrix_to_quaternion_wxyz(ry)
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(qy[0], qy[1], qy[2], qy[3]), ry, atol=1e-6)
    # 180 deg about z
    rz = np.diag([-1.0, -1.0, 1.0])
    qz = matrix_to_quaternion_wxyz(rz)
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(qz[0], qz[1], qz[2], qz[3]), rz, atol=1e-6)


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_matrix_to_quaternion_xx_branch():
    """Rx(150deg) has max diagonal at [0,0] with negative trace -> xx branch."""
    q = matrix_to_quaternion_wxyz(_rx(math.radians(150.0)))
    np.testing.assert_allclose(q, [0.258819, 0.965926, 0.0, 0.0], atol=1e-6)


def test_matrix_to_quaternion_yy_branch():
    q = matrix_to_quaternion_wxyz(_ry(math.radians(150.0)))
    np.testing.assert_allclose(q, [0.258819, 0.0, 0.965926, 0.0], atol=1e-6)


def test_matrix_to_quaternion_zz_branch():
    q = matrix_to_quaternion_wxyz(_rz(math.radians(150.0)))
    np.testing.assert_allclose(q, [0.258819, 0.0, 0.0, 0.965926], atol=1e-6)


def test_matrix_to_quaternion_trace_positive_antisymmetric():
    """Rz(90deg) has positive trace with nonzero antisymmetric part."""
    q = matrix_to_quaternion_wxyz(_rz(math.radians(90.0)))
    np.testing.assert_allclose(q, [0.7071068, 0.0, 0.0, 0.7071068], atol=1e-6)
