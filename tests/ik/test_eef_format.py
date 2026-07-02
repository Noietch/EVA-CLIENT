"""Tests for EEFPose derived-dimension methods + normalize_chunk_to_canonical."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.app.handlers.space import EEFPose


@pytest.mark.parametrize(
    "rotation,expected_rot_dim",
    [("quat", 4), ("euler", 3), ("rotvec", 3), ("rot6d", 6)],
)
def test_rotation_dim(rotation, expected_rot_dim):
    fmt = EEFPose(rotation=rotation)
    assert fmt.rotation_dim() == expected_rot_dim


@pytest.mark.parametrize(
    "rotation,include_gripper,expected_per_arm",
    [
        ("quat", True, 3 + 4 + 1),
        ("quat", False, 3 + 4),
        ("euler", True, 3 + 3 + 1),
        ("rot6d", True, 3 + 6 + 1),
        ("rot6d", False, 3 + 6),
    ],
)
def test_per_arm_dim(rotation, include_gripper, expected_per_arm):
    fmt = EEFPose(rotation=rotation, include_gripper=include_gripper)
    assert fmt.per_arm_dim() == expected_per_arm


@pytest.mark.parametrize("n_arms", [1, 2, 3])
def test_n_arms(n_arms):
    assert EEFPose(n_arms=n_arms).n_arms == n_arms


def test_total_dim_dual_quat_is_16():
    fmt = EEFPose(n_arms=2, rotation="quat", include_gripper=True)
    assert fmt.total_dim() == 16


def test_total_dim_single_quat_is_8():
    fmt = EEFPose(n_arms=1, rotation="quat", include_gripper=True)
    assert fmt.total_dim() == 8


def test_total_dim_dual_euler_no_gripper():
    fmt = EEFPose(n_arms=2, rotation="euler", include_gripper=False)
    assert fmt.total_dim() == (3 + 3) * 2


def test_default_format_is_dual_quat_with_gripper():
    fmt = EEFPose()
    assert fmt.n_arms == 2
    assert fmt.rotation == "quat"
    assert fmt.include_gripper is True
    assert fmt.total_dim() == 16


def test_total_dim_consistency():
    """total_dim must equal per_arm_dim * n_arms for any combination."""
    for n_arms in (1, 2):
        for rotation in ("quat", "euler", "rotvec", "rot6d"):
            for grip in (True, False):
                fmt = EEFPose(n_arms=n_arms, rotation=rotation, include_gripper=grip)
                assert fmt.total_dim() == fmt.per_arm_dim() * fmt.n_arms


def test_normalize_eef_chunk_euler_to_quat_nonzero_yaw():
    fmt = EEFPose(n_arms=1, rotation="euler", include_gripper=True)
    row = np.array([0.1, 0.2, 0.3, 0.0, 0.0, math.pi / 2, 0.9], dtype=np.float32)
    out = fmt.normalize_chunk_to_canonical(row[np.newaxis, :])
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0, :3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(out[0, 3:7], [0.7071068, 0.0, 0.0, 0.7071068], atol=1e-6)
    assert out[0, 7] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "rotation,expected_order",
    [
        ("quat", "wxyz"),
        ("euler", "xyz"),
        ("rot6d", None),
        ("rotvec", None),
    ],
)
def test_rot_order_default(rotation, expected_order):
    fmt = EEFPose(rotation=rotation)
    assert fmt.rot_order() == expected_order


def test_rotation_order_override():
    fmt = EEFPose(rotation="euler", rotation_order="zyx", degrees=True)
    assert fmt.rotation == "euler"
    assert fmt.rot_order() == "zyx"
    assert fmt.degrees is True


def test_quat_order_override_xyzw():
    fmt = EEFPose(rotation="quat", rotation_order="xyzw")
    assert fmt.rot_order() == "xyzw"


def test_unsupported_rotation_raises():
    fmt = EEFPose(rotation="nope")
    with pytest.raises(ValueError):
        fmt.rotation_dim()


def test_total_dim_dual_rotvec():
    fmt = EEFPose(n_arms=2, rotation="rotvec", include_gripper=True)
    assert fmt.total_dim() == (3 + 3 + 1) * 2
