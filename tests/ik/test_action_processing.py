"""Tests for action processing utilities (core.app.handlers.imaging)."""

from __future__ import annotations

import numpy as np
import pytest

from core.app.handlers.imaging import (
    build_linear_trajectory,
    normalize_action_chunk,
    prepare_image,
    resize_with_pad,
    snap_gripper_dims,
)
from core.app.handlers.space import EEFPose

# --- normalize_action_chunk ---


def test_normalize_1d_to_2d():
    chunk = normalize_action_chunk(np.arange(7.0))
    assert chunk.shape == (1, 7)
    assert chunk.dtype == np.float32


def test_normalize_2d_passthrough():
    chunk = normalize_action_chunk(np.zeros((5, 7)))
    assert chunk.shape == (5, 7)


def test_normalize_3d_raises():
    with pytest.raises(ValueError):
        normalize_action_chunk(np.zeros((2, 3, 4)))


def test_normalize_returns_writable_copy_of_readonly_input():
    # The policy socket can hand back a read-only buffer (msgpack-backed); the chunk is
    # mutated in place downstream (gripper snapping), so normalize must return a writable copy.
    source = np.zeros((5, 7), dtype=np.float32)
    source.setflags(write=False)
    chunk = normalize_action_chunk(source)
    chunk[0, 0] = 1.0  # would raise "assignment destination is read-only" if a view
    assert chunk[0, 0] == 1.0
    assert source[0, 0] == 0.0


def test_snap_gripper_dims_uses_configured_open_close_values():
    action = np.array([25.0, 75.0], dtype=np.float32)
    out = snap_gripper_dims(
        action.copy(),
        (True, True),
        threshold=50.0,
        open_value=100.0,
        close_value=0.0,
    )
    np.testing.assert_allclose(out, [0.0, 100.0], atol=1e-6)


# --- build_linear_trajectory ---


def test_linear_trajectory_endpoints():
    traj = build_linear_trajectory(np.zeros(3), np.array([1.0, 2.0, 3.0]), steps=5)
    assert traj.shape == (5, 3)
    np.testing.assert_allclose(traj[0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(traj[-1], [1.0, 2.0, 3.0], atol=1e-6)


def test_linear_trajectory_min_two_steps():
    traj = build_linear_trajectory(np.zeros(2), np.ones(2), steps=1)
    assert traj.shape[0] == 2


def test_linear_trajectory_shape_mismatch_raises():
    with pytest.raises(ValueError):
        build_linear_trajectory(np.zeros(3), np.zeros(4), steps=3)


# --- normalize_eef_chunk_to_canonical ---


def test_eef_canonical_quat_passthrough():
    fmt = EEFPose(n_arms=2, rotation="quat", include_gripper=True)
    chunk = np.zeros((2, fmt.total_dim()), dtype=np.float32)
    out = fmt.normalize_chunk_to_canonical(chunk)
    assert out.shape == (2, fmt.total_dim())


def test_eef_canonical_dim_too_small_raises():
    fmt = EEFPose(n_arms=2, rotation="quat", include_gripper=True)
    with pytest.raises(ValueError):
        fmt.normalize_chunk_to_canonical(np.zeros((1, 3)))


def test_eef_canonical_euler_to_quat_single_arm():
    fmt = EEFPose(n_arms=1, rotation="euler", include_gripper=True)
    # per_arm = 3(xyz)+3(euler)+1(grip) = 7
    row = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    out = fmt.normalize_chunk_to_canonical(row[np.newaxis, :])
    # canonical single-arm quat dim = 3+4+1 = 8
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0, :3], [0.1, 0.2, 0.3], atol=1e-6)
    # identity euler -> identity quat (wxyz)
    np.testing.assert_allclose(out[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert out[0, 7] == pytest.approx(0.9)


def test_eef_canonical_rot6d_to_quat_single_arm():
    fmt = EEFPose(n_arms=1, rotation="rot6d", include_gripper=True)
    # identity rotation as rot6d (first two columns of identity matrix)
    row = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.9], dtype=np.float32)
    out = fmt.normalize_chunk_to_canonical(row[np.newaxis, :])
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0, :3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(out[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert out[0, 7] == pytest.approx(0.9)


def test_eef_canonical_rotvec_to_quat_single_arm():
    fmt = EEFPose(n_arms=1, rotation="rotvec", include_gripper=True)
    # 90deg about z as a rotation vector
    row = np.array([0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.9], dtype=np.float32)
    out = fmt.normalize_chunk_to_canonical(row[np.newaxis, :])
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0, 3:7], [0.7071068, 0.0, 0.0, 0.7071068], atol=1e-6)


def test_eef_canonical_quat_xyzw_to_wxyz_single_arm():
    fmt = EEFPose(n_arms=1, rotation="quat", rotation_order="xyzw")
    # 90deg about z in xyzw order
    row = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.7071068, 0.7071068, 0.9], dtype=np.float32)
    out = fmt.normalize_chunk_to_canonical(row[np.newaxis, :])
    np.testing.assert_allclose(out[0, 3:7], [0.7071068, 0.0, 0.0, 0.7071068], atol=1e-6)


def test_eef_canonical_euler_degrees_zyx_single_arm():
    fmt = EEFPose(n_arms=1, rotation="euler", rotation_order="zyx", degrees=True)
    row = np.array([0.1, 0.2, 0.3, 90.0, 0.0, 0.0, 0.9], dtype=np.float32)
    out = fmt.normalize_chunk_to_canonical(row[np.newaxis, :])
    np.testing.assert_allclose(out[0, 3:7], [0.7071068, 0.0, 0.0, 0.7071068], atol=1e-6)


def test_linear_trajectory_interior_values():
    traj = build_linear_trajectory(np.array([0.0, 0.0]), np.array([4.0, 8.0]), steps=5)
    np.testing.assert_allclose(
        traj, [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]], atol=1e-6
    )


def test_resize_with_pad_non_square_centers_vertically():
    img = np.full((4, 8, 3), 100, dtype=np.uint8)
    out = resize_with_pad(img, 8, 8)
    assert out.shape == (8, 8, 3)
    assert np.all(out[:2] == 0)
    assert np.all(out[6:] == 0)
    assert np.all(out[2:6] == 100)
    assert not np.any(np.all(out == 0, axis=(0, 2)))  # no fully-zero column


def test_prepare_image_grayscale_expand_chw():
    out = prepare_image(
        np.full((4, 4), 50, dtype=np.uint8),
        False,
        4,
        4,
        resize_pad=False,
        image_layout="chw",
    )
    assert out.shape == (3, 4, 4)
    assert np.all(out == 50)


def test_prepare_image_bgr_to_rgb_hwc():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 10  # B
    img[..., 2] = 30  # R
    out = prepare_image(img, True, 2, 2, resize_pad=False, image_layout="hwc")
    np.testing.assert_array_equal(out[0, 0], [30, 0, 10])


def test_prepare_image_bgr_to_rgb_chw():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 10  # B
    img[..., 2] = 30  # R
    out = prepare_image(img, True, 2, 2, resize_pad=False, image_layout="chw")
    assert out.shape == (3, 2, 2)
    assert np.all(out[0] == 30)
