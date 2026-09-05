"""Tests for action processing utilities (core.app.handlers.imaging)."""

from __future__ import annotations

import numpy as np
import pytest

from core.app.handlers.imaging import (
    snap_gripper_dims,
)
from core.app.handlers.space import EEFPose

pytestmark = pytest.mark.unit


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


def test_eef_canonical_rot6d_to_quat_single_arm():
    fmt = EEFPose(n_arms=1, rotation="rot6d", include_gripper=True)
    # identity rotation as rot6d (first two columns of identity matrix)
    row = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.9], dtype=np.float32)
    out = fmt.normalize_chunk_to_canonical(row[np.newaxis, :])
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0, :3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(out[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert out[0, 7] == pytest.approx(0.9)


def test_eef_canonical_euler_degrees_zyx_single_arm():
    fmt = EEFPose(n_arms=1, rotation="euler", rotation_order="zyx", degrees=True)
    row = np.array([0.1, 0.2, 0.3, 90.0, 0.0, 0.0, 0.9], dtype=np.float32)
    out = fmt.normalize_chunk_to_canonical(row[np.newaxis, :])
    np.testing.assert_allclose(out[0, 3:7], [0.7071068, 0.0, 0.0, 0.7071068], atol=1e-6)
