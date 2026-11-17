"""Tests for action processing utilities (core.app.handlers.imaging)."""

from __future__ import annotations

import numpy as np
import pytest

from core.app.handlers.imaging import (
    prepare_image,
    snap_gripper_dims,
)
from core.app.handlers.space import EEFPose

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("resize_pad", [False, True])
def test_prepare_image_preserves_original_pixels_by_default(resize_pad, monkeypatch):
    image = np.arange(23 * 37 * 3, dtype=np.uint8).reshape(23, 37, 3)

    def unexpected_resize(*_args):
        pytest.fail("Raw policy images must not enter a resize operation")

    monkeypatch.setattr("core.app.handlers.imaging.resize_direct", unexpected_resize)
    monkeypatch.setattr("core.app.handlers.imaging.resize_with_pad", unexpected_resize)
    output = prepare_image(image, False, 224, 224, resize_pad)
    np.testing.assert_array_equal(output, image)
    assert output.dtype == np.uint8


@pytest.mark.parametrize("resize_pad", [False, True])
def test_prepare_image_resizes_when_explicitly_enabled(resize_pad):
    image = np.full((23, 37, 3), 100, dtype=np.uint8)
    output = prepare_image(image, False, 8, 10, resize_pad, resize=True)
    assert output.shape == (8, 10, 3)
    assert output.dtype == np.uint8


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
