"""Solver correctness against a synthetic pinhole camera + ChArUco board."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tools.calibration import CharucoBoardSpec, build_cv_board
from tools.calibration import (
    CameraSample,
    package_sdk_intrinsics,
    solve_intrinsics,
)


def _spec() -> CharucoBoardSpec:
    return CharucoBoardSpec(cols=5, rows=7, square_length=0.03, marker_length=0.023)


def _synthetic_samples(
    spec: CharucoBoardSpec,
    K: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
    n_frames: int,
    rng: np.random.Generator,
) -> tuple[CameraSample, ...]:
    """Project the board's chessboard corners through a virtual camera at N poses."""
    board = build_cv_board(spec)
    objp_all = np.asarray(board.getChessboardCorners(), dtype=np.float64)  # (Nc, 3)
    w, h = image_size
    samples: list[CameraSample] = []
    for _ in range(n_frames):
        rvec = rng.uniform(-0.4, 0.4, size=3).astype(np.float64)
        tvec = np.array([
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.05, 0.05),
            rng.uniform(0.4, 0.7),
        ], dtype=np.float64)
        imgp, _ = cv2.projectPoints(objp_all, rvec, tvec, K, dist)
        imgp = np.asarray(imgp, dtype=np.float64).reshape(-1, 2)
        in_bounds = (imgp[:, 0] > 5) & (imgp[:, 0] < w - 5) & (imgp[:, 1] > 5) & (imgp[:, 1] < h - 5)
        if in_bounds.sum() < 8:
            continue
        samples.append(
            CameraSample(
                image_shape=(h, w),
                objp=objp_all[in_bounds].astype(np.float64),
                imgp=imgp[in_bounds],
                thumbnail_jpeg=b"",
            )
        )
    return tuple(samples)


def test_solve_intrinsics_recovers_ground_truth():
    spec = _spec()
    K_gt = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    dist_gt = np.zeros(5)
    rng = np.random.default_rng(42)
    samples = _synthetic_samples(spec, K_gt, dist_gt, (640, 480), 15, rng)
    assert len(samples) >= 8
    result = solve_intrinsics(samples)
    assert result.rms < 0.5, f"expected sub-pixel RMS, got {result.rms}"
    fx, fy = result.K[0, 0], result.K[1, 1]
    cx, cy = result.K[0, 2], result.K[1, 2]
    assert abs(fx - 600) / 600 < 0.02
    assert abs(fy - 600) / 600 < 0.02
    assert abs(cx - 320) / 320 < 0.02
    assert abs(cy - 240) / 240 < 0.02
    assert np.allclose(result.dist, 0, atol=5e-3)
    assert result.method == "solve"
    assert result.image_size == (640, 480)


def test_solve_intrinsics_rejects_too_few_frames():
    spec = _spec()
    K_gt = np.eye(3); K_gt[0, 0] = K_gt[1, 1] = 600; K_gt[0, 2] = 320; K_gt[1, 2] = 240
    dist_gt = np.zeros(5)
    samples = _synthetic_samples(spec, K_gt, dist_gt, (640, 480), 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least 4 frames"):
        solve_intrinsics(samples)


def test_package_sdk_intrinsics_sets_sdk_method():
    K = np.eye(3); K[0, 0] = K[1, 1] = 700
    dist = np.zeros(5)
    result = package_sdk_intrinsics(K, dist, (1280, 720))
    assert result.method == "sdk"
    assert result.n_frames == 0
    assert result.image_size == (1280, 720)
    assert np.allclose(result.K, K)
