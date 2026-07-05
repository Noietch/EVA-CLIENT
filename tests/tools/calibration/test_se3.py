"""Tests for tools.calibration — tiny SE(3) helpers."""

from __future__ import annotations

import numpy as np

from tools.calibration import (
    compose_se3,
    invert_se3,
    rvec_tvec_to_se3,
    so3_exp,
    so3_log,
)


def _random_rvec(rng: np.random.Generator) -> np.ndarray:
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    return axis * rng.uniform(0.05, 1.5)


def test_invert_compose_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        A = rvec_tvec_to_se3(_random_rvec(rng), rng.uniform(-1, 1, size=3))
        B = rvec_tvec_to_se3(_random_rvec(rng), rng.uniform(-1, 1, size=3))
        AB = compose_se3(A, B)
        expected = invert_se3(B) @ invert_se3(A)
        assert np.allclose(invert_se3(AB), expected, atol=1e-9)


def test_so3_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        rvec = _random_rvec(rng)
        R = so3_exp(rvec)
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-9)


def test_so3_log_zero_rotation():
    assert np.allclose(so3_log(np.eye(3)), np.zeros(3), atol=1e-12)


def test_so3_log_near_pi():
    rvec = np.array([0.0, 0.0, np.pi - 1e-4])
    R = so3_exp(rvec)
    recovered = so3_log(R)
    assert np.isclose(np.linalg.norm(recovered), np.pi - 1e-4, atol=1e-3)


def test_rvec_tvec_to_se3_matches_manual_assembly():
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([0.5, -0.1, 0.7])
    T = rvec_tvec_to_se3(rvec, tvec)
    R = so3_exp(rvec)
    assert np.allclose(T[:3, :3], R)
    assert np.allclose(T[:3, 3], tvec)
    assert T[3, 3] == 1.0
