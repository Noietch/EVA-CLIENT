"""Scene extrinsic solver correctness."""

from __future__ import annotations

import numpy as np

from tools.calibration import _average_se3, solve_scene_extrinsic
from tools.calibration import invert_se3, rvec_tvec_to_se3, so3_log


def _rotation_diff_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(so3_log(R1.T @ R2))))


def test_scene_extrinsic_recovers_identity_when_all_same():
    T = rvec_tvec_to_se3(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.0, 0.4]))
    poses = tuple(T.copy() for _ in range(5))
    result = solve_scene_extrinsic(poses)
    expected = invert_se3(T)
    assert np.allclose(result, expected, atol=1e-6)


def test_scene_extrinsic_averages_noisy_poses():
    rng = np.random.default_rng(0)
    T_true = rvec_tvec_to_se3(np.array([0.1, 0.0, 0.05]), np.array([0.0, 0.0, 0.5]))
    poses: list[np.ndarray] = []
    for _ in range(20):
        drvec = rng.normal(scale=0.02, size=3)
        dt = rng.normal(scale=0.005, size=3)
        T_noise = rvec_tvec_to_se3(drvec, dt)
        poses.append(T_true @ T_noise)
    result = solve_scene_extrinsic(tuple(poses))
    expected = invert_se3(T_true)
    assert np.allclose(result[:3, 3], expected[:3, 3], atol=0.01)
    assert _rotation_diff_deg(result[:3, :3], expected[:3, :3]) < 3.0


def test_average_se3_single_matrix_passthrough():
    T = rvec_tvec_to_se3(np.array([0.3, 0.1, 0.2]), np.array([0.1, 0.2, 0.3]))
    result = _average_se3((T,))
    assert np.allclose(result, T)
