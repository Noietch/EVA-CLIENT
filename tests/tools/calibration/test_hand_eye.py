"""Hand-eye solver correctness on a synthetic AX=XB dataset."""

from __future__ import annotations

import numpy as np
import pytest

from tools.calibration import PoseSample, invert_se3, rvec_tvec_to_se3, so3_log, solve_hand_eye


def _random_se3(rng: np.random.Generator, trans_scale: float = 0.4) -> np.ndarray:
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    rvec = axis * rng.uniform(0.1, 1.0)
    tvec = rng.uniform(-trans_scale, trans_scale, size=3)
    return rvec_tvec_to_se3(rvec, tvec)


def _synth_pairs(
    T_cam_gripper: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[PoseSample, ...]:
    """AX=XB from a ground-truth T_cam_gripper: T_target_cam = X^-1 A^-1 B_common."""
    # Fix the board in world space; T_target_base = identity.
    T_target_base = np.eye(4)
    pairs: list[PoseSample] = []
    for _ in range(n):
        T_gripper_base = _random_se3(rng)
        # camera in base = T_gripper_base @ T_cam_gripper
        T_cam_base = T_gripper_base @ T_cam_gripper
        T_target_cam = invert_se3(T_cam_base) @ T_target_base
        pairs.append(
            PoseSample(
                T_target_cam=T_target_cam,
                T_gripper_base=T_gripper_base,
                camera_key="c",
                robot_link="ee",
                ts=0.0,
            )
        )
    return tuple(pairs)


def _rotation_diff_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(so3_log(R1.T @ R2))))


@pytest.mark.parametrize("method", ["TSAI", "PARK", "HORAUD", "DANIILIDIS"])
def test_hand_eye_recovers_ground_truth(method: str):
    rng = np.random.default_rng(123)
    T_gt = rvec_tvec_to_se3(np.array([0.1, -0.2, 0.3]), np.array([0.05, 0.02, 0.1]))
    pairs = _synth_pairs(T_gt, 12, rng)
    result = solve_hand_eye(pairs, method=method)
    assert result.method == method
    assert result.n_pairs == 12
    trans_err = float(np.linalg.norm(result.T_cam_gripper[:3, 3] - T_gt[:3, 3]))
    rot_err = _rotation_diff_deg(result.T_cam_gripper[:3, :3], T_gt[:3, :3])
    assert trans_err < 1e-3, f"{method}: trans err {trans_err} m"
    assert rot_err < 0.5, f"{method}: rot err {rot_err} deg"
    assert result.translation_rms_m < 5e-3
    assert result.rotation_rms_deg < 1.0


def test_hand_eye_too_few_pairs_raises():
    T_gt = np.eye(4)
    pairs = _synth_pairs(T_gt, 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least 3 pose pairs"):
        solve_hand_eye(pairs)


def test_hand_eye_falls_back_when_method_missing(monkeypatch):
    import tools.calibration as HE

    real = HE._method_flags

    def fake_flags():
        flags = real()
        flags.pop("TSAI", None)
        return flags

    monkeypatch.setattr(HE, "_method_flags", fake_flags)
    T_gt = rvec_tvec_to_se3(np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.1]))
    pairs = _synth_pairs(T_gt, 6, np.random.default_rng(0))
    result = solve_hand_eye(pairs, method="TSAI")
    assert result.method == "PARK"
