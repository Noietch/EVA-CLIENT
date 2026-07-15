import numpy as np

from core.utils.smooth_dataset import smooth_arm_qpos


def normalized_acceleration(values: np.ndarray) -> float:
    diff = np.diff(values, axis=0)
    second = diff[1:] - diff[:-1]
    return float(np.linalg.norm(second, axis=1).sum() / np.linalg.norm(diff, axis=1).sum())


def test_smooth_arm_qpos_keeps_grippers_and_reduces_acceleration() -> None:
    base = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    qpos = np.zeros((31, 14), dtype=np.float64)
    qpos[:, 0] = base
    qpos[:, 1] = base**2
    qpos[8, 0] += 0.4
    qpos[16, 1] -= 0.35
    qpos[:, 6] = np.where(np.arange(31) < 15, 0.0, 100.0)
    qpos[:, 13] = 50.0

    smoothed = smooth_arm_qpos(qpos, window=9, polyorder=2)

    assert np.allclose(smoothed[:, 6], qpos[:, 6])
    assert np.allclose(smoothed[:, 13], qpos[:, 13])
    assert np.allclose(smoothed[0], qpos[0])
    assert np.allclose(smoothed[-1], qpos[-1])
    assert normalized_acceleration(smoothed[:, [0, 1]]) < normalized_acceleration(qpos[:, [0, 1]])
