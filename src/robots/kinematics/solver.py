# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""Backend-agnostic chunk orchestration over whole-chunk per-arm solvers.

Wraps a per-arm whole-chunk solver into the ``KinematicsSolver`` protocol: the
FK joint->EEF chunk map, a lock guarding the seed state, reset/default_seed/close,
tail passthrough, and (dual-arm) splitting an EEF chunk across two arms. The
per-arm solver only needs to expose

    fk_arm(qpos[n_arm + 1]) -> eef[8]
    solve_arm_chunk(eef_chunk[T, 8], start_arm[n_arm], rest_arm[n_arm]) -> arm[T, n_arm]

IK solves the WHOLE chunk jointly (the per-arm solver enforces smoothness and a
frame-0 start anchor), so there is no per-frame loop, jump rejection, or
best-effort fallback here. ``DualArmChunkSolver`` drives two such solvers;
``SingleArmChunkSolver`` drives one. Both satisfy ``robots.base.KinematicsSolver``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


class IkError(RuntimeError):
    """IK solver did not converge.

    When *result* is not ``None`` it carries the best-effort joint configuration
    the solver reached before giving up, so callers can fall back to it instead
    of snapping to a default pose.
    """

    result: np.ndarray | None

    def __init__(self, msg: str, result: np.ndarray | None = None) -> None:
        super().__init__(msg)
        self.result = result


def build_initial_seed(initial_qpos_groups: Sequence[Sequence[float]]) -> np.ndarray:
    """Concatenate per-group initial qpos into one flat float64 seed vector.

    Args:
        initial_qpos_groups: per-actuator-group initial joint values.

    Returns:
        Flat float64 seed [sum of group dims].
    """
    return np.concatenate([np.asarray(g, dtype=np.float64) for g in initial_qpos_groups])


class DualArmChunkSolver:
    """Dual-arm whole-chunk orchestration over two per-arm chunk solvers.

    Splits an EEF chunk into the two arms, solves each arm's whole trajectory
    jointly (frame 0 anchored to the seed state), and recombines. FK maps a joint
    chunk to an EEF chunk; a lock guards the seed state; reset/default_seed/close.

    The persistent state width may exceed ``2 * arm_width``: when
    ``carry_full_state`` is True the solver keeps a full-body seed (e.g. agibot's
    24-DoF state with frozen head/body), updates only the two arm slices, and FK
    re-appends the trailing body columns; otherwise it tracks exactly the two arms
    and passes any EEF-chunk tail beyond index 16 through unchanged.
    """

    def __init__(
        self,
        left_solver: Any,
        right_solver: Any,
        arm_width: int,
        default_seed: np.ndarray,
        dt: float,
        position_tolerance: float,
        orientation_tolerance: float,
        carry_full_state: bool,
    ) -> None:
        self._left_solver = left_solver
        self._right_solver = right_solver
        self._arm_width = arm_width
        self._joint_dof = arm_width - 1
        self._dt = dt
        self._position_tolerance = position_tolerance
        self._orientation_tolerance = orientation_tolerance
        self._carry_full_state = carry_full_state
        self._default_seed = np.asarray(default_seed, dtype=np.float64)
        self._seed_qpos = self._default_seed.copy()
        self._lock = threading.Lock()

    def fk_chunk(self, qpos_chunk: np.ndarray) -> np.ndarray:
        """Forward kinematics over a dual-arm joint chunk.

        Args:
            qpos_chunk: joint chunk [T, >=2*arm_width] (arm_width per arm); 1D
                input is treated as a single frame. With ``carry_full_state`` any
                trailing columns beyond the two arms are re-appended unchanged.

        Returns:
            EEF chunk, float32 [T, 16] (8D per arm: xyz + quat wxyz + gripper),
            plus the trailing body columns when ``carry_full_state``.
        """
        chunk = np.asarray(qpos_chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        both = 2 * self._arm_width
        if chunk.ndim != 2 or chunk.shape[1] < both:
            raise IkError(f"Expected qpos chunk shaped [T, >={both}], got {chunk.shape}")
        result = []
        for qpos in chunk:
            left = self._left_solver.fk_arm(qpos[: self._arm_width])
            right = self._right_solver.fk_arm(qpos[self._arm_width : both])
            if self._carry_full_state:
                result.append(np.concatenate([left, right, qpos[both:]]))
            else:
                result.append(np.concatenate([left, right]))
        return np.asarray(result, dtype=np.float32)

    def reset(
        self,
        initial_qpos_groups: Sequence[Sequence[float]] | None = None,
        **kwargs: object,
    ) -> None:
        """Reset the seed state to the default (or a new) seed."""
        del kwargs
        with self._lock:
            if initial_qpos_groups is not None:
                self._default_seed = build_initial_seed(initial_qpos_groups)
            self._seed_qpos = self._default_seed.copy()

    def default_seed(self) -> np.ndarray:
        """Return a copy of the default joint-space seed."""
        return self._default_seed.copy()

    def _check_chunk(
        self, solver: Any, arm_traj: np.ndarray, eef_targets: np.ndarray, label: str
    ) -> None:
        """Log frames whose FK tracking error exceeds tolerance (not rejected)."""
        for t, arm in enumerate(arm_traj):
            eef = solver.fk_arm(np.concatenate([arm, [0.0]]))
            pos_err = float(np.linalg.norm(eef[:3] - eef_targets[t, :3]))
            if pos_err >= self._position_tolerance:
                _logger.warning(
                    "%s frame %d tracking pos err %.4f >= tol %.4f",
                    label, t, pos_err, self._position_tolerance,
                )

    def solve_chunk(self, chunk: np.ndarray, seed_qpos: np.ndarray | None = None) -> np.ndarray:
        """Whole-chunk inverse kinematics over a dual-arm EEF chunk.

        Each arm's whole trajectory is solved jointly with smoothness, frame 0
        anchored to the seed state (the measured current joints when ``seed_qpos``
        is given). With ``carry_full_state`` the full-body seed is preserved and
        only the two arm slices are updated; otherwise trailing columns beyond
        index 16 are passed through unchanged.

        Args:
            chunk: EEF chunk [T, >=16] (8D per arm); 1D input is one frame.
            seed_qpos: measured current joint state [state_width] anchoring frame 0;
                falls back to the persistent seed when None.

        Returns:
            Joint chunk, float32 [T, state_width] (with any pass-through tail re-appended).
        """
        normalized = np.asarray(chunk, dtype=np.float64)
        if normalized.ndim == 1:
            normalized = normalized[np.newaxis, :]
        if normalized.ndim != 2 or normalized.shape[1] < 16:
            raise IkError(f"Expected eef chunk shaped [T, >=16], got {normalized.shape}")

        eef_primary = normalized[:, :16]
        tail = (
            normalized[:, 16:]
            if (normalized.shape[1] > 16 and not self._carry_full_state)
            else None
        )
        n_frames = eef_primary.shape[0]
        n = self._arm_width

        with self._lock:
            if seed_qpos is not None:
                self._seed_qpos = np.asarray(seed_qpos, dtype=np.float64).copy()
            start = self._seed_qpos

            left_eef = eef_primary[:, :8]
            right_eef = eef_primary[:, 8:16]
            left_arm = self._left_solver.solve_arm_chunk(
                left_eef, start[: self._joint_dof], self._default_seed[: self._joint_dof]
            )
            right_arm = self._right_solver.solve_arm_chunk(
                right_eef,
                start[n : n + self._joint_dof],
                self._default_seed[n : n + self._joint_dof],
            )
            self._check_chunk(self._left_solver, left_arm, left_eef, "left")
            self._check_chunk(self._right_solver, right_arm, right_eef, "right")

            left = np.concatenate([left_arm, left_eef[:, 7:8]], axis=1)
            right = np.concatenate([right_arm, right_eef[:, 7:8]], axis=1)
            if self._carry_full_state:
                solved = np.tile(self._seed_qpos, (n_frames, 1))
                solved[:, :n] = left
                solved[:, n : 2 * n] = right
            else:
                solved = np.concatenate([left, right], axis=1)
            self._seed_qpos = solved[-1].astype(np.float64).copy()

        solved = solved.astype(np.float32)
        if tail is None:
            return solved
        return np.concatenate([solved, tail.astype(np.float32)], axis=1)

    def close(self) -> None:
        """No-op; the solver holds no resources needing release."""
        return


class SingleArmChunkSolver:
    """Single-arm whole-chunk orchestration over one per-arm chunk solver.

    The single-arm counterpart of ``DualArmChunkSolver``: whole-trajectory IK with
    a frame-0 start anchor, the FK joint->EEF chunk map, a lock guarding the seed
    state, and reset/default_seed/close. Tracks one arm (``arm_width`` = joints +
    gripper) and passes any EEF-chunk tail beyond index 8 through unchanged.
    """

    def __init__(
        self,
        solver: Any,
        arm_width: int,
        default_seed: np.ndarray,
        dt: float,
        position_tolerance: float,
        orientation_tolerance: float,
    ) -> None:
        self._solver = solver
        self._arm_width = arm_width
        self._joint_dof = arm_width - 1
        self._dt = dt
        self._position_tolerance = position_tolerance
        self._orientation_tolerance = orientation_tolerance
        self._default_seed = np.asarray(default_seed, dtype=np.float64)
        self._seed_qpos = self._default_seed.copy()
        self._lock = threading.Lock()

    def fk_chunk(self, qpos_chunk: np.ndarray) -> np.ndarray:
        """Forward kinematics over a single-arm joint chunk.

        Args:
            qpos_chunk: joint chunk [T, >=arm_width]; 1D input is one frame.

        Returns:
            EEF chunk, float32 [T, 8] (xyz + quat wxyz + gripper).
        """
        chunk = np.asarray(qpos_chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        if chunk.ndim != 2 or chunk.shape[1] < self._arm_width:
            raise IkError(f"Expected qpos chunk shaped [T, >={self._arm_width}], got {chunk.shape}")
        rows = [self._solver.fk_arm(qpos[: self._arm_width]) for qpos in chunk]
        return np.asarray(rows, dtype=np.float32)

    def reset(
        self,
        initial_qpos_groups: Sequence[Sequence[float]] | None = None,
        **kwargs: object,
    ) -> None:
        """Reset the seed state to the default (or a new) seed."""
        del kwargs
        with self._lock:
            if initial_qpos_groups is not None:
                self._default_seed = build_initial_seed(initial_qpos_groups)
            self._seed_qpos = self._default_seed.copy()

    def default_seed(self) -> np.ndarray:
        """Return a copy of the default joint-space seed."""
        return self._default_seed.copy()

    def solve_chunk(self, chunk: np.ndarray, seed_qpos: np.ndarray | None = None) -> np.ndarray:
        """Whole-chunk inverse kinematics over a single-arm EEF chunk.

        The whole trajectory is solved jointly with smoothness, frame 0 anchored to
        the seed state (the measured current joints when ``seed_qpos`` is given).
        Trailing columns beyond index 8 are passed through unchanged.

        Args:
            chunk: EEF chunk [T, >=8] (xyz + quat wxyz + gripper); 1D is one frame.
            seed_qpos: measured current joint state [arm_width] anchoring frame 0;
                falls back to the persistent seed when None.

        Returns:
            Joint chunk, float32 [T, arm_width] (with any pass-through tail re-appended).
        """
        normalized = np.asarray(chunk, dtype=np.float64)
        if normalized.ndim == 1:
            normalized = normalized[np.newaxis, :]
        if normalized.ndim != 2 or normalized.shape[1] < 8:
            raise IkError(f"Expected eef chunk shaped [T, >=8], got {normalized.shape}")

        eef_primary = normalized[:, :8]
        tail = normalized[:, 8:] if normalized.shape[1] > 8 else None

        with self._lock:
            if seed_qpos is not None:
                self._seed_qpos = np.asarray(seed_qpos, dtype=np.float64).copy()
            arm_traj = self._solver.solve_arm_chunk(
                eef_primary,
                self._seed_qpos[: self._joint_dof],
                self._default_seed[: self._joint_dof],
            )
            for t, arm in enumerate(arm_traj):
                eef = self._solver.fk_arm(np.concatenate([arm, [0.0]]))
                pos_err = float(np.linalg.norm(eef[:3] - eef_primary[t, :3]))
                if pos_err >= self._position_tolerance:
                    _logger.warning(
                        "frame %d tracking pos err %.4f >= tol %.4f",
                        t, pos_err, self._position_tolerance,
                    )
            solved = np.concatenate([arm_traj, eef_primary[:, 7:8]], axis=1)
            self._seed_qpos = solved[-1].astype(np.float64).copy()

        solved = solved.astype(np.float32)
        if tail is None:
            return solved
        return np.concatenate([solved, tail.astype(np.float32)], axis=1)

    def close(self) -> None:
        """No-op; the solver holds no resources needing release."""
        return
