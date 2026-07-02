# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""PyRoki (JAX + jaxls) per-frame IK/FK backend covering every robot.

Forward kinematics uses ``pyroki.Robot.forward_kinematics``; inverse kinematics
solves each action-chunk frame as its own single-config jaxls Levenberg-Marquardt
problem, chained for continuity:

    for each frame t:
      minimize  pose_cost_analytic_jac(q_t, target_t)
              + rest_cost(q_t - q_home)
              + velocity_cost(q_t - q_{t-1})
      subject to limit_constraint

Frames are solved one at a time. The velocity cost ties each frame to the
previous frame's solution (so the trajectory stays continuous and each solve is
warm-started from its predecessor); the first frame uses the measured current
state as its predecessor, so execution begins without a jump. The rest_cost
biases every frame toward the init/home pose to fight null-space drift. There is
no jump rejection or best-effort fallback — an over-tolerance frame is logged,
not rejected.

One ``PyrokiSingleArm`` covers a single arm; ``PyrokiDualArm`` /
``PyrokiSingleArmChunk`` drive one or two arms through the thin
``DualArmChunkSolver`` / ``SingleArmChunkSolver`` orchestration (FK chunk map,
reset/default_seed, tail passthrough). The single-arm core covers every feature
the robot zoo needs:

* **joint_mask freezing** — non-arm actuated joints (gripper fingers, opposite
  arm, torso) are zeroed in the pose Jacobian and pinned at their seed value, a
  soft ``buildReducedModel``.
* **fixed non-zero joints** — declared joints frozen at a given value (r1_lite
  torso, agibot frozen body/head) embedded in the full-actuated seed.
* **reference-frame change-of-basis** — targets/FK expressed in a reference link
  frame: ``T_ref_eef = SE3(fk[ref]).inverse() @ SE3(fk[eef])`` and the inverse
  for IK targets (r1_lite torso_link3, agibot chest body_link5).
* **fixed mount SE3** — a per-arm rigid base transform composed outside FK so two
  arms can share one URDF (dual_franka).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

from robots.kinematics.solver import (
    DualArmChunkSolver,
    IkError,
    SingleArmChunkSolver,
    build_initial_seed,
)

# jaxls logs an INFO block on every problem.analyze(); silence it for streaming.
logging.getLogger("jaxls").setLevel(logging.WARNING)

_logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=("max_iterations",))
def _solve_frame_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_pos: jax.Array,
    joint_mask: jax.Array,
    rest_seed: jax.Array,
    prev_cfg: jax.Array,
    init_cfg: jax.Array,
    pos_weight: float,
    ori_weight: float,
    reg_weight: float,
    vel_weight: float,
    dt: float,
    max_iterations: int,
) -> jax.Array:
    """Solve a single-frame IK config as one jaxls problem.

    Frames are solved one at a time and chained: the LM solve is warm-started
    from ``prev_cfg`` (the previous frame's solution), which is what keeps the
    trajectory continuous. A velocity cost additionally penalizes only
    over-speed steps (|q - prev|/dt beyond the joint velocity limits). The first
    frame passes the measured state as ``prev_cfg``, so execution begins without
    a jump.

    Args:
        robot: PyRoki robot.
        target_link_index: int32 scalar index of the controlled link.
        target_wxyz: target orientation [4] (wxyz), URDF base frame.
        target_pos: target position [3], URDF base frame.
        joint_mask: per-actuated-joint mask [n_act] (1 active, 0 frozen).
        rest_seed: rest_cost target [n_act] (frozen-joint values pinned here).
        prev_cfg: previous frame's full config [n_act] for the velocity cost.
        init_cfg: initial-guess config [n_act] (LM warm start).
        pos_weight / ori_weight / reg_weight / vel_weight: cost weights.
        dt: timestep used by the velocity cost.
        max_iterations: LM/AL iteration cap (static).

    Returns:
        Solved config [n_act].
    """
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_pos
            ),
            target_link_index,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            joint_mask=joint_mask,
        ),
        pk.costs.limit_constraint(robot, joint_var),
        pk.costs.rest_cost(joint_var, rest_seed, reg_weight),
    ]

    @jaxls.Cost.factory(name="limit_velocity")
    def limit_velocity_cost(
        vals: jaxls.VarValues, var: jaxls.Var[jax.Array]
    ) -> jax.Array:
        joint_vel = (vals[var] - prev_cfg) / dt
        residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
        return (residual * vel_weight).flatten()

    costs.append(limit_velocity_cost(joint_var))

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(max_iterations=max_iterations),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(init_cfg)]),
        )
    )
    return sol[joint_var]


class PyrokiSingleArm:
    """Single-arm per-frame IK/FK using PyRoki FK + jaxls LM.

    Exposes ``fk_arm`` (per-frame FK) and ``solve_arm_chunk`` (per-frame IK
    chained by a velocity cost) for the chunk orchestration. Externally the
    arm has ``n_arm`` solved joints (+1 gripper); internally the PyRoki robot keeps
    all actuated joints and freezes the non-arm ones via ``joint_mask`` pinned to
    the seed, with declared ``fixed_joints`` pinned at non-zero values.

    EEF poses are expressed in ``reference_frame`` (a URDF link); when None they
    are in the URDF base frame. A fixed ``mount`` SE3 (pos + wxyz) is composed
    outside FK on top of the reference frame, letting two arms share one URDF.
    """

    def __init__(
        self,
        urdf_path: Path,
        arm_joint_names: Sequence[str],
        eef_link_name: str,
        reference_frame: str | None = None,
        fixed_joints: dict[str, float] | None = None,
        mount: tuple[Sequence[float], Sequence[float]] | None = None,
        pos_weight: float = 50.0,
        ori_weight: float = 10.0,
        reg_weight: float = 1.0,
        vel_weight: float = 0.1,
        dt: float = 1.0 / 30.0,
        max_iterations: int = 100,
    ) -> None:
        """Build a single-arm per-frame PyRoki solver.

        Args:
            urdf_path: path to the robot URDF (loaded with build_scene_graph=True).
            arm_joint_names: actuated joints actively solved, in order [n_arm].
            eef_link_name: URDF link whose pose is the end-effector.
            reference_frame: URDF link the EEF pose is expressed relative to;
                None = URDF base frame.
            fixed_joints: actuated joint name -> pinned value for frozen non-arm
                joints (e.g. a torso held at a non-zero pose); others freeze at 0.
            mount: (pos[3], wxyz[4]) rigid base transform applied on top of the
                reference frame, outside FK; None = identity.
            pos_weight / ori_weight / reg_weight: per-frame pose and rest weights.
            vel_weight: velocity-cost weight tying each frame to the previous one
                (continuity); higher = smoother, lower tracking precision.
            dt: timestep used by the velocity cost.
            max_iterations: LM/AL iteration cap per frame solve.
        """
        urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False)
        self._robot = pk.Robot.from_urdf(urdf)
        actuated = list(self._robot.joints.actuated_names)
        self._n_act = self._robot.joints.num_actuated_joints
        self._arm_cols = np.array([actuated.index(n) for n in arm_joint_names], dtype=np.int64)
        self._n_arm = len(self._arm_cols)
        self._eef_idx = jnp.array(self._robot.links.names.index(eef_link_name), dtype=jnp.int32)

        self._reference_frame = reference_frame
        self._ref_idx = (
            jnp.array(self._robot.links.names.index(reference_frame), dtype=jnp.int32)
            if reference_frame is not None
            else None
        )

        mask = np.zeros(self._n_act, dtype=np.float64)
        mask[self._arm_cols] = 1.0
        self._joint_mask = jnp.array(mask)
        self._pos_weight = pos_weight
        self._ori_weight = ori_weight
        self._reg_weight = reg_weight
        self._vel_weight = vel_weight
        self._dt = dt
        self._max_iterations = max_iterations

        lower = np.asarray(self._robot.joints.lower_limits, dtype=np.float64)
        upper = np.asarray(self._robot.joints.upper_limits, dtype=np.float64)
        self._arm_lower = lower[self._arm_cols]
        self._arm_upper = upper[self._arm_cols]

        # Full actuated config holding frozen (non-arm) joints at their reference
        # value: declared fixed_joints at their value, all others at 0.
        self._full_seed = np.zeros(self._n_act, dtype=np.float64)
        for name, value in (fixed_joints or {}).items():
            self._full_seed[actuated.index(name)] = float(value)

        # Mount transform composed outside FK (SE3 mount @ ref_T_eef).
        if mount is not None:
            mount_pos, mount_wxyz = mount
            self._mount = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(jnp.asarray(mount_wxyz)),
                jnp.asarray(mount_pos),
            )
        else:
            self._mount = None

        # Reference link pose is upstream of the arm joints, so it depends only on
        # the frozen config and is constant across frames: evaluate it once.
        if self._ref_idx is not None:
            fk = self._robot.forward_kinematics(jnp.array(self._full_seed))
            self._ref_T = jaxlie.SE3(fk[int(self._ref_idx)])
        else:
            self._ref_T = None

    def _full_config(self, arm_qpos: np.ndarray) -> np.ndarray:
        """Embed arm joints into the full actuated config (frozen joints kept)."""
        full = self._full_seed.copy()
        full[self._arm_cols] = arm_qpos
        return full

    def _arm_pose_se3(self, arm_qpos: np.ndarray) -> jaxlie.SE3:
        """EEF SE3 in the reference frame (with mount) for arm joints [n_arm]."""
        full = self._full_config(arm_qpos)
        fk = self._robot.forward_kinematics(jnp.array(full))
        eef = jaxlie.SE3(fk[int(self._eef_idx)])
        if self._ref_idx is not None:
            eef = jaxlie.SE3(fk[int(self._ref_idx)]).inverse() @ eef
        if self._mount is not None:
            eef = self._mount @ eef
        return eef

    def _targets_to_base(
        self, target_wxyz: np.ndarray, target_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map reference-frame (+mount) EEF targets to the URDF base frame.

        Args:
            target_wxyz: per-frame target orientation [T, 4] (wxyz).
            target_pos: per-frame target position [T, 3].

        Returns:
            (base_wxyz [T, 4], base_pos [T, 3]) in the URDF base frame.
        """
        target = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.asarray(target_wxyz)),
            jnp.asarray(target_pos),
        )
        if self._mount is not None:
            target = self._mount.inverse() @ target
        if self._ref_T is not None:
            target = self._ref_T @ target
        base_wxyz = np.asarray(target.rotation().wxyz, dtype=np.float64)
        base_pos = np.asarray(target.translation(), dtype=np.float64)
        return base_wxyz, base_pos

    def fk_arm(self, qpos: Sequence[float]) -> np.ndarray:
        """Arm qpos (n_arm joints + 1 gripper) -> 8D EEF [x,y,z, qw,qx,qy,qz, grip]."""
        v = np.asarray(qpos, dtype=np.float64)
        if v.shape[0] != self._n_arm + 1:
            raise IkError(f"Expected {self._n_arm + 1}D arm qpos, got {v.shape}")
        pose = self._arm_pose_se3(v[: self._n_arm])
        quat_wxyz = np.asarray(pose.rotation().wxyz, dtype=np.float64)
        trans = np.asarray(pose.translation(), dtype=np.float64)
        return np.array(
            [
                float(trans[0]),
                float(trans[1]),
                float(trans[2]),
                float(quat_wxyz[0]),
                float(quat_wxyz[1]),
                float(quat_wxyz[2]),
                float(quat_wxyz[3]),
                float(v[self._n_arm]),
            ],
            dtype=np.float64,
        )

    def solve_arm_chunk(
        self,
        eef_chunk: np.ndarray,
        start_arm: np.ndarray,
        rest_arm: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve an arm trajectory one frame at a time, chained for continuity.

        Each frame is a separate single-config IK solve whose velocity cost ties
        it to the previous frame's solution (warm start + continuity). The first
        frame uses the measured ``start_arm`` as its predecessor, so execution
        begins without a jump. The rest_cost biases every frame toward ``rest_arm``
        (init/home) to fight null-space drift.

        Args:
            eef_chunk: EEF targets [T, 8] ([x,y,z, qw,qx,qy,qz, grip]) in the
                reference (+mount) frame; the gripper column is ignored here.
            start_arm: measured current arm joints [n_arm], the first frame's predecessor.
            rest_arm: arm joints [n_arm] the rest_cost biases toward (the init/home
                pose); None falls back to ``start_arm`` (no homing).

        Returns:
            Solved arm joint trajectory [T, n_arm]. Frames whose tracking error
            exceeds tolerance are logged (not rejected).
        """
        chunk = np.asarray(eef_chunk, dtype=np.float64)
        n_frames = chunk.shape[0]
        base_wxyz, base_pos = self._targets_to_base(chunk[:, 3:7], chunk[:, :3])

        start_full = self._full_config(np.clip(start_arm, self._arm_lower, self._arm_upper))
        if rest_arm is None:
            rest_full = start_full
        else:
            rest_full = self._full_config(np.clip(rest_arm, self._arm_lower, self._arm_upper))

        sol = np.empty((n_frames, self._n_act), dtype=np.float64)
        prev_cfg = start_full
        t_start = time.perf_counter()
        for t in range(n_frames):
            t0 = time.perf_counter()
            cfg = np.asarray(
                _solve_frame_jax(
                    self._robot,
                    self._eef_idx,
                    jnp.array(base_wxyz[t]),
                    jnp.array(base_pos[t]),
                    self._joint_mask,
                    jnp.array(rest_full),
                    jnp.array(prev_cfg),
                    jnp.array(prev_cfg),
                    self._pos_weight,
                    self._ori_weight,
                    self._reg_weight,
                    self._vel_weight,
                    self._dt,
                    self._max_iterations,
                )
            )
            _logger.info(
                "ik frame %d/%d solved in %.1f ms", t + 1, n_frames,
                (time.perf_counter() - t0) * 1e3,
            )
            sol[t] = cfg
            prev_cfg = cfg
        _logger.info(
            "ik chunk: %d frames in %.1f ms", n_frames,
            (time.perf_counter() - t_start) * 1e3,
        )
        return sol[:, self._arm_cols]


class PyrokiDualArm(DualArmChunkSolver):
    """Dual-arm whole-chunk kinematics for EEF control (IK) + FK, PyRoki-backed.

    Two ``PyrokiSingleArm`` solvers (built by ``single_arm_factory``, called once
    per arm with the arm index so per-arm config like end_link/mount can differ)
    wrapped by the thin ``DualArmChunkSolver`` orchestration (FK chunk, reset,
    tail passthrough). Set ``carry_full_state`` for whole-body robots that keep a
    wider seed than the two arms (e.g. agibot's 24-DoF state with frozen
    head/body); FK then re-appends the trailing body columns and IK preserves them.
    """

    def __init__(
        self,
        initial_qpos_groups: Sequence[Sequence[float]],
        single_arm_factory: Callable[[int], PyrokiSingleArm],
        arm_width: int | None = None,
        carry_full_state: bool = False,
        frame_name: str | None = None,
        dt: float = 1.0 / 30.0,
        position_tolerance: float = 5e-3,
        orientation_tolerance: float = 5e-2,
    ) -> None:
        """Build the dual-arm PyRoki solver.

        Args:
            initial_qpos_groups: per-actuator-group initial joint values; flattened
                into the default seed.
            single_arm_factory: builds one ``PyrokiSingleArm`` given the arm index
                (0 = left, 1 = right).
            arm_width: per-arm joints + gripper; defaults to the left arm's
                ``n_arm + 1`` (override only for ``carry_full_state`` robots whose
                arm group width differs from the solved-joint count).
            carry_full_state: keep a full-body seed and update only the arm slices.
            frame_name: unused (reference frame is baked into each arm at build).
            dt: unused (kept for orchestration-signature compatibility).
            position_tolerance / orientation_tolerance: over-tolerance logging gate.
        """
        del frame_name
        left_solver = single_arm_factory(0)
        right_solver = single_arm_factory(1)
        width = arm_width if arm_width is not None else left_solver._n_arm + 1
        super().__init__(
            left_solver,
            right_solver,
            width,
            build_initial_seed(initial_qpos_groups),
            dt,
            position_tolerance,
            orientation_tolerance,
            carry_full_state=carry_full_state,
        )


class PyrokiSingleArmChunk(SingleArmChunkSolver):
    """Single-arm whole-chunk kinematics for EEF control (IK) + FK, PyRoki-backed.

    One ``PyrokiSingleArm`` (built by ``single_arm_factory``) wrapped by the thin
    ``SingleArmChunkSolver`` orchestration (FK chunk, reset, tail passthrough).
    """

    def __init__(
        self,
        initial_qpos_groups: Sequence[Sequence[float]],
        single_arm_factory: Callable[[], PyrokiSingleArm],
        frame_name: str | None = None,
        dt: float = 1.0 / 30.0,
        position_tolerance: float = 5e-3,
        orientation_tolerance: float = 5e-2,
    ) -> None:
        """Build the single-arm PyRoki solver.

        Args:
            initial_qpos_groups: per-actuator-group initial joint values (one group
                for a single arm); flattened into the default seed.
            single_arm_factory: builds the ``PyrokiSingleArm``.
            frame_name: unused (reference frame is baked into the arm at build).
            dt: unused (kept for orchestration-signature compatibility).
            position_tolerance / orientation_tolerance: over-tolerance logging gate.
        """
        del frame_name
        solver = single_arm_factory()
        super().__init__(
            solver,
            solver._n_arm + 1,
            build_initial_seed(initial_qpos_groups),
            dt,
            position_tolerance,
            orientation_tolerance,
        )


def pyroki_arms(
    urdf_path: Path,
    arms: list[dict[str, Any]],
    *,
    fixed_joints: dict[str, float] | None = None,
    default_reference_frame: str | None = None,
    supported_reference_frames: tuple[str, ...] = (),
    **kwargs: Any,
) -> DualArmChunkSolver | SingleArmChunkSolver:
    """Build a shared-PyRoki solver from a declarative arm spec (1 arm -> single, 2 -> dual).

    The common kinematics case, so a robot's ``build_kinematics`` is one call. Each arm
    is a ``PyrokiSingleArm`` over its declared chain; both arms may share one URDF.

    Args:
        urdf_path: absolute path to the robot's URDF.
        arms: one dict per arm (1 = single, 2 = dual), each with ``joints`` (ordered
            actuated joints solved), ``eef_link`` (the EEF link), and optional
            ``reference_frame``.
        fixed_joints: actuated joint name -> pinned value, applied to every arm.
        default_reference_frame: reference frame used when the runtime kwarg is unset,
            active only when ``supported_reference_frames`` is non-empty.
        supported_reference_frames: when non-empty, the runtime ``reference_frame`` kwarg
            overrides each arm's declared frame (validated against this set).
        **kwargs: runtime solver params forwarded to the orchestrator — ``dt``,
            ``initial_qpos_groups``, and a popped ``reference_frame``/``frame_name``.

    Returns:
        A PyrokiDualArm (2 arms) or PyrokiSingleArmChunk (1 arm).
    """
    fixed = {k: float(v) for k, v in (fixed_joints or {}).items()}
    kwargs.pop("frame_name", None)
    reference_frame = kwargs.pop("reference_frame", None)

    def resolve_frame(arm_default: str | None) -> str | None:
        if not supported_reference_frames:
            return arm_default
        rf = reference_frame or default_reference_frame
        if rf not in supported_reference_frames:
            allowed = ", ".join(sorted(supported_reference_frames))
            raise ValueError(f"Unsupported EEF reference frame {rf!r}; expected: {allowed}")
        return rf

    def make_arm(arm_cfg: dict[str, Any]) -> PyrokiSingleArm:
        return PyrokiSingleArm(
            urdf_path,
            tuple(arm_cfg["joints"]),
            arm_cfg["eef_link"],
            reference_frame=resolve_frame(arm_cfg.get("reference_frame")),
            fixed_joints=fixed,
        )

    if len(arms) == 2:
        return PyrokiDualArm(single_arm_factory=lambda i: make_arm(arms[i]), **kwargs)
    return PyrokiSingleArmChunk(single_arm_factory=lambda: make_arm(arms[0]), **kwargs)
