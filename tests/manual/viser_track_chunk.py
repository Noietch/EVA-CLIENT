"""Interactive viser tool: record a dragged EEF trajectory, solve it as one chunk,
replay the solved joint trajectory, and inspect tracking quality.

Unlike the pyroki per-frame-IK demos, the project's chunk solver hard-anchors
frame 0 to the seed state and solves the WHOLE trajectory jointly with smoothness
costs, so live drag-and-follow is meaningless. Instead this tool RECORDS a target
trajectory: tick "Record", drag each arm's transform control around to trace a
path, then "Solve & Replay" feeds the recorded ``[T, 8*n_arms]`` EEF chunk through
``solver.solve_chunk`` once. It then animates the solved joint trajectory and
overlays the recorded target frames (green) against the solved-FK frames (red),
printing per-frame position / orientation tracking error.

Robot-agnostic: pick any registered robot whose package self-registers and whose
``build_kinematics`` returns a PyRoki solver. Each arm is rendered with viser's
``ViserUrdf`` (geometry once, joints per frame); the project's ``qpos_to_cfg``
maps each arm's solved qpos slice (single gripper scalar) to the URDF's actuated
joints (two finger joints). Targets/axes are children of each arm's base node, so
their local poses ARE the solver reference frame — exact for robots whose EEF
frame is the URDF base (piper / arx / ur5e); robots with a mount or non-base
reference frame (franka / r1_lite) get a constant offset on the overlay axes only
(the printed error is computed in the solver frame and stays correct).

Run (proxy off, venv active)::

    python tests/manual/viser_track_chunk.py --robot agilex_piper
    python tests/manual/viser_track_chunk.py --robot ur5e --frames 60
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _import_robot(robot_name: str):
    """Import the robot's zoo package so it self-registers, then build it.

    The ``robots`` package __init__ is mid-refactor and may fail to import; we
    import the specific zoo subpackage directly to trigger its registration.

    Args:
        robot_name: registry key matching its ``zoo/<robot_name>/`` directory.

    Returns:
        The built Robot instance from ROBOT_REGISTRY.
    """
    try:
        importlib.import_module(f"robots.zoo.{robot_name}")
    except Exception as exc:
        raise SystemExit(
            f"Failed to import robots.zoo.{robot_name!r}: {exc}\n"
            "The robots package is mid-refactor; this tool needs the robot's zoo "
            "subpackage to import and self-register, and build_kinematics to return "
            "a PyRoki solver. Pick a migrated robot (e.g. agilex_piper, ur5e, arx_r5)."
        ) from exc
    from core.registry import ROBOT_REGISTRY

    return ROBOT_REGISTRY.build(robot_name)


def _arm_eef_width(eef_dim: int) -> int:
    """Number of arms from a flat per-frame EEF width (8 per arm)."""
    if eef_dim % 8 != 0:
        raise SystemExit(f"EEF width {eef_dim} is not a multiple of 8 (8D per arm)")
    return eef_dim // 8


def _arm_layout_offsets(seed_eef: np.ndarray, n_arms: int) -> list[np.ndarray]:
    """Pure-translation offset per arm so shared-URDF arms don't overlap.

    Robots whose arms share one URDF report the SAME seed EEF for every arm (the
    URDF has no per-arm base), so rendering them all at the origin overlaps them.
    When the seed EEFs coincide we fan the arms out along the world Y axis; when
    they already differ (arms with distinct bases) we leave them in place.

    Args:
        seed_eef: flat per-arm EEF at the seed pose [8*n_arms].
        n_arms: number of arms.

    Returns:
        One [3] world translation per arm.
    """
    if n_arms == 1:
        return [np.zeros(3)]
    positions = np.stack([seed_eef[i * 8 : i * 8 + 3] for i in range(n_arms)], axis=0)
    spread = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    if spread > 0.05:
        return [np.zeros(3) for _ in range(n_arms)]
    step = 0.5
    mid = (n_arms - 1) / 2.0
    return [np.array([0.0, (i - mid) * step, 0.0]) for i in range(n_arms)]


def main() -> None:
    """Launch the viser server and run the record / solve / replay loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="agilex_piper", help="registered robot name")
    parser.add_argument("--frames", type=int, default=40, help="recorded chunk length T")
    parser.add_argument("--port", type=int, default=8080, help="viser server port")
    args = parser.parse_args()

    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    robot = _import_robot(args.robot)
    if robot.vis_config is None:
        raise SystemExit(f"Robot {args.robot!r} has no vis_config; cannot render.")

    solver = robot.build_kinematics(initial_qpos_groups=robot.initial_qpos_by_group())
    if solver is None:
        raise SystemExit(f"Robot {args.robot!r} has no kinematics solver.")

    seed = solver.default_seed()
    seed_eef = solver.fk_chunk(seed)[0]  # [8*n_arms] per-arm EEF at the seed pose
    parts = robot.vis_config.parts
    n_arms = _arm_eef_width(seed_eef.shape[0])
    if len(parts) < n_arms:
        raise SystemExit(
            f"{len(parts)} vis parts but solver reports {n_arms} arms; mismatch."
        )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)

    # Render in the SOLVER reference frame, not the project frontend's layout frame:
    # the solver's fk_chunk EEF lives in each arm's reference frame, so the URDF mesh
    # must too, or the target controls drift off the gripper. vis_config.base_wxyz is
    # the Three.js frontend's layout rotation and does NOT apply here. We only need a
    # pure translation to pull apart arms that share one URDF (their seed EEFs would
    # otherwise coincide); the offset preserves "control sits on the gripper" because
    # mesh and EEF translate together.
    arm_offsets = _arm_layout_offsets(seed_eef, n_arms)

    urdf_vises: list[ViserUrdf] = []
    target_handles: list = []
    for i in range(n_arms):
        part = parts[i]
        base_node = f"/robot/arm{i}"
        server.scene.add_frame(
            base_node,
            position=(float(arm_offsets[i][0]), float(arm_offsets[i][1]), float(arm_offsets[i][2])),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            show_axes=False,
        )
        urdf = yourdfpy.URDF.load(str(part.urdf_path), load_meshes=True)
        urdf_vises.append(ViserUrdf(server, urdf, root_node_name=base_node))

        eef = seed_eef[i * 8 : i * 8 + 8]
        target_handles.append(
            server.scene.add_transform_controls(
                f"{base_node}/target",
                scale=0.15,
                position=(float(eef[0]), float(eef[1]), float(eef[2])),
                wxyz=(float(eef[3]), float(eef[4]), float(eef[5]), float(eef[6])),
            )
        )

    # Overlay axes: recorded target trajectory (green) vs solved-FK trajectory (red).
    target_axes = [
        server.scene.add_batched_axes(
            f"/robot/arm{i}/target_traj",
            axes_length=0.04,
            axes_radius=0.004,
            batched_positions=np.zeros((1, 3), dtype=np.float32),
            batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        for i in range(n_arms)
    ]
    fk_axes = [
        server.scene.add_batched_axes(
            f"/robot/arm{i}/fk_traj",
            axes_length=0.05,
            axes_radius=0.005,
            batched_positions=np.zeros((1, 3), dtype=np.float32),
            batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        for i in range(n_arms)
    ]

    record_btn = server.gui.add_checkbox("Record (drag targets)", False)
    solve_btn = server.gui.add_button("Solve & Replay")
    clear_btn = server.gui.add_button("Clear recording")
    rec_count = server.gui.add_number("Recorded frames", 0, disabled=True)
    pos_err_h = server.gui.add_number("Max pos err (mm)", 0.0, disabled=True)
    ori_err_h = server.gui.add_number("Max ori err (deg)", 0.0, disabled=True)

    # Recorded targets: per frame, flat [8*n_arms] in each arm's reference frame.
    recorded: list[np.ndarray] = []
    replay: dict[str, Any] = {"traj": None, "frame": 0}

    def _current_targets() -> np.ndarray:
        """Sample the current control poses as one flat EEF frame [8*n_arms]."""
        row = np.zeros(8 * n_arms, dtype=np.float64)
        for i, handle in enumerate(target_handles):
            pos = np.asarray(handle.position, dtype=np.float64)
            wxyz = np.asarray(handle.wxyz, dtype=np.float64)
            row[i * 8 : i * 8 + 3] = pos
            row[i * 8 + 3 : i * 8 + 7] = wxyz
            row[i * 8 + 7] = 0.0  # gripper: ignored by IK, passed through
        return row

    @clear_btn.on_click
    def _on_clear(_) -> None:
        recorded.clear()
        replay["traj"] = None
        rec_count.value = 0

    @solve_btn.on_click
    def _on_solve(_) -> None:
        if len(recorded) < 2:
            print("Need at least 2 recorded frames; drag targets with Record on.")
            return
        chunk = np.stack(recorded, axis=0)  # [T, 8*n_arms]
        solved = np.asarray(solver.solve_chunk(chunk, solver.default_seed()))
        fk = np.asarray(solver.fk_chunk(solved))  # [T, 8*n_arms]

        # Tracking error per arm, in the solver reference frame.
        max_pos_mm, max_ori_deg = 0.0, 0.0
        print(f"\n=== Tracking report: {args.robot}, {chunk.shape[0]} frames ===")
        for i in range(n_arms):
            tgt = chunk[:, i * 8 : i * 8 + 7]
            got = fk[:, i * 8 : i * 8 + 7]
            pos_err = np.linalg.norm(tgt[:, :3] - got[:, :3], axis=1)
            ori_err = np.array(
                [_quat_angle_deg(tgt[t, 3:7], got[t, 3:7]) for t in range(len(tgt))]
            )
            max_pos_mm = max(max_pos_mm, float(pos_err.max() * 1000.0))
            max_ori_deg = max(max_ori_deg, float(ori_err.max()))
            print(
                f"  arm{i}: pos err mean {pos_err.mean() * 1000:.2f} mm "
                f"max {pos_err.max() * 1000:.2f} mm | "
                f"ori err mean {ori_err.mean():.2f} deg max {ori_err.max():.2f} deg"
            )
            target_axes[i].batched_positions = np.ascontiguousarray(tgt[:, :3], dtype=np.float32)
            target_axes[i].batched_wxyzs = np.ascontiguousarray(tgt[:, 3:7], dtype=np.float32)
            fk_axes[i].batched_positions = np.ascontiguousarray(got[:, :3], dtype=np.float32)
            fk_axes[i].batched_wxyzs = np.ascontiguousarray(got[:, 3:7], dtype=np.float32)

        pos_err_h.value = round(max_pos_mm, 2)
        ori_err_h.value = round(max_ori_deg, 2)
        replay["traj"] = solved
        replay["frame"] = 0

    while True:
        if record_btn.value:
            recorded.append(_current_targets())
            rec_count.value = len(recorded)
            if len(recorded) >= args.frames:
                record_btn.value = False
            time.sleep(0.05)
            continue

        traj = replay["traj"]
        if traj is not None:
            f = int(replay["frame"])
            qpos = np.asarray(traj)[f]
            for i, urdf_vis in enumerate(urdf_vises):
                part = parts[i]
                slice_q = qpos[part.qpos_offset : part.qpos_offset + part.n_joints]
                urdf_vis.update_cfg(np.asarray(part.qpos_to_cfg(slice_q)))
            replay["frame"] = (f + 1) % len(np.asarray(traj))
            time.sleep(0.05)
        else:
            time.sleep(0.02)


def _quat_angle_deg(wxyz_a: np.ndarray, wxyz_b: np.ndarray) -> float:
    """Geodesic angle in degrees between two wxyz quaternions."""
    a = np.asarray(wxyz_a, dtype=np.float64)
    b = np.asarray(wxyz_b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


if __name__ == "__main__":
    main()
