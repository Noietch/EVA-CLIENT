"""Robot visualization layer — vis-config structs and the console 3D scene helper.

Holds the declarative vis structs (``VisPart`` / ``RobotVisConfig``) a robot exposes
via ``Robot.vis_config``, plus ``UrdfScene``: it parses each part's URDF with yourdfpy,
exposes the static mesh list once, then per-frame computes each mesh's 4x4 world
transform from the current robot qpos (the frontend loads geometry once and applies
these matrices each tick). Parts may share a URDF (dual-arm robots reusing one
single-arm URDF) — shared files are loaded once.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import struct
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from core.utils.math import quaternion_wxyz_to_matrix

if TYPE_CHECKING:
    from robots.base import Robot

logger = logging.getLogger(__name__)
_MESH_MAGIC = b"EVAMESH1"
_XFRM_MAGIC = b"EVAXFRM1"


def _compile_segments(segments: list[dict[str, Any]]) -> Callable[[np.ndarray], np.ndarray]:
    """Compile a declarative qpos-slice -> URDF cfg segment table into a mapping function.

    Each segment appends values to the output cfg vector, in declaration order:
      {copy: [start, end]}   passthrough input[start:end] (half-open) — arm joints.
      {fixed: [v, ...]}      constant values — e.g. locked wheels / fixed torso pose.
      {gripper: i, range: [lo, hi], stroke: s, invert: bool, fingers: [signs...]}
                             map gripper scalar input[i] to finger joints:
                             f = (clip(input[i], lo, hi) - lo) / (hi - lo) * s,
                             optionally inverted before scaling, one joint per sign as sign*f.

    Covers single/dual-arm robots reusing one URDF (one copy + one gripper) and
    whole-body robots (fixed base/torso prefix + per-arm segments).
    """

    def qpos_to_cfg(qpos: np.ndarray) -> np.ndarray:
        vector = np.asarray(qpos, dtype=np.float64)
        parts: list[np.ndarray] = []
        for seg in segments:
            if "copy" in seg:
                start, end = seg["copy"]
                parts.append(vector[start:end])
            elif "fixed" in seg:
                parts.append(np.asarray(seg["fixed"], dtype=np.float64))
            elif "gripper" in seg:
                lo, hi = seg["range"]
                clamped = float(np.clip(vector[seg["gripper"]], lo, hi))
                alpha = (clamped - lo) / (hi - lo)
                if seg.get("invert", False):
                    alpha = 1.0 - alpha
                finger = alpha * seg["stroke"]
                parts.append(np.asarray([s * finger for s in seg["fingers"]], dtype=np.float64))
            else:
                raise ValueError(f"Unknown qpos_to_cfg segment: {seg}")
        return np.concatenate(parts)

    return qpos_to_cfg


@dataclasses.dataclass(frozen=True)
class VisPart:
    """One renderable URDF instance placed in the console 3D scene.

    Each part self-describes the URDF it renders, where it sits, and which slice of
    the full robot qpos drives it — so the same mechanism covers single-arm, dual-arm
    reusing one URDF, and whole-body robots. Meshes resolve from ``urdf_path.parent``.

    name: logical key surfaced to the frontend (e.g. "left_arm", "body").
    urdf_path: this part's URDF (parts may share the same file).
    base_position / base_wxyz: this part's root pose in the world (xyz, quat wxyz).
    qpos_offset / n_joints: this part's slice into the full robot qpos vector.
    qpos_to_cfg: maps this part's qpos slice (n_joints,) to its URDF actuated-joint
        cfg (array or named dict) for yourdfpy ``update_cfg``.
    gripper_visual_ranges: (index, lo, hi) per gripper dim, for visual range remap.
    """

    name: str
    urdf_path: Path
    base_position: tuple[float, float, float]
    base_wxyz: tuple[float, float, float, float]
    qpos_offset: int
    n_joints: int
    qpos_to_cfg: Callable[[np.ndarray], Any]
    gripper_visual_ranges: tuple[tuple[int, float, float], ...] = ()
    # Raw geom names to skip when rendering the console 3D scene (e.g. mobile base /
    # wheels that never move during a fixed-base manipulation task). Pure render-layer
    # cull: kinematics, IK and data collection are untouched.
    render_exclude: tuple[str, ...] = ()

    @classmethod
    def from_segments(
        cls,
        name: str,
        urdf_path: Path,
        base_position: tuple[float, float, float],
        base_wxyz: tuple[float, float, float, float],
        qpos_offset: int,
        n_joints: int,
        segments: list[dict[str, Any]],
    ) -> VisPart:
        """Build a VisPart from a declarative qpos->cfg segment table.

        The segments compile into the ``qpos_to_cfg`` mapping (see ``_compile_segments``)
        and the gripper visual ranges are derived from the gripper segments. Use the
        plain constructor with a callable ``qpos_to_cfg`` for mappings the segment table
        cannot express.

        Args:
            name / urdf_path / base_position / base_wxyz / qpos_offset / n_joints: see
                the class fields.
            segments: ordered ``{copy}|{fixed}|{gripper}`` entries (see ``_compile_segments``).

        Returns:
            A VisPart whose qpos_to_cfg runs the compiled segments.
        """
        gripper_visual_ranges = tuple(
            (int(seg["gripper"]), float(seg["range"][0]), float(seg["range"][1]))
            for seg in segments
            if "gripper" in seg
        )
        return cls(
            name=name,
            urdf_path=urdf_path,
            base_position=base_position,
            base_wxyz=base_wxyz,
            qpos_offset=qpos_offset,
            n_joints=n_joints,
            qpos_to_cfg=_compile_segments(segments),
            gripper_visual_ranges=gripper_visual_ranges,
        )


@dataclasses.dataclass(frozen=True)
class RobotVisConfig:
    """The console 3D scene's render config: one VisPart per renderable URDF instance."""

    parts: tuple[VisPart, ...]


def _as_rgb(color: object) -> list[float] | None:
    if color is None:
        return None
    arr = np.asarray(color, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 3:
        return None
    rgb = arr[:3]
    if np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0).tolist()


def _wxyz_to_matrix(
    pos: tuple[float, float, float], wxyz: tuple[float, float, float, float]
) -> np.ndarray:
    # Build a 4x4 homogeneous transform from position + quaternion (w, x, y, z).
    w, x, y, z = wxyz
    T = np.eye(4, dtype=np.float64)
    if (w, x, y, z) != (0.0, 0.0, 0.0, 0.0):
        T[:3, :3] = quaternion_wxyz_to_matrix(w, x, y, z)
    T[:3, 3] = pos
    return T


def _remap_gripper_qpos_for_visual(
    part: VisPart,
    qpos_part: np.ndarray,
    gripper_open: float | None,
    gripper_close: float | None,
) -> np.ndarray:
    """Map app-level gripper open/close scalars into the URDF visual range."""
    if (
        gripper_open is None
        or gripper_close is None
        or gripper_open == gripper_close
        or not part.gripper_visual_ranges
    ):
        return qpos_part

    visual_qpos = np.asarray(qpos_part, dtype=np.float64).copy()
    denominator = gripper_open - gripper_close
    for index, visual_close, visual_open in part.gripper_visual_ranges:
        if index >= len(visual_qpos):
            continue
        alpha = (visual_qpos[index] - gripper_close) / denominator
        alpha = float(np.clip(alpha, 0.0, 1.0))
        visual_qpos[index] = visual_close + alpha * (visual_open - visual_close)
    return visual_qpos


class UrdfScene:
    """Parses each part's URDF once and computes per-mesh world transforms per frame.

    Parts may share a URDF (dual-arm robots reusing one single-arm URDF) — shared
    files are loaded a single time and reused across parts. Geometry keys are kept
    raw when only one URDF is loaded (frontend-transparent regression) and prefixed
    with the URDF stem when multiple distinct URDFs are present, to disambiguate
    same-named geoms across files.
    """

    def __init__(
        self,
        robot: Robot,
        gripper_open: float | None = None,
        gripper_close: float | None = None,
    ) -> None:
        import yourdfpy

        cfg: RobotVisConfig | None = robot.vis_config
        if cfg is None:
            raise ValueError(f"Robot '{robot.name}' has no vis_config; 3D canvas unavailable")
        self._cfg = cfg
        self._mesh_prefix = robot.name
        self._gripper_open = gripper_open
        self._gripper_close = gripper_close
        self._lock = threading.RLock()

        # Load each distinct URDF once, keyed by resolved path.
        self._urdfs: dict[str, object] = {}
        for part in cfg.parts:
            key = str(part.urdf_path)
            if key in self._urdfs:
                continue
            urdf_dir = part.urdf_path.parent
            self._urdfs[key] = yourdfpy.URDF.load(
                str(part.urdf_path),
                build_scene_graph=True,
                load_meshes=True,
                filename_handler=partial(yourdfpy.filename_handler_magic, dir=str(urdf_dir)),
            )

        # Prefix geom keys only when more than one distinct URDF is loaded, so the
        # single-URDF case (all current robots) stays byte-for-byte as before.
        self._multi_urdf = len(self._urdfs) > 1
        # Raw geom names to cull from the rendered scene, unioned across parts.
        self._render_exclude = {g for part in cfg.parts for g in part.render_exclude}
        self._mesh_bytes: dict[str, bytes] = {}
        # geom_key -> (urdf_path_str, raw_geom_name) for mesh_bytes/color lookup.
        self._geom_source: dict[str, tuple[str, str]] = {}
        for key, urdf in self._urdfs.items():
            stem = Path(key).stem
            for raw in urdf.scene.geometry.keys():  # type: ignore[attr-defined]
                if raw in self._render_exclude:
                    continue
                geom_key = f"{stem}/{raw}" if self._multi_urdf else raw
                self._geom_source[geom_key] = (key, raw)

        self._bases = {
            part.name: _wxyz_to_matrix(part.base_position, part.base_wxyz) for part in cfg.parts
        }
        logger.info(
            "UrdfScene ready for '%s': %d parts, %d URDFs, %d meshes",
            robot.name,
            len(cfg.parts),
            len(self._urdfs),
            len(self._geom_source),
        )

    @property
    def arm_names(self) -> list[str]:
        """Names of the configured vis parts, in part order."""
        return [part.name for part in self._cfg.parts]

    def static_meshes(self) -> list[dict]:
        """List each geometry once as {name, file, color} for the frontend to load.

        ``name`` is the geom key matched against ``transforms()`` output (kept raw).
        ``file`` is a robot-namespaced mesh URL so the browser cache can't return
        another robot's same-named mesh.
        """
        return [
            {"name": g, "file": f"{self._mesh_prefix}/{g}", "color": self._geom_color(g)}
            for g in self._geom_source
        ]

    def _geom_color(self, name: str) -> list[float] | None:
        source = self._geom_source.get(name)
        if source is None:
            return None
        urdf_key, raw = source
        mesh = self._urdfs[urdf_key].scene.geometry.get(raw)  # type: ignore[attr-defined]
        if mesh is None:
            return None
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None)
        for owner in (visual, material):
            if owner is None:
                continue
            for attr in ("main_color", "baseColorFactor", "diffuse"):
                try:
                    rgb = _as_rgb(getattr(owner, attr))
                except Exception:
                    continue
                if rgb is not None:
                    return rgb
        return None

    def mesh_bytes(self, name: str) -> bytes | None:
        """Encode one loaded trimesh geometry as a compact binary payload.

        Args:
            name: mesh URL path from ``static_meshes()`` ``file`` —
                "<robot_name>/<geom>", where <geom> is the raw geom name or
                "<urdf_stem>/<geom>" when multiple distinct URDFs are loaded.

        Returns:
            Bytes with header, float32 vertices [N, 3], float32 normals [N, 3],
            and uint32 triangle faces [F, 3], or None when the key is unknown.
        """
        geom = (
            name[len(self._mesh_prefix) + 1 :] if name.startswith(f"{self._mesh_prefix}/") else name
        )
        source = self._geom_source.get(geom)
        if source is None:
            return None
        if geom not in self._mesh_bytes:
            urdf_key, raw = source
            mesh = self._urdfs[urdf_key].scene.geometry[raw]  # type: ignore[attr-defined]
            vertices = np.ascontiguousarray(mesh.vertices, dtype="<f4")
            normals = np.ascontiguousarray(mesh.vertex_normals, dtype="<f4")
            faces = np.ascontiguousarray(mesh.faces, dtype="<u4")
            header = _MESH_MAGIC + struct.pack("<II", len(vertices), len(faces))
            self._mesh_bytes[geom] = (
                header + vertices.tobytes() + normals.tobytes() + faces.tobytes()
            )
        return self._mesh_bytes[geom]

    def _part_transforms(
        self, part: VisPart, qpos_part: np.ndarray, base: np.ndarray
    ) -> dict[str, list[list[float]]]:
        # Returns geom_key -> 4x4 world matrix (row-major) for one part.
        urdf_key = str(part.urdf_path)
        urdf = self._urdfs[urdf_key]
        visual_qpos = _remap_gripper_qpos_for_visual(
            part,
            qpos_part,
            self._gripper_open,
            self._gripper_close,
        )
        urdf.update_cfg(part.qpos_to_cfg(visual_qpos))  # type: ignore[attr-defined]
        stem = Path(urdf_key).stem
        out: dict[str, list[list[float]]] = {}
        for raw in urdf.scene.geometry.keys():  # type: ignore[attr-defined]
            if raw in self._render_exclude:
                continue
            geom_key = f"{stem}/{raw}" if self._multi_urdf else raw
            T_local, _ = urdf.scene.graph.get(frame_to=raw)  # type: ignore[attr-defined]
            T_world = base @ np.asarray(T_local, dtype=np.float64)
            out[geom_key] = T_world.tolist()
        return out

    def transforms(self, qpos: np.ndarray | None) -> dict[str, dict[str, list[list[float]]]]:
        """Compute per-mesh world transforms for all parts at a given robot qpos.

        Args:
            qpos: Full robot joint vector [total_joints] float; None or short
                vectors fall back to zeros (and zeros for any missing part slice).

        Returns:
            ``{part_name: {geom_key: 4x4 row-major matrix}}`` for every part.
        """
        with self._lock:
            total = sum(part.n_joints for part in self._cfg.parts)
            q = (
                np.zeros(total, dtype=np.float64)
                if qpos is None
                else np.asarray(qpos, dtype=np.float64)
            )
            out: dict[str, dict[str, list[list[float]]]] = {}
            for part in self._cfg.parts:
                lo, hi = part.qpos_offset, part.qpos_offset + part.n_joints
                slice_q = q[lo:hi] if len(q) >= hi else np.zeros(part.n_joints, dtype=np.float64)
                out[part.name] = self._part_transforms(part, slice_q, self._bases[part.name])
            return out

    def all_transforms_blob(self, qpos_seq: np.ndarray) -> bytes:
        """Pack per-mesh world transforms for a whole qpos sequence into one binary blob.

        Lets the console fetch an entire replay episode's URDF poses in a single request
        and play it back locally, instead of one FK + round-trip per frame.

        Args:
            qpos_seq: [n_frames, total_joints] float; each row a full robot qpos.

        Returns:
            magic "EVAXFRM1" + uint32 n_frames + uint32 n_geoms + uint32 hdr_len
            + JSON header (ordered "part/geom" key list, length n_geoms)
            + float32[n_frames * n_geoms * 16] row-major 4x4 matrices, geom order
            matching the header. The key order is derived from the first frame's
            ``transforms`` output so it always matches the float layout.
        """
        seq = np.asarray(qpos_seq, dtype=np.float64)
        n_frames = seq.shape[0]
        # Derive the geom key order from the float source itself, never from static_meshes
        # (that walks a separate URDF load whose dict order need not agree).
        first = self.transforms(seq[0]) if n_frames else {}
        keys = [(part, geom) for part in first for geom in first[part]]
        n_geoms = len(keys)
        flat = np.empty((n_frames, n_geoms, 4, 4), dtype="<f4")
        for f in range(n_frames):
            arms = self.transforms(seq[f])
            for g, (part, geom) in enumerate(keys):
                flat[f, g] = arms[part][geom]
        header = json.dumps([f"{part}/{geom}" for part, geom in keys]).encode("utf-8")
        meta = _XFRM_MAGIC + struct.pack("<III", n_frames, n_geoms, len(header))
        return meta + header + flat.tobytes()

    def solve_ik(self, robot: Robot, eef_target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Solve IK by delegating to the robot's own kinematics solver.

        Args:
            robot: Robot config providing the registered kinematics solver.
            eef_target: EEF target chunk [T, >=16] (canonical xyz+quat[+gripper] per arm).
            seed: Seed joint configuration for the solver.

        Returns:
            Joint chunk [T, n_joints] from the solver.
        """
        solver = robot.build_kinematics(initial_qpos_groups=robot.initial_qpos_by_group())
        if solver is None:
            raise ValueError(f"Robot '{robot.name}' has no kinematics solver")
        return np.asarray(solver.solve_chunk(np.asarray(eef_target), np.asarray(seed)))
