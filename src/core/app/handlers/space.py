"""Observation / action space types.

Two concrete shapes the policy can produce or the transport reads back:

  JointState — joint-space angles; no extra layout fields needed.
  EEFPose    — end-effector pose; per arm xyz(3) + rotation + optional gripper,
               concatenated over n_arms arms.

YAML/.py configs reference these by class name:

    inference_cfg = dict(
        obs_space    = dict(type="JointState"),
        action_space = dict(type="EEFPose", n_arms=2, rotation="quat",
                            include_gripper=True),
        ...
    )

``load_config`` (src/core/config.py) calls ``build_space`` on the obs_space and
action_space dicts so consumers receive the class instance and can call methods
directly: ``cfg.inference_cfg.obs_space.is_eef()``,
``cfg.inference_cfg.action_space.total_dim()``.

Both classes self-register on ``SPACE_REGISTRY`` under their ``type`` name;
``build_space`` is a thin wrapper over ``SPACE_REGISTRY.build_from_cfg``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

from core.registry import SPACE_REGISTRY

_ROT_TYPE_DIMS = {"quat": 4, "euler": 3, "rotvec": 3, "rot6d": 6}
_DEFAULT_ROT_ORDER = {"quat": "wxyz", "euler": "xyz"}


@SPACE_REGISTRY.register("JointState")
@dataclasses.dataclass
class JointState:
    """Joint-space observation or action — no extra layout fields needed."""

    type: str = "JointState"

    def is_eef(self) -> bool:
        """Whether this space describes an EEF (end-effector) chunk."""
        return False


@SPACE_REGISTRY.register("EEFPose")
@dataclasses.dataclass
class EEFPose:
    """End-effector pose observation or action.

    Per arm the layout is xyz(3) + rotation(rotation_dim) [+ gripper(1)],
    concatenated over ``n_arms`` arms. Used to convert policy output (or the
    matching observation read from the transport) into the canonical
    xyz + quat_wxyz [+ gripper] representation before IK / state updates.

    ``rotation`` names the source rotation type: ``quat`` / ``euler`` /
    ``rotvec`` / ``rot6d``. ``rotation_order`` refines it — quaternion order
    (``wxyz`` / ``xyzw``) or Euler order (``xyz`` / ``zyx`` ...); when omitted a
    type default is used (quat→wxyz, euler→xyz). ``degrees`` only applies to
    Euler input.
    """

    type: str = "EEFPose"
    n_arms: int = 2
    rotation: str = "quat"
    rotation_order: str | None = None
    include_gripper: bool = True
    degrees: bool = False

    def is_eef(self) -> bool:
        """Whether this space describes an EEF (end-effector) chunk."""
        return True

    def rotation_dim(self) -> int:
        """Flat dimension of the source rotation (quat 4, euler/rotvec 3, rot6d 6)."""
        rot = self.rotation.lower()
        if rot not in _ROT_TYPE_DIMS:
            raise ValueError(
                f"Unsupported rotation '{self.rotation}'. Expected one of {sorted(_ROT_TYPE_DIMS)}."
            )
        return _ROT_TYPE_DIMS[rot]

    def rot_order(self) -> str | None:
        """Resolved order: explicit ``rotation_order``, else the rotation type's default."""
        if self.rotation_order is not None:
            return self.rotation_order
        return _DEFAULT_ROT_ORDER.get(self.rotation.lower())

    def per_arm_dim(self) -> int:
        """Per-arm vector size: xyz(3) + rotation_dim [+ gripper(1)]."""
        return 3 + self.rotation_dim() + (1 if self.include_gripper else 0)

    def total_dim(self) -> int:
        """Full chunk dimension across all arms: ``per_arm_dim * n_arms``."""
        return self.per_arm_dim() * self.n_arms

    def normalize_chunk_to_canonical(self, eef_chunk: np.ndarray) -> np.ndarray:
        """Convert an EEF chunk in this layout to the canonical IK input layout.

        Canonical layout is per arm: xyz(3) + quat_wxyz(4) [+ gripper(1)],
        concatenated over all arms. Any supported source representation
        (quat wxyz/xyzw, euler with order, rotvec, rot6d) maps to the same
        canonical output.

        Args:
            eef_chunk: Policy EEF actions, shape [D] or [T, D] with
                D >= ``self.total_dim()``.

        Returns:
            Canonical chunk, shape [T, (3 + 4 + include_gripper) * n_arms].
        """
        chunk = self._to_2d_float32(eef_chunk)
        total_dim = self.total_dim()
        rot_order = self.rot_order()
        if chunk.shape[1] < total_dim:
            raise ValueError(
                f"EEF chunk dim {chunk.shape[1]} < expected {total_dim} "
                f"(arms={self.n_arms}, rotation={self.rotation}, gripper={self.include_gripper})"
            )

        # Already canonical (scalar-first quat wxyz): skip the pose round-trip so
        # all-zero placeholder rows pass through without scipy rejecting non-unit.
        if self.rotation == "quat" and rot_order == "wxyz":
            return chunk.copy()

        per_arm = self.per_arm_dim()
        rot_dim = self.rotation_dim()
        converted_rows = []
        for row in chunk:
            arms = []
            for arm_idx in range(self.n_arms):
                offset = arm_idx * per_arm
                xyz = row[offset : offset + 3]
                rot = row[offset + 3 : offset + 3 + rot_dim]
                quat_wxyz = self._rot_to_quat_wxyz(rot, rot_order)
                parts = [xyz, quat_wxyz.astype(np.float32)]
                if self.include_gripper:
                    parts.append(row[offset + 3 + rot_dim : offset + per_arm])
                arms.append(np.concatenate(parts, axis=0))
            converted_rows.append(np.concatenate(arms, axis=0))
        return np.asarray(converted_rows, dtype=np.float32)

    @staticmethod
    def _to_2d_float32(actions: object) -> np.ndarray:
        """Coerce an action input to a 2-D [T, D] float32 array."""
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with 1 or 2 dims, got {chunk.shape}")
        return chunk

    def _rot_to_quat_wxyz(self, rot: np.ndarray, rotation_order: str | None) -> np.ndarray:
        """Convert a flat rotation vector of ``self.rotation`` type to a wxyz quaternion.

        Args:
            rot: Flat rotation for one arm; len 4 (quat), 3 (euler/rotvec), or 6 (rot6d).
            rotation_order: Quat order ("wxyz"/"xyzw") or Euler axis order; ignored otherwise.

        Returns:
            [4] quaternion in wxyz (scalar-first) order.
        """
        rot = np.asarray(rot, dtype=np.float64)
        rt = self.rotation.lower()
        if rt == "quat":
            if (rotation_order or "wxyz").lower() == "wxyz":
                rotation = Rotation.from_quat([rot[1], rot[2], rot[3], rot[0]])
            else:
                rotation = Rotation.from_quat(rot)
        elif rt == "euler":
            rotation = Rotation.from_euler(
                (rotation_order or "xyz").lower(), rot, degrees=self.degrees
            )
        elif rt == "rotvec":
            rotation = Rotation.from_rotvec(rot)
        elif rt == "rot6d":
            a1, a2 = rot[0:3], rot[3:6]
            b1 = a1 / (np.linalg.norm(a1) + 1e-8)
            b2 = a2 - np.dot(b1, a2) * b1
            b2 = b2 / (np.linalg.norm(b2) + 1e-8)
            b3 = np.cross(b1, b2)
            rotation = Rotation.from_matrix(np.stack([b1, b2, b3], axis=-1))
        else:
            raise ValueError(f"Unsupported rotation type: {self.rotation}")
        qx, qy, qz, qw = rotation.as_quat()
        return np.array([qw, qx, qy, qz])


ActionSpace = JointState | EEFPose


def build_space(cfg: dict) -> ActionSpace:
    """Construct a space instance from a YAML/.py config dict.

    cfg is a plain dict with a ``type`` key naming the class
    (e.g. ``dict(type="EEFPose", n_arms=2, rotation="quat")``). Unknown types
    raise ``KeyError`` listing valid choices.
    """
    args = dict(cfg)
    type_name = args.pop("type")
    return SPACE_REGISTRY.build(type_name, **args)
