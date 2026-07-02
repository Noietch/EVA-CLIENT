"""Rotation and quaternion math utilities — conversions between Euler angles,
quaternions (wxyz convention), and 3x3 rotation matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_xyzw_to_wxyz(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Reorder a quaternion from xyzw to wxyz convention.

    Args:
        qx, qy, qz, qw: Quaternion components in xyzw order.

    Returns:
        [4] float32 quaternion in wxyz order.
    """
    return np.array([qw, qx, qy, qz], dtype=np.float32)


def build_eef_pose_from_xyzw(
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    gripper: float,
) -> np.ndarray:
    """Assemble a canonical EEF pose from position, xyzw quaternion, and gripper.

    Args:
        x, y, z: End-effector translation.
        qx, qy, qz, qw: Orientation quaternion in xyzw order.
        gripper: Gripper scalar.

    Returns:
        [8] float32 canonical EEF pose as xyz + wxyz + gripper.
    """
    quaternion = quaternion_xyzw_to_wxyz(qx, qy, qz, qw)
    return np.array(
        [x, y, z, quaternion[0], quaternion[1], quaternion[2], quaternion[3], gripper],
        dtype=np.float32,
    )


def quaternion_wxyz_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix.

    Args:
        qw, qx, qy, qz: Quaternion components in wxyz order (normalized internally).

    Returns:
        [3, 3] float64 rotation matrix.
    """
    if qw == 0.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:
        raise ValueError("Quaternion norm must be non-zero")
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion.

    Args:
        rotation: [3, 3] rotation matrix.

    Returns:
        [4] float64 quaternion in wxyz order.
    """
    qx, qy, qz, qw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    return np.array([qw, qx, qy, qz], dtype=np.float64)
