"""Isolated whole-episode URDF transform computation.

The console process has ROS, camera streams, polling handlers, and the recorder all
sharing the Python GIL. A dedicated process keeps the CPU-heavy yourdfpy frame loop
from starving those real-time threads and reuses its parsed URDF between episodes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SCENES: dict[tuple[str, float | None, float | None], Any] = {}


def build_transform_blob(
    robot_type: str,
    gripper_open: float | None,
    gripper_close: float | None,
    qpos: np.ndarray,
) -> bytes:
    """Build an EVAXFRM1 blob in a worker process, caching the parsed URDF scene."""
    import robots  # noqa: F401  (register robot types in this fresh process)
    from core.registry import ROBOT_REGISTRY
    from robots.utils import UrdfScene

    key = (robot_type, gripper_open, gripper_close)
    scene = _SCENES.get(key)
    if scene is None:
        scene = UrdfScene(
            ROBOT_REGISTRY.build(robot_type),
            gripper_open=gripper_open,
            gripper_close=gripper_close,
        )
        _SCENES[key] = scene
    return scene.all_transforms_blob(np.asarray(qpos, dtype=np.float32))
