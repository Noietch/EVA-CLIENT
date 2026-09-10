"""Motion-aware camera QC, independent of transport frame liveness."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from core.types import Observation
from robots.base import Robot


def frozen_camera_issues(
    frames: list[Observation], robot: Robot, camera_keys: Iterable[str]
) -> list[tuple[str, str]]:
    """Reject all-zero image diffs for external cameras and cameras on moving arms.

    Comparing every image with the first is equivalent to checking whether all
    adjacent diffs are zero, without allocating full-sized difference arrays.
    Joint travel below 0.02 rad is ignored as feedback noise. Gripper-only motion
    does not move a wrist camera. Liveness is checked separately: even identical
    pixels may be fresh frames, and sensor noise may conceal a frozen scene.
    """
    if len(frames) < 2:
        return []
    states = np.asarray([frame.state_qpos for frame in frames])
    moving = set()
    offset = 0
    for group in robot.actuator_groups:
        indices = [offset + i for i in range(group.dof) if i != group.gripper_index]
        if indices and np.any(np.ptp(states[:, indices], axis=0) >= 0.02):
            moving.add(group.name)
        offset += group.dof

    enabled = set(camera_keys)
    issues = []
    for camera in robot.observation_schema.cameras:
        if camera.observation_key not in enabled:
            continue
        if camera.attached_to is not None and camera.attached_to not in moving:
            continue
        reference = None
        changed = False
        sample_count = 0
        for frame in frames:
            image = frame.images.get(camera.observation_key)
            if image is None:
                continue  # Missing images are handled by the alignment checks.
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[2] != 3:
                continue
            sample_count += 1
            if reference is None:
                reference = image
            elif not np.array_equal(image, reference):
                changed = True
                break
        if sample_count >= 2 and not changed:
            reason = (
                f"attached arm {camera.attached_to} moved"
                if camera.attached_to
                else "external camera must show scene changes"
            )
            issues.append(
                (
                    "frozen_camera",
                    f"{camera.observation_key}: {reason}, "
                    "but image diffs were zero throughout the episode",
                )
            )
    return issues
