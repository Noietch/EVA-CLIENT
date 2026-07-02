from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from robots.utils import UrdfScene


def test_r1lite_scene_gripper_uses_config_open_close_range():
    robot = ROBOT_REGISTRY.build("r1_lite")
    scene = UrdfScene(robot, gripper_open=0.0, gripper_close=1.0)

    qpos_open = np.zeros(14, dtype=np.float64)
    qpos_close = np.zeros(14, dtype=np.float64)
    qpos_close[6] = 1.0
    qpos_close[13] = 1.0

    open_transforms = scene.transforms(qpos_open)["body"]
    close_transforms = scene.transforms(qpos_close)["body"]

    left_finger_1 = "left_gripper_finger_link1.STL"
    left_finger_2 = "left_gripper_finger_link2.STL"
    assert open_transforms[left_finger_1][1][3] - close_transforms[left_finger_1][1][3] == (
        pytest.approx(0.035)
    )
    assert close_transforms[left_finger_2][1][3] - open_transforms[left_finger_2][1][3] == (
        pytest.approx(0.035)
    )


def test_scene_transforms_serialize_shared_urdf_updates():
    scene = UrdfScene(ROBOT_REGISTRY.build("agilex_piper"))
    active = 0
    max_active = 0
    guard = threading.Lock()
    entered = threading.Event()

    def part_transforms(_part, _qpos_part: np.ndarray, _base: np.ndarray) -> dict:
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
            entered.set()
        time.sleep(0.05)
        with guard:
            active -= 1
        return {}

    scene._part_transforms = part_transforms  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(scene.transforms, np.zeros(14, dtype=np.float32))
        assert entered.wait(timeout=1.0)
        second = executor.submit(scene.transforms, np.ones(14, dtype=np.float32))
        first.result()
        second.result()

    assert max_active == 1
