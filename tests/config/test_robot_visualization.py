from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from robots.utils import UrdfScene


def test_arx_x5_uses_ac_one_arm_visuals_without_whole_robot_mesh():
    robot = ROBOT_REGISTRY.build("arx_x5")
    parts = robot.vis_config.parts

    assert [part.name for part in parts] == ["left_arm", "right_arm"]
    assert [part.base_position for part in parts] == [(-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)]
    assert parts[0].urdf_path == parts[1].urdf_path

    root = ET.parse(parts[0].urdf_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    assert set(links) == {"base_link", *(f"link{i}" for i in range(1, 9))}

    for index in range(1, 9):
        link = links[f"link{index}"]
        visual_mesh = link.find("visual/geometry/mesh")
        collision_mesh = link.find("collision/geometry/mesh")
        collision_origin = link.find("collision/origin")

        assert visual_mesh is not None
        assert collision_mesh is not None
        assert collision_origin is not None
        visual_filename = f"link{index}.STL" if index >= 6 else f"x5_one_link{index}.STL"
        assert visual_mesh.attrib["filename"].endswith(f"/{visual_filename}")
        assert collision_mesh.attrib["filename"].endswith(f"/link{index}.STL")
        assert collision_origin.attrib == {"xyz": "0 0 0", "rpy": "0 0 0"}


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


def test_r1lite_home_qpos_zeroes_arms_and_opens_grippers():
    robot = ROBOT_REGISTRY.build("r1_lite")

    arm_mask = np.ones(robot.total_action_dim, dtype=bool)
    arm_mask[list(robot.gripper_indices)] = False

    assert robot.initial_qpos[arm_mask] == pytest.approx(np.zeros(12, dtype=np.float32))
    assert robot.initial_qpos[list(robot.gripper_indices)] == pytest.approx(
        np.array([100.0, 100.0], dtype=np.float32)
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
