from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from robots.utils import UrdfScene

pytestmark = pytest.mark.integration


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
