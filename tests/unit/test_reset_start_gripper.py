"""A reset returns each gripper to its episode-start value, not its live value."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.app.handlers.control import reset_target_qpos, set_gripper_open_close

pytestmark = pytest.mark.unit

HOME = np.array([0.0, 1.5, 1.0, 0.0, 1.5, 1.0], dtype=np.float32)
START_POSE = np.array([0.3, 0.9, 0.0, -0.3, 0.9, 0.0], dtype=np.float32)
CLOSED_LIVE = np.array([0.2, 0.8, 0.0, -0.2, 0.8, 0.0], dtype=np.float32)


def _runtime(init_qpos: np.ndarray | None = START_POSE) -> SimpleNamespace:
    published: list[np.ndarray] = []
    groups = [
        SimpleNamespace(name="left_arm", dof=3, gripper_index=2),
        SimpleNamespace(name="right_arm", dof=3, gripper_index=2),
    ]
    return SimpleNamespace(
        init_qpos=init_qpos,
        init_gripper={},
        published=published,
        robot=SimpleNamespace(actuator_groups=groups, initial_qpos=HOME),
        transport=SimpleNamespace(
            get_latest_qpos=lambda: CLOSED_LIVE,
            create_rate=lambda hz: SimpleNamespace(sleep=lambda: None),
            publish_action=lambda action, target: published.append(action),
            record_collection_gripper_action=lambda group_name, value: None,
        ),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        robot=SimpleNamespace(gripper_open=1.0, gripper_close=0.0),
        inference_cfg=SimpleNamespace(publish_rate=10),
    )


def test_reset_reopens_grippers_left_closed_by_the_previous_episode() -> None:
    target = reset_target_qpos(_runtime())

    np.testing.assert_allclose(target, [0.3, 0.9, 1.0, -0.3, 0.9, 1.0])


def test_reset_keeps_the_start_gripper_the_operator_chose_per_side() -> None:
    runtime = _runtime()

    assert set_gripper_open_close(_config(), runtime, open_gripper=False, side="l")

    assert runtime.init_gripper == {"left_arm": 0.0}
    np.testing.assert_allclose(reset_target_qpos(runtime), [0.3, 0.9, 0.0, -0.3, 0.9, 1.0])


def test_reset_without_an_init_pose_still_goes_home() -> None:
    assert reset_target_qpos(_runtime(init_qpos=None)) is HOME
