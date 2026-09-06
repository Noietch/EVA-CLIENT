from types import SimpleNamespace

import pytest

from core.app.console.server import _serialize_gripper_controls
from core.app.handlers.control import opposite_gripper_value

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("config", "current", "expected"),
    [
        (SimpleNamespace(open=1.0, close=0.0), 1.0, 0.0),
        (SimpleNamespace(open=1.0, close=0.0), 0.0, 1.0),
        (SimpleNamespace(open=0.0, close=1.0), 0.0, 1.0),
        (SimpleNamespace(open=0.0, close=1.0), 1.0, 0.0),
    ],
)
def test_opposite_gripper_value_handles_both_directions(config, current, expected):
    assert opposite_gripper_value(current, config.open, config.close) == expected


def test_gripper_controls_expose_feedback_mapping() -> None:
    groups = (
        SimpleNamespace(name="left_arm", dof=7, gripper_index=6),
        SimpleNamespace(name="right_arm", dof=7, gripper_index=6),
    )
    config = SimpleNamespace(robot=SimpleNamespace(gripper_open=1.0, gripper_close=0.0))
    ctx = SimpleNamespace(
        config=config,
        runtime=SimpleNamespace(
            active_config=None,
            robot=SimpleNamespace(
                actuator_groups=groups,
                initial_qpos=[0.0] * 6 + [1.0] + [0.0] * 6 + [1.0],
            ),
        ),
    )

    assert _serialize_gripper_controls(ctx) == [
        {
            "group": "left_arm",
            "qpos_index": 6,
            "initial_value": 1.0,
            "open_value": 1.0,
            "close_value": 0.0,
            "side": "l",
            "label": "LEFT",
        },
        {
            "group": "right_arm",
            "qpos_index": 13,
            "initial_value": 1.0,
            "open_value": 1.0,
            "close_value": 0.0,
            "side": "r",
            "label": "RIGHT",
        },
    ]
