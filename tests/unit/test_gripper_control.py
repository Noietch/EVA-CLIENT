from types import SimpleNamespace

import pytest

from core.app.handlers.control import opposite_gripper_value

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("config", "current", "expected"),
    [
        (SimpleNamespace(open=1.0, close=0.0), 1.0, 0.0),
        (SimpleNamespace(open=0.0, close=1.0), 0.0, 1.0),
    ],
)
def test_opposite_gripper_value_handles_both_directions(config, current, expected):
    assert opposite_gripper_value(current, config.open, config.close) == expected


@pytest.mark.parametrize(
    ("arm_error", "ignore_gripper", "expected_waits"), [(0, True, 0), (0.2, True, 3), (0, False, 3)]
)
def test_sync_wait_can_ignore_a_blocked_gripper_without_ignoring_arm_error(
    arm_error, ignore_gripper, expected_waits
):
    import numpy as np

    from core.app.handlers.control import _wait_until_reached

    waits = []
    runtime = SimpleNamespace(
        robot=SimpleNamespace(gripper_mask=[False, True]),
        transport=SimpleNamespace(
            create_rate=lambda hz: SimpleNamespace(sleep=lambda: waits.append(hz)),
            get_latest_qpos=lambda: np.zeros(2),
        ),
    )
    assert _wait_until_reached(
        runtime, np.array([arm_error, 1.0]), 20, max_ticks=3, ignore_gripper_error=ignore_gripper
    )
    assert len(waits) == expected_waits
