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
