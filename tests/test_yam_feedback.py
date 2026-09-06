"""Offline cached poses must not authorize motion in EVA."""

from types import SimpleNamespace

import numpy as np
import pytest

from examples.hardware.yam import node

pytestmark = pytest.mark.unit


def test_publisher_omits_offline_cached_state(monkeypatch):
    captured = []
    monkeypatch.setattr(node, "pack_observation", lambda observation: observation)
    monkeypatch.setattr(
        node,
        "split_group_eef",
        lambda *args: {
            "left_arm": np.zeros(8),
            "right_arm": np.zeros(8),
        },
    )
    fake = SimpleNamespace(
        _latest_leader_action=None,
        _followers=SimpleNamespace(
            snapshot_state=lambda: {"left_arm": np.zeros(7), "right_arm": np.zeros(7)},
            hardware_status=lambda: {"left_arm": "online", "right_arm": "retrying"},
        ),
        _config=SimpleNamespace(group_names=("left_arm", "right_arm"), leader_can_channels={}),
        _fk=lambda qpos: np.zeros(16),
        _camera_snapshot=lambda: {},
        _hil_active=False,
        _hil_error="",
        _operator_event="",
        _operator_event_id=0,
        _obs_pub=SimpleNamespace(send=captured.append),
        _published_observations=0,
    )
    node.YamZmqNode._publish_observation(fake)
    assert set(captured[0].state) == {"left_arm"}
    assert set(captured[0].eef) == {"left_arm"}
