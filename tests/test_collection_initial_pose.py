"""Collection defaults to init, but never replaces live or replay poses with it."""

from types import SimpleNamespace

import numpy as np
import pytest

import robots  # noqa: F401
from core.app.console import server
from core.app.state import SessionMode, SessionState
from core.cfg import ConfigDict
from core.devices.hardware import HardwareCatalog
from core.registry import ROBOT_REGISTRY

pytestmark = pytest.mark.unit


def test_yam_zoo_matches_configured_initial_pose():
    robot = ROBOT_REGISTRY.build("dual_yam")
    config = HardwareCatalog().catalog["robot"]["dual_yam"]["config"]["robot"]
    np.testing.assert_allclose(robot.initial_qpos, config["initial_qpos"])
    assert not np.allclose(robot.initial_qpos, config["safe_qpos"])


@pytest.mark.parametrize("pose_source", ["initial", "live", "replay"])
def test_collection_initial_pose_ignores_stale_manual_preview(monkeypatch, pose_source):
    robot = ROBOT_REGISTRY.build("dual_yam")
    live = np.full(14, 0.2) if pose_source == "live" else None
    replay = np.full(14, 0.3) if pose_source == "replay" else None
    recorded = []
    scene = SimpleNamespace(transforms=lambda pose: recorded.append(np.asarray(pose).copy()) or {})
    runtime = SimpleNamespace(
        robot=robot,
        replay_source=None,
        collection_replay_qpos=replay,
    )
    session = SessionState()
    session.mode = SessionMode.MANUAL
    session.manual_qpos = np.zeros(14)
    session.sim_preview_qpos = np.ones(14)
    ctx = server.ConsoleContext(
        config=ConfigDict(),
        runtime=runtime,
        session=session,
        device_settings=SimpleNamespace(),
        scene=scene,
        active_tab="collect",
    )
    monkeypatch.setattr(
        server, "_observation_reader", lambda ctx: SimpleNamespace(get_latest_qpos=lambda: live)
    )
    monkeypatch.setattr(server, "_active_collection_replay_qpos", lambda runtime: (replay, 0))
    monkeypatch.setattr(server, "_collection_replay_progress", lambda *args: None)
    result = server._serialize_scene(ctx)
    assert result["available"]
    expected = replay if replay is not None else live if live is not None else robot.initial_qpos
    np.testing.assert_allclose(recorded[-1], expected)
