"""Normal Stop must park successfully before releasing the hardware process."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.cfg import ConfigDict
from core.devices import startup

pytestmark = pytest.mark.unit


def make_config():
    return ConfigDict(
        robot=dict(initial_qpos=[0.5, 1.0], safe_qpos=[0.0, 1.0]), inference_cfg=dict(publish_rate=30)
    )


def test_pose_defaults_and_validation():
    robot = SimpleNamespace(initial_qpos=np.array([0.2, 1.0]))
    config = make_config()
    np.testing.assert_allclose(startup.resolve_pose(config, robot, "safe_qpos"), [0, 1])
    config.robot.safe_qpos = None
    np.testing.assert_allclose(startup.resolve_pose(config, robot, "safe_qpos"), [0.5, 1])
    config.robot.safe_qpos = [float("nan"), 1]
    with pytest.raises(ValueError, match="finite"):
        startup.resolve_pose(config, robot, "safe_qpos")


def test_every_robot_has_explicit_pose_overrides():
    import robots  # noqa: F401
    from core.devices.hardware import HardwareCatalog
    from core.registry import ROBOT_REGISTRY

    catalog = HardwareCatalog().catalog["robot"]
    assert catalog
    for name, spec in catalog.items():
        robot = ROBOT_REGISTRY.build(name)
        configured = spec["config"]["robot"]
        assert isinstance(configured.get("initial_qpos"), list), name
        assert isinstance(configured.get("safe_qpos"), list), name
        initial = startup.resolve_pose(ConfigDict(robot=configured), robot, "initial_qpos")
        safe = startup.resolve_pose(ConfigDict(robot=configured), robot, "safe_qpos")
        assert initial.shape == robot.initial_qpos.shape, name
        assert safe.shape == robot.initial_qpos.shape, name
        np.testing.assert_allclose(initial, robot.initial_qpos, err_msg=name)
        assert np.isfinite(safe).all(), name
        if name in {"dual_yam", "arx_x5"}:
            joint_mask = np.ones(safe.shape, dtype=bool)
            joint_mask[list(robot.gripper_indices)] = False
            np.testing.assert_array_equal(safe[joint_mask], 0, err_msg=name)
        else:
            np.testing.assert_allclose(safe, robot.initial_qpos, err_msg=name)


@pytest.mark.parametrize("fails", [True, False])
def test_stop_order_and_failure_keeps_power(monkeypatch, fails):
    calls = []

    def request(action, payload=None):
        calls.append(action)
        return {"processes": {"robot": None}}

    def park(*args):
        calls.append("park")
        if fails:
            raise ValueError("Safe position not reached")

    monkeypatch.setattr(startup, "park_robot", park)
    session = SimpleNamespace(interrupt_requested=True)
    service = SimpleNamespace(request=request)
    if fails:
        with pytest.raises(ValueError, match="not reached"):
            startup.stop_devices(None, None, session, service, "robot")
        assert calls == ["status", "park"]
    else:
        startup.stop_devices(None, None, session, service, "robot")
        assert calls == ["status", "park", "stop"]


def test_camera_stop_never_moves_robot(monkeypatch):
    monkeypatch.setattr(startup, "park_robot", lambda *args: pytest.fail("Unexpected motion"))
    service = SimpleNamespace(request=lambda *args: {"processes": {"robot": None}})
    startup.stop_devices(None, None, None, service, "camera")


@pytest.mark.parametrize(
    "age,actual,message",
    [
        (1.0, [0.0, 0.4], "stale"),
        (0.0, [float("nan"), 0.4], "invalid"),
        (0.0, [0.2, 0.4], "did not settle"),
        (0.0, [0.0, 0.4], None),
    ],
)
def test_parking_feedback_gate(monkeypatch, age, actual, message):
    clock = iter(np.arange(0, 30, 0.05))
    monkeypatch.setattr(startup.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(startup, "poll_motion_commands", lambda *args: False)
    targets = []

    def move(config, runtime, session, target, *, feedback_guard):
        feedback_guard()
        targets.append(target.copy())
        return True

    monkeypatch.setattr(startup, "_publish_manual_real_qpos", move)
    runtime = SimpleNamespace(
        robot=SimpleNamespace(initial_qpos=np.array([0.5, 1.0]), gripper_indices=(1,)),
        transport=SimpleNamespace(
            get_latest_qpos=lambda: np.array(actual),
            seconds_since_last_recv=lambda: age,
            create_rate=lambda hz: SimpleNamespace(sleep=lambda: None),
            publish_action=lambda *args, **kwargs: None,
        ),
    )
    session = SimpleNamespace(manual_publish_active=False, manual_qpos=None)
    if message:
        with pytest.raises(ValueError, match=message):
            startup.park_robot(make_config(), runtime, session)
    else:
        startup.park_robot(make_config(), runtime, session)
        np.testing.assert_allclose(targets[0], [0.0, 0.4])
        np.testing.assert_allclose(session.manual_qpos, [0.0, 0.4])
    assert not session.manual_publish_active
