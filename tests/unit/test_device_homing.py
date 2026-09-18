"""Exercise startup homing without opening a transport or sending hardware commands."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.devices import startup

pytestmark = pytest.mark.unit


def fixture_state(feedback):
    runtime = SimpleNamespace(
        robot=SimpleNamespace(initial_qpos=np.array([0.0, 1.0]), gripper_indices=[1]),
        transport=SimpleNamespace(get_latest_qpos=lambda: feedback),
    )
    session = SimpleNamespace(manual_publish_active=False, manual_qpos=None)
    return runtime, session


@pytest.mark.parametrize("live", [np.array([np.nan, 0.4])])
def test_invalid_feedback_never_publishes(monkeypatch, live):
    runtime, session = fixture_state(None)

    def unexpected(*args):
        pytest.fail("Invalid feedback must not trigger motion")

    monkeypatch.setattr(startup, "_publish_manual_real_qpos", unexpected)
    with pytest.raises(ValueError, match="Invalid joint or gripper"):
        startup._home_robot(None, runtime, session, live)
    assert not session.manual_publish_active


def test_home_preserves_live_gripper(monkeypatch):
    runtime, session = fixture_state(np.array([0.0, 0.4]))
    targets = []

    def publish(config, runtime, session, target):
        assert session.manual_publish_active
        targets.append(target.copy())
        return True

    monkeypatch.setattr(startup, "_publish_manual_real_qpos", publish)
    startup._home_robot(None, runtime, session, np.array([0.2, 0.4]))
    np.testing.assert_allclose(targets[0], [0.0, 0.4])
    np.testing.assert_allclose(session.manual_qpos, [0.0, 0.4])
    assert not session.manual_publish_active


@pytest.mark.parametrize(
    "feedback,message",
    [
        (np.array([0.5, 0.4]), "did not reach"),
    ],
)
def test_failed_arrival_is_not_ready(monkeypatch, feedback, message):
    runtime, session = fixture_state(feedback)
    monkeypatch.setattr(startup, "_publish_manual_real_qpos", lambda *args: True)
    with pytest.raises(ValueError, match=message):
        startup._home_robot(None, runtime, session, np.array([0.2, 0.4]))
    assert session.manual_qpos is None
    assert not session.manual_publish_active


def test_real_robot_already_at_startup_pose_skips_duplicate_homing(monkeypatch):
    qpos = np.array([0.0, 1.0])
    workspace = SimpleNamespace(
        initial_selection=lambda config: {
            "robot": "dual_yam",
            "teleop": "vr_webxr",
            "camera": "none",
        },
        resolve=lambda selected: {"robot": {"mode": "real"}},
    )
    runtime = SimpleNamespace(
        robot=SimpleNamespace(initial_qpos=qpos.copy(), gripper_indices=[1]),
        console_ctx=SimpleNamespace(device_settings=SimpleNamespace(workspace=workspace)),
        command_queue=None,
        transport=SimpleNamespace(
            get_latest_qpos=lambda: qpos,
            seconds_since_last_recv=lambda: 0.1,
        ),
    )
    session = SimpleNamespace(interrupt_requested=False, manual_qpos=None, manual_real_qpos=None)
    calls = []

    def request(action, payload=None):
        calls.append((action, payload))
        if action == "status":
            return {"pids": {"robot": 123}, "processes": {"robot": None}}
        return {}

    service = SimpleNamespace(request=request)
    monkeypatch.setattr(
        startup,
        "_home_robot",
        lambda *args, **kwargs: pytest.fail("Already homed robot must not move again"),
    )

    startup.prepare_device(
        SimpleNamespace(robot={"initial_qpos": qpos.tolist()}),
        runtime,
        session,
        service,
        "robot",
        123,
    )

    np.testing.assert_allclose(session.manual_qpos, qpos)
    assert calls[-1] == ("ready", {"component": "robot", "pid": 123})
