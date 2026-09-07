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


@pytest.mark.parametrize("live", [np.array([np.nan, 0.4]), np.array([0.2])])
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
        (None, "feedback lost"),
        (np.array([np.nan, 0.4]), "Invalid joint"),
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


def test_interrupted_home_is_not_ready(monkeypatch):
    runtime, session = fixture_state(np.array([0.0, 0.4]))
    monkeypatch.setattr(startup, "_publish_manual_real_qpos", lambda *args: False)
    with pytest.raises(ValueError, match="interrupted"):
        startup._home_robot(None, runtime, session, np.array([0.2, 0.4]))
    assert not session.manual_publish_active
    assert session.manual_qpos is None


def test_fake_robot_readiness_skips_homing(monkeypatch):
    qpos = np.array([0.0, 0.4])
    workspace = SimpleNamespace(
        initial_selection=lambda config: {
            "robot": "dual_yam",
            "teleop": "vr_webxr",
            "camera": "none",
        },
        resolve=lambda selected: {"robot": {"mode": "fake"}},
    )
    runtime = SimpleNamespace(
        console_ctx=SimpleNamespace(device_settings=SimpleNamespace(workspace=workspace)),
        command_queue=None,
        transport=SimpleNamespace(get_latest_qpos=lambda: qpos),
    )
    session = SimpleNamespace(interrupt_requested=False)
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
        lambda *args, **kwargs: pytest.fail("Fake robot must not run hardware homing"),
    )

    startup.prepare_device(SimpleNamespace(), runtime, session, service, "robot", 123)

    assert calls[-1] == ("ready", {"component": "robot", "pid": 123})


def test_real_robot_readiness_waits_for_fresh_feedback(monkeypatch):
    qpos = np.array([0.0, 0.4])
    workspace = SimpleNamespace(
        initial_selection=lambda config: {
            "robot": "dual_yam",
            "teleop": "vr_webxr",
            "camera": "none",
        },
        resolve=lambda selected: {"robot": {"mode": "real"}},
    )
    feedback_ages = iter([5.0, None, 0.1])
    observed_ages = []

    def seconds_since_last_recv():
        age = next(feedback_ages)
        observed_ages.append(age)
        return age

    runtime = SimpleNamespace(
        console_ctx=SimpleNamespace(device_settings=SimpleNamespace(workspace=workspace)),
        command_queue=None,
        transport=SimpleNamespace(
            get_latest_qpos=lambda: qpos,
            seconds_since_last_recv=seconds_since_last_recv,
        ),
    )
    session = SimpleNamespace(interrupt_requested=False)
    calls = []

    def request(action, payload=None):
        calls.append((action, payload))
        if action == "status":
            return {"pids": {"robot": 123}, "processes": {"robot": None}}
        return {}

    def home(*args):
        assert observed_ages == [5.0, None, 0.1]

    service = SimpleNamespace(request=request)
    monkeypatch.setattr(startup, "_home_robot", home)
    monkeypatch.setattr(startup.time, "sleep", lambda _: None)

    startup.prepare_device(SimpleNamespace(), runtime, session, service, "robot", 123)

    assert calls[-1] == ("ready", {"component": "robot", "pid": 123})


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


def test_camera_readiness_accepts_frames_arriving_in_separate_messages(monkeypatch):
    snapshots = iter(
        [
            {"cam_left_wrist": object()},
            {"cam_right_wrist": object()},
            {"cam_high": object()},
        ]
    )

    class FakeCamera:
        def __init__(self, endpoint):
            assert endpoint == "tcp://127.0.0.1:5557"

        def snapshot(self):
            return next(snapshots)

        def close(self):
            pass

    workspace = SimpleNamespace(
        initial_selection=lambda config: {
            "robot": "dual_yam",
            "teleop": "vr_webxr",
            "camera": "yam_orbbec",
        },
        resolve=lambda selected: {
            "robot": {"settings": {"camera_endpoint": "tcp://127.0.0.1:5557"}},
            "camera": {
                "settings": {
                    "camera": ["cam_high=260422275306"],
                    "orbbec_camera": [
                        "cam_left_wrist=CV2R1610003Z",
                        "cam_right_wrist=CV2L360000CL",
                    ]
                }
            },
        },
    )
    runtime = SimpleNamespace(
        console_ctx=SimpleNamespace(device_settings=SimpleNamespace(workspace=workspace)),
        command_queue=None,
    )
    session = SimpleNamespace(interrupt_requested=False)
    calls = []

    def request(action, payload=None):
        calls.append((action, payload))
        if action == "status":
            return {"pids": {"camera": 123}, "processes": {"camera": None}}
        return {}

    clock = iter([index * 0.1 for index in range(1, 30)])
    monkeypatch.setattr(startup, "CameraSource", FakeCamera)
    monkeypatch.setattr(startup.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(startup.time, "sleep", lambda _: None)

    config = SimpleNamespace(
        collection=SimpleNamespace(
            schema=SimpleNamespace(
                cameras={
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                }
            )
        )
    )
    startup.prepare_device(
        config,
        runtime,
        session,
        SimpleNamespace(request=request),
        "camera",
        123,
    )

    assert calls[-1] == ("ready", {"component": "camera", "pid": 123})
