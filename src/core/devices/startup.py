"""Readiness checks for explicit device starts from the console."""

import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import numpy as np

from core.app.handlers.control import _publish_manual_real_qpos, poll_motion_commands
from core.devices.camera import CameraSource


def resolve_pose(config, robot, name):
    """Resolve configured initial or safe qpos and validate its shape."""
    robot_config = config.robot
    if name == "initial_qpos":
        values = robot_config.get("initial_qpos")
        target = np.asarray(robot.initial_qpos if values is None else values, dtype=np.float32)
    elif name == "safe_qpos":
        values = robot_config.get("safe_qpos")
        target = np.asarray(
            robot_config.get("initial_qpos") if values is None else values,
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown robot pose: {name}")
    if target.shape != robot.initial_qpos.shape or not np.isfinite(target).all():
        raise ValueError(f"robot.{name} must contain finite values for every joint")
    return target.copy()


def park_robot(config, runtime, session):
    """Normal shutdown: retain power unless fresh feedback confirms the safe pose."""
    target = resolve_pose(config, runtime.robot, "safe_qpos")

    def feedback():
        qpos = runtime.transport.get_latest_qpos()
        age = runtime.transport.seconds_since_last_recv()
        if qpos is None or age is None or age > 0.5:
            raise ValueError("Safe stop aborted: missing or stale feedback; power-off withheld")
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != target.shape or not np.isfinite(qpos).all():
            raise ValueError("Safe stop aborted: invalid feedback; power-off withheld")
        return qpos

    live = feedback()
    grippers = list(runtime.robot.gripper_indices)
    target[grippers] = live[grippers]
    session.manual_real_qpos = live.copy()
    session.manual_publish_active = True
    try:
        if not _publish_manual_real_qpos(config, runtime, session, target, feedback_guard=feedback):
            raise ValueError("Safe stop interrupted; power-off withheld")
        # Require multiple fresh observations, not a single cached target echo.
        previous_stamp = None
        previous_qpos = None
        stable = 0
        deadline = time.monotonic() + 3.0
        rate = runtime.transport.create_rate(config.inference_cfg.publish_rate)
        while time.monotonic() < deadline:
            if poll_motion_commands(config, runtime, session):
                raise ValueError("Safe stop interrupted; power-off withheld")
            actual = feedback()
            age = runtime.transport.seconds_since_last_recv()
            stamp = time.monotonic() - age
            if previous_stamp is None or stamp > previous_stamp + 0.0001:
                slow = (
                    previous_qpos is not None
                    and np.max(np.abs(actual - previous_qpos)) / max(stamp - previous_stamp, 0.0001)
                    < 0.05
                )
                stable = (
                    stable + 1 if slow and np.allclose(actual, target, atol=0.03, rtol=0) else 0
                )
                previous_stamp, previous_qpos = stamp, actual.copy()
                if stable >= 3:
                    session.manual_qpos = actual.copy()
                    return
            runtime.transport.publish_action(target, target="real")
            rate.sleep()
        raise ValueError("Robot did not settle at safe_qpos; power-off withheld")
    finally:
        session.manual_publish_active = False


def stop_devices(config, runtime, session, service, component):
    current = service.request("status")
    if component in {None, "robot"} and current["processes"].get("robot", -1) is None:
        session.interrupt_requested = False
        park_robot(config, runtime, session)
    return service.request("stop", {"component": component})


def prepare_device(config, runtime, session, service, component, pid, *, timeout=60):
    """Wait for live input; robot readiness also requires a verified move home."""
    deadline = time.monotonic() + timeout
    camera = None
    workspace = runtime.console_ctx.device_settings.workspace
    selected = workspace.initial_selection(config)
    values = workspace.resolve(selected)
    robot_mode = str(values["robot"].get("mode", "real")).lower()
    http = build_opener(ProxyHandler({}))
    if component == "camera":
        camera = CameraSource(values["robot"]["settings"]["camera_endpoint"])
    try:
        while time.monotonic() < deadline:
            if poll_motion_commands(config, runtime, session):
                raise ValueError("Device startup interrupted")
            status = service.request("status", {"component": component})
            if (
                status["pids"].get(component) != pid
                or status["processes"].get(component, -1) is not None
            ):
                raise ValueError(status["error"] or "Device process exited during startup")
            if component == "robot":
                qpos = runtime.transport.get_latest_qpos()
                if qpos is not None:
                    # The fake node starts at its configured initial pose and does
                    # not need real-hardware homing or a freshness-gated command.
                    if robot_mode != "fake":
                        _home_robot(config, runtime, session, qpos)
                    break
            elif component == "camera":
                images = camera.snapshot()
                expected = set(config.collection.schema.cameras)
                if images and expected.issubset(images):
                    break
            elif status.get("browser_url"):
                try:
                    with http.open(status["browser_url"], timeout=0.5) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError, OSError):
                    pass
            else:
                raise ValueError("No readiness check is available for this operation")
            time.sleep(0.1)
        else:
            raise ValueError(f"{component} startup timed out waiting for live feedback")
        service.request("ready", {"component": component, "pid": pid})
    finally:
        if camera is not None:
            camera.close()


def _home_robot(config, runtime, session, qpos):
    robot = runtime.robot
    live = np.asarray(qpos, dtype=np.float32)
    target = (
        resolve_pose(config, robot, "initial_qpos")
        if config is not None
        else np.asarray(robot.initial_qpos, dtype=np.float32).copy()
    )
    if live.shape != target.shape or not np.isfinite(live).all() or not np.isfinite(target).all():
        raise ValueError("Invalid joint or gripper feedback; robot startup cancelled")
    # Homing must not squeeze an object: inspect, then preserve live gripper positions.
    for index in robot.gripper_indices:
        target[index] = live[index]
    session.manual_real_qpos = live.copy()
    session.manual_publish_active = True
    try:

        def check_feedback():
            current = runtime.transport.get_latest_qpos()
            age = runtime.transport.seconds_since_last_recv()
            if current is None or age is None or age > 0.5:
                raise ValueError("Robot homing aborted: missing or stale joint feedback")
            current = np.asarray(current, dtype=np.float32)
            if current.shape != target.shape or not np.isfinite(current).all():
                raise ValueError("Robot homing aborted: invalid joint or gripper feedback")

        kwargs = {"feedback_guard": check_feedback} if config is not None else {}
        if not _publish_manual_real_qpos(config, runtime, session, target, **kwargs):
            raise ValueError("Robot homing interrupted")
        feedback = runtime.transport.get_latest_qpos()
        if feedback is None:
            raise ValueError("Robot feedback lost during homing")
        feedback = np.asarray(feedback, dtype=np.float32)
        if feedback.shape != target.shape or not np.isfinite(feedback).all():
            raise ValueError("Invalid joint or gripper feedback after homing")
        if not np.allclose(feedback, target, atol=0.08, rtol=0):
            raise ValueError("Robot did not reach its initial position")
        session.manual_qpos = feedback.copy()
    finally:
        session.manual_publish_active = False
