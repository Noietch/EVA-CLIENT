"""Workstation composition, persisted overrides and local node lifecycle."""

import importlib
import json
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import numpy as np
import pytest
import yaml

import robots  # noqa: F401
from core.app.state import SessionStatus
from core.cfg import Config
from core.config import load_config
from core.devices import REPOSITORY_ROOT, DeviceWorkspace
from core.devices.service import DeviceProcesses, DeviceService
from core.registry import ROBOT_REGISTRY
from tests.integration.web._harness import console_config, serve_console

pytestmark = pytest.mark.integration


@pytest.fixture
def device_daemon(tmp_path):
    workspace_path = tmp_path / ("nested-workstation-" * 6) / "workstation.yaml"
    service = DeviceService(workspace_path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "core.devices.service",
            "--workspace",
            str(workspace_path),
        ],
        cwd=REPOSITORY_ROOT,
    )
    try:
        deadline = time.monotonic() + 10
        while not service.socket_path.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        yield workspace_path
    finally:
        process.terminate()
        process.wait(timeout=10)
        service.socket_path.unlink(missing_ok=True)
        service.socket_path.with_suffix(".lock").unlink(missing_ok=True)


def test_device_catalog_composes_and_restores_each_robot(tmp_path, monkeypatch):
    workspace = DeviceWorkspace(tmp_path / "workstation.yaml")
    for name in workspace.catalog["robot"]:
        robot = ROBOT_REGISTRY.build(name)
        selected = dict(robot=name, teleop="vr_webxr", camera="none")
        values = workspace.resolve(selected)
        values["teleop"]["client"]["position_scale"] = 0.7
        workspace.save(selected, values)
        restored = DeviceWorkspace(workspace.path)
        assert restored.saved["selected"] == selected
        assert restored.saved["overrides"]["robot"][name] == {}
        config = load_config(
            REPOSITORY_ROOT / "configs/00_base/defaults.py", workspace=restored
        )
        assert config.robot.type == name
        robot_defaults = workspace.catalog["robot"][name]["config"]
        assert set(robot_defaults["collection"]["schema"]["cameras"]) == {
            camera.observation_key for camera in robot.observation_schema.cameras
        }
        for key, value in robot_defaults.get("collection", {}).get("transport", {}).items():
            assert config.collection.transport[key] == value
        assert config.collection.teleop.client.position_scale == 0.7
        assert set(config.collection.teleop.client.arms) == {
            group.name for group in robot.arm_groups
        }
        assert config.transport.disabled_cameras == [
            camera.observation_key for camera in robot.observation_schema.cameras
        ]
        assert "collection_teleop_armed" not in restored.saved
        for kind, command in restored.commands().items():
            spec = restored.hardware.options(name)[kind][selected[kind]]["launch"]
            module = importlib.import_module(spec["module"])
            parser = getattr(module, spec.get("parser", "build_arg_parser"))()
            parsed = parser.parse_args(command[3:])
            if kind == "robot":
                assert parsed.obs_endpoint == config.transport.sub_endpoint

    selected = dict(robot="dual_yam", teleop="yam_leader", camera="yam_d405")
    values = workspace.resolve(selected)
    profile = workspace.catalog["camera"][selected["camera"]]["profiles"][0]
    data, settings = workspace.profile_values(selected["camera"], profile)
    values["camera"]["settings"].update(settings)
    assert all("index:" not in value for value in settings["camera"])
    values["robot"]["settings"]["gripper_limits_override"] = [0.0, 1.0]
    workspace.save(selected, values)
    from examples.hardware.yam.node import build_arg_parser, build_config

    node_config = build_config(build_arg_parser().parse_args(workspace.commands()["robot"][3:]))
    assert node_config.startup_position == "current"
    assert not node_config.direct_leader_control
    assert node_config.leader_can_channels
    camera_config = build_config(build_arg_parser().parse_args(workspace.commands()["camera"][3:]))
    assert len(camera_config.cameras) == 3
    assert camera_config.cameras[0].auto_exposure_limit_us == 33000
    selected["camera"] = "x5_d405"
    with pytest.raises(ValueError, match="not supported"):
        workspace.resolve(selected)

    from main import parse_args

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVA_WORKSTATION_PATH", str(workspace.path))
    monkeypatch.setattr(sys, "argv", ["eva"])
    config, _, _, _ = parse_args()
    assert config.robot.type == "dual_yam" and config.console.initial_tab == "manual"


def test_robot_hardware_defaults_and_saved_overrides_preserve_business_config(tmp_path):
    workspace = DeviceWorkspace(tmp_path / "workstation.yaml")
    selected = dict(robot="agilex_piper", teleop="vr_webxr", camera="external")
    values = workspace.resolve(selected)
    client = values["teleop"]["client"]
    assert client["gripper"]["open_value"] == 0.1
    assert client["arms"]["left_arm"]["workspace"]["min"] == [0.0, -0.55, -0.10]
    g2 = workspace.resolve(dict(robot="agibot_g2", teleop="vr_webxr", camera="none"))
    assert g2["teleop"]["client"]["gripper"]["close_value"] == -0.785
    values["robot"]["config"]["robot"]["gripper_threshold"] = 0.03
    workspace.save(selected, values)
    restored = DeviceWorkspace(workspace.path)
    raw = Config.fromfile(str(REPOSITORY_ROOT / "configs/00_base/defaults.py"))._cfg_dict
    raw.transport.update(image_mode="on_demand", image_height=320, convert_bgr_to_rgb=False)
    raw.collection.schema.cameras = {"cam_high": "observation.images.front"}
    config = restored.configure(raw)
    assert config.robot.gripper_threshold == 0.03
    assert config.transport.image_mode == "on_demand"
    assert config.transport.image_height == 320
    assert config.transport.convert_bgr_to_rgb is False
    assert config.collection.schema.cameras.cam_high == "observation.images.front"
    assert restored.resolve(selected)["robot"]["config"]["robot"]["gripper_threshold"] == 0.03
    raw.transport.type = "dataset"
    raw.robot.type = "agibot_g2"
    assert restored.configure(raw).robot.type == "agibot_g2"
    with pytest.raises(ValueError, match="not supported"):
        workspace.resolve(dict(robot="agibot_g2", teleop="yam_leader", camera="none"))


def test_disabled_cameras_use_camera_ids_independently_of_dataset_column_names(tmp_path):
    workspace = DeviceWorkspace(tmp_path / "workstation.yaml")
    selected = dict(robot="dual_franka", teleop="joint", camera="external")
    workspace.save(selected, workspace.resolve(selected))
    raw = Config.fromfile(str(REPOSITORY_ROOT / "configs/00_base/defaults.py"))._cfg_dict
    raw.collection.schema.cameras = {
        "cam_high": "observation.images.cam_left_wrist",
        "cam_left_wrist": "left_view",
    }
    config = workspace.configure(raw)
    assert config.collection.schema.cameras.cam_high == "observation.images.cam_left_wrist"
    assert "cam_left_wrist" not in config.collection.schema.cameras


def test_device_http_profiles_rejection_and_restart_request(tmp_path, monkeypatch, device_daemon):
    monkeypatch.setenv("EVA_WORKSTATION_PATH", str(device_daemon))
    with serve_console(console_config()) as console:
        response = console.get("/api/devices?robot=arx_x5&teleop=vr_webxr&camera=x5_d405")
        assert response.status == 200
        payload = response.json
        spec = payload["catalog"]["camera"]["x5_d405"]
        response = console.post(
            "/api/camera_profile", {"camera": "x5_d405", "path": spec["profiles"][0]}
        )
        assert response.status == 200
        data = response.json["data"]
        data["day"]["cam_high"]["exposure"] = 5000
        response = console.post(
            "/api/camera_profile", {"camera": "x5_d405", "content": yaml.safe_dump(data)}
        )
        assert response.status == 200
        payload["values"]["camera"]["settings"]["realsense_profile_path"] = response.json["path"]
        assert (
            console.post("/api/camera_profile", {"camera": "x5_d405", "path": "/etc/passwd"}).status
            == 400
        )
        assert (
            console.post("/api/camera_profile", {"camera": "x5_d405", "content": "[]"}).status
            == 400
        )
        console.runtime.collection_teleop_armed = True
        console.do("/api/device_selection", payload)
        assert console.get("/api/device_settings").json["state"] == "failed"
        assert not device_daemon.exists()
        console.runtime.collection_teleop_armed = False
        console.do("/api/device_selection", payload)
        assert console.session.status is SessionStatus.EXIT
        assert console.runtime.console_ctx.device_settings.restart_requested
        workspace = DeviceWorkspace(device_daemon)
        from examples.hardware.arx_x5.node import build_arg_parser, build_config

        config = build_config(build_arg_parser().parse_args(workspace.commands()["camera"][3:]))
        assert config.realsense_cameras[0].color_profile.exposure == 5000


def test_cli_and_eva_restart_preserve_real_vr_service(tmp_path, monkeypatch, device_daemon):
    monkeypatch.setenv("EVA_WORKSTATION_PATH", str(device_daemon))
    workspace = DeviceWorkspace(device_daemon)
    selected = dict(robot="agibot_g2", teleop="vr_webxr", camera="none")
    values = workspace.resolve(selected)
    ports = []
    for _ in range(6):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    values["teleop"]["settings"]["port"] = ports[0]
    values["teleop"]["client"]["endpoint"] = f"tcp://127.0.0.1:{ports[1]}"
    values["teleop"]["client"]["ack_endpoint"] = f"tcp://127.0.0.1:{ports[2]}"
    values["robot"]["settings"].update(
        obs_endpoint=f"tcp://127.0.0.1:{ports[3]}",
        action_endpoint=f"tcp://127.0.0.1:{ports[4]}",
    )
    workspace.save(selected, values)
    service = DeviceService(workspace.path)
    http = build_opener(ProxyHandler({}))
    cli = [sys.executable, "src/main.py", "device", "--workspace", str(workspace.path)]
    eva = None
    try:
        started = subprocess.run(
            cli + ["start", "teleop"], capture_output=True, text=True, check=True
        )
        assert "token=" not in started.stdout + started.stderr
        pid = service.request("status")["pids"]["teleop"]
        subprocess.run(cli + ["start", "teleop"], capture_output=True, check=True)
        assert service.request("status")["pids"]["teleop"] == pid
        for attempt in range(2):
            with (tmp_path / f"eva-{attempt}.log").open("wb") as log:
                eva = subprocess.Popen(
                    [sys.executable, "src/main.py", "--web-port", str(ports[5])],
                    stdout=log,
                    stderr=log,
                    cwd=REPOSITORY_ROOT,
                )
            deadline = time.monotonic() + 45
            while True:
                try:
                    with http.open(
                        f"http://127.0.0.1:{ports[5]}/api/status", timeout=1
                    ) as response:
                        status = json.load(response)
                    break
                except URLError:
                    assert eva.poll() is None, (tmp_path / f"eva-{attempt}.log").read_text()
                    assert time.monotonic() < deadline
                    time.sleep(0.1)
            assert not status["collection_teleop_armed"]
            assert not status["manual_publish_active"]
            with http.open(f"http://127.0.0.1:{ports[5]}/api/devices", timeout=2) as response:
                assert json.load(response)["selected"] == selected
            subprocess.run(
                cli + ["--url", f"http://127.0.0.1:{ports[5]}", "start", "teleop"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            assert service.request("status")["pids"]["teleop"] == pid
            if attempt == 0:
                subprocess.run(
                    cli
                    + [
                        "--url",
                        f"http://127.0.0.1:{ports[5]}",
                        "set",
                        "teleop.position_scale",
                        "0.85",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
                assert (
                    DeviceWorkspace(workspace.path).resolve(selected)["teleop"]["client"][
                        "position_scale"
                    ]
                    == 0.85
                )
            eva.kill()
            eva.wait(timeout=10)
            eva = None
            status = service.request("status")
            assert status["pids"]["teleop"] == pid and status["processes"]["teleop"] is None
            with http.open(status["browser_url"], timeout=2) as response:
                assert response.status == 200
            assert "token=" not in status["log"]
    finally:
        if eva is not None:
            eva.kill()
            eva.wait(timeout=10)
        service.request("stop")
    assert service.request("status")["processes"] == {}


def test_camera_restart_keeps_robot_process_and_expires_frames(tmp_path, monkeypatch):
    from core.devices.camera import CameraSource

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{sock.getsockname()[1]}"
    workspace = DeviceWorkspace(tmp_path / "workstation.yaml")
    selected = dict(robot="dual_yam", teleop="joint", camera="yam_d405")
    workspace.save(selected, workspace.resolve(selected))
    camera_script = (
        "from types import SimpleNamespace; import numpy as np; "
        "from core.devices.camera import CameraPublisher; "
        "cache = SimpleNamespace("
        "snapshot=lambda: {'cam_high': np.full((8, 12, 3), VALUE, np.uint8)}, close=lambda: None); "
        f"CameraPublisher((cache,), {endpoint!r}).run()"
    )
    commands = {
        "robot": [
            sys.executable,
            "-c",
            "import time; print('robot ready', flush=True); time.sleep(60)",
        ],
        "camera": [sys.executable, "-c", camera_script.replace("VALUE", "20")],
    }
    monkeypatch.setattr(DeviceWorkspace, "commands", lambda self: dict(commands))
    manager = DeviceProcesses(workspace.path)
    source = CameraSource(endpoint)
    try:
        manager.start()
        robot = manager.processes["robot"]
        first_camera = manager.processes["camera"]
        for value in (20, 80):
            deadline = time.monotonic() + 10
            while True:
                images = source.snapshot()
                if images and np.all(images["cam_high"] == value):
                    break
                assert time.monotonic() < deadline, manager.status()["log"]
                time.sleep(0.05)
            assert manager.processes["robot"] is robot and robot.poll() is None
            if value == 20:
                manager.stop("camera")
                assert first_camera.poll() is not None
                deadline = time.monotonic() + 0.7
                while time.monotonic() < deadline:
                    source.snapshot()
                    time.sleep(0.05)
                assert source.snapshot() == {}
                assert manager.status()["state"] == "started"
                commands["camera"] = [sys.executable, "-c", camera_script.replace("VALUE", "80")]
                manager.start("camera")
                assert manager.processes["camera"].pid != first_camera.pid
        assert "robot ready" in manager.status("robot")["log"]
        assert "robot ready" not in manager.status("camera")["log"]
    finally:
        manager.stop()
        source.close()
    assert robot.poll() is not None and manager.status()["state"] == "stopped"
