from __future__ import annotations

import dataclasses
import fcntl
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from examples.hardware.yam.robot import (
    YamFollowers,
    map_leader_gripper,
)
from examples.hardware.yam.wire import (
    WireAction as YamWireAction,
)
from robots.utils import UrdfScene

pytestmark = pytest.mark.integration


def _rendered_finger_separation(urdf) -> float:
    centroids = []
    for geometry_name in ("geometry_7", "geometry_8"):
        transform, _ = urdf.scene.graph.get(frame_to=geometry_name)
        mesh = urdf.scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        centroids.append(mesh.centroid)
    return float(abs(centroids[0][1] - centroids[1][1]))


@dataclasses.dataclass(frozen=True)
class _Config:
    group_names: tuple[str, ...] = ("left_arm", "right_arm")
    follower_can_channels: dict[str, str] = dataclasses.field(
        default_factory=lambda: {"left_arm": "can0", "right_arm": "can1"}
    )
    leader_can_channels: dict[str, str] = dataclasses.field(default_factory=dict)
    leader_gripper_encoder_endpoints: dict[str, tuple[float, float]] = dataclasses.field(
        default_factory=dict
    )
    arm_type: str = "yam"
    gripper_type: str = "linear_4310"
    enable_auto_recovery: bool = False
    command_timeout_s: float = 0.5
    idle_mode: str = "gravity_comp"
    startup_position: str = "current"
    startup_duration_s: float = 5.0
    joint4_kp: float | None = None
    end_effector_mass: float | None = None
    gravity_comp_factor: tuple[float, ...] | None = None
    gripper_limits_override: tuple[float, float] | None = None
    gripper_kp: float = 5.0
    gripper_damping: float = 0.5
    gripper_close_torque_limit: float = 0.29
    gripper_open_torque_limit: float = 0.29
    gripper_max_speed: float = 0.0
    tracking_ki: float = 0.0
    tracking_trim_limit: float = 0.12
    tracking_deadband: float = 0.002
    tracking_settle_delay_s: float = 0.15
    startup_trim_duration_s: float = 3.0


class _FakeYam:
    def __init__(self) -> None:
        self.qpos = np.asarray([0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.idle_count = 0
        self.closed = False
        self.last_command: np.ndarray | None = None
        self.last_kp: np.ndarray | None = None
        self.last_kd: np.ndarray | None = None

    def num_dofs(self) -> int:
        return 7

    def get_joint_pos(self) -> np.ndarray:
        return self.qpos.copy()

    def command_joint_pos(self, qpos: np.ndarray) -> None:
        self.last_command = np.asarray(qpos, dtype=np.float64).copy()
        self.qpos = self.last_command.copy()

    def enter_gravity_comp_idle(self) -> None:
        self.idle_count += 1

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self.last_kp = np.asarray(kp, dtype=np.float64).copy()
        self.last_kd = np.asarray(kd, dtype=np.float64).copy()

    def close(self) -> None:
        self.closed = True


def test_yam_scene_matches_black_joint_white_arm_finish() -> None:
    robot = ROBOT_REGISTRY.build("dual_yam")
    scene = UrdfScene(robot)
    urdf = scene._urdfs[str(robot.urdf)]

    for name, link in urdf.link_map.items():
        visual = link.visuals[0]
        assert visual.geometry.mesh is not None
        assert (robot.urdf.parent / visual.geometry.mesh.filename).is_file()
        expected = [0.88] * 3 if name in {"link2", "link3"} else [0.2] * 3
        assert visual.material.color.rgba[:3] == pytest.approx(expected)
    meshes = scene.static_meshes()
    assert len(meshes) == 9
    assert sum(np.mean(mesh["color"]) > 0.8 for mesh in meshes) == 2
    assert all(len(mesh.faces) > 1000 for mesh in urdf.scene.geometry.values())


def test_yam_official_model_gripper_opening_and_independent_arms() -> None:
    robot = ROBOT_REGISTRY.build("dual_yam")
    scene = UrdfScene(robot)
    urdf = scene._urdfs[str(robot.urdf)]
    part = robot.vis_config.parts[0]
    assert urdf.actuated_joint_names == [f"joint{index}" for index in range(1, 9)]
    separations = []
    for value in (0.0, 1.0):
        urdf.update_cfg(part.qpos_to_cfg(np.array([0.0] * 6 + [value])))
        separations.append(_rendered_finger_separation(urdf))
    assert separations[1] - separations[0] > 0.08
    assert separations[1] > 0.08

    physical_scene = UrdfScene(robot, gripper_open=1.0, gripper_close=0.0)
    physical_urdf = physical_scene._urdfs[str(robot.urdf)]
    physical_part = robot.vis_config.parts[0]
    physical_separations = []
    for value in (0.0, 1.0):
        visual_qpos = physical_scene._part_transforms(
            physical_part,
            np.array([0.0] * 6 + [value]),
            np.eye(4),
        )
        assert visual_qpos
        physical_separations.append(_rendered_finger_separation(physical_urdf))
    assert physical_separations[1] > physical_separations[0]
    assert physical_separations[1] > 0.08

    qpos = robot.initial_qpos.copy()
    before = scene.transforms(qpos)
    qpos[0] = 0.5
    after = scene.transforms(qpos)
    assert before["right_arm"] == after["right_arm"]
    assert any(
        not np.allclose(before["left_arm"][name], after["left_arm"][name])
        for name in before["left_arm"]
    )


def test_yam_leader_gripper_uses_one_for_open() -> None:
    endpoints = (-0.7, 0.0)
    assert map_leader_gripper(-0.7, endpoints) == pytest.approx(1.0)
    assert map_leader_gripper(0.0, endpoints) == pytest.approx(0.0)
    assert map_leader_gripper(-0.35, endpoints) == pytest.approx(0.5)


def test_follower_commands_both_arms_and_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    robots_by_channel: dict[str, _FakeYam] = {}
    factory_kwargs: dict[str, dict[str, object]] = {}

    def factory(**kwargs: object) -> _FakeYam:
        robot = _FakeYam()
        channel = str(kwargs["channel"])
        robots_by_channel[channel] = robot
        factory_kwargs[channel] = kwargs
        return robot

    clock = 10.0
    monkeypatch.setattr("examples.hardware.yam.robot.time.monotonic", lambda: clock)
    followers = YamFollowers(_Config(joint4_kp=40.0), factory=factory)
    target = np.asarray(
        [0.1, 0.4, 0.5, 0.1, 0.1, 0.1, 0.25] * 2,
        dtype=np.float32,
    )
    followers.apply_action(YamWireAction(t=clock, action=target, target="real"))

    assert robots_by_channel["can0"].qpos == pytest.approx(target[:7])
    assert robots_by_channel["can1"].qpos == pytest.approx(target[7:])
    assert robots_by_channel["can0"].last_kp == pytest.approx([80, 80, 80, 40, 10, 10, 5])
    assert robots_by_channel["can0"].last_kd == pytest.approx([5, 5, 5, 1.5, 1.5, 1.5, 0.5])
    for kwargs in factory_kwargs.values():
        assert kwargs["gripper_close_torque_limit"] == 0.29
        assert kwargs["gripper_open_torque_limit"] == 0.29
        assert kwargs["gripper_damping"] == 0.5
    assert factory_kwargs["can0"]["gripper_kp"] == 5.0
    assert factory_kwargs["can1"]["gripper_kp"] == 5.0

    clock = 10.6
    followers.watchdog_tick()
    assert robots_by_channel["can0"].idle_count == 1
    assert robots_by_channel["can1"].idle_count == 1


def test_follower_limits_only_gripper_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    robots_by_channel: dict[str, _FakeYam] = {}

    def factory(**kwargs: object) -> _FakeYam:
        robot = _FakeYam()
        robots_by_channel[str(kwargs["channel"])] = robot
        return robot

    clock = 10.0
    monkeypatch.setattr("examples.hardware.yam.robot.time.monotonic", lambda: clock)
    followers = YamFollowers(_Config(gripper_max_speed=2.0), factory=factory)
    target = np.asarray(
        [0.1, 0.4, 0.5, 0.1, 0.1, 0.1, 0.0] * 2,
        dtype=np.float32,
    )

    followers.apply_action(YamWireAction(t=clock, action=target, target="real"))
    assert robots_by_channel["can0"].qpos[:6] == pytest.approx(target[:6])
    assert robots_by_channel["can0"].qpos[6] == pytest.approx(1.0)

    clock = 10.05
    followers.watchdog_tick()
    assert robots_by_channel["can0"].qpos[:6] == pytest.approx(target[:6])
    assert robots_by_channel["can0"].qpos[6] == pytest.approx(0.9)

    clock = 10.1
    followers.watchdog_tick()
    assert robots_by_channel["can0"].qpos[6] == pytest.approx(0.8)
    assert robots_by_channel["can1"].qpos[6] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--gripper-close-torque-limit", "0"),
        ("--gripper-close-torque-limit", "nan"),
        ("--gripper-open-torque-limit", "-0.1"),
        ("--gripper-open-torque-limit", "inf"),
        ("--gripper-damping", "0"),
        ("--gripper-kp", "nan"),
    ],
)
def test_gripper_impedance_rejects_invalid_parameters(flag: str, value: str) -> None:
    from examples.hardware.yam.node import build_arg_parser, build_config

    args = build_arg_parser().parse_args(["--gripper-limits-override", "0", "6.57", flag, value])
    with pytest.raises(ValueError, match=flag):
        build_config(args)


def test_run_hardware_defaults_to_dual_leaders_and_gripper_calibration(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]

    fake_venv = tmp_path / "venv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)
    captured_args = tmp_path / "node-args.txt"
    captured_environment = tmp_path / "node-environment.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$@" > "$YAM_TEST_ARGS_PATH"\n'
        'printf \'%s\\n\' "${PYTHONPATH-}" "${VIRTUAL_ENV-}" "${PATH-}" '
        '"${PYTHONHOME-}" "${PYTHONNOUSERSITE-}" > "$YAM_TEST_ENV_PATH"\n'
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    for name in (
        "ENABLE_YAM_LEADERS",
        "YAM_ALLOW_GRIPPER_CALIBRATION",
        "YAM_GRIPPER_CLOSE_TORQUE_LIMIT",
        "YAM_GRIPPER_OPEN_TORQUE_LIMIT",
        "YAM_GRIPPER_DAMPING",
        "YAM_GRIPPER_MAX_SPEED",
        "YAM_GRIPPER_KP",
        "YAM_GRIPPER_LIMITS",
        "LEFT_LEADER_GRIPPER_ENDPOINTS",
        "RIGHT_LEADER_GRIPPER_ENDPOINTS",
        "YAM_CAN_SERIAL_LEFT_FOLLOWER",
        "YAM_CAN_SERIAL_RIGHT_FOLLOWER",
        "YAM_CAN_SERIAL_LEFT_LEADER",
        "YAM_CAN_SERIAL_RIGHT_LEADER",
        "YAM_STARTUP_POSITION",
        "D405_CAM_HIGH_SERIAL",
        "D405_CAMERA_WIDTH",
        "D405_CAMERA_HEIGHT",
        "D405_CAMERA_FPS",
        "D405_CAMERA_TIMEOUT_MS",
        "D405_CAMERA_PROFILE",
        "D405_ENABLED_CAMERAS",
        "D405_CAMERA_AUTO_EXPOSURE_LIMIT_US",
        "D405_CAMERA_AUTO_GAIN_LIMIT",
        "D405_CAMERA_EXPOSURE_US",
        "D405_CAMERA_WARMUP_FRAMES",
        "ORBBEC_CAM_LEFT_WRIST_SERIAL",
        "ORBBEC_CAM_RIGHT_WRIST_SERIAL",
        "ORBBEC_CAM_HIGH_SERIAL",
        "ORBBEC_CAMERA_WIDTH",
        "ORBBEC_CAMERA_HEIGHT",
        "ORBBEC_CAMERA_FPS",
        "ORBBEC_CAMERA_FORMAT",
        "ORBBEC_CAMERA_TIMEOUT_MS",
        "ORBBEC_CAMERA_WARMUP_FRAMES",
        "ORBBEC_CAMERA_BRIGHTNESS",
        "ORBBEC_POWER_LINE_FREQUENCY",
        "ORBBEC_ENABLED_CAMERAS",
        "ORBBEC_USB_RESET",
        "YAM_CPU_AFFINITY",
        "YAM_SIM",
        "ENABLE_YAM_CAMERAS",
    ):
        env.pop(name, None)
    env.update(
        YAM_VENV_DIR=str(fake_venv),
        YAM_TEST_ARGS_PATH=str(captured_args),
        YAM_TEST_ENV_PATH=str(captured_environment),
        PYTHONHOME="/tmp/forbidden-python-home",
        PYTHONPATH="/tmp/forbidden-python-path",
        LEFT_FOLLOWER_CAN="test_follower_l",
        RIGHT_FOLLOWER_CAN="test_follower_r",
        LEFT_LEADER_CAN="test_leader_l",
        RIGHT_LEADER_CAN="test_leader_r",
    )

    subprocess.run(
        ["bash", "examples/hardware/yam/run_hardware.sh", "--help"],
        cwd=root,
        env=env,
        check=True,
    )
    args = captured_args.read_text().splitlines()
    runtime_environment = captured_environment.read_text().splitlines()

    assert runtime_environment[0] == f"{root}/src:{root}"
    assert runtime_environment[1] == str(fake_venv)
    assert runtime_environment[2].split(":", 1)[0] == str(fake_bin)
    assert runtime_environment[3:] == ["", "1"]
    assert "--allow-gripper-calibration" in args
    follower_index = args.index("--follower-can")
    assert args[follower_index + 1] == "left_arm=test_follower_l"
    assert args[follower_index + 3] == "right_arm=test_follower_r"
    leader_index = args.index("--leader-cans")
    assert args[leader_index + 1 : leader_index + 3] == ["test_leader_l", "test_leader_r"]
    endpoint_indexes = [
        index for index, value in enumerate(args) if value == "--leader-gripper-endpoint"
    ]
    assert [args[index + 1] for index in endpoint_indexes] == [
        "left_arm=-0.007669904,-0.708699124",
        "right_arm=0.030679616,-0.648873873",
    ]
    assert "--direct-leader-control" in args
    startup_index = args.index("--startup-position")
    assert args[startup_index + 1] == "zero"
    for flag, expected in (
        ("--gripper-close-torque-limit", "0.29"),
        ("--gripper-open-torque-limit", "0.29"),
        ("--gripper-damping", "0.5"),
        ("--gripper-max-speed", "2.0"),
    ):
        assert args[args.index(flag) + 1] == expected
    gripper_kp_index = args.index("--gripper-kp")
    assert args[gripper_kp_index + 1] == "5.0"
    camera_indexes = [index for index, value in enumerate(args) if value == "--camera"]
    assert [args[index + 1] for index in camera_indexes] == ["cam_high=260422275306"]
    orbbec_indexes = [index for index, value in enumerate(args) if value == "--orbbec-camera"]
    assert [args[index + 1] for index in orbbec_indexes] == [
        "cam_left_wrist=CV2R1610003Z",
        "cam_right_wrist=CV2L360000CL",
    ]
    camera_width_index = args.index("--camera-width")
    camera_height_index = args.index("--camera-height")
    camera_fps_index = args.index("--camera-fps")
    camera_timeout_index = args.index("--camera-timeout-ms")
    assert args[camera_width_index + 1] == "640"
    assert args[camera_height_index + 1] == "480"
    assert args[camera_fps_index + 1] == "30"
    assert args[camera_timeout_index + 1] == "3000"
    orbbec_width_index = args.index("--orbbec-width")
    orbbec_height_index = args.index("--orbbec-height")
    orbbec_fps_index = args.index("--orbbec-fps")
    orbbec_format_index = args.index("--orbbec-color-format")
    orbbec_warmup_index = args.index("--orbbec-warmup-frames")
    orbbec_brightness_index = args.index("--orbbec-brightness")
    orbbec_power_line_index = args.index("--orbbec-power-line-frequency")
    assert args[orbbec_width_index + 1] == "640"
    assert args[orbbec_height_index + 1] == "480"
    assert args[orbbec_fps_index + 1] == "30"
    assert args[orbbec_format_index + 1] == "MJPG"
    assert args[orbbec_warmup_index + 1] == "30"
    assert args[orbbec_brightness_index + 1] == "25"
    assert args[orbbec_power_line_index + 1] == "50"
    assert "--camera-profile" not in args
    exposure_limit_index = args.index("--camera-auto-exposure-limit-us")
    gain_limit_index = args.index("--camera-auto-gain-limit")
    warmup_index = args.index("--camera-warmup-frames")
    assert args[exposure_limit_index + 1] == "16000"
    assert args[gain_limit_index + 1] == "32"
    assert args[warmup_index + 1] == "90"
    assert not list((root / "examples/hardware/yam").glob("run_*leader.sh"))

    env["ORBBEC_ENABLED_CAMERAS"] = ""
    subprocess.run(
        ["bash", "examples/hardware/yam/run_hardware.sh", "--help"],
        cwd=root,
        env=env,
        check=True,
    )
    args = captured_args.read_text().splitlines()
    assert "--orbbec-camera" not in args


def test_run_hardware_rejects_a_second_instance_before_hardware_setup(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    fake_venv = tmp_path / "venv"
    fake_python = fake_venv / "bin/python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    lock_path = tmp_path / "hardware.lock"

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", "examples/hardware/yam/run_hardware.sh"],
            cwd=root,
            env={
                **os.environ,
                "YAM_VENV_DIR": str(fake_venv),
                "YAM_LOCK_FILE": str(lock_path),
            },
            check=False,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 1
    assert "YAM hardware is already running" in result.stderr
    assert "Resetting Orbbec" not in result.stdout
