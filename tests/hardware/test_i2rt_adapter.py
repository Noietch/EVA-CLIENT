from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import robots  # noqa: F401
from core.config import load_config
from core.registry import ROBOT_REGISTRY
from examples.hardware.i2rt.camera import parse_camera_specs
from examples.hardware.i2rt.node import (
    I2RTZmqNode,
    _RisingEdgeDebouncer,
    build_arg_parser,
    build_config,
)
from examples.hardware.i2rt.robot import (
    I2RTYamFollowers,
    I2RTYamLeaders,
    clip_group_action,
    split_action,
)
from examples.hardware.i2rt.wire import (
    WireAction as I2RTWireAction,
)
from examples.hardware.i2rt.wire import (
    WireObservation as I2RTWireObservation,
)
from examples.hardware.i2rt.wire import (
    pack_observation as pack_i2rt_observation,
)
from examples.hardware.i2rt.wire import (
    unpack_action as unpack_i2rt_action,
)
from robots.utils import UrdfScene
from transport.zmq import (
    WireAction,
    pack_action,
    unpack_observation,
)


@dataclasses.dataclass(frozen=True)
class _Config:
    group_names: tuple[str, ...] = ("left_arm", "right_arm")
    follower_can_channels: dict[str, str] = dataclasses.field(
        default_factory=lambda: {"left_arm": "can0", "right_arm": "can1"}
    )
    leader_can_channels: dict[str, str] = dataclasses.field(default_factory=dict)
    disabled_groups: tuple[str, ...] = ()
    arm_type: str = "yam"
    gripper_type: str = "linear_4310"
    sim: bool = False
    enable_auto_recovery: bool = False
    command_timeout_s: float = 0.5
    idle_mode: str = "gravity_comp"
    startup_position: str = "current"
    startup_duration_s: float = 5.0
    joint4_kp: float | None = None
    end_effector_mass: float | None = None
    gravity_comp_factor: tuple[float, ...] | None = None
    gripper_limits_override: tuple[float, float] | None = None
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


class _FakeSimYam:
    def __init__(self) -> None:
        self.qpos = np.asarray([0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.gravity_comp_count = 0

    def num_dofs(self) -> int:
        return 7

    def get_joint_pos(self) -> np.ndarray:
        return self.qpos.copy()

    def command_joint_pos(self, qpos: np.ndarray) -> None:
        self.qpos = np.asarray(qpos, dtype=np.float64).copy()

    def enable_gravity_comp(self) -> None:
        self.gravity_comp_count += 1

    def close(self) -> None:
        pass


class _FakeLeader:
    def __init__(self) -> None:
        self.qpos = np.asarray([0.1, 0.4, 0.5, 0.1, 0.1, 0.1], dtype=np.float64)
        self.closed = False

    def num_dofs(self) -> int:
        return 6

    def get_joint_pos(self) -> np.ndarray:
        return self.qpos.copy()

    def close(self) -> None:
        self.closed = True


class _FakeSagYam(_FakeYam):
    def __init__(self) -> None:
        super().__init__()
        self.sag = np.asarray([0.004, 0.030, 0.010, -0.060, 0.009, 0.008], dtype=np.float64)

    def command_joint_pos(self, qpos: np.ndarray) -> None:
        self.last_command = np.asarray(qpos, dtype=np.float64).copy()
        self.qpos = self.last_command.copy()
        self.qpos[:6] += self.sag


def test_i2rt_robot_registry_layouts() -> None:
    single = ROBOT_REGISTRY.build("i2rt_yam")
    dual = ROBOT_REGISTRY.build("i2rt_dual_yam")

    assert single.total_action_dim == 7
    assert [group.name for group in single.actuator_groups] == ["arm"]
    assert single.gripper_indices == (6,)
    assert [camera.observation_key for camera in single.observation_schema.cameras] == [
        "cam_high",
        "cam_wrist",
    ]

    assert dual.total_action_dim == 14
    assert [group.name for group in dual.actuator_groups] == ["left_arm", "right_arm"]
    assert dual.gripper_indices == (6, 13)
    assert dual.vis_config is not None
    left_part, right_part = dual.vis_config.parts
    assert left_part.base_position == pytest.approx((0.0, 0.25, 0.0))
    assert right_part.base_position == pytest.approx((0.0, -0.25, 0.0))


def test_i2rt_scene_matches_black_joint_white_arm_finish() -> None:
    robot = ROBOT_REGISTRY.build("i2rt_dual_yam")
    scene = UrdfScene(robot)
    mesh_colors = {mesh["name"]: mesh["color"] for mesh in scene.static_meshes()}

    # Official YAM URDF order: base, gripper, link1..5, left/right finger tips.
    dark = np.asarray([0.25, 0.25, 0.25])
    white = np.asarray([0.8, 0.8, 0.8])
    np.testing.assert_array_less(mesh_colors["geometry_0"], dark)  # base
    np.testing.assert_array_less(mesh_colors["geometry_1"], dark)  # gripper body
    np.testing.assert_array_less(mesh_colors["geometry_2"], dark)  # first joint housing
    np.testing.assert_array_less(white, mesh_colors["geometry_3"])  # first long arm shell
    np.testing.assert_array_less(white, mesh_colors["geometry_4"])  # second long arm shell
    np.testing.assert_array_less(mesh_colors["geometry_5"], dark)  # wrist joint
    np.testing.assert_array_less(mesh_colors["geometry_6"], dark)  # wrist roll
    np.testing.assert_array_less(mesh_colors["geometry_7"], dark)  # left finger
    np.testing.assert_array_less(mesh_colors["geometry_8"], dark)  # right finger


def test_action_split_clips_joint_and_gripper_limits() -> None:
    raw = np.asarray([-9.0, -1.0, 9.0, -9.0, 9.0, -9.0, 2.0], dtype=np.float32)
    clipped = clip_group_action(raw)

    assert clipped == pytest.approx([-2.61799, 0.0, 3.14159, -1.69297, 1.5708, -2.0944, 1.0])
    parts = split_action(np.concatenate([raw, raw]), ("left_arm", "right_arm"))
    assert parts["left_arm"] == pytest.approx(clipped)
    assert parts["right_arm"] == pytest.approx(clipped)


def test_follower_commands_both_arms_and_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    robots_by_channel: dict[str, _FakeYam] = {}

    def factory(**kwargs: object) -> _FakeYam:
        robot = _FakeYam()
        robots_by_channel[str(kwargs["channel"])] = robot
        return robot

    clock = 10.0
    monkeypatch.setattr("examples.hardware.i2rt.robot.time.monotonic", lambda: clock)
    followers = I2RTYamFollowers(_Config(joint4_kp=40.0), factory=factory)
    target = np.asarray(
        [0.1, 0.4, 0.5, 0.1, 0.1, 0.1, 0.25] * 2,
        dtype=np.float32,
    )
    followers.apply_action(I2RTWireAction(t=clock, action=target, target="real"))

    assert robots_by_channel["can0"].qpos == pytest.approx(target[:7])
    assert robots_by_channel["can1"].qpos == pytest.approx(target[7:])
    assert robots_by_channel["can0"].last_kp == pytest.approx([80, 80, 80, 40, 10, 10, 20])
    assert robots_by_channel["can0"].last_kd == pytest.approx([5, 5, 5, 1.5, 1.5, 1.5, 0.5])

    clock = 10.6
    followers.watchdog_tick()
    assert robots_by_channel["can0"].idle_count == 1
    assert robots_by_channel["can1"].idle_count == 1


def test_sim_watchdog_uses_simrobot_gravity_comp_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = _FakeSimYam()
    clock = 10.0
    monkeypatch.setattr("examples.hardware.i2rt.robot.time.monotonic", lambda: clock)
    followers = I2RTYamFollowers(
        _Config(sim=True),
        factory=lambda **_kwargs: robot,
    )
    target = np.asarray([0.1, 0.4, 0.5, 0.1, 0.1, 0.1, 0.25] * 2, dtype=np.float32)
    followers.apply_action(I2RTWireAction(t=clock, action=target, target="real"))

    clock = 10.6
    followers.watchdog_tick()

    assert robot.gravity_comp_count == 2
    assert followers.hardware_status() == {"left_arm": "online", "right_arm": "online"}


def test_left_only_leader_holds_right_follower_anchor() -> None:
    leader = _FakeLeader()
    config = _Config(
        leader_can_channels={"left_arm": "can2"},
        sim=True,
    )
    leaders = I2RTYamLeaders(config, factory=lambda **_kwargs: leader)
    anchor = np.asarray(
        [0.0, 0.3, 0.4, 0.0, 0.0, 0.0, 0.5, 0.2, 0.6, 0.7, 0.1, 0.1, 0.1, 0.8],
        dtype=np.float32,
    )

    action = leaders.read_action(anchor)

    assert action[:6] == pytest.approx(leader.qpos)
    assert action[6] == pytest.approx(1.0)
    assert action[7:] == pytest.approx(anchor[7:])


def test_collection_maps_leader_delta_onto_follower_anchor() -> None:
    follower_anchor = np.asarray(
        [0.0, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0] * 2,
        dtype=np.float32,
    )
    leader_anchor = np.asarray(
        [0.1, 0.5, 0.6, 0.1, 0.1, 0.1, 1.0] * 2,
        dtype=np.float32,
    )
    leader = leader_anchor.copy()
    leader[1] += 0.2

    class _Followers:
        action: np.ndarray | None = None

        def apply_action(self, action: I2RTWireAction) -> None:
            self.action = action.action.copy()

    node = object.__new__(I2RTZmqNode)
    node._config = _Config()
    node._collection_active = True
    node._hil_active = False
    node._hil_mode = "relative"
    node._hil_error = ""
    node._follower_anchor = follower_anchor
    node._leader_anchor = leader_anchor
    node._leaders = type(
        "_Leaders",
        (),
        {
            "read_action": lambda self, _anchor: leader,
            "read_buttons": lambda self: {"left_arm": (False, False)},
        },
    )()
    node._followers = _Followers()
    node._leader_control_updates = 0
    node._record_button = _RisingEdgeDebouncer(0.08)
    node._operator_event = ""
    node._operator_event_id = 0

    action = node._leader_action()

    assert action is not None
    assert action[1] == pytest.approx(follower_anchor[1] + 0.2)
    assert action[8] == pytest.approx(follower_anchor[8])
    assert node._followers.action == pytest.approx(action)
    assert node._leader_control_updates == 1


def test_leader_record_button_requires_stable_release_then_press() -> None:
    button = _RisingEdgeDebouncer(0.08)

    assert button.update(True, 1.00) is False  # held while leader connects
    assert button.update(False, 1.10) is False
    assert button.update(False, 1.19) is False
    assert button.update(True, 1.20) is False
    assert button.update(True, 1.27) is False
    assert button.update(True, 1.29) is True
    assert button.update(True, 1.40) is False


def test_hold_position_idle_starts_and_returns_to_measured_position() -> None:
    robots_by_channel: dict[str, _FakeYam] = {}
    factory_kwargs: list[dict[str, object]] = []

    def factory(**kwargs: object) -> _FakeYam:
        factory_kwargs.append(kwargs)
        robot = _FakeYam()
        robots_by_channel[str(kwargs["channel"])] = robot
        return robot

    followers = I2RTYamFollowers(
        _Config(
            idle_mode="hold_position",
            end_effector_mass=0.7,
            gravity_comp_factor=(1.0, 0.62, 0.98, 1.2, 1.0, 1.0),
            gripper_limits_override=(0.071, -5.072),
        ),
        factory=factory,
    )
    followers.read_state()
    robots_by_channel["can0"].qpos[1] = 0.75
    robots_by_channel["can1"].qpos[1] = 0.65

    followers.enter_safe_idle()

    assert all(kwargs["zero_gravity_mode"] is False for kwargs in factory_kwargs)
    assert all(kwargs["ee_mass"] == pytest.approx(0.7) for kwargs in factory_kwargs)
    assert all(
        kwargs["gravity_comp_factor"] == pytest.approx([1.0, 0.62, 0.98, 1.2, 1.0, 1.0])
        for kwargs in factory_kwargs
    )
    assert all(
        kwargs["gripper_limits_override"] == pytest.approx([0.071, -5.072])
        for kwargs in factory_kwargs
    )
    assert robots_by_channel["can0"].last_command == pytest.approx(robots_by_channel["can0"].qpos)
    assert robots_by_channel["can1"].last_command == pytest.approx(robots_by_channel["can1"].qpos)


def test_startup_zero_smoothly_commands_both_followers_with_grippers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robots_by_channel: dict[str, _FakeYam] = {}

    def factory(**kwargs: object) -> _FakeYam:
        robot = _FakeYam()
        robots_by_channel[str(kwargs["channel"])] = robot
        return robot

    monkeypatch.setattr("examples.hardware.i2rt.robot.time.sleep", lambda _seconds: None)
    followers = I2RTYamFollowers(
        _Config(
            idle_mode="hold_position",
            startup_position="zero",
            startup_duration_s=5.0,
            joint4_kp=40.0,
        ),
        factory=factory,
    )

    followers.move_to_startup_position()

    expected = np.asarray([0.0] * 6 + [1.0])
    assert robots_by_channel["can0"].qpos == pytest.approx(expected)
    assert robots_by_channel["can1"].qpos == pytest.approx(expected)
    assert robots_by_channel["can0"].last_kp == pytest.approx([80, 80, 80, 40, 10, 10, 20])
    assert robots_by_channel["can1"].last_kp == pytest.approx([80, 80, 80, 40, 10, 10, 20])
    state = followers.read_state()
    assert state["left_arm"] == pytest.approx([0.0] * 6 + [1.0])
    assert state["right_arm"] == pytest.approx([0.0] * 6 + [1.0])


def test_startup_integral_trim_ignores_error_into_lower_joint_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robots_by_channel: dict[str, _FakeSagYam] = {}

    def factory(**kwargs: object) -> _FakeSagYam:
        robot = _FakeSagYam()
        robots_by_channel[str(kwargs["channel"])] = robot
        return robot

    monkeypatch.setattr("examples.hardware.i2rt.robot.time.sleep", lambda _seconds: None)
    followers = I2RTYamFollowers(
        _Config(
            startup_position="zero",
            tracking_ki=2.0,
            tracking_trim_limit=0.12,
            tracking_deadband=0.002,
            startup_trim_duration_s=3.0,
        ),
        factory=factory,
    )

    followers.move_to_startup_position()

    expected = np.zeros(6)
    expected[1:3] = robots_by_channel["can0"].sag[1:3]
    for robot in robots_by_channel.values():
        assert robot.qpos[:6] == pytest.approx(expected, abs=0.0021)
        assert robot.qpos[6] == pytest.approx(1.0)
    trims = followers.tracking_trim()
    expected_trim = -robots_by_channel["can0"].sag
    expected_trim[1:3] = 0.0
    assert trims["left_arm"] == pytest.approx(expected_trim, abs=0.003)
    assert trims["right_arm"] == pytest.approx(expected_trim, abs=0.003)


def test_runtime_integral_trim_only_learns_stationary_tracking_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robots_by_channel: dict[str, _FakeSagYam] = {}

    def factory(**kwargs: object) -> _FakeSagYam:
        robot = _FakeSagYam()
        robots_by_channel[str(kwargs["channel"])] = robot
        return robot

    clock = 10.0
    monkeypatch.setattr("examples.hardware.i2rt.robot.time.monotonic", lambda: clock)
    followers = I2RTYamFollowers(
        _Config(
            tracking_ki=2.0,
            tracking_trim_limit=0.12,
            tracking_deadband=0.002,
            tracking_settle_delay_s=0.15,
        ),
        factory=factory,
    )
    target = np.asarray([0.2, 0.4, 0.5, 0.1, 0.1, 0.1, 1.0] * 2, dtype=np.float32)

    for _ in range(600):
        followers.apply_action(I2RTWireAction(t=clock, action=target, target="real"))
        clock += 0.005

    for robot in robots_by_channel.values():
        assert robot.qpos[:6] == pytest.approx(target[:6], abs=0.0021)
        assert robot.qpos[6] == pytest.approx(1.0)


def test_isolated_i2rt_wire_is_eva_compatible() -> None:
    action_payload = pack_action(
        WireAction(t=1.25, action=np.arange(7, dtype=np.float32), target="real")
    )
    decoded_action = unpack_i2rt_action(action_payload)
    assert decoded_action.t == pytest.approx(1.25)
    assert decoded_action.action == pytest.approx(np.arange(7, dtype=np.float32))

    observation_payload = pack_i2rt_observation(
        I2RTWireObservation(
            t=2.5,
            images={"cam_high": np.zeros((4, 6, 3), dtype=np.uint8)},
            state={"arm": np.arange(7, dtype=np.float32)},
            hil_supported=True,
            operator_event="collection_record_toggle",
            operator_event_id=3,
        )
    )
    decoded_observation = unpack_observation(observation_payload)
    assert decoded_observation.t == pytest.approx(2.5)
    assert decoded_observation.state["arm"] == pytest.approx(np.arange(7))
    assert decoded_observation.images["cam_high"].shape == (4, 6, 3)
    assert decoded_observation.hil_supported is True
    assert decoded_observation.operator_event == "collection_record_toggle"
    assert decoded_observation.operator_event_id == 3


def test_d405_specs_and_node_config() -> None:
    specs = parse_camera_specs(
        ["cam_high=255323073172", "cam_left_wrist=index:1"],
        width=848,
        height=480,
        fps=30,
    )
    assert specs[0].serial == "255323073172"
    assert specs[1].device_index == 1

    args = build_arg_parser().parse_args(
        [
            "--robot",
            "i2rt_dual_yam",
            "--follower-can",
            "left_arm=can_follower_l",
            "--follower-can",
            "right_arm=can_follower_r",
            "--camera",
            "cam_high=255323073172",
            "--allow-gripper-calibration",
        ]
    )
    config = build_config(args)
    assert config.follower_can_channels == {
        "left_arm": "can_follower_l",
        "right_arm": "can_follower_r",
    }
    assert config.cameras[0].serial == "255323073172"
    assert config.control_rate_hz == pytest.approx(200.0)
    assert config.tracking_ki == pytest.approx(0.0)
    assert config.end_effector_mass is None
    assert config.gravity_comp_factor is None
    assert config.gripper_limits_override is None
    assert config.allow_gripper_calibration is True

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--arm-type", "yam_pro"])

    unsafe_args = build_arg_parser().parse_args(["--gripper-type", "linear_4310"])
    with pytest.raises(ValueError, match="requires calibrated"):
        build_config(unsafe_args)


@pytest.mark.parametrize(
    ("robot_name", "disabled_cameras"),
    [
        ("i2rt_yam", ["cam_wrist"]),
        ("i2rt_dual_yam", ["cam_left_wrist", "cam_right_wrist"]),
    ],
)
def test_i2rt_presets_default_to_one_d405(
    robot_name: str,
    disabled_cameras: list[str],
) -> None:
    root = Path(__file__).resolve().parents[2]
    deploy = load_config(root / "configs" / "01_deploy" / robot_name / "openpi_qpos.py")
    collection = load_config(root / "configs" / "02_collection" / f"{robot_name}.py")

    assert deploy.transport.disabled_cameras == disabled_cameras
    assert collection.transport.disabled_cameras == disabled_cameras
    assert collection.inference_cfg.manual_max_qpos_step == pytest.approx(0.005)
    assert collection.inference_cfg.manual_settle_duration == pytest.approx(2.0)
    assert dict(collection.collection.schema.cameras) == {"cam_high": "observation.images.cam_high"}
