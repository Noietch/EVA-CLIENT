from __future__ import annotations

import dataclasses
import fcntl
import os
import subprocess
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import robots  # noqa: F401
from core.config import load_config
from core.registry import ROBOT_REGISTRY
from examples.hardware.yam.camera import parse_camera_specs
from examples.hardware.yam.node import (
    YamZmqNode,
    _RisingEdgeDebouncer,
    build_arg_parser,
    build_config,
    split_group_eef,
)
from examples.hardware.yam.robot import (
    YamFollowers,
    YamLeaders,
    clip_group_action,
    map_leader_gripper,
    split_action,
)
from examples.hardware.yam.wire import (
    WireAction as YamWireAction,
)
from examples.hardware.yam.wire import (
    WireObservation as YamWireObservation,
)
from examples.hardware.yam.wire import (
    pack_observation as pack_yam_observation,
)
from examples.hardware.yam.wire import (
    unpack_action as unpack_yam_action,
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


class _FakeLeader:
    def __init__(self) -> None:
        self.qpos = np.asarray([0.1, 0.4, 0.5, 0.1, 0.1, 0.1], dtype=np.float64)
        self.closed = False
        self.gravity_comp_count = 0
        self.motor_chain = self
        self.encoder_position = 0.0
        self.raw_encoder_position = 0.0

    def num_dofs(self) -> int:
        return 6

    def get_joint_pos(self) -> np.ndarray:
        return self.qpos.copy()

    def command_joint_pos(self, qpos: np.ndarray) -> None:
        self.qpos = np.asarray(qpos, dtype=np.float64).copy()

    def enter_gravity_comp_idle(self) -> None:
        self.gravity_comp_count += 1

    def get_same_bus_device_states(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                position=self.encoder_position,
                raw_position=self.raw_encoder_position,
                io_inputs=(False, False),
            )
        ]

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


def test_yam_robot_registry_layout() -> None:
    dual = ROBOT_REGISTRY.build("dual_yam")

    assert dual.total_action_dim == 14
    assert [group.name for group in dual.actuator_groups] == ["left_arm", "right_arm"]
    assert dual.gripper_indices == (6, 13)
    assert dual.vis_config is not None
    left_part, right_part = dual.vis_config.parts
    assert left_part.base_position == pytest.approx((0.0, 0.25, 0.0))
    assert right_part.base_position == pytest.approx((0.0, -0.25, 0.0))


def test_yam_observation_uses_eva_fk_for_state_and_action() -> None:
    state = {
        "left_arm": np.asarray([0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4], dtype=np.float32),
        "right_arm": np.asarray([0.1, 0.3, 0.4, 0.1, 0.1, 0.1, 0.6], dtype=np.float32),
    }
    action = np.concatenate([state["left_arm"], state["right_arm"]]).astype(np.float32)
    action[0] += 0.15

    class _Solver:
        def __init__(self) -> None:
            self.inputs: list[np.ndarray] = []

        def fk_chunk(self, chunk: np.ndarray) -> np.ndarray:
            self.inputs.append(np.asarray(chunk, dtype=np.float32).copy())
            offset = 10.0 * len(self.inputs)
            return (np.arange(16, dtype=np.float32) + offset)[np.newaxis, :]

    class _Publisher:
        payload: bytes | None = None

        def send(self, payload: bytes) -> None:
            self.payload = payload

    solver = _Solver()
    publisher = _Publisher()
    node = object.__new__(YamZmqNode)
    node._config = _Config()
    node._latest_leader_action = action
    node._followers = type("_Followers", (), {"snapshot_state": lambda self: state})()
    node._fk_solver = solver
    node._camera_caches = (type("_Cameras", (), {"snapshot": lambda self: {}})(),)
    node._obs_pub = publisher
    node._tracking_error = {}
    node._hil_active = False
    node._hil_error = ""
    node._operator_event = ""
    node._operator_event_id = 0
    node._published_observations = 0

    node._publish_observation()

    assert publisher.payload is not None
    observation = unpack_observation(publisher.payload)
    assert len(solver.inputs) == 2
    assert solver.inputs[0][0] == pytest.approx(
        np.concatenate([state["left_arm"], state["right_arm"]])
    )
    assert solver.inputs[1][0] == pytest.approx(action)
    assert observation.eef is not None
    assert observation.eef["left_arm"] == pytest.approx(np.arange(8) + 10.0)
    assert observation.eef["right_arm"] == pytest.approx(np.arange(8, 16) + 10.0)
    assert observation.action_eef == pytest.approx(np.arange(16) + 20.0)
    assert node._published_observations == 1


def test_yam_observation_publisher_owns_socket_for_thread_lifetime() -> None:
    class _Publisher:
        endpoint: str | None = None
        closed = False

        def bind(self, endpoint: str) -> None:
            self.endpoint = endpoint

        def close(self, *, linger: int) -> None:
            assert linger == 0
            self.closed = True

    class _Context:
        publisher = _Publisher()

        def socket(self, _socket_type: object) -> _Publisher:
            return self.publisher

    node = object.__new__(YamZmqNode)
    node._config = SimpleNamespace(
        publish_rate_hz=30.0,
        observation_endpoint="tcp://127.0.0.1:5555",
    )
    node._ctx = _Context()
    node._obs_pub = None
    node._publisher_ready = threading.Event()
    node._publisher_error = None
    node._stop = threading.Event()
    publish_count = 0

    def publish_observation() -> None:
        nonlocal publish_count
        assert node._obs_pub is node._ctx.publisher
        publish_count += 1
        node._stop.set()

    node._publish_observation = publish_observation
    node._log_status_if_due = lambda: None

    node._publish_loop()

    assert publish_count == 1
    assert node._publisher_ready.is_set()
    assert node._publisher_error is None
    assert node._ctx.publisher.endpoint == node._config.observation_endpoint
    assert node._ctx.publisher.closed is True
    assert node._obs_pub is None


def test_split_group_eef_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="16-D"):
        split_group_eef(np.zeros(8, dtype=np.float32), ("left_arm", "right_arm"))


def test_yam_scene_matches_black_joint_white_arm_finish() -> None:
    robot = ROBOT_REGISTRY.build("dual_yam")
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
    monkeypatch.setattr("examples.hardware.yam.robot.time.monotonic", lambda: clock)
    followers = YamFollowers(_Config(joint4_kp=40.0), factory=factory)
    target = np.asarray(
        [0.1, 0.4, 0.5, 0.1, 0.1, 0.1, 0.25] * 2,
        dtype=np.float32,
    )
    followers.apply_action(YamWireAction(t=clock, action=target, target="real"))

    assert robots_by_channel["can0"].qpos == pytest.approx(target[:7])
    assert robots_by_channel["can1"].qpos == pytest.approx(target[7:])
    assert robots_by_channel["can0"].last_kp == pytest.approx([80, 80, 80, 40, 10, 10, 20])
    assert robots_by_channel["can0"].last_kd == pytest.approx([5, 5, 5, 1.5, 1.5, 1.5, 0.5])

    clock = 10.6
    followers.watchdog_tick()
    assert robots_by_channel["can0"].idle_count == 1
    assert robots_by_channel["can1"].idle_count == 1


def test_follower_rejects_sdk_motor_loop_that_stopped() -> None:
    def factory(**_kwargs: object) -> _FakeYam:
        robot = _FakeYam()
        robot.motor_chain = SimpleNamespace(running=False)
        return robot

    followers = YamFollowers(_Config(), factory=factory)

    state = followers.read_state()

    assert state["left_arm"][6] == pytest.approx(1.0)
    assert followers.hardware_status()["left_arm"] == "retrying"


def test_dual_leaders_read_both_arms() -> None:
    leaders_by_channel = {"can2": _FakeLeader(), "can3": _FakeLeader()}
    leaders_by_channel["can3"].qpos += 0.2
    factory_kwargs: list[dict[str, object]] = []
    config = _Config(
        leader_can_channels={"left_arm": "can2", "right_arm": "can3"},
    )

    def factory(**kwargs: object) -> _FakeLeader:
        factory_kwargs.append(kwargs)
        return leaders_by_channel[str(kwargs["channel"])]

    leaders = YamLeaders(
        config,
        factory=factory,
    )

    action = leaders.read_action()

    assert action[:6] == pytest.approx(leaders_by_channel["can2"].qpos)
    assert action[6] == pytest.approx(1.0)
    assert action[7:13] == pytest.approx(leaders_by_channel["can3"].qpos)
    assert action[13] == pytest.approx(1.0)
    assert all(kwargs["zero_gravity_mode"] is True for kwargs in factory_kwargs)
    assert all(leader.gravity_comp_count == 1 for leader in leaders_by_channel.values())


def test_leader_rejects_sdk_motor_loop_that_stopped() -> None:
    def factory(**_kwargs: object) -> _FakeLeader:
        robot = _FakeLeader()
        robot.motor_chain.running = False
        return robot

    leaders = YamLeaders(
        _Config(leader_can_channels={"left_arm": "can2", "right_arm": "can3"}),
        factory=factory,
    )

    with pytest.raises(RuntimeError, match="CAN motor loop stopped"):
        leaders.read_action()


def test_dual_leaders_map_independent_raw_gripper_endpoints() -> None:
    leaders_by_channel = {"can2": _FakeLeader(), "can3": _FakeLeader()}
    endpoints = {
        "left_arm": (-0.007669904, -0.708699124),
        "right_arm": (0.030679616, -0.648873873),
    }
    leaders = YamLeaders(
        _Config(
            leader_can_channels={"left_arm": "can2", "right_arm": "can3"},
            leader_gripper_encoder_endpoints=endpoints,
        ),
        factory=lambda **kwargs: leaders_by_channel[str(kwargs["channel"])],
    )

    for group_name, channel in (("left_arm", "can2"), ("right_arm", "can3")):
        opened, _ = endpoints[group_name]
        leaders_by_channel[channel].raw_encoder_position = opened
    opened_action = leaders.read_action()

    assert opened_action[[6, 13]] == pytest.approx([1.0, 1.0])

    for group_name, channel in (("left_arm", "can2"), ("right_arm", "can3")):
        opened, closed = endpoints[group_name]
        leaders_by_channel[channel].raw_encoder_position = (opened + closed) / 2
    midpoint_action = leaders.read_action()

    assert midpoint_action[[6, 13]] == pytest.approx([0.5, 0.5])

    for group_name, channel in (("left_arm", "can2"), ("right_arm", "can3")):
        _, closed = endpoints[group_name]
        leaders_by_channel[channel].raw_encoder_position = closed
    closed_action = leaders.read_action()

    assert closed_action[[6, 13]] == pytest.approx([0.0, 0.0])


def test_map_leader_gripper_clips_beyond_calibrated_endpoints() -> None:
    endpoints = (0.03, -0.65)

    assert map_leader_gripper(0.04, endpoints) == pytest.approx(1.0)
    assert map_leader_gripper(-0.66, endpoints) == pytest.approx(0.0)


def test_dual_leaders_move_to_zero_then_restore_gravity_compensation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaders_by_channel = {"can2": _FakeLeader(), "can3": _FakeLeader()}
    monkeypatch.setattr("examples.hardware.yam.robot.time.sleep", lambda _seconds: None)
    leaders = YamLeaders(
        _Config(
            leader_can_channels={"left_arm": "can2", "right_arm": "can3"},
            startup_position="zero",
            startup_duration_s=1.0,
        ),
        factory=lambda **kwargs: leaders_by_channel[str(kwargs["channel"])],
    )

    leaders.move_to_startup_position()

    assert all(leader.qpos == pytest.approx(np.zeros(6)) for leader in leaders_by_channel.values())
    assert all(leader.gravity_comp_count == 2 for leader in leaders_by_channel.values())


def test_single_leader_configuration_is_rejected() -> None:
    leaders = YamLeaders(
        _Config(leader_can_channels={"left_arm": "can2"}),
        factory=lambda **_kwargs: _FakeLeader(),
    )

    with pytest.raises(RuntimeError, match="requires both"):
        leaders.read_action()


def test_direct_leader_control_captures_anchors_and_commands_follower_delta() -> None:
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
    leader[6] = 0.25
    leader[13] = 0.75

    class _Followers:
        action: np.ndarray | None = None

        def read_state(self) -> dict[str, np.ndarray]:
            return split_action(follower_anchor, ("left_arm", "right_arm"))

        def apply_action(self, action: YamWireAction) -> None:
            self.action = action.action.copy()

    class _Leaders:
        reads = 0

        def read_action(self) -> np.ndarray:
            self.reads += 1
            return leader_anchor if self.reads == 1 else leader

        def read_buttons(self) -> dict[str, tuple[bool, bool]]:
            return {
                "left_arm": (False, False),
                "right_arm": (False, False),
            }

    node = object.__new__(YamZmqNode)
    node._config = _Config()
    node._collection_active = False
    node._hil_active = False
    node._direct_leader_control = True
    node._hil_mode = "relative"
    node._hil_error = ""
    node._follower_anchor = None
    node._leader_anchor = None
    node._next_leader_connect_time = 0.0
    node._leaders = _Leaders()
    node._followers = _Followers()
    node._leader_control_updates = 0
    node._record_button = _RisingEdgeDebouncer(0.08)
    node._cancel_button = _RisingEdgeDebouncer(0.08)
    node._operator_event = ""
    node._operator_event_id = 0

    assert node._ensure_direct_leader_control() is True
    action = node._leader_action()

    assert action is not None
    assert action[1] == pytest.approx(follower_anchor[1] + 0.2)
    assert action[8] == pytest.approx(follower_anchor[8])
    assert action[6] == pytest.approx(0.25)
    assert action[13] == pytest.approx(0.75)
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


def test_leader_record_and_cancel_buttons_emit_separate_events() -> None:
    node = object.__new__(YamZmqNode)
    node._record_button = _RisingEdgeDebouncer(0.08)
    node._cancel_button = _RisingEdgeDebouncer(0.08)
    node._operator_event = ""
    node._operator_event_id = 0

    node._update_operator_buttons({"left_arm": (False, False)}, 1.00)
    node._update_operator_buttons({"left_arm": (False, True)}, 1.10)
    node._update_operator_buttons({"left_arm": (False, True)}, 1.19)

    assert node._operator_event == "collection_record_toggle"
    assert node._operator_event_id == 1

    node._update_operator_buttons({"left_arm": (False, False)}, 1.30)
    node._update_operator_buttons({"left_arm": (False, False)}, 1.39)
    node._update_operator_buttons({"right_arm": (True, False)}, 1.40)
    node._update_operator_buttons({"right_arm": (True, False)}, 1.49)

    assert node._operator_event == "collection_cancel"
    assert node._operator_event_id == 2


def test_hold_position_idle_starts_and_returns_to_measured_position() -> None:
    robots_by_channel: dict[str, _FakeYam] = {}
    factory_kwargs: list[dict[str, object]] = []

    def factory(**kwargs: object) -> _FakeYam:
        factory_kwargs.append(kwargs)
        robot = _FakeYam()
        robots_by_channel[str(kwargs["channel"])] = robot
        return robot

    followers = YamFollowers(
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

    monkeypatch.setattr("examples.hardware.yam.robot.time.sleep", lambda _seconds: None)
    followers = YamFollowers(
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

    monkeypatch.setattr("examples.hardware.yam.robot.time.sleep", lambda _seconds: None)
    followers = YamFollowers(
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
    monkeypatch.setattr("examples.hardware.yam.robot.time.monotonic", lambda: clock)
    followers = YamFollowers(
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
        followers.apply_action(YamWireAction(t=clock, action=target, target="real"))
        clock += 0.005

    for robot in robots_by_channel.values():
        assert robot.qpos[:6] == pytest.approx(target[:6], abs=0.0021)
        assert robot.qpos[6] == pytest.approx(1.0)


def test_isolated_yam_wire_is_eva_compatible() -> None:
    action_payload = pack_action(
        WireAction(t=1.25, action=np.arange(7, dtype=np.float32), target="real")
    )
    decoded_action = unpack_yam_action(action_payload)
    assert decoded_action.t == pytest.approx(1.25)
    assert decoded_action.action == pytest.approx(np.arange(7, dtype=np.float32))

    observation_payload = pack_yam_observation(
        YamWireObservation(
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


def test_mixed_camera_specs_and_node_config() -> None:
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
            "--follower-can",
            "left_arm=can_follower_l",
            "--follower-can",
            "right_arm=can_follower_r",
            "--camera",
            "cam_high=255323073172",
            "--orbbec-camera",
            "cam_left_wrist=CV2L360000CL",
            "--leader-cans",
            "can_leader_l",
            "can_leader_r",
            "--leader-gripper-endpoint",
            "left_arm=-0.01,-0.71",
            "--leader-gripper-endpoint",
            "right_arm=0.03,-0.65",
            "--direct-leader-control",
            "--allow-gripper-calibration",
        ]
    )
    config = build_config(args)
    assert config.follower_can_channels == {
        "left_arm": "can_follower_l",
        "right_arm": "can_follower_r",
    }
    assert config.leader_can_channels == {
        "left_arm": "can_leader_l",
        "right_arm": "can_leader_r",
    }
    assert config.leader_gripper_encoder_endpoints == {
        "left_arm": pytest.approx((-0.01, -0.71)),
        "right_arm": pytest.approx((0.03, -0.65)),
    }
    assert config.direct_leader_control is True
    assert config.cameras[0].serial == "255323073172"
    assert config.orbbec_cameras[0].serial == "CV2L360000CL"
    assert config.control_rate_hz == pytest.approx(200.0)
    assert config.tracking_ki == pytest.approx(0.0)
    assert config.end_effector_mass is None
    assert config.gravity_comp_factor is None
    assert config.gripper_limits_override is None
    assert config.allow_gripper_calibration is True

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--arm-type", "yam_pro"])

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--leader-cans", "can2"])

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--sim"])

    direct_without_leaders = build_arg_parser().parse_args(
        ["--direct-leader-control", "--allow-gripper-calibration"]
    )
    with pytest.raises(ValueError, match="requires --leader-cans"):
        build_config(direct_without_leaders)

    unsafe_args = build_arg_parser().parse_args(["--gripper-type", "linear_4310"])
    with pytest.raises(ValueError, match="requires calibrated"):
        build_config(unsafe_args)


def test_yam_preset_enables_three_camera_observations() -> None:
    root = Path(__file__).resolve().parents[2]
    deploy = load_config(root / "configs" / "01_deploy/dual_yam/openpi_qpos.py")

    assert deploy.transport.disabled_cameras == []
    assert deploy.inference_cfg.manual_max_qpos_step == pytest.approx(0.005)
    assert deploy.inference_cfg.manual_settle_duration == pytest.approx(2.0)


def test_yam_dual_leader_collection_preset() -> None:
    root = Path(__file__).resolve().parents[2]
    collection = load_config(root / "configs" / "02_collection" / "dual_yam.py")

    assert collection.transport.disabled_cameras == []
    assert collection.inference_cfg.manual_max_qpos_step == pytest.approx(0.005)
    assert collection.inference_cfg.manual_settle_duration == pytest.approx(2.0)
    assert collection.collection.storage.image_skew_tolerance_sec == pytest.approx(0.020)
    assert dict(collection.collection.schema.cameras) == {
        "cam_high": "observation.images.cam_high",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }


def test_run_hardware_defaults_to_dual_leaders_and_gripper_calibration(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "examples/hardware/yam/run_hardware.sh").read_text()
    assert "choose_can can_follower_l can1" in launcher
    assert "choose_can can_follower_r can2" in launcher
    assert "choose_can can_leader_l can0" in launcher
    assert "choose_can can_leader_r can3" in launcher

    fake_venv = tmp_path / "venv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)
    captured_args = tmp_path / "node-args.txt"
    captured_environment = tmp_path / "node-environment.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$YAM_TEST_ARGS_PATH\"\n"
        "printf '%s\\n' \"${PYTHONPATH-}\" \"${VIRTUAL_ENV-}\" \"${PATH-}\" "
        "\"${PYTHONHOME-}\" \"${PYTHONNOUSERSITE-}\" > \"$YAM_TEST_ENV_PATH\"\n"
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    for name in (
        "ENABLE_YAM_LEADERS",
        "YAM_ALLOW_GRIPPER_CALIBRATION",
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
    camera_indexes = [index for index, value in enumerate(args) if value == "--camera"]
    assert camera_indexes == []
    orbbec_indexes = [index for index, value in enumerate(args) if value == "--orbbec-camera"]
    assert [args[index + 1] for index in orbbec_indexes] == [
        "cam_left_wrist=CV2R1610003Z",
        "cam_right_wrist=CV2L360000CL",
        "cam_high=CP0HC530000Z",
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
    assert args[orbbec_width_index + 1] == "640"
    assert args[orbbec_height_index + 1] == "480"
    assert args[orbbec_fps_index + 1] == "30"
    assert args[orbbec_format_index + 1] == "MJPG"
    assert args[orbbec_warmup_index + 1] == "30"
    assert args[orbbec_brightness_index + 1] == "5"
    camera_profile_index = args.index("--camera-profile")
    assert args[camera_profile_index + 1].endswith(
        "examples/hardware/yam/profiles/d405_workcell.json"
    )
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


def test_yam_project_and_setup_cover_the_isolated_runtime_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    yam_dir = root / "examples/hardware/yam"
    project = tomllib.loads((yam_dir / "pyproject.toml").read_text())["project"]
    dependency_names = {
        dependency.split(" ", 1)[0].split("=", 1)[0].split(">", 1)[0].lower()
        for dependency in project["dependencies"]
    }
    setup = (yam_dir / "setup_sdk.sh").read_text()

    assert {"addict", "packaging", "pyyaml", "rich"} <= dependency_names
    assert 'SDK_DIR="$REPOSITORY_ROOT/$SDK_RELATIVE_DIR"' in setup
    assert "YAM_SDK_DIR" not in setup
    assert "dpkg-query" not in setup
    assert "linux-headers" not in setup
    assert "env -u PYTHONHOME -u PYTHONPATH -u VIRTUAL_ENV" in setup
    assert "UV_CONCURRENT_INSTALLS=1" in setup
    assert 'UV_PROJECT_ENVIRONMENT="$VENV_DIR"' in setup
    assert 'uv sync --no-cache --project "$PROJECT_DIR"' in setup
    assert 'PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT"' in setup
    assert "from examples.hardware.yam.node import build_arg_parser" in setup
    assert 'find_spec("pyrealsense2")' in setup


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
