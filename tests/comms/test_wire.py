"""Tests for the zmq wire protocol (transport.wire) — pack/unpack roundtrips."""

from __future__ import annotations

import collections
import sys
import threading
import types
from typing import cast

import numpy as np
import pytest

import robots  # noqa: F401  (registers piper)
import transport.utils as transport_utils
import transport.zmq as zmq_transport
from core.config import ConfigDict
from core.registry import ROBOT_REGISTRY
from core.types import CollectionRawImage
from robots.base import Robot
from transport.utils import ImageRateTracker
from transport.zmq import (
    COLLECTION_CONTROL_REPEATS,
    WireAction,
    WireObservation,
    ZmqTransport,
    _CollectionImageSpool,
    _ObservationReader,
    pack_action,
    pack_observation,
    unpack_action,
    unpack_observation,
)


def _build_zmq_transport_with_fake_readers(monkeypatch):
    readers = []

    class _Reader:
        def __init__(
            self,
            _config,
            _robot,
            _zmq_mod,
            preserve_collection_backlog=False,
            conflate=False,
        ) -> None:
            self.frame = None
            self.collection_frame = None
            self.qpos = None
            self.age = None
            self.calls = []
            self.closed = False
            self.conflate = conflate
            readers.append(self)

        def get_frame(self):
            self.calls.append("frame")
            return self.frame

        def get_collection_frame(self):
            self.calls.append("collection")
            return self.collection_frame

        def get_latest_qpos(self):
            self.calls.append("qpos")
            return self.qpos

        def seconds_since_last_recv(self):
            return self.age

        def close(self):
            self.closed = True

    class _Socket:
        def __init__(self) -> None:
            self.closed = False

        def connect(self, _endpoint) -> None:
            return None

        def close(self, linger=0) -> None:
            self.closed = True

    class _Context:
        def __init__(self) -> None:
            self.sockets = []

        def socket(self, _kind):
            socket = _Socket()
            self.sockets.append(socket)
            return socket

    ctx = _Context()
    fake_zmq = types.SimpleNamespace(
        PUB=1,
        Context=types.SimpleNamespace(instance=lambda: ctx),
    )
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    monkeypatch.setattr(zmq_transport, "_ObservationReader", _Reader)

    config = types.SimpleNamespace(
        transport=types.SimpleNamespace(
            sub_endpoint="tcp://127.0.0.1:5555",
            pub_endpoint="tcp://127.0.0.1:5556",
        )
    )
    robot = types.SimpleNamespace(name="test_robot")
    return (
        ZmqTransport(cast(ConfigDict, config), cast(Robot, robot)),
        readers,
        ctx,
    )


def test_observation_reader_enables_conflate_before_connect():
    operations = []

    class _Socket:
        def setsockopt(self, option, value):
            operations.append(("setsockopt", option, value))

        def connect(self, endpoint):
            operations.append(("connect", endpoint))

    socket = _Socket()
    fake_zmq = types.SimpleNamespace(
        SUB=1,
        CONFLATE=2,
        SUBSCRIBE=3,
        RCVTIMEO=4,
        Context=types.SimpleNamespace(
            instance=lambda: types.SimpleNamespace(socket=lambda _kind: socket)
        ),
    )
    config = types.SimpleNamespace(
        transport=types.SimpleNamespace(
            disabled_cameras=[],
            disabled_groups=[],
            sub_endpoint="tcp://127.0.0.1:5555",
        )
    )

    _ObservationReader(
        cast(ConfigDict, config),
        ROBOT_REGISTRY.build("agilex_piper"),
        fake_zmq,
        conflate=True,
    )

    assert operations == [
        ("setsockopt", fake_zmq.CONFLATE, 1),
        ("setsockopt", fake_zmq.SUBSCRIBE, b""),
        ("setsockopt", fake_zmq.RCVTIMEO, 0),
        ("connect", config.transport.sub_endpoint),
    ]


def test_observation_roundtrip_preserves_images_state_and_timestamp():
    images = {
        "cam_high": np.random.randint(0, 256, (12, 16, 3), dtype=np.uint8),
        "cam_left_wrist": np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8),
    }
    state = {
        "left_arm": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0], dtype=np.float32),
        "right_arm": np.array([-0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    obs = WireObservation(t=123.456, images=images, state=state)

    back = unpack_observation(pack_observation(obs))

    assert back.t == 123.456
    assert set(back.images) == set(images)
    for key, img in images.items():
        np.testing.assert_array_equal(back.images[key], img)
        assert back.images[key].dtype == np.uint8
    for key, vec in state.items():
        np.testing.assert_allclose(back.state[key], vec)


def test_observation_roundtrip_preserves_collection_fields():
    obs = WireObservation(
        t=3.0,
        images={"cam_high": np.zeros((4, 4, 3), dtype=np.uint8)},
        state={"left_arm": np.ones(7, dtype=np.float32)},
        eef={"left_arm": np.ones(8, dtype=np.float32) * 2},
        action=np.ones(7, dtype=np.float32) * 5,
        action_eef=np.ones(8, dtype=np.float32) * 6,
    )

    back = unpack_observation(pack_observation(obs))

    assert back.eef is not None
    assert back.action is not None
    assert back.action_eef is not None
    np.testing.assert_allclose(back.eef["left_arm"], np.ones(8) * 2)
    np.testing.assert_allclose(back.action, np.ones(7) * 5)
    np.testing.assert_allclose(back.action_eef, np.ones(8) * 6)


def test_wire_observation_converts_to_collection_frame_by_robot_group_order():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    reader = object.__new__(_ObservationReader)
    reader._robot = robot

    left_qpos = np.ones(7, dtype=np.float32)
    right_qpos = np.ones(7, dtype=np.float32) * 2
    left_eef = np.ones(8, dtype=np.float32) * 3
    right_eef = np.ones(8, dtype=np.float32) * 4
    action_qpos = np.arange(14, dtype=np.float32)
    action_eef = np.arange(16, dtype=np.float32)
    wire_obs = WireObservation(
        t=3.0,
        images={"cam_high": np.zeros((4, 4, 3), dtype=np.uint8)},
        state={"right_arm": right_qpos, "left_arm": left_qpos},
        eef={"right_arm": right_eef, "left_arm": left_eef},
        action=action_qpos,
        action_eef=action_eef,
    )

    frame = reader._wire_to_observation(wire_obs)

    assert frame.timestamp == 3.0
    assert frame.state_eef is not None
    assert frame.action_qpos is not None
    assert frame.action_eef is not None
    np.testing.assert_allclose(frame.state_qpos, np.concatenate([left_qpos, right_qpos]))
    np.testing.assert_allclose(frame.state_eef, np.concatenate([left_eef, right_eef]))
    np.testing.assert_allclose(frame.action_qpos, action_qpos)
    np.testing.assert_allclose(frame.action_eef, action_eef)


def test_zmq_collection_reader_preserves_backlog_order():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    payloads = collections.deque(
        pack_observation(
            WireObservation(
                t=float(index),
                images={"cam_high": np.zeros((4, 4, 3), dtype=np.uint8)},
                state={
                    group.name: np.zeros(group.dof, dtype=np.float32)
                    for group in robot.actuator_groups
                },
                action=np.ones(robot.total_action_dim, dtype=np.float32) * index,
            )
        )
        for index in range(3)
    )

    class _Again(Exception):
        pass

    class _Sub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.popleft()

    reader = object.__new__(_ObservationReader)
    reader._robot = robot
    reader._zmq = types.SimpleNamespace(NOBLOCK=object(), Again=_Again)
    reader._sub = _Sub()
    reader._latest = None
    reader._disabled_cameras = set()
    reader._latest_images = {}
    reader._image_rate = ImageRateTracker()
    reader._preserve_collection_backlog = True
    reader._collection_queue = collections.deque()
    reader._freshness = types.SimpleNamespace(mark=lambda: None)  # type: ignore[reportAttributeAccessIssue]
    reader._image_rate = transport_utils.ImageRateTracker()
    reader._lock = threading.Lock()

    frames = [reader.get_collection_frame() for _ in range(3)]

    assert all(frame is not None for frame in frames)
    assert [frame.timestamp for frame in frames if frame is not None] == [0.0, 1.0, 2.0]
    last = frames[2]
    assert last is not None and last.action_qpos is not None
    np.testing.assert_allclose(last.action_qpos, np.ones(robot.total_action_dim) * 2)


def test_zmq_collection_reader_defaults_to_latest_frame():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    payloads = collections.deque(
        pack_observation(
            WireObservation(
                t=float(index),
                images={"cam_high": np.zeros((4, 4, 3), dtype=np.uint8)},
                state={
                    group.name: np.zeros(group.dof, dtype=np.float32)
                    for group in robot.actuator_groups
                },
                action=np.ones(robot.total_action_dim, dtype=np.float32) * index,
            )
        )
        for index in range(3)
    )

    class _Again(Exception):
        pass

    class _Sub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.popleft()

    reader = object.__new__(_ObservationReader)
    reader._robot = robot
    reader._zmq = types.SimpleNamespace(NOBLOCK=object(), Again=_Again)
    reader._sub = _Sub()
    reader._latest = None
    reader._disabled_cameras = set()
    reader._latest_images = {}
    reader._image_rate = ImageRateTracker()
    reader._preserve_collection_backlog = False
    reader._collection_queue = collections.deque()
    reader._freshness = types.SimpleNamespace(mark=lambda: None)  # type: ignore[reportAttributeAccessIssue]
    reader._image_rate = transport_utils.ImageRateTracker()
    reader._lock = threading.Lock()

    frame = reader.get_collection_frame()

    assert frame is not None and frame.action_qpos is not None
    assert frame.timestamp == 2.0
    np.testing.assert_allclose(frame.action_qpos, np.ones(robot.total_action_dim) * 2)


def test_zmq_raw_collection_reader_keeps_latest_with_bounded_socket_drain():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    budget = 64
    payloads = collections.deque(
        pack_observation(
            WireObservation(
                t=float(index),
                images={"cam_high": np.zeros((4, 4, 3), dtype=np.uint8)},
                state={
                    group.name: np.zeros(group.dof, dtype=np.float32)
                    for group in robot.actuator_groups
                },
                action=np.ones(robot.total_action_dim, dtype=np.float32) * index,
            )
        )
        for index in range(budget + 5)
    )

    class _Again(Exception):
        pass

    class _Sub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.popleft()

    reader = object.__new__(_ObservationReader)
    reader._zmq = types.SimpleNamespace(NOBLOCK=object(), Again=_Again)
    reader._sub = _Sub()
    reader._raw_collection_queue = collections.deque()
    reader._freshness = types.SimpleNamespace(mark=lambda: None)  # type: ignore[reportAttributeAccessIssue]
    reader._lock = threading.Lock()

    snapshot = reader.acquire_collection_raw()

    assert snapshot is not None
    assert snapshot.timestamp == float(budget - 1)
    assert not reader._raw_collection_queue
    assert len(payloads) == 5


_PIPER_CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _build_piper_raw_collection_payloads(robot):
    state = {group.name: np.zeros(group.dof, dtype=np.float32) for group in robot.actuator_groups}
    enabled_camera_keys = {camera.observation_key for camera in robot.observation_schema.cameras}
    assert enabled_camera_keys == set(_PIPER_CAMERA_KEYS)
    images = {
        key: np.full((4, 4, 3), index + 1, dtype=np.uint8)
        for index, key in enumerate(_PIPER_CAMERA_KEYS)
    }
    return collections.deque(
        [
            pack_observation(
                WireObservation(
                    t=1.0,
                    images=images,
                    state=state,
                    action=np.ones(robot.total_action_dim, dtype=np.float32),
                )
            ),
            pack_observation(
                WireObservation(
                    t=2.0,
                    images={},
                    state=state,
                    action=np.ones(robot.total_action_dim, dtype=np.float32) * 2,
                )
            ),
        ]
    )


def _build_raw_collection_reader(robot, payloads):
    class _Again(Exception):
        pass

    class _Sub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.popleft()

    reader = object.__new__(_ObservationReader)
    reader._robot = robot
    reader._zmq = types.SimpleNamespace(NOBLOCK=object(), Again=_Again)
    reader._sub = _Sub()
    reader._raw_collection_queue = collections.deque()
    reader._freshness = types.SimpleNamespace(mark=lambda: None)  # type: ignore[reportAttributeAccessIssue]
    reader._lock = threading.Lock()
    return reader


def test_zmq_raw_collection_stream_keeps_newer_state_only_payload():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    reader = _build_raw_collection_reader(robot, _build_piper_raw_collection_payloads(robot))

    snapshot = reader.acquire_collection_raw()

    assert snapshot is not None
    assert snapshot.timestamp == 2.0
    action = snapshot.decode_raw().vectors["action_qpos"][0].value
    np.testing.assert_array_equal(action, np.ones(robot.total_action_dim) * 2)


def test_zmq_raw_collection_spools_full_resolution_images_off_heap():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    shape = (480, 640, 3)
    images = {
        key: np.full(shape, 40 + index * 40, dtype=np.uint8)
        for index, key in enumerate(_PIPER_CAMERA_KEYS)
    }
    payloads = collections.deque(
        [
            pack_observation(
                WireObservation(
                    t=3.0,
                    images=images,
                    state={
                        group.name: np.zeros(group.dof, dtype=np.float32)
                        for group in robot.actuator_groups
                    },
                )
            )
        ]
    )
    reader = _build_raw_collection_reader(robot, payloads)

    snapshot = reader.acquire_collection_raw()

    assert snapshot is not None
    batch = snapshot.decode_raw()
    reader.rotate_collection_image_spool()
    raw_bytes = sum(image.nbytes for image in images.values())
    stored_images = [sample.value for samples in batch.images.values() for sample in samples]
    assert all(isinstance(image, CollectionRawImage) for image in stored_images)
    assert all(image.encoded is None and image.has_encoded for image in stored_images)
    assert sum(len(image.load_encoded()) for image in stored_images) < raw_bytes // 20
    assert all(image._decoded is None for image in stored_images)
    decoded = stored_images[0].decode()
    assert decoded.shape == shape
    assert abs(float(decoded.mean()) - 40.0) < 2.0


def test_collection_image_spool_appends_after_interleaved_read():
    spool = _CollectionImageSpool()
    first = spool.append(b"first")
    second = spool.append(b"second")

    assert first.read() == b"first"
    third = spool.append(b"third")

    assert first.read() == b"first"
    assert second.read() == b"second"
    assert third.read() == b"third"


def test_zmq_clear_collection_backlog_drops_raw_queue_and_returns_cutoff(monkeypatch):
    monkeypatch.setattr(zmq_transport.time, "monotonic", lambda: 123.0)
    payloads = collections.deque(
        [
            pack_observation(WireObservation(t=7.0, images={}, state={})),
            pack_observation(WireObservation(t=8.0, images={}, state={})),
        ]
    )

    class _Again(Exception):
        pass

    class _Sub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.popleft()

    reader = object.__new__(_ObservationReader)
    reader._zmq = types.SimpleNamespace(NOBLOCK=object(), Again=_Again)
    reader._sub = _Sub()
    reader._collection_queue = collections.deque([WireObservation(t=6.0, images={}, state={})])
    reader._raw_collection_queue = collections.deque(
        [pack_observation(WireObservation(t=6.5, images={}, state={}))]
    )
    reader._lock = threading.Lock()

    cutoff = reader.clear_collection_backlog()

    assert cutoff == 8.0
    assert not payloads
    assert not reader._collection_queue
    assert not reader._raw_collection_queue


def test_zmq_clear_collection_backlog_without_source_time_returns_none(monkeypatch):
    monkeypatch.setattr(zmq_transport.time, "monotonic", lambda: 123.0)

    class _Again(Exception):
        pass

    class _Sub:
        def recv(self, _flags):
            raise _Again

    reader = object.__new__(_ObservationReader)
    reader._zmq = types.SimpleNamespace(NOBLOCK=object(), Again=_Again)
    reader._sub = _Sub()
    reader._collection_queue = collections.deque()
    reader._raw_collection_queue = collections.deque()
    reader._lock = threading.Lock()

    assert reader.clear_collection_backlog() is None


def test_zmq_reader_exposes_partial_camera_frames_for_visualization():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    image = np.full((4, 4, 3), 127, dtype=np.uint8)
    wire_obs = WireObservation(
        t=1.0,
        images={"cam_high": image},
        state={
            group.name: np.zeros(group.dof, dtype=np.float32) for group in robot.actuator_groups
        },
    )

    reader = object.__new__(_ObservationReader)
    reader._robot = robot
    reader._disabled_cameras = set()
    reader._disabled_groups = set()
    reader._latest_images = {}
    reader._lock = threading.Lock()
    reader._drain_latest = lambda: wire_obs

    assert reader.get_camera_keys() == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    cam_high_frame = reader.get_camera_frame("cam_high")
    assert cam_high_frame is not None
    np.testing.assert_array_equal(cam_high_frame, image)
    assert reader.get_camera_frame("cam_left_wrist") is None
    assert reader.get_frame() is None


def test_zmq_reader_camera_keys_respect_disabled_cameras():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    reader = object.__new__(_ObservationReader)
    reader._robot = robot
    reader._disabled_cameras = {"cam_right_wrist"}
    reader._drain_latest = lambda: None

    assert reader.get_camera_keys() == ["cam_high", "cam_left_wrist"]


def test_zmq_reader_keeps_multi_camera_visualization_cache():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    cam_high = np.full((4, 4, 3), 40, dtype=np.uint8)
    cam_left = np.full((4, 4, 3), 80, dtype=np.uint8)
    state = {group.name: np.zeros(group.dof, dtype=np.float32) for group in robot.actuator_groups}
    payloads = collections.deque(
        [
            pack_observation(WireObservation(t=1.0, images={"cam_high": cam_high}, state=state)),
            pack_observation(
                WireObservation(t=2.0, images={"cam_left_wrist": cam_left}, state=state)
            ),
        ]
    )

    class _Again(Exception):
        pass

    class _Sub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.popleft()

    reader = object.__new__(_ObservationReader)
    reader._robot = robot
    reader._zmq = types.SimpleNamespace(NOBLOCK=object(), Again=_Again)
    reader._sub = _Sub()
    reader._latest = None
    reader._latest_images = {}
    reader._image_rate = ImageRateTracker()
    reader._disabled_cameras = set()
    reader._disabled_groups = set()
    reader._freshness = types.SimpleNamespace(mark=lambda: None)  # type: ignore[reportAttributeAccessIssue]
    reader._image_rate = transport_utils.ImageRateTracker()
    reader._lock = threading.Lock()

    assert reader.get_camera_keys() == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    cam_high_frame = reader.get_camera_frame("cam_high")
    cam_left_frame = reader.get_camera_frame("cam_left_wrist")
    assert cam_high_frame is not None
    assert cam_left_frame is not None
    np.testing.assert_array_equal(cam_high_frame, cam_high)
    np.testing.assert_array_equal(cam_left_frame, cam_left)
    assert reader.get_camera_frame("cam_right_wrist") is None


def test_zmq_reader_reports_minimum_image_hz_from_received_images():
    robot = ROBOT_REGISTRY.build("agilex_piper")
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    state = {group.name: np.zeros(group.dof, dtype=np.float32) for group in robot.actuator_groups}
    payloads = collections.deque(
        [
            pack_observation(
                WireObservation(
                    t=0.0,
                    images={"cam_high": image, "cam_left_wrist": image},
                    state=state,
                )
            ),
            pack_observation(WireObservation(t=0.05, images={"cam_high": image}, state=state)),
            pack_observation(
                WireObservation(t=0.10, images={"cam_left_wrist": image}, state=state)
            ),
        ]
    )

    class _Again(Exception):
        pass

    class _Sub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.popleft()

    reader = object.__new__(_ObservationReader)
    reader._robot = robot
    reader._zmq = types.SimpleNamespace(NOBLOCK=object(), Again=_Again)
    reader._sub = _Sub()
    reader._latest = None
    reader._latest_images = {}
    reader._disabled_cameras = {"cam_right_wrist"}
    reader._disabled_groups = set()
    reader._freshness = types.SimpleNamespace(mark=lambda: None)  # type: ignore[reportAttributeAccessIssue]
    reader._image_rate = transport_utils.ImageRateTracker()
    reader._lock = threading.Lock()

    reader.get_camera_keys()

    assert reader.image_min_hz() == pytest.approx(10.0)


def test_action_roundtrip_preserves_vector_target_and_timestamp():
    action = WireAction(t=9.0, action=np.arange(14, dtype=np.float32), target="sim")

    back = unpack_action(pack_action(action))

    assert back.t == 9.0
    assert back.target == "sim"
    np.testing.assert_allclose(back.action, np.arange(14))


def test_action_default_target_is_real():
    back = unpack_action(pack_action(WireAction(t=0.0, action=np.zeros(14))))
    assert back.target == "real"


def test_zmq_transport_uses_separate_internal_readers(monkeypatch):
    transport, readers, _ctx = _build_zmq_transport_with_fake_readers(monkeypatch)

    assert [reader.conflate for reader in readers] == [False, True, False]
    readers[0].frame = "frame"
    readers[1].collection_frame = "collection"
    readers[2].qpos = "qpos"

    assert transport.get_frame() == "frame"
    assert transport.get_collection_frame() == "collection"
    assert transport.get_latest_qpos() == "qpos"
    assert readers[0].calls == ["frame"]
    assert readers[1].calls == ["collection"]
    assert readers[2].calls == ["qpos"]


def test_zmq_transport_reports_freshest_reader_and_closes_all_readers(monkeypatch):
    transport, readers, ctx = _build_zmq_transport_with_fake_readers(monkeypatch)
    extra = transport.create_observation_reader()

    readers[0].age = 3.0
    readers[1].age = 2.0
    readers[2].age = None
    extra.age = 1.0  # type: ignore[reportAttributeAccessIssue]

    assert transport.seconds_since_last_recv() == 1.0

    transport.close()

    assert all(reader.closed for reader in readers)
    assert ctx.sockets[0].closed


@pytest.mark.parametrize("control_source", ["transport", "client"])
def test_zmq_transport_sends_collection_control_actions(monkeypatch, control_source):
    monkeypatch.setattr(zmq_transport.time, "sleep", lambda _seconds: None)

    class _Publisher:
        def __init__(self):
            self.payloads = []

        def send(self, payload):
            self.payloads.append(payload)

    class _Robot:
        total_action_dim = 7

    transport = object.__new__(ZmqTransport)
    transport._pub = _Publisher()
    transport._robot = cast(Robot, _Robot())
    transport._config = ConfigDict(
        collection=ConfigDict(teleop=ConfigDict(control_source=control_source))
    )

    transport.start_collection()
    transport.stop_collection()

    assert [unpack_action(payload).target for payload in transport._pub.payloads] == (
        ["collect_start"] * COLLECTION_CONTROL_REPEATS
        + ["collect_stop"] * COLLECTION_CONTROL_REPEATS
    )
    np.testing.assert_allclose(unpack_action(transport._pub.payloads[0]).action, np.zeros(7))
    assert unpack_action(transport._pub.payloads[0]).mode == control_source
    assert unpack_action(transport._pub.payloads[-1]).mode is None


def test_zmq_transport_policy_collection_uses_client_mode(monkeypatch):
    monkeypatch.setattr(zmq_transport.time, "sleep", lambda _seconds: None)

    class _Publisher:
        def __init__(self):
            self.payloads = []

        def send(self, payload):
            self.payloads.append(payload)

    class _Robot:
        total_action_dim = 7

    transport = object.__new__(ZmqTransport)
    transport._pub = _Publisher()
    transport._robot = cast(Robot, _Robot())
    transport._config = ConfigDict(
        collection=ConfigDict(teleop=ConfigDict(control_source="transport"))
    )

    transport.start_policy_collection()

    assert len(transport._pub.payloads) == COLLECTION_CONTROL_REPEATS
    assert all(unpack_action(payload).mode == "client" for payload in transport._pub.payloads)


def test_wire_hil_status_and_control_mode_round_trip():
    observation = WireObservation(
        t=1.0,
        images={},
        state={"arm": np.zeros(2, dtype=np.float32)},
        hil_supported=True,
        hil_active=True,
        hil_error="leader stale",
    )
    restored = unpack_observation(pack_observation(observation))
    assert restored.hil_supported is True
    assert restored.hil_active is True
    assert restored.hil_error == "leader stale"

    action = WireAction(
        t=2.0,
        action=np.zeros(2, dtype=np.float32),
        target="hil_start",
        mode="relative",
    )
    restored_action = unpack_action(pack_action(action))
    assert restored_action.target == "hil_start"
    assert restored_action.mode == "relative"
