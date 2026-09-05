"""Tests for the zmq wire protocol (transport.wire) — pack/unpack roundtrips."""

from __future__ import annotations

import collections
import threading
import types

import numpy as np
import pytest

import robots  # noqa: F401  (registers piper)
import transport.zmq as zmq_transport
from core.registry import ROBOT_REGISTRY
from transport.zmq import (
    WireObservation,
    _ObservationReader,
    pack_observation,
)

pytestmark = pytest.mark.unit


_PIPER_CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _build_raw_collection_reader(robot, payloads, *, image_mode="stream"):
    class _Again(Exception):
        pass

    class _Sub:
        def recv(self, _flags):
            if not payloads:
                raise _Again
            return payloads.popleft()

    reader = object.__new__(_ObservationReader)
    reader._robot = robot
    reader._image_mode = image_mode
    reader._disabled_cameras = set()
    reader._zmq = types.SimpleNamespace(NOBLOCK=object(), Again=_Again)
    reader._sub = _Sub()
    reader._raw_collection_queue = collections.deque()
    reader._freshness = types.SimpleNamespace(mark=lambda: None)  # type: ignore[reportAttributeAccessIssue]
    reader._lock = threading.Lock()
    return reader


def test_zmq_raw_collection_snapshot_defers_decode_until_requested(monkeypatch, tmp_path):
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
                    action=np.ones(robot.total_action_dim, dtype=np.float32),
                )
            )
        ]
    )
    reader = _build_raw_collection_reader(robot, payloads)
    reader.prepare_collection_capture(str(tmp_path))

    decode_calls = []
    real_unpack_observation = zmq_transport.unpack_observation

    def _wrapped_unpack_observation(payload):
        decode_calls.append(payload)
        return real_unpack_observation(payload)

    monkeypatch.setattr(zmq_transport, "unpack_observation", _wrapped_unpack_observation)

    snapshot = reader.acquire_collection_raw()

    assert snapshot is not None
    assert decode_calls == []

    reader.finish_collection_capture()
    batch = snapshot.decode_raw()

    assert len(decode_calls) == 1
    assert list(batch.images) == list(_PIPER_CAMERA_KEYS)
    stored_images = [sample.value for samples in batch.images.values() for sample in samples]
    assert all(isinstance(image, np.ndarray) for image in stored_images)
    assert all(image.shape == shape for image in stored_images)
    assert all(image.dtype == np.uint8 for image in stored_images)
    np.testing.assert_array_equal(stored_images[0], images[_PIPER_CAMERA_KEYS[0]])
    np.testing.assert_allclose(
        batch.vectors["action_qpos"][0].value, np.ones(robot.total_action_dim, dtype=np.float32)
    )


def test_zmq_wire_journal_writes_off_the_capture_thread(monkeypatch, tmp_path):
    real_named_temporary_file = zmq_transport.tempfile.NamedTemporaryFile
    write_threads = []

    class _TrackedFile:
        def __init__(self):
            self._file = real_named_temporary_file(dir=tmp_path, buffering=0)

        def write(self, payload):
            write_threads.append(threading.current_thread().name)
            return self._file.write(payload)

        def __getattr__(self, name):
            return getattr(self._file, name)

    monkeypatch.setattr(
        zmq_transport.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: _TrackedFile(),
    )
    journal = zmq_transport._WireCaptureJournal(tmp_path)

    entry = journal.append(b"payload")
    journal.finish()

    assert entry.read() == b"payload"
    assert write_threads == ["eva-zmq-wire-journal"]


def test_zmq_journal_finish_cannot_overtake_pending_append(monkeypatch, tmp_path):
    real_queue = zmq_transport.queue.SimpleQueue
    appending = threading.Event()
    finishing = threading.Event()
    finished = threading.Event()
    enqueued = []

    class PausedQueue:
        def __init__(self):
            self.queue = real_queue()

        def put(self, payload):
            if payload is not None:
                appending.set()
                assert finishing.wait(2)
                finished.wait(0.2)
            enqueued.append(payload)
            self.queue.put(payload)

        def get(self):
            return self.queue.get()

    monkeypatch.setattr(zmq_transport.queue, "SimpleQueue", PausedQueue)
    journal = zmq_transport._WireCaptureJournal(tmp_path)

    # Race the final sentinel against a producer paused just before enqueueing
    def finish():
        assert appending.wait(2)
        finishing.set()
        journal.finish()
        finished.set()

    thread = threading.Thread(target=finish, daemon=True)
    thread.start()
    entry = journal.append(b"payload")
    thread.join(timeout=3)
    assert finished.is_set()
    assert enqueued == [b"payload", None]
    assert entry.read() == b"payload"
