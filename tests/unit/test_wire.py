"""Tests for the zmq wire protocol (transport.wire) — pack/unpack roundtrips."""

from __future__ import annotations

import collections
import threading
import types

import pytest

import robots  # noqa: F401  (registers piper)
import transport.zmq as zmq_transport
from transport.zmq import (
    _ObservationReader,
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
