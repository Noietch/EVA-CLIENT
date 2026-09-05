"""Tests for async inference action buffers (strategy.action_buffer)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import strategy.action_buffer as action_buffer
from strategy.action_buffer import StreamActionBuffer

pytestmark = pytest.mark.unit


def _chunk(values):
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def test_stream_exact_linear_blend_across_three_chunks():
    """Each chunk blends against the ALREADY-SMOOTHED buffer, not the raw prior chunk."""
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(_chunk([0.0, 0.0, 0.0, 0.0]))
    buf.integrate_new_chunk(_chunk([10.0, 10.0, 10.0, 10.0]), max_k=0, min_m=4)
    np.testing.assert_allclose(
        [float(a[0]) for a in buf._cur_chunk],
        [0.0, 3.33333, 6.66667, 10.0],
        atol=1e-4,
    )
    buf.integrate_new_chunk(_chunk([20.0, 20.0, 20.0, 20.0]), max_k=0, min_m=4)
    values = []
    while buf.has_actions():
        action = buf.pop_next_action()
        assert action is not None
        values.append(float(action[0]))
    np.testing.assert_allclose(values, [0.0, 8.88889, 15.55556, 20.0], atol=1e-4)


def _run_concurrent(writer, reader):
    """Run writer and reader threads, collecting per-thread exceptions; return the list."""
    errors: list[BaseException] = []

    def guard(fn):
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure to the test
            errors.append(exc)

    w = threading.Thread(target=guard, args=(writer,))
    r = threading.Thread(target=guard, args=(reader,))
    w.start()
    r.start()
    w.join()
    r.join()
    return errors


def test_stream_concurrent_pop_and_integrate():
    buf = StreamActionBuffer()
    buf.seed(np.zeros(7, dtype=np.float32))

    def writer():
        for j in range(2000):
            buf.integrate_new_chunk(np.full((16, 7), float(j), dtype=np.float32), max_k=4, min_m=8)

    def reader():
        for _ in range(20000):
            a = buf.pop_next_action()
            if a is not None:
                assert a.shape == (7,)
                assert np.all(np.isfinite(a))

    assert _run_concurrent(writer, reader) == []


def test_stream_vector_blending_does_not_hold_pop_lock(monkeypatch):
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(np.ones((16, 7), dtype=np.float32))
    blending = threading.Event()
    release = threading.Event()
    popped = threading.Event()
    original_deque = action_buffer.deque

    def blocked_deque(values):
        blending.set()
        release.wait(timeout=1.0)
        return original_deque(values)

    monkeypatch.setattr(action_buffer, "deque", blocked_deque)
    writer = threading.Thread(
        target=buf.integrate_new_chunk,
        args=(np.full((16, 7), 2.0, dtype=np.float32),),
    )
    reader = threading.Thread(target=lambda: (buf.pop_next_action(), popped.set()))
    writer.start()
    assert blending.wait(timeout=1.0)
    reader.start()

    assert popped.wait(timeout=0.1)

    release.set()
    writer.join(timeout=1.0)
    reader.join(timeout=1.0)
