"""Tests for async inference action buffers (strategy.action_buffer)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import strategy.action_buffer as action_buffer
from strategy.action_buffer import StreamActionBuffer


def _chunk(values):
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def _pop(buf) -> np.ndarray:
    """Pop an action that is expected to be present (narrows Optional for type checkers)."""
    action = buf.pop_next_action()
    assert action is not None
    return action


# --- StreamActionBuffer ---


def test_stream_first_chunk_becomes_buffer():
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(_chunk([1.0, 2.0, 3.0]))
    assert buf.has_actions()
    np.testing.assert_allclose(_pop(buf), [1.0], atol=1e-6)
    np.testing.assert_allclose(_pop(buf), [2.0], atol=1e-6)
    np.testing.assert_allclose(_pop(buf), [3.0], atol=1e-6)
    assert not buf.has_actions()


def test_stream_empty_chunk_ignored():
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(np.zeros((0, 1), dtype=np.float32))
    assert not buf.has_actions()
    assert buf.pop_next_action() is None


def test_stream_pop_increments_k_and_drop_n_compensates():
    """After consuming k steps, a new chunk is trimmed by min(k, max_k) at the front."""
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(_chunk([0.0, 0.0, 0.0, 0.0]))
    buf.pop_next_action()  # k=1
    buf.pop_next_action()  # k=2
    # cur_chunk now [0,0], k=2. New chunk trimmed by min(2,2)=2 at front.
    new = _chunk([10.0, 11.0, 12.0, 13.0])
    buf.integrate_new_chunk(new, max_k=2, min_m=2)
    # remaining new after trim: [12, 13]; smoothing blends with old [0,0]
    actions = []
    while buf.has_actions():
        actions.append(float(_pop(buf)[0]))
    assert len(actions) == 2
    # end of overlap weighted fully toward new
    assert actions[-1] == pytest.approx(13.0, abs=1e-5)


def test_stream_overlap_is_smoothed_monotonic_blend():
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(_chunk([0.0, 0.0, 0.0, 0.0]))
    # do not pop -> k=0, no trim
    buf.integrate_new_chunk(_chunk([10.0, 10.0, 10.0, 10.0]), max_k=0, min_m=4)
    vals = []
    while buf.has_actions():
        vals.append(float(_pop(buf)[0]))
    # linear blend from old(0) to new(10): increasing, start near 0, end == 10
    assert vals[0] == pytest.approx(0.0, abs=1e-5)
    assert vals[-1] == pytest.approx(10.0, abs=1e-5)
    assert all(a <= b + 1e-6 for a, b in zip(vals, vals[1:], strict=False))


def test_stream_seed_then_smooths_against_seed():
    buf = StreamActionBuffer()
    buf.seed(np.array([5.0], dtype=np.float32))
    assert not buf.has_actions()
    buf.integrate_new_chunk(_chunk([5.0, 5.0, 5.0, 5.0]), max_k=0, min_m=4)
    assert buf.has_actions()


def test_stream_reset_clears_state():
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(_chunk([1.0, 2.0]))
    buf.reset()
    assert not buf.has_actions()
    assert buf.pop_next_action() is None


# --- StreamActionBuffer: exact numeric blend ---


def _drain(buf) -> list[float]:
    vals = []
    while buf.has_actions():
        vals.append(float(_pop(buf)[0]))
    return vals


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
    np.testing.assert_allclose(_drain(buf), [0.0, 8.88889, 15.55556, 20.0], atol=1e-4)


def test_stream_min_m_tail_pads_short_old_buffer():
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(_chunk([1.0, 2.0]))
    buf.integrate_new_chunk(_chunk([10.0, 10.0, 10.0, 10.0]), max_k=0, min_m=4)
    drained = _drain(buf)
    assert len(drained) == 4
    np.testing.assert_allclose(drained, [1.0, 4.66667, 7.33333, 10.0], atol=1e-4)


def test_stream_drop_n_ge_len_early_returns_untouched():
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(_chunk([1.0, 2.0, 3.0, 4.0, 5.0]))
    for _ in range(4):
        buf.pop_next_action()  # k=4, buffer holds [5]
    # drop_n = min(4, 4) = 4 >= len(new)=3 -> early return, buffer unchanged
    buf.integrate_new_chunk(_chunk([90.0, 91.0, 92.0]), max_k=4, min_m=2)
    assert _drain(buf) == [5.0]


def test_stream_old_longer_than_new_is_trimmed():
    buf = StreamActionBuffer()
    buf.integrate_new_chunk(_chunk([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    buf.integrate_new_chunk(_chunk([10.0, 10.0]), max_k=0, min_m=2)
    drained = _drain(buf)
    assert len(drained) == 2
    np.testing.assert_allclose(drained, [0.0, 10.0], atol=1e-6)


def test_stream_seed_then_integrate_blends_then_clears_seed():
    buf = StreamActionBuffer()
    buf.seed(np.array([0.0], dtype=np.float32))
    buf.integrate_new_chunk(_chunk([10.0, 10.0, 10.0, 10.0]), max_k=0, min_m=4)
    np.testing.assert_allclose(
        [float(a[0]) for a in buf._cur_chunk],
        [0.0, 3.33333, 6.66667, 10.0],
        atol=1e-4,
    )
    drained = _drain(buf)
    np.testing.assert_allclose(drained, [0.0, 3.33333, 6.66667, 10.0], atol=1e-4)
    # second integrate must not re-pad from the now-cleared seed
    buf.integrate_new_chunk(_chunk([20.0, 20.0, 20.0, 20.0]), max_k=0, min_m=4)
    # last_action from final pop was 10 -> pads to [10,10,10,10] then blends to 20
    np.testing.assert_allclose(_drain(buf), [10.0, 13.33333, 16.66667, 20.0], atol=1e-4)


# --- Concurrency ---


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
