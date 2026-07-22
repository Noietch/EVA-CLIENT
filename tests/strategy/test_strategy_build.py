"""Tests for inference strategy classes (OOP-direct, no intermediate RuntimeConfig).

Strategy classes are dataclasses whose fields are the YAML config keys; the
registered builder IS the class itself. STRATEGY_REGISTRY.build(class_name, **args)
constructs the instance directly.
"""

from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np
import pytest

from core.registry import STRATEGY_REGISTRY
from strategy.async_strategy import AsyncLinearOverlapInferStrategy
from strategy.base_strategy import BaseInferStrategy
from strategy.rtc_strategy import RtcInferStrategy


def _build(spec, rate: float = 3.0):
    """Mirror handlers.replace_inference_strategy: inject inference_rate only for
    classes that declare it.
    """
    args = dict(spec["args"])
    cls = STRATEGY_REGISTRY.get(spec["type"])
    if "inference_rate" in {f.name for f in dataclasses.fields(cls)}:  # pyright: ignore[reportArgumentType]
        args.setdefault("inference_rate", rate)
    return STRATEGY_REGISTRY.build(spec["type"], **args)


# --- registry dispatch by class-name key, instance fields ---


def test_build_sync_basic():
    strategy = _build({"type": "BaseInferStrategy", "args": {"execute_horizon": 5}})
    assert type(strategy) is BaseInferStrategy
    assert strategy.execute_horizon == 5
    assert strategy.sync_wait_ignore_gripper is False
    assert strategy.runs_background_loop is False


def test_build_sync_ignore_gripper_wait_flag():
    strategy = _build(
        {"type": "BaseInferStrategy", "args": {"sync_wait_ignore_gripper": True}}
    )
    assert strategy.sync_wait_ignore_gripper is True


def test_build_execute_horizon_floored_to_one():
    strategy = _build({"type": "BaseInferStrategy", "args": {"execute_horizon": 0}})
    assert strategy.execute_horizon == 1


def test_build_async_fields_and_clamps():
    strategy = _build(
        {
            "type": "AsyncLinearOverlapInferStrategy",
            "args": {
                "execute_horizon": 8,
                "latency_k": -3,
                "exp_weight_m": -1.0,
            },
        },
        rate=10.0,
    )
    assert isinstance(strategy, AsyncLinearOverlapInferStrategy)  # pyright: ignore[reportArgumentType]
    assert strategy.execute_horizon == 8
    assert strategy.inference_rate == 10.0
    assert strategy.latency_k == 0
    assert strategy.exp_weight_m == 0.0
    assert strategy.runs_background_loop is True


def test_build_async_defaults():
    strategy = _build({"type": "AsyncLinearOverlapInferStrategy", "args": {}})
    assert strategy.latency_k is None
    assert strategy.exp_weight_m == 0.01


def test_build_negative_inference_rate_clamped():
    strategy = _build({"type": "AsyncLinearOverlapInferStrategy", "args": {}}, rate=-5.0)
    assert strategy.inference_rate == 0.0


def test_build_rtc_fields():
    strategy = _build(
        {"type": "RtcInferStrategy", "args": {"execute_horizon": 4, "latency_k": 2}},
        rate=3.0,
    )
    assert isinstance(strategy, RtcInferStrategy)  # pyright: ignore[reportArgumentType]
    assert strategy.execute_horizon == 4
    assert strategy.latency_k == 2
    assert strategy.runs_background_loop is True


def test_build_unknown_type_raises_with_available():
    with pytest.raises(KeyError) as exc:
        STRATEGY_REGISTRY.build("NoSuchStrategy")
    msg = str(exc.value)
    assert "NoSuchStrategy" in msg
    assert "available" in msg.lower()
    for name in ("BaseInferStrategy", "AsyncLinearOverlapInferStrategy", "RtcInferStrategy"):
        assert name in msg


@pytest.mark.parametrize("legacy", ["sync", "async", "async_prefetch", "prefetch", "rtc"])
def test_legacy_short_names_no_longer_registered(legacy):
    with pytest.raises(KeyError):
        STRATEGY_REGISTRY.build(legacy)


# --- BaseInferStrategy._crop_chunk ---


def _sync(execute_horizon=None) -> BaseInferStrategy:  # pyright: ignore[reportGeneralTypeIssues]
    return BaseInferStrategy(execute_horizon=execute_horizon)


def test_crop_chunk_promotes_1d_to_2d():
    out = _sync()._crop_chunk(np.arange(7.0))
    assert out.shape == (1, 7)
    assert out.dtype == np.float32


def test_crop_chunk_3d_raises():
    with pytest.raises(ValueError):
        _sync()._crop_chunk(np.zeros((2, 3, 4)))


def test_crop_chunk_truncates_to_horizon():
    out = _sync(execute_horizon=3)._crop_chunk(np.zeros((10, 7)))
    assert out.shape == (3, 7)


def test_crop_chunk_horizon_longer_than_chunk_is_noop():
    out = _sync(execute_horizon=100)._crop_chunk(np.zeros((4, 7)))
    assert out.shape == (4, 7)


def test_crop_chunk_returns_copy():
    src = np.zeros((4, 2), dtype=np.float32)
    out = _sync()._crop_chunk(src)
    out[0, 0] = 9.0
    assert src[0, 0] == 0.0


# --- sync vs async vs rtc loop efficacy ---

_CHUNK_N = 6


class _FakeFetch:
    """Thread-safe fetch_chunk stub recording every (prompt, obs) it sees."""

    def __init__(self, n: int = _CHUNK_N) -> None:
        self._lock = threading.Lock()
        self.calls: list[tuple[str, dict | None]] = []
        self._n = n

    def __call__(self, prompt: str, obs: dict | None) -> np.ndarray:
        with self._lock:
            self.calls.append((prompt, obs))
        return np.arange(self._n).reshape(self._n, 1).astype(np.float32)

    def count(self) -> int:
        with self._lock:
            return len(self.calls)


_FAKE_OBS = {"state": np.zeros(7, dtype=np.float32)}


def _fake_get_obs() -> dict:
    return _FAKE_OBS


def _build_loop(spec):
    return _build(spec, rate=20.0)


def _loop_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name.startswith("eva-")}


def _poll(predicate, deadline: float = 1.0, interval: float = 0.005) -> bool:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


_ASYNC_SPEC = {
    "type": "AsyncLinearOverlapInferStrategy",
    "args": {},
}
_RTC_SPEC = {"type": "RtcInferStrategy", "args": {}}
_LOOP_SPECS = [_ASYNC_SPEC, _RTC_SPEC]
_LOOP_IDS = ["async_linear", "rtc"]


def test_sync_start_loop_is_true_noop():
    strategy = _build_loop({"type": "BaseInferStrategy", "args": {}})
    fetch = _FakeFetch()
    baseline = _loop_threads()
    try:
        strategy.start_loop("p", fetch, _fake_get_obs)
        time.sleep(0.05)
        assert strategy.is_loop_running() is False
        assert strategy.pop_next_action() is None
        assert fetch.count() == 0
        assert _loop_threads() - baseline == set()
    finally:
        strategy.stop_loop()
        strategy.close()


@pytest.mark.parametrize("spec", _LOOP_SPECS, ids=_LOOP_IDS)
def test_loop_spawns_named_thread_and_fetches(spec):
    strategy = _build_loop(spec)
    fetch = _FakeFetch()
    baseline = _loop_threads()
    try:
        strategy.start_loop("hello", fetch, _fake_get_obs)
        assert _poll(lambda: fetch.count() >= 2, deadline=1.0)
        assert strategy.is_loop_running() is True
        prompt, obs = fetch.calls[0]
        assert prompt == "hello"
        assert obs is _FAKE_OBS
        new_threads = _loop_threads() - baseline
        assert len(new_threads) == 1
        loop_thread = new_threads.pop()
        assert loop_thread.daemon is True
        assert loop_thread.name in {"eva-async-loop", "eva-rtc-loop"}
    finally:
        strategy.stop_loop()
        strategy.close()


@pytest.mark.parametrize("spec", _LOOP_SPECS, ids=_LOOP_IDS)
def test_reset_joins_thread_and_drops_stale_chunks(spec):
    strategy = _build_loop(spec)

    in_fetch = {"n": 0}
    lock = threading.Lock()

    def fetch(_prompt: str, _obs: dict | None) -> np.ndarray:
        with lock:
            in_fetch["n"] += 1
        return np.arange(_CHUNK_N).reshape(_CHUNK_N, 1).astype(np.float32)

    baseline = _loop_threads()
    try:
        strategy.start_loop("p", fetch, _fake_get_obs)
        assert _poll(lambda: in_fetch["n"] >= 2, deadline=1.0)
        own_threads = _loop_threads() - baseline
        strategy.reset()
        with lock:
            snapshot = in_fetch["n"]
        time.sleep(0.1)
        with lock:
            assert in_fetch["n"] == snapshot
        assert strategy.is_loop_running() is False
        assert all(not t.is_alive() for t in own_threads)
    finally:
        strategy.stop_loop()
        strategy.close()


@pytest.mark.parametrize("spec", _LOOP_SPECS, ids=_LOOP_IDS)
def test_take_or_fetch_chunk_fallback_first_chunk(spec):
    strategy = _build_loop(spec)
    fetch = _FakeFetch()
    expected = np.arange(_CHUNK_N).reshape(_CHUNK_N, 1).astype(np.float32)
    try:
        out = strategy.take_or_fetch_chunk("p", fetch)
        assert out.ndim == 2
        assert out.dtype == np.float32
        assert len(out) >= 1
        np.testing.assert_array_equal(out, expected)
    finally:
        strategy.stop_loop()
        strategy.close()


@pytest.mark.parametrize("spec", _LOOP_SPECS, ids=_LOOP_IDS)
def test_prepare_warmup_chunk_remains_available_for_immediate_publish(spec):
    strategy = _build_loop(spec)
    fetch = _FakeFetch()
    strategy.seed_buffer(np.array([0.0], dtype=np.float32))

    try:
        strategy.prepare_warmup_chunk("p", fetch)
        action = strategy.pop_next_action()

        assert action is not None
        assert action.dtype == np.float32
        assert action.shape == (1,)
    finally:
        strategy.stop_loop()
        strategy.close()


@pytest.mark.parametrize("spec", _LOOP_SPECS, ids=_LOOP_IDS)
def test_loop_integrates_chunks_for_pop(spec):
    strategy = _build_loop(spec)
    fetch = _FakeFetch()
    try:
        strategy.start_loop("p", fetch, _fake_get_obs)
        action = None

        def _got():
            nonlocal action
            action = strategy.pop_next_action()
            return action is not None

        assert _poll(_got, deadline=1.0)
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.float32
        assert action.shape == (1,)
    finally:
        strategy.stop_loop()
        strategy.close()


@pytest.mark.parametrize("spec", _LOOP_SPECS, ids=_LOOP_IDS)
def test_double_start_loop_is_idempotent(spec):
    strategy = _build_loop(spec)
    fetch = _FakeFetch()
    baseline = _loop_threads()
    try:
        strategy.start_loop("p", fetch, _fake_get_obs)
        strategy.start_loop("p", fetch, _fake_get_obs)
        assert _poll(strategy.is_loop_running, deadline=1.0)
        assert len(_loop_threads() - baseline) == 1
    finally:
        strategy.stop_loop()
        strategy.close()


@pytest.mark.parametrize("spec", _LOOP_SPECS, ids=_LOOP_IDS)
def test_stop_loop_joins_promptly(spec):
    strategy = _build_loop(spec)
    fetch = _FakeFetch()
    baseline = _loop_threads()
    try:
        strategy.start_loop("p", fetch, _fake_get_obs)
        assert _poll(strategy.is_loop_running, deadline=1.0)
        own_threads = _loop_threads() - baseline
        start = time.monotonic()
        strategy.stop_loop()
        assert time.monotonic() - start < 2.0
        assert strategy.is_loop_running() is False
        assert all(not t.is_alive() for t in own_threads)
    finally:
        strategy.stop_loop()
        strategy.close()


@pytest.mark.parametrize("spec", _LOOP_SPECS, ids=_LOOP_IDS)
def test_loop_gates_fetch_on_observation(spec):
    strategy = _build_loop(spec)
    fetch = _FakeFetch()
    ready = threading.Event()

    def get_obs() -> dict | None:
        return _FAKE_OBS if ready.is_set() else None

    try:
        strategy.start_loop("p", fetch, get_obs)
        time.sleep(0.1)
        assert fetch.count() == 0
        assert strategy.is_loop_running() is True
        ready.set()
        assert _poll(lambda: fetch.count() > 0, deadline=2.0)
    finally:
        strategy.stop_loop()
        strategy.close()


# --- naive_async + act_ensemble (added strategy coverage) ---


def test_build_naive_async_inherits_background_loop_fields():
    from strategy.naive_strategy import NaiveAsyncInferStrategy

    strategy = _build(
        {
            "type": "NaiveAsyncInferStrategy",
            "args": {"execute_horizon": 4, "latency_k": 2},
        },
        rate=10.0,
    )
    assert isinstance(strategy, NaiveAsyncInferStrategy)  # pyright: ignore[reportArgumentType]
    assert strategy.execute_horizon == 4
    assert strategy.inference_rate == 10.0
    assert strategy.latency_k == 2
    assert strategy.runs_background_loop is True


def test_build_act_ensemble_with_exp_weight_clamped():
    from strategy.act_strategy import ActEnsembleInferStrategy

    strategy = _build(
        {
            "type": "ActEnsembleInferStrategy",
            "args": {"execute_horizon": 6, "exp_weight_m": -0.5},
        },
        rate=5.0,
    )
    assert isinstance(strategy, ActEnsembleInferStrategy)  # pyright: ignore[reportArgumentType]
    assert strategy.execute_horizon == 6
    assert strategy.exp_weight_m == 0.0
    assert strategy.runs_background_loop is True


def test_rtc_strategy_builds_with_latency_k():
    strategy = _build(
        {"type": "RtcInferStrategy", "args": {"latency_k": 7}},
        rate=3.0,
    )
    assert strategy.latency_k == 7
    assert strategy.runs_background_loop is True
