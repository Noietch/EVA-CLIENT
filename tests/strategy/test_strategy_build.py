"""Background strategy reset and warmup regressions."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.registry import STRATEGY_REGISTRY

pytestmark = pytest.mark.unit


_CHUNK_N = 6
_STRATEGIES = ["AsyncLinearOverlapInferStrategy", "RtcInferStrategy"]


def _loop_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name.startswith("eva-")}


@pytest.mark.parametrize("strategy_type", _STRATEGIES)
def test_reset_joins_thread_and_drops_stale_chunks(strategy_type):
    strategy = STRATEGY_REGISTRY.build(strategy_type, inference_rate=20.0)

    in_fetch = {"n": 0}
    lock = threading.Lock()
    fetched = threading.Event()
    observation = {"state": np.zeros(7, dtype=np.float32)}

    def fetch(_prompt: str, _obs: dict | None) -> np.ndarray:
        with lock:
            in_fetch["n"] += 1
            if in_fetch["n"] >= 2:
                fetched.set()
        return np.arange(_CHUNK_N).reshape(_CHUNK_N, 1).astype(np.float32)

    baseline = _loop_threads()
    try:
        strategy.start_loop("p", fetch, lambda: observation)
        assert fetched.wait(timeout=1.0)
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


@pytest.mark.parametrize("strategy_type", _STRATEGIES)
def test_prepare_warmup_chunk_remains_available_for_immediate_publish(strategy_type):
    strategy = STRATEGY_REGISTRY.build(strategy_type, inference_rate=20.0)
    strategy.seed_buffer(np.array([0.0], dtype=np.float32))

    try:
        strategy.prepare_warmup_chunk(
            "p", lambda *_args: np.arange(_CHUNK_N, dtype=np.float32).reshape(_CHUNK_N, 1)
        )
        action = strategy.pop_next_action()

        assert action is not None
        assert action.dtype == np.float32
        assert action.shape == (1,)
    finally:
        strategy.stop_loop()
        strategy.close()
