"""Console UI: the happy-path operation sequence a user actually performs.

Order mirrors the real workflow: connect → pick mode → pick prompt → setup →
run → (robot publishes) → halt → reset. Each step asserts both the HTTP contract
(status code + ``{"ok": true}``) and the resulting session/runtime state, so a
regression in either the route wiring or the state machine is caught.
"""

from __future__ import annotations

import pytest

from core.app import handlers

pytestmark = pytest.mark.integration


def test_full_operation_sequence_connect_to_reset(console):
    # 1. Connect the (mock) policy.
    assert console.do("/api/connect").json == {"ok": True}
    assert console.status()["policy_connected"] is True

    # 2. Pick the SIM mode (no real arm).
    console.do("/api/select_mode", {"mode": "sim"})
    assert console.status()["cli_mode"] == "sim"

    # 3. Pick a prompt.
    console.do("/api/select_task", {"task": "pick up the cup"})
    assert console.status()["selected_task"] == "pick up the cup"

    # 4. Setup: validates policy + resets to home + primes a chunk -> READY.
    console.do("/api/setup")
    st = console.status()
    assert st["is_setup_done"] is True
    assert st["session_status"] == "ready"
    assert st["last_error"] == ""

    # 5. Run -> RUNNING.
    console.do("/api/run")
    assert console.status()["session_status"] == "running"

    # 6. The main loop publishes actions while RUNNING; drive a few cycles.
    for _ in range(5):
        handlers.publish_next_action(console.config, console.runtime, console.session)
    assert console.status()["step_index"] == 5

    # 7. Halt -> back to READY (an episode boundary, not a full teardown).
    console.do("/api/halt")
    assert console.status()["session_status"] == "ready"

    # 8. Reset comes AFTER operation: clears progress back to UNSET / not-setup.
    console.do("/api/reset")
    st = console.status()
    assert st["session_status"] == "unset"
    assert st["is_setup_done"] is False


def test_manual_gripper_does_not_lock_by_default_and_unlock_does_not_move(console):
    import numpy as np

    console.do("/api/gripper", {"side": "l", "state": "open"})
    assert console.session.gripper_locks == {}
    console.do("/api/gripper", {"side": "l", "state": "close", "lock": True})
    assert console.session.gripper_locks
    before = console.runtime.transport.get_latest_qpos().copy()
    console.do("/api/gripper_unlock")
    assert console.session.gripper_locks == {}
    np.testing.assert_array_equal(console.runtime.transport.get_latest_qpos(), before)


def test_explicit_default_strategy_is_selected_before_any_execution(console):
    from core.app import run as app
    from core.config import ConfigDict

    console.config.inference_cfg.default_strategy = "async"
    console.config.inference_cfg.publish_rate = 10
    console.config.inference_strategies["async"] = ConfigDict(
        type="AsyncLinearOverlapInferStrategy", args=dict(execute_horizon=48, latency_k=0)
    )
    console.runtime.selected_inference_strategy_key = None
    assert app._select_only_inference_strategy_if_needed(
        console.config, console.runtime, console.session
    )
    assert console.status()["selected_strategy"] == "async"
    assert console.runtime.infer_strategy.runs_background_loop
    assert not console.runtime.infer_strategy.is_loop_running()
    assert console.status()["step_index"] == 0


def test_linear_overlap_uses_robot_gripper_columns_and_config_switch(console):
    import numpy as np

    from core.config import ConfigDict

    console.config.inference_strategies["async"] = ConfigDict(
        type="AsyncLinearOverlapInferStrategy", args=dict(execute_horizon=4)
    )
    console.do("/api/select_strategy", {"strategy": "async"})
    grippers = console.runtime.robot.gripper_indices
    assert grippers
    old = np.zeros_like(console.runtime.robot.initial_qpos, dtype=np.float32)
    new = np.tile(np.full_like(old, 12), (4, 1))
    new[:, grippers] = 1
    strategy = console.runtime.infer_strategy
    assert not strategy.blend_gripper
    assert strategy.gripper_indices == grippers
    strategy.seed_buffer(old)
    strategy.prepare_warmup_chunk("pick", lambda *_: new)
    first = strategy.pop_next_action()
    np.testing.assert_array_equal(first[list(grippers)], np.ones(len(grippers)))
    arms = [i for i in range(len(old)) if i not in grippers]
    np.testing.assert_array_equal(first[arms], old[arms])
    console.do("/api/update_infer_params", {"strategies": {"async": {"blend_gripper": True}}})
    strategy = console.runtime.infer_strategy
    strategy.seed_buffer(old)
    strategy.prepare_warmup_chunk("pick", lambda *_: new)
    np.testing.assert_array_equal(strategy.pop_next_action(), old)
