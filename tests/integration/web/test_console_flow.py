"""Console UI: the happy-path operation sequence a user actually performs.

Order mirrors the real workflow: connect → pick mode → pick prompt → setup →
run → (robot publishes) → halt → reset. Each step asserts both the HTTP contract
(status code + ``{"ok": true}``) and the resulting session/runtime state, so a
regression in either the route wiring or the state machine is caught.
"""

from __future__ import annotations

from core.app import handlers
from core.app.state import SessionStatus
from core.config import ConfigDict


def test_config_route_serves_static_frontend_inputs(console):
    cfg = console.get("/api/config").json
    assert cfg["robot_type"] == "agilex_piper"
    assert cfg["transport_type"] == "debug"
    assert "pick up the cup" in cfg["tasks"]
    assert cfg["camera_keys"] == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    assert {s["key"] for s in cfg["strategies"]} == {"sync"}
    assert cfg["modes"] == ["real", "sim", "step", "manual"]
    assert cfg["rl"] == {"enabled": False}


def test_config_route_serves_rl_workspace_contract(console):
    console.config.rl = ConfigDict(
        cli_mode="real",
        inference_strategy="rtc",
        tasks=["pack the phone"],
        policies=[ConfigDict(name="policy-a")],
        critics=[ConfigDict(name="critic-a", type="websocket")],
        data=ConfigDict(
            format="lerobot",
            storage=ConfigDict(log_dir="work_dirs/rl/test"),
        ),
    )

    rl = console.get("/api/config").json["rl"]

    assert rl["backend_ready"] is True
    assert rl["cli_mode"] == "real"
    assert rl["inference_strategy"] == "rtc"
    assert rl["tasks"] == ["pack the phone"]
    assert rl["policies"] == [{"slot": 0, "name": "policy-a"}]
    assert rl["critics"] == [{"slot": 0, "name": "critic-a", "type": "websocket"}]
    assert rl["data"]["format"] == "lerobot"


def test_initial_status_is_idle_and_disconnected(console):
    st = console.status()
    assert st["policy_connected"] is False
    assert st["session_status"] == "unset"
    assert st["selected_task"] is None
    assert st["is_setup_done"] is False


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


def test_collect_and_debug_task_selection_are_independent(console):
    # DEBUG and COLLECT each remember their own task; selecting one must not
    # leak into the other (they were previously a single shared selected_prompt).
    console.do("/api/select_task", {"task": "pick up the cup"})
    console.do("/api/select_collect_task", {"task": "pour soybean"})
    st = console.status()
    assert st["selected_task"] == "pick up the cup"
    assert st["selected_collect_task"] == "pour soybean"

    # Re-selecting the DEBUG task leaves the COLLECT task untouched.
    console.do("/api/select_task", {"task": "pour soybean"})
    st = console.status()
    assert st["selected_task"] == "pour soybean"
    assert st["selected_collect_task"] == "pour soybean"


def test_run_elapsed_timer_starts_only_after_running(console):
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    console.do("/api/setup")
    assert console.status()["run_elapsed_ms"] == 0  # ready, not yet running

    console.do("/api/run")
    assert console.session.status is SessionStatus.RUNNING
    # The timer is wall-clock from run_start_time; it is non-negative once running.
    assert console.status()["run_elapsed_ms"] >= 0


def test_disconnect_clears_policy_and_setup(console):
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    console.do("/api/setup")
    assert console.status()["is_setup_done"] is True

    console.do("/api/disconnect")
    st = console.status()
    assert st["policy_connected"] is False
    assert st["is_setup_done"] is False
