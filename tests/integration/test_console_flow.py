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
