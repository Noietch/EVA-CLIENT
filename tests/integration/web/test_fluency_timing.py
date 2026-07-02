"""Operation-latency attribution + UI fluency guardrails.

The user's question: *where does the time go, and does the console feel smooth?*
These tests answer it quantitatively by timing the two halves of every click:

- ``time_post``  — what the browser waits for on click. POST only enqueues, so this
  must stay tiny for *every* verb, even the expensive ones. That's what keeps the UI
  responsive: clicking "Setup" returns instantly; the work happens behind a poll.
- ``time_pump``  — the operation's real cost, paid by the main loop. This is where
  setup/run spend their inference latency and reset spends its trajectory.

Bounds are deliberately loose (≈10× the observed cost on a dev box) — they are
regression tripwires for "this got pathologically slow", not micro-benchmarks.
Per-component performance numbers live in tests/perf/ (owned by the other agent).
"""

from __future__ import annotations

import pytest
from _harness import WebHarness

# Observed on a dev box (see commit notes): POSTs ~1.5ms; setup ~320ms, run ~200ms,
# reset ~105ms — all dominated by the mock policy's 150–250ms inference sleep.
POST_LATENCY_CEILING_MS = 50.0
INSTANT_VERB_CEILING_MS = 50.0
INFERENCE_PHASE_CEILING_MS = 3000.0


@pytest.fixture
def ready_console(console: WebHarness) -> WebHarness:
    """A console connected + mode/prompt selected, ready for setup/run timing."""
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    return console


def test_every_post_returns_immediately(ready_console: WebHarness):
    """The click latency must be tiny for *every* verb — even setup/run, whose real
    work is deferred to the queue. This is the core of the async UI contract."""
    h = ready_console
    for path in ("/api/setup", "/api/run", "/api/halt", "/api/reset"):
        latency = h.time_post(path)
        h.pump()  # settle the work so the next verb's precondition holds
        assert latency < POST_LATENCY_CEILING_MS, f"{path} POST blocked for {latency:.0f}ms"


def test_instant_verbs_have_no_processing_cost(console: WebHarness):
    """Connect / mode / prompt selection are pure state flips — their pump cost is
    effectively zero. A regression here means one of them started doing real work."""
    assert console.time_post("/api/connect") < POST_LATENCY_CEILING_MS
    assert console.time_pump() < INSTANT_VERB_CEILING_MS

    console.post("/api/select_mode", {"mode": "sim"})
    assert console.time_pump() < INSTANT_VERB_CEILING_MS

    console.post("/api/select_task", {"task": "pick up the cup"})
    assert console.time_pump() < INSTANT_VERB_CEILING_MS


def test_setup_cost_is_attributed_to_inference(ready_console: WebHarness):
    """Setup's wall-clock should be dominated by policy inference (the mock sleeps
    150–250ms), not by the HTTP layer. We assert the cost lands in pump, not post."""
    h = ready_console
    post_ms = h.time_post("/api/setup")
    pump_ms = h.time_pump()

    assert h.status()["is_setup_done"] is True
    assert post_ms < POST_LATENCY_CEILING_MS
    # The real cost is in the operation, and it's bounded (no runaway / hang).
    assert pump_ms > post_ms
    assert pump_ms < INFERENCE_PHASE_CEILING_MS


def test_full_sequence_latency_breakdown(ready_console: WebHarness):
    """Time the whole connect→reset sequence and assert the shape of the breakdown:
    the inference phases (setup, run) are the heavy ones; everything else is light.
    Surfaces the answer to 'where does the time go' as a structured profile."""
    h = ready_console
    profile: dict[str, float] = {}
    for verb, body in [
        ("/api/setup", None),
        ("/api/run", None),
        ("/api/halt", None),
        ("/api/reset", None),
    ]:
        h.post(verb, body)
        profile[verb] = h.time_pump()

    # Heavy phases (involve inference / trajectory) dwarf the instant ones.
    assert profile["/api/setup"] > profile["/api/halt"]
    assert profile["/api/run"] > profile["/api/halt"]
    # None of them runs away.
    for verb, cost in profile.items():
        assert cost < INFERENCE_PHASE_CEILING_MS, f"{verb} took {cost:.0f}ms"


def test_repeated_run_halt_cycles_stay_responsive(ready_console: WebHarness):
    """An operator runs many trials in a session. POST latency must not degrade as
    state churns — no per-cycle leak that gradually makes clicks feel sluggish."""
    h = ready_console
    h.do("/api/setup")

    latencies = []
    for _ in range(5):
        latencies.append(h.time_post("/api/run"))
        h.pump()
        latencies.append(h.time_post("/api/halt"))
        h.pump()

    assert max(latencies) < POST_LATENCY_CEILING_MS, f"slowest click {max(latencies):.0f}ms"
