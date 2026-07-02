"""Tests for CLI state data helpers (core.app.state)."""

from __future__ import annotations

from core.app.state import (
    default_inference_strategy_key,
    format_task_label,
    resolve_inference_strategy_label,
)
from core.config import ConfigDict

# --- format_task_label ---


def test_format_task_label_none():
    assert format_task_label(None) == "unset"


def test_format_task_label_empty():
    assert format_task_label("") == "<empty>"


def test_format_task_label_normal():
    assert format_task_label("pick up cup") == "pick up cup"


# --- inference strategy key helpers ---


def _config_with_strategies(strategies):
    return ConfigDict(inference_strategies=strategies)


def test_default_inference_strategy_key_first_entry():
    cfg = _config_with_strategies(
        {"sync": {"type": "sync", "args": {}}, "async": {"type": "async_prefetch", "args": {}}}
    )
    assert default_inference_strategy_key(cfg) == "sync"


def test_resolve_label_none_falls_back_to_default():
    cfg = _config_with_strategies({"async": {"type": "async_prefetch", "args": {}}})
    assert resolve_inference_strategy_label(cfg, None) == "async"


def test_resolve_label_explicit_key_passthrough():
    cfg = _config_with_strategies(
        {"sync": {"type": "sync", "args": {}}, "rtc": {"type": "rtc", "args": {}}}
    )
    assert resolve_inference_strategy_label(cfg, "rtc") == "rtc"
