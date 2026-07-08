"""Console inference strategy selection tests."""

from __future__ import annotations

from typing import cast

from core.app import run as app
from core.app.state import RuntimeState, SessionMode, SessionState
from core.config import ConfigDict
from robots.base import Robot
from transport.base import TransportBridge


def _config_with_strategies(strategies):
    return ConfigDict(inference_strategies=strategies)


def _runtime_and_session() -> tuple[RuntimeState, SessionState]:
    runtime = RuntimeState(
        robot=cast(Robot, object()),
        transport=cast(TransportBridge, object()),
    )
    session = SessionState(selected_task="push black block", mode=SessionMode.REAL)
    return runtime, session


def test_console_autoselects_the_only_inference_strategy():
    config = _config_with_strategies(
        {"sync": {"type": "BaseInferStrategy", "args": {"execute_horizon": 5}}}
    )
    runtime, session = _runtime_and_session()

    assert app._select_only_inference_strategy_if_needed(config, runtime, session)
    assert runtime.selected_inference_strategy_key == "sync"
    assert runtime.infer_strategy is not None


def test_console_setup_uses_only_inference_strategy(monkeypatch):
    config = _config_with_strategies(
        {"sync": {"type": "BaseInferStrategy", "args": {"execute_horizon": 5}}}
    )
    runtime, session = _runtime_and_session()
    calls = []

    def fake_run_setup(config, runtime, session, fail_fast=False):
        calls.append(fail_fast)
        return True

    monkeypatch.setattr(app, "run_setup", fake_run_setup)

    app._handle_web_command("web:setup", config, runtime, session)

    assert runtime.selected_inference_strategy_key == "sync"
    assert calls == [True]


def test_console_setup_keeps_multi_strategy_choice_explicit(monkeypatch):
    config = _config_with_strategies(
        {
            "sync": {"type": "BaseInferStrategy", "args": {}},
            "rtc": {"type": "RtcInferStrategy", "args": {}},
        }
    )
    runtime, session = _runtime_and_session()

    def fail_run_setup(*args, **kwargs):
        raise AssertionError("run_setup should not run without an explicit strategy")

    monkeypatch.setattr(app, "run_setup", fail_run_setup)

    app._handle_web_command("web:setup", config, runtime, session)

    assert runtime.selected_inference_strategy_key is None
    assert session.last_error == "Select inference strategy first: 1.sync  2.rtc"


def test_eval_ssh_start_requires_explicit_ssh_fields():
    ckpt = cast(ConfigDict, object())
    eval_cfg = ConfigDict(checkpoints=(ckpt,))

    assert not app._should_start_eval_ssh(eval_cfg)

    eval_cfg = ConfigDict(
        checkpoints=(ckpt,),
        enable_ssh_forward=True,
        ssh=ConfigDict(host="127.0.0.1", user="remote-user", port=8000),
    )

    assert app._should_start_eval_ssh(eval_cfg)


def test_eval_bootstrap_uses_default_ckpt_slot(monkeypatch):
    ckpt_config = ConfigDict(policy=ConfigDict(port=9080))
    ckpt = ConfigDict(
        name="local_policy",
        host="127.0.0.1",
        port=9080,
        config=ckpt_config,
    )
    eval_cfg = ConfigDict(
        checkpoints=(ckpt,),
        inference_strategy="sync",
        cli_mode="real",
    )
    config = ConfigDict(eval=eval_cfg, eval_cfg=eval_cfg)
    runtime, session = _runtime_and_session()
    runtime.ckpt_order = [0]
    connected_ports = []

    def fake_connect_policy(config, runtime, session, **kwargs):
        connected_ports.append(config.policy.port)
        return True

    monkeypatch.setattr(app, "connect_policy", fake_connect_policy)
    monkeypatch.setattr(app, "select_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "select_inference_strategy", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "rebuild_eval_episode_logger", lambda *args, **kwargs: None)

    app._handle_web_command("web:bootstrap", config, runtime, session)

    assert runtime.active_ckpt_slot == 0
    assert runtime.active_config is not None
    assert runtime.active_config.policy.port == 9080
    assert connected_ports == [9080]
