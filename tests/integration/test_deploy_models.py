"""Named deploy policy choices: config loading and idle model switching."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from core.app import run as app
from core.app.handlers.io import connect_policy
from core.app.state import SessionStatus
from core.config import ConfigDict, deployment_models, load_config
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration
DEFAULTS = Path(__file__).resolve().parents[2] / "configs/00_base/defaults.py"


def _load(tmp_path, policy):
    path = tmp_path / "models.py"
    path.write_text(f"_base_ = [{str(DEFAULTS)!r}]\npolicy = {policy!r}\n")
    return load_config(path)


def _config():
    cfg = console_config()
    cfg.policy_models = [
        ConfigDict(name="Alpha", type="mock", host="alpha", port=5555,
                   backend_options=dict(chunk_size=5)),
        ConfigDict(name="Beta", type="mock", host="beta", port=5556,
                   backend_options=dict(chunk_size=9)),
    ]
    cfg.policy = ConfigDict({k: copy.deepcopy(v) for k, v in cfg.policy_models[0].items()
                             if k != "name"})
    return cfg


def test_policy_list_loads_first_model_and_keeps_choices_independent(tmp_path):
    cfg = _load(tmp_path, [
        dict(name="Alpha", type="mock", port=5555, backend_options=dict(chunk_size=5)),
        dict(name="Beta", type="mock", host="beta", port=5556),
    ])
    assert [model.name for model in deployment_models(cfg)] == ["Alpha", "Beta"]
    assert cfg.policy.type == "mock"
    assert cfg.policy.host == "127.0.0.1"
    assert cfg.policy.port == 5555
    assert "name" not in cfg.policy
    cfg.policy.backend_options.chunk_size = 99
    assert cfg.policy_models[0].backend_options.chunk_size == 5


def test_legacy_policy_dict_remains_usable(tmp_path):
    cfg = _load(tmp_path, dict(type="mock", port=5555))
    assert cfg.policy.port == 5555
    assert deployment_models(cfg)[0].name == "mock"


@pytest.mark.parametrize("policy,match", [
    ([], "non-empty list"),
    (["server"], "must be a dict"),
    ([dict(type="mock")], "name"),
    ([dict(name=" ")], "name"),
    ([dict(name="A"), dict(name="A")], "duplicate"),
    ([dict(name="A", port=0)], "port"),
    ([dict(name="A", port=True)], "port"),
    ([dict(name="A", backend_options=[])], "backend_options"),
])
def test_policy_list_rejects_invalid_models(tmp_path, policy, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, policy)


def test_http_switch_clears_old_client_and_preview_then_uses_selected_backend():
    cfg = _config()
    with serve_console(cfg) as h:
        assert h.get("/api/config").json["models"] == [
            dict(index=0, name="Alpha"), dict(index=1, name="Beta")]
        assert h.get("/api/status").json["model_index"] == 0
        h.session.selected_task = "pick up the cup"
        h.session.is_setup_done = True
        h.session.status = SessionStatus.READY
        h.session.pending_real_chunk = np.ones((2, h.runtime.robot.total_action_dim))
        h.runtime.policy = object()
        h.runtime.policy_metadata = {"old": True}
        h.runtime.last_policy_error = "old endpoint failed"
        h.runtime.web_phase = "ready"
        assert h.do("/api/select_model", dict(index=1)).status == 200
        status = h.get("/api/status").json
        assert status["model_index"] == 1
        assert status["is_setup_done"] is False
        assert status["pending_chunk"] is None
        assert status["policy_connected"] is False
        assert h.session.selected_task == "pick up the cup"
        assert h.runtime.active_config.policy.host == "beta"
        assert h.runtime.active_config.policy.port == 5556
        assert cfg.policy.host == "alpha"
        assert connect_policy(h.runtime.active_config, h.runtime, h.session)
        actions = h.runtime.policy.infer({})["actions"]
        assert actions.shape[0] == 9
        h.runtime.active_config.policy.backend_options.chunk_size = 99
        assert cfg.policy_models[1].backend_options.chunk_size == 9
        assert h.do("/api/select_model", dict(index=0)).status == 200
        assert h.runtime.active_config.policy.host == "alpha"


@pytest.mark.parametrize("index", [-1, 2, "1", True, None])
def test_http_rejects_invalid_model_index(index):
    with serve_console(_config()) as h:
        assert h.post("/api/select_model", dict(index=index)).status == 400
        assert h.runtime.command_queue.empty()


@pytest.mark.parametrize("phase", ["starting", "running", "stopping", "resetting"])
def test_model_switch_rejected_during_active_operations(phase):
    with serve_console(_config()) as h:
        h.runtime.web_phase = phase
        assert h.post("/api/select_model", dict(index=1)).status == 409
        app.handle_command("web:select_model:1", h.config, h.runtime, h.session)
        assert h.runtime.active_model_index == 0
        assert h.runtime.active_config is None


def test_queued_switch_is_rechecked_before_execution_and_same_model_is_noop():
    with serve_console(_config()) as h:
        h.session.is_setup_done = True
        assert h.do("/api/select_model", dict(index=0)).status == 200
        assert h.session.is_setup_done is True
        assert h.post("/api/select_model", dict(index=1)).status == 200
        h.session.status = SessionStatus.RUNNING
        h.pump()
        assert h.runtime.active_model_index == 0
