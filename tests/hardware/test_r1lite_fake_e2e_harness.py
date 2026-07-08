from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "tools" / "run_r1lite_fake_e2e.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_r1lite_fake_e2e", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_e2e_configs_pins_ports_and_output_dirs(tmp_path):
    harness = _load_harness()

    paths = harness.write_e2e_configs(tmp_path, policy_port=19000)

    rollout = paths.deploy_config.read_text()
    rollout_hil = paths.rollout_hil_config.read_text()
    hil = paths.hil_config.read_text()
    collection = paths.collection_config.read_text()
    collection_no_hil = paths.collection_no_hil_config.read_text()
    eval_text = paths.eval_config.read_text()
    assert "policy = dict(" in rollout
    assert "port=19000" in rollout
    assert str(paths.rollout_no_hil_dir) in rollout
    assert "intervention=dict(enabled=False" in rollout
    assert str(paths.rollout_hil_dir) in rollout_hil
    assert "intervention=dict(enabled=True" in rollout_hil
    assert str(paths.hil_dir) in hil
    assert "intervention=dict(enabled=True" in hil
    assert "policy = dict(" in collection
    assert "port=19000" in collection
    assert str(paths.collection_hil_dir) in collection
    assert "intervention=dict(enabled=True" in collection
    assert f"_base_ = ['{paths.collection_config}']" in collection_no_hil
    assert str(paths.collection_no_hil_dir) in collection_no_hil
    assert "intervention=dict(enabled=False" in collection_no_hil
    assert "inference_strategy='sync'" in eval_text
    assert "port=19000" in eval_text
    assert str(paths.eval_dir) in eval_text


def test_assert_fixed_clock_accepts_exact_grid_and_rejects_jitter():
    harness = _load_harness()
    exact = pa.table({"timestamp": [0.0, 1.0 / 15.0, 2.0 / 15.0]})
    jitter = pa.table({"timestamp": [0.0, 0.08, 0.12]})

    harness.assert_fixed_clock(exact, fps=15)
    try:
        harness.assert_fixed_clock(jitter, fps=15)
    except AssertionError as exc:
        assert "timestamp is not fixed-clock" in str(exc)
    else:
        raise AssertionError("jittered timestamp table was accepted")


def test_assert_vector_varies_requires_motion_in_selected_columns():
    harness = _load_harness()
    fixed = np.zeros((4, 14), dtype=np.float32)
    moving = fixed.copy()
    moving[:, 2] = [0.0, 0.1, 0.2, 0.3]

    harness.assert_vector_varies(moving, columns=(2, 10), label="action.qpos")
    try:
        harness.assert_vector_varies(fixed, columns=(2, 10), label="action.qpos")
    except AssertionError as exc:
        assert "action.qpos did not vary" in str(exc)
    else:
        raise AssertionError("fixed action vector was accepted")


def test_poll_status_for_keeps_eva_watchdog_alive(monkeypatch):
    harness = _load_harness()
    now = [0.0]
    calls = []

    def fake_monotonic():
        return now[0]

    def fake_sleep(duration):
        now[0] += duration

    monkeypatch.setattr(harness.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(harness.time, "sleep", fake_sleep)
    monkeypatch.setattr(harness, "get_json", lambda base_url, path: calls.append((base_url, path)))

    harness.poll_status_for("http://eva", duration_s=0.7, interval_s=0.2)

    assert calls == [
        ("http://eva", "/api/status"),
        ("http://eva", "/api/status"),
        ("http://eva", "/api/status"),
        ("http://eva", "/api/status"),
    ]


def test_drive_motion_target_publishes_direct_command_deltas(monkeypatch):
    harness = _load_harness()
    calls = []

    monkeypatch.setattr(
        harness,
        "post_json",
        lambda base_url, path, payload: calls.append((base_url, path, payload)),
    )
    monkeypatch.setattr(harness.time, "sleep", lambda _duration: None)

    harness.drive_motion_target("http://fake", repeats=2, delay_s=0.1)

    assert calls == [
        (
            "http://fake",
            "/api/command_delta",
            {"group": "left_arm", "index": 2, "delta": 0.03},
        ),
        (
            "http://fake",
            "/api/command_delta",
            {"group": "right_arm", "index": 3, "delta": -0.02},
        ),
        (
            "http://fake",
            "/api/command_delta",
            {"group": "left_arm", "index": 2, "delta": 0.03},
        ),
        (
            "http://fake",
            "/api/command_delta",
            {"group": "right_arm", "index": 3, "delta": -0.02},
        ),
    ]
