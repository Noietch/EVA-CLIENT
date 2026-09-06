"""Tests for config loading and helpers (core.config)."""

from __future__ import annotations

from pathlib import Path

import pytest

import robots  # noqa: F401  (registers robots)
import transport  # noqa: F401  (registers transport backends)
from core.config import load_config
from core.registry import ROBOT_REGISTRY

pytestmark = pytest.mark.integration

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_DEFAULTS = _CONFIGS_DIR / "00_base" / "defaults.py"


def _write_config(path: Path, body: str) -> Path:
    """Write a .py config inheriting the real defaults via an absolute ``_base_``."""
    path.write_text(f"_base_ = [{str(_DEFAULTS)!r}]\n{body}", encoding="utf-8")
    return path


def test_load_config_uses_configured_collection_task_set(tmp_path):
    task_set = tmp_path / "task_set"
    task_set.mkdir()
    (task_set / "tasks.csv").write_text(
        "task_id,prompt_en,total_epsiodes_count\nTASK-001,place the cup,5\n",
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path / "collection.py",
        f"collection = dict(task_set_dir={str(task_set)!r}, task_set_name='demo_set')\n",
    )

    cfg = load_config(config_path)

    assert dict(cfg.collection.tasks) == {"demo_set": [("place the cup", 5)]}


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            "collection = dict(teleop=dict(control_source='invalid'))\n",
            "control_source",
        ),
        (
            "collection = dict(teleop=dict(client=dict(type='vr_webxr')))\n",
            "transport-driven",
        ),
        (
            "collection = dict(teleop=dict("
            "control_source='client', client=dict(type='unknown')))\n",
            "unsupported teleop client type",
        ),
        (
            "rl_cfg = dict(data=dict(format='lerobot'), intervention=dict(source='invalid'))\n",
            "rl.intervention.source",
        ),
        (
            "rl_cfg = dict(\n"
            "    data=dict(format='lerobot'),\n"
            "    intervention=dict(source='teleop_client'),\n"
            ")\n",
            "collection.teleop.control_source='client'",
        ),
    ],
)
def test_teleop_config_validation_rejects_unsafe_combinations(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write_config(tmp_path / "invalid_teleop.py", body))


@pytest.mark.parametrize(
    "preset",
    [
        "agibot_g2/openpi_eef.py",
        "arx_r5/openpi_qpos.py",
        "arx_x5/openpi_eef.py",
        "dual_agilex_piper/xpolicylab.py",
        "dual_franka/openpi_qpos.py",
        "dual_yam/openpi_qpos.py",
        "r1lite/eva_eef.py",
    ],
)
def test_deploy_config_builds_robot_contract(preset):
    cfg = load_config(_CONFIGS_DIR / "01_deploy" / preset)
    robot = ROBOT_REGISTRY.build(cfg.robot.type)
    assert robot.initial_qpos.shape == (robot.total_action_dim,)
    assert robot.observation_schema.cameras
    assert cfg.inference_cfg.publish_rate > 0


@pytest.mark.parametrize(
    "preset",
    ["arx_r5.py", "arx_x5_tasks_set.py", "dual_yam.py", "r1lite.py", "ur5e.py"],
)
def test_collection_config_resolves_recording_schema(preset):
    cfg = load_config(_CONFIGS_DIR / "02_collection" / preset)
    assert cfg.collection.schema.robot_type
    assert set(cfg.collection.schema.columns) == {"qpos", "eef", "action_qpos", "action_eef"}
    assert isinstance(cfg.collection.tasks, dict)


@pytest.mark.parametrize(
    "preset",
    ["dual_agilex_piper_eva_sim_eval.py", "ur5e_eval.py"],
)
def test_eval_config_resolves_checkpoint_contract(preset):
    cfg = load_config(_CONFIGS_DIR / "03_evaluation" / preset)
    assert cfg.eval is not None
