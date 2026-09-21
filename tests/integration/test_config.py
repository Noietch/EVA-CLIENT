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


def test_loaded_task_sets_allow_duplicate_prompts_across_datasets(tmp_path):
    prompt = "insert the plate into the rack"
    task_set_dirs = []
    for name in ("insert", "insert_withdraw"):
        task_set = tmp_path / name
        task_set.mkdir()
        (task_set / "tasks.csv").write_text(
            "task_id,prompt_en,total_epsiodes_count\n"
            f"{name.upper()},\"{prompt}\",5\n",
            encoding="utf-8",
        )
        task_set_dirs.append(str(task_set))

    config_path = _write_config(
        tmp_path / "collection.py",
        f"collection = dict(task_set_dir={task_set_dirs!r}, tasks={{}})\n",
    )

    cfg = load_config(config_path)

    assert set(cfg.collection.tasks) == {"insert", "insert_withdraw"}
    assert cfg.collection.tasks["insert"] == [(prompt, 5)]
    assert cfg.collection.tasks["insert_withdraw"] == [(prompt, 5)]


def test_collection_allows_huggingface_with_one_delivery_remote(tmp_path):
    config_path = _write_config(
        tmp_path / "collection.py",
        "collection = dict(\n"
        "    storage=dict(\n"
        "        huggingface=dict(repo_id='team/data'),\n"
        "        sftp=dict(\n"
        "            host='upload.example.com',\n"
        "            port=22,\n"
        "            remote_dir='/datasets/collection',\n"
        "        ),\n"
        "    ),\n"
        ")\n",
    )

    cfg = load_config(config_path)

    assert cfg.collection.storage.huggingface.repo_id == "team/data"
    assert cfg.collection.storage.sftp.host == "upload.example.com"


def test_collection_rejects_multiple_delivery_remotes(tmp_path):
    config_path = _write_config(
        tmp_path / "invalid_storage.py",
        "collection = dict(\n"
        "    storage=dict(\n"
        "        sftp=dict(host='upload.example.com', remote_dir='/datasets/collection'),\n"
        "        loopback=dict(remote_dir='/tmp/collection-upload'),\n"
        "    ),\n"
        ")\n",
    )

    with pytest.raises(ValueError, match="multiple remotes"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            "collection = dict(teleop=dict(client=dict(type='vr_webxr')))\n",
            "transport-driven",
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
        "dual_yam/openpi_qpos.py",
    ],
)
def test_deploy_config_builds_robot_contract(preset):
    cfg = load_config(_CONFIGS_DIR / "01_deploy" / preset)
    robot = ROBOT_REGISTRY.build(cfg.robot.type)
    assert robot.initial_qpos.shape == (robot.total_action_dim,)
    assert robot.observation_schema.cameras
    assert cfg.inference_cfg.publish_rate > 0


def test_agilex_piper_deploy_converts_ros_bgr_frames_to_rgb():
    cfg = load_config(_CONFIGS_DIR / "01_deploy" / "dual_agilex_piper" / "openpi_qpos.py")

    assert cfg.transport.convert_bgr_to_rgb is True
