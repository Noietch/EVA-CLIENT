"""Tests for config loading and helpers (core.config)."""

from __future__ import annotations

from pathlib import Path

import pytest

import robots  # noqa: F401  (registers robots)
import transport  # noqa: F401  (registers transport backends)
from core.config import ConfigDict, load_config, resolve_video_key
from core.registry import TRANSPORT_REGISTRY
from transport.base import resolve_topics

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_DEFAULTS = _CONFIGS_DIR / "00_base" / "defaults.py"
def _write_config(path: Path, body: str) -> Path:
    """Write a .py config inheriting the real defaults via an absolute ``_base_``."""
    path.write_text(f"_base_ = [{str(_DEFAULTS)!r}]\n{body}", encoding="utf-8")
    return path


def test_resolve_video_key_default_convention():
    assert resolve_video_key(ConfigDict(video_keys={}), "cam_high") == "observation.images.cam_high"


def test_resolve_video_key_explicit_override():
    keys = ConfigDict(video_keys={"cam_high": "videos.top"})
    assert resolve_video_key(keys, "cam_high") == "videos.top"


def test_resolve_topics_builds_camera_and_group_maps():
    cameras, groups = resolve_topics(
        {
            "camera_topics": {"front": "/front", "left_wrist": "/left"},
            "group_topics": {
                "left_arm": {"state_topic": "/state", "command_topic": "/cmd"},
            },
        }
    )
    assert cameras == {"front": "/front", "left_wrist": "/left"}
    assert groups["left_arm"].state_topic == "/state"
    assert groups["left_arm"].command_topic == "/cmd"
    assert groups["left_arm"].eef_state_topic is None


def test_resolve_topics_empty_raises():
    with pytest.raises(ValueError, match="transport.topics is required"):
        resolve_topics({})


def test_transport_registry_only_exposes_supported_backends():
    available = [name for name in TRANSPORT_REGISTRY.available() if name != "debug"]
    assert available == ["dataset", "ros1", "ros2", "zmq"]


def test_load_deploy_config_resolves_spaces_and_defaults():
    cfg = load_config(_CONFIGS_DIR / "01_deploy" / "dual_agilex_piper" / "openpi_qpos.py")
    assert cfg.robot.type == "agilex_piper"
    assert cfg.policy.type == "openpi_rtc"
    assert cfg.policy.backend_options["latency_k"] == 4
    assert cfg.inference_cfg.publish_rate > 0
    assert not cfg.inference_cfg.obs_space.is_eef()
    assert cfg.eval_cfg is None
    assert cfg.eval is None


def test_load_eef_config_builds_eef_space():
    cfg = load_config(_CONFIGS_DIR / "01_deploy" / "dual_agilex_piper" / "openpi_eef.py")
    assert cfg.inference_cfg.obs_space.is_eef()
    assert cfg.inference_cfg.action_space.is_eef()


def test_load_collection_config_exposes_schema():
    cfg = load_config(_CONFIGS_DIR / "02_collection" / "arx_r5.py")
    assert cfg.collection.schema.robot_type == "arx_r5"
    assert set(cfg.collection.schema.columns) == {"qpos", "eef", "action_qpos", "action_eef"}
    assert cfg.collection.schema.cameras["cam_high"] == "observation.images.cam_high"
    assert cfg.collection.storage.log_dir


@pytest.mark.parametrize(
    "robot_name",
    ["r1lite", "ur5e", "arx_r5", "dual_agilex_piper"],
)
def test_deploy_configs_enable_relative_hil_rollout(robot_name: str):
    cfg = load_config(_CONFIGS_DIR / "01_deploy" / robot_name / "openpi_qpos.py")
    assert cfg.rollout.storage.enabled is True
    assert "enabled" not in cfg.rollout.intervention
    assert cfg.rollout.intervention.control_mode == "relative"


@pytest.mark.parametrize("robot_name", ["dual_franka", "agibot_g2"])
def test_deploy_configs_without_leader_keep_hil_disabled(robot_name: str):
    cfg = load_config(_CONFIGS_DIR / "01_deploy" / robot_name / "openpi_qpos.py")
    assert cfg.rollout.storage.enabled is False
    assert cfg.rollout.intervention.control_mode == "absolute"


def test_rollout_config_parses(tmp_path):
    cfg_path = _write_config(
        tmp_path / "rollout.py",
        "rollout = dict(\n"
        "    storage=dict(enabled=True, log_dir='work_dirs/rollout/test', fps=15),\n"
        "    intervention=dict(control_mode='relative'),\n"
        ")\n",
    )
    cfg = load_config(cfg_path)
    assert cfg.rollout.storage.enabled is True
    assert cfg.rollout.storage.log_dir == "work_dirs/rollout/test"
    assert cfg.rollout.storage.fps == 15
    assert cfg.rollout.intervention.control_mode == "relative"


def test_r1lite_collection_configures_common_recording():
    cfg = load_config(_CONFIGS_DIR / "02_collection" / "r1lite.py")
    assert "gate" not in cfg.collection
    assert cfg.collection.storage.image_height == 360
    assert cfg.collection.storage.image_width == 640
    assert cfg.rollout.storage.image_height == 360
    assert cfg.rollout.storage.image_width == 640
    assert cfg.rollout.intervention.control_mode == "relative"


def test_operator_control_defaults_disabled():
    cfg = load_config(_CONFIGS_DIR / "00_base" / "defaults.py")

    assert cfg.operator_control.enabled is False
    assert cfg.operator_control.button_topic == "/eva/operator_button"


def test_r1lite_enables_operator_control():
    cfg = load_config(_CONFIGS_DIR / "01_deploy" / "r1lite" / "openpi_qpos.py")

    assert "hardware" not in cfg
    assert cfg.operator_control.enabled is True
    assert cfg.operator_control.button_topic == "/eva/operator_button"
    assert cfg.rollout.storage.image_height == 360
    assert cfg.rollout.storage.image_width == 640


def test_r1lite_collection_enables_operator_control():
    cfg = load_config(_CONFIGS_DIR / "02_collection" / "r1lite.py")

    assert cfg.operator_control.enabled is True
    assert cfg.operator_control.button_topic == "/eva/operator_button"


def test_load_eval_config_resolves_checkpoints():
    cfg = load_config(_CONFIGS_DIR / "03_evaluation" / "arx_r5_eval.py")
    assert cfg.eval_cfg is cfg.eval
    assert cfg.eval is not None
    assert len(cfg.eval.checkpoints) == 2
    for checkpoint in cfg.eval.checkpoints:
        resolved = checkpoint["config"]
        assert isinstance(resolved, ConfigDict)
        assert resolved.policy.port == checkpoint["port"]


def test_collection_log_dir_derives_from_filename(tmp_path):
    cfg_path = _write_config(
        tmp_path / "myrun.py",
        "collection = dict(schema=dict(\n"
        "    robot_type='ur5e', arms=dict(arm='arm'),\n"
        "    cameras=dict(cam_high='observation.images.cam_high'),\n"
        "    columns=dict(qpos='o.q', eef='o.e', action_qpos='a.q', action_eef='a.e'),\n"
        "))\n",
    )
    cfg = load_config(cfg_path)
    assert cfg.collection.storage.log_dir.endswith("myrun")


def test_collection_schema_requires_core_columns(tmp_path):
    cfg_path = _write_config(
        tmp_path / "bad.py",
        "collection = dict(schema=dict(\n"
        "    robot_type='ur5e', arms=dict(arm='arm'),\n"
        "    cameras=dict(cam_high='observation.images.cam_high'),\n"
        "    columns=dict(qpos='o.q', action_qpos='a.q'),\n"
        "))\n",
    )
    with pytest.raises(ValueError, match="collection.schema.columns"):
        load_config(cfg_path)


def test_collection_schema_requires_cameras(tmp_path):
    cfg_path = _write_config(
        tmp_path / "no_cam.py",
        "collection = dict(schema=dict(\n"
        "    robot_type='ur5e', arms=dict(arm='arm'), cameras=dict(),\n"
        "    columns=dict(qpos='o.q', eef='o.e', action_qpos='a.q', action_eef='a.e'),\n"
        "))\n",
    )
    with pytest.raises(ValueError, match="collection.schema.cameras"):
        load_config(cfg_path)


@pytest.mark.parametrize(
    "preset",
    sorted(
        str(p)
        for p in _CONFIGS_DIR.glob("01_deploy/**/*.py")
        if not p.name.startswith("_")
    ),
)
def test_all_deploy_presets_load_without_error(preset):
    cfg = load_config(preset)
    assert cfg.robot.type
    assert cfg.inference_cfg.publish_rate > 0


@pytest.mark.parametrize(
    "preset",
    sorted(str(p) for p in _CONFIGS_DIR.glob("02_collection/*.py")),
)
def test_all_collection_presets_load_without_error(preset):
    cfg = load_config(preset)
    assert cfg.collection.schema.robot_type
    assert set(cfg.collection.schema.columns) == {"qpos", "eef", "action_qpos", "action_eef"}


@pytest.mark.parametrize(
    "preset",
    sorted(str(p) for p in _CONFIGS_DIR.glob("03_evaluation/*.py")),
)
def test_all_eval_presets_load_without_error(preset):
    cfg = load_config(preset)
    assert cfg.eval is not None
