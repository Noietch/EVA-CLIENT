"""Tests for config loading and helpers (core.config)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import robots  # noqa: F401  (registers robots)
import transport  # noqa: F401  (registers transport backends)
from core.app.handlers.io import ensure_ik_solver, normalize_policy_action_chunk
from core.app.handlers.recording import (
    _fill_record_eef,
    build_policy_observation,
    record_rollout_intervention_step,
)
from core.app.state import RuntimeState, SessionState
from core.config import load_config
from core.registry import ROBOT_REGISTRY
from core.types import Observation, RolloutInterventionSegment
from policy_client.base import PolicyBuildContext
from policy_client.xpolicylab import XPolicyLabPolicyClient

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = ROOT / "configs"
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
            f'task_id,prompt_en,total_epsiodes_count\n{name.upper()},"{prompt}",5\n',
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
        "dual_agilex_piper/pi05_qpos.py",
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


def test_pi05_piper_config_owns_joint_qpos_contract():
    cfg = load_config(_CONFIGS_DIR / "01_deploy/dual_agilex_piper/pi05_qpos.py")
    robot = ROBOT_REGISTRY.build(cfg.robot.type)

    assert cfg.robot.type == "agilex_piper"
    assert cfg.policy.backend_options.arm_dims == [6, 6]
    assert cfg.policy.backend_options.gripper_dims == [1, 1]
    assert cfg.policy.backend_options.action_type == "joint"
    assert cfg.inference_cfg.obs_space.type == "JointState"
    assert cfg.inference_cfg.action_space.type == "JointState"
    assert robot.total_action_dim == 14


@pytest.mark.parametrize(
    ("preset", "robot_type"),
    [
        ("arx_r5", "arx_r5"),
        ("arx_x5", "arx_x5"),
        ("dual_agilex_piper", "agilex_piper"),
        ("dual_yam", "dual_yam"),
        ("r1lite", "r1_lite"),
        ("ur5e", "ur5e"),
    ],
)
def test_real_robot_evaluation_presets_keep_checkpoint_and_task_contract(preset, robot_type):
    cfg = load_config(_CONFIGS_DIR / "03_evaluation" / f"{preset}_eval.py")

    assert cfg.robot.type == robot_type
    assert cfg.console.initial_tab == "eval"
    assert cfg.eval.checkpoints
    assert cfg.eval.tasks


def test_dual_yam_eva_sim_config_selects_multiple_task_sets():
    cfg = load_config(_CONFIGS_DIR / "02_collection" / "dual_yam_eva_sim.py")

    assert len(cfg.collection.task_set_dir) == 12
    assert "datasets/data_collection/task_sets/larybenchv2.v1.2/train_v1.2" in (
        cfg.collection.task_set_dir
    )
    assert (
        "datasets/data_collection/task_sets/larybenchv2.v1.2/larybench.v1.2.train.supplement"
    ) in (cfg.collection.task_set_dir)
    assert (
        "datasets/data_collection/task_sets/larybench2_20260908_eva_sim_dual_yam_pickplace_last200"
    ) not in cfg.collection.task_set_dir
    assert cfg.collection.storage.log_dir == "work_dirs/collection/eva_sim_dual_yam/"
    assert cfg.transport.sub_endpoint == "tcp://127.0.0.1:5555"
    assert cfg.transport.pub_endpoint == "tcp://127.0.0.1:5556"
    assert cfg.transport.disabled_cameras == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]


def test_dual_yam_composite_config_reuses_sim_transport_with_rendering_disabled():
    cfg = load_config(_CONFIGS_DIR / "02_collection/dual_yam_eva_sim_composite.py")
    assert len(cfg.collection.task_set_dir) == 10
    assert all(
        Path(path).name.startswith("larybench.v1.2.train.composite_")
        for path in cfg.collection.task_set_dir
    )
    assert cfg.collection.render_service.url == ""
    assert cfg.transport.type == "zmq"


def test_dual_yam_sim_evaluation_uses_eef_and_on_demand_transport():
    cfg = load_config(_CONFIGS_DIR / "03_evaluation/dual_yam_eva_sim_eval.py")
    assert cfg.robot.type == "dual_yam"
    assert cfg.inference_cfg.obs_space.is_eef()
    assert cfg.inference_cfg.action_space.is_eef()
    assert cfg.transport.type == "zmq"
    assert cfg.transport.image_mode == "on_demand"
    assert cfg.transport.convert_bgr_to_rgb is False
    assert cfg.control_channel.enabled is True


@pytest.mark.parametrize(
    ("preset", "links"),
    [
        ("dual_yam", ["tcp", "tcp"]),
        ("arx_x5", ["tcp", "tcp"]),
        ("dual_agilex_piper", ["tcp", "tcp"]),
    ],
)
def test_tcp_preset_builds_actual_target_links(preset, links):
    config = load_config(ROOT / "configs/01_deploy" / preset / "openpi_tcp.py")
    robot = ROBOT_REGISTRY.build(config.robot.type)
    runtime = RuntimeState(
        robot=robot,
        transport=SimpleNamespace(get_latest_qpos=lambda: robot.initial_qpos.copy()),
    )
    solver = ensure_ik_solver(config, runtime)

    for arm, link in zip((solver._left_solver, solver._right_solver), links, strict=True):
        assert arm._robot.links.names[int(arm._eef_idx)] == link
    assert config.inference_cfg.obs_space.rotation_order == "wxyz"
    assert config.inference_cfg.action_space.rotation_order == "wxyz"
    assert ensure_ik_solver(config, runtime) is solver


def test_yam_tcp_inference_replaces_wire_gripper_pose_and_solves_tcp_action():
    config = load_config(ROOT / "configs/01_deploy/dual_yam/openpi_tcp.py")
    robot = ROBOT_REGISTRY.build(config.robot.type)
    qpos = robot.initial_qpos.copy()
    runtime = RuntimeState(
        robot=robot,
        transport=SimpleNamespace(get_latest_qpos=lambda: qpos.copy()),
    )
    solver = ensure_ik_solver(config, runtime)
    tcp_pose = solver.fk_chunk(qpos)[0]
    gripper_solver = robot.build_kinematics(initial_qpos_groups=robot.initial_qpos_by_group())
    gripper_pose = gripper_solver.fk_chunk(qpos)[0]
    frame = Observation(images={}, state_qpos=qpos, state_eef=gripper_pose)
    observation = build_policy_observation(frame, "pick the cup", config, runtime)

    np.testing.assert_allclose(observation["state"], tcp_pose, atol=1e-6)
    assert np.linalg.norm(observation["state"][:3] - gripper_pose[:3]) > 0.14
    solved = normalize_policy_action_chunk(config, runtime, tcp_pose[np.newaxis])
    np.testing.assert_allclose(solved, qpos[np.newaxis], atol=1e-4)
    np.testing.assert_array_equal(solved[:, [6, 13]], qpos[np.newaxis, [6, 13]])

    frame.action_qpos = qpos.copy()
    frame.action_eef = gripper_pose.copy()
    _fill_record_eef(config, runtime, frame, qpos)
    np.testing.assert_allclose(frame.state_eef, tcp_pose, atol=1e-6)
    np.testing.assert_allclose(frame.action_eef, tcp_pose, atol=1e-6)

    frame.state_eef = gripper_pose.copy()
    frame.action_eef = gripper_pose.copy()
    runtime.transport.get_hil_frame = lambda: frame
    runtime.rollout_intervention_active = True
    segment = RolloutInterventionSegment(0, 0, qpos.copy())
    runtime.rollout_intervention_active_segment = segment
    assert record_rollout_intervention_step(config, runtime, SessionState())
    np.testing.assert_allclose(segment.frames[0].state_eef, tcp_pose, atol=1e-6)
    np.testing.assert_allclose(segment.frames[0].action_eef, tcp_pose, atol=1e-6)
    np.testing.assert_array_equal(frame.state_eef, gripper_pose)
    np.testing.assert_array_equal(frame.action_eef, gripper_pose)


def test_tcp_target_count_mismatch_is_rejected():
    robot = ROBOT_REGISTRY.build("dual_yam")
    with pytest.raises(ValueError, match="zip"):
        robot.build_kinematics(initial_qpos_groups=robot.initial_qpos_by_group(), eef_links=["tcp"])


class _TcpEchoPolicy:
    """Echo each arm's received TCP pose through the real server protocol."""

    def __init__(self):
        self.observation = None

    def reset(self):
        self.observation = None

    def infer(self, observation):
        self.observation = observation
        return {"actions": [observation["state"], observation["state"]]}


@pytest.mark.skipif(
    not (ROOT.parent / "eva-xpolicylab/XPolicyLab.py").is_file(),
    reason="requires the sibling eva-xpolicylab checkout",
)
def test_yam_tcp_eval_round_trip_through_actual_xpolicylab_server(monkeypatch):
    config = load_config(ROOT / "configs/03_evaluation/dual_yam_eva_sim_eval.py")
    monkeypatch.syspath_prepend(str(ROOT.parent / "eva-xpolicylab"))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    robot = ROBOT_REGISTRY.build(config.robot.type)
    qpos = robot.initial_qpos.copy()
    runtime = RuntimeState(
        robot=robot, transport=SimpleNamespace(get_latest_qpos=lambda: qpos.copy())
    )
    solver = ensure_ik_solver(config, runtime)
    tcp = solver.fk_chunk(qpos)[0]
    for arm in (solver._left_solver, solver._right_solver):
        assert arm._robot.links.names[int(arm._eef_idx)] == "tcp"

    model = _TcpEchoPolicy()
    server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        asyncio.run_coroutine_threadsafe(server.start(), loop).result(timeout=5)
        config.policy.port = int(server.url.rsplit(":", 1)[1])
        context = PolicyBuildContext(action_dim=robot.total_action_dim, retry_until_connected=False)
        with XPolicyLabPolicyClient.from_config(config.policy, context) as client:
            runtime.policy_metadata = client.metadata
            for episode in range(2):
                client.reset()
                assert model.observation is None
                for step in range(3):
                    images = {
                        camera.observation_key: np.full(
                            (32, 40, 3), 60 * index + 3 * episode + step, dtype=np.uint8
                        )
                        for index, camera in enumerate(robot.observation_schema.cameras)
                    }
                    observation = build_policy_observation(
                        Observation(images=images, state_qpos=qpos), "pick the cup", config, runtime
                    )
                    response = client.infer(observation)
                    assert response["action_mode"] == "eef"
                    assert client.metadata["action_dim"] == 16
                    np.testing.assert_allclose(response["actions"], np.tile(tcp, (2, 1)), atol=1e-6)
                    assert model.observation["state"]["left_ee_pose"].shape == (7,)
                    assert model.observation["state"]["right_ee_pose"].shape == (7,)
                    assert list(model.observation["vision"]) == list(images)
                    for key, image in observation["images"].items():
                        np.testing.assert_array_equal(
                            model.observation["vision"][key]["color"], image
                        )
            solved = normalize_policy_action_chunk(config, runtime, response["actions"], response)
            np.testing.assert_allclose(solved, np.tile(qpos, (2, 1)), atol=1e-4)
    finally:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
