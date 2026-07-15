from __future__ import annotations

from pathlib import Path

import numpy as np

from core.app.handlers import enable_default_recording, rebuild_eval_episode_logger
from core.app.state import RuntimeState
from core.config import ConfigDict
from core.utils.paths import resolve_output_dir
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_resolve_output_dir_is_repo_rooted(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = ConfigDict(eval=ConfigDict())

    out = resolve_output_dir(config, "configs/03_evaluation/ur5e_eval.py")

    assert out == _REPO_ROOT / "work_dirs" / "03_evaluation" / "ur5e_eval"


def test_non_eval_recording_keeps_custom_log_dir(tmp_path):
    config = ConfigDict(
        eval=None,
        log=ConfigDict(enabled=True, log_dir="work_dirs/logs/debug", fps=10),
    )
    output_dir = tmp_path / "debug_run"

    out = enable_default_recording(config, str(output_dir))

    assert out.log.log_dir == "work_dirs/logs/debug"


def test_rebuild_eval_episode_logger_applies_storage_image_size(tmp_path):
    robot = Robot(
        name="fake",
        actuator_groups=(ActuatorGroup("arm", 6, tuple(f"j{i}" for i in range(6))),),
        initial_qpos=np.zeros(6, dtype=np.float32),
        observation_schema=ObservationSchema(
            cameras=(CameraSpec("front", "cam_high"),),
            state_composition=("arm",),
        ),
    )
    runtime = RuntimeState(robot=robot, transport=object())
    runtime.eval_output_dir = str(tmp_path)
    config = ConfigDict(
        eval=ConfigDict(
            storage=ConfigDict(
                fps=15,
                save_queue_max=15,
                image_height=120,
                image_width=160,
            ),
            checkpoints=[ConfigDict(name="fake_policy")],
        ),
        transport=ConfigDict(
            convert_bgr_to_rgb=False,
            dataset_keys=ConfigDict(
                state_key="observations.state.qpos",
                eef_key="observations.state.eef",
                action_key="action",
                video_keys={},
            ),
        ),
    )

    rebuild_eval_episode_logger(config, runtime)

    assert runtime.episode_logger is not None
    assert runtime.episode_logger._save_image_height == 120
    assert runtime.episode_logger._save_image_width == 160
