from __future__ import annotations

from pathlib import Path

from core.app.handlers import enable_default_recording
from core.config import ConfigDict
from core.utils.paths import resolve_output_dir

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
