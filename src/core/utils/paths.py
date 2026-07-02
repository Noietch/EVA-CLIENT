"""Shared path helpers for the web layer."""

from __future__ import annotations

from pathlib import Path

from core.config import ConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def resolve_output_dir(config: ConfigDict, config_path: str | None) -> Path:
    """Default output dir = <work_dir>/<config path relative to configs/>.

    e.g. configs/eval/arx_r5_eval_v1.yaml -> work_dirs/eval/arx_r5_eval_v1/.
    An explicit eval.output_dir overrides this.
    """
    work_dir = config.get("work_dir") or "work_dirs"
    eval_cfg = config.get("eval")
    output_dir = eval_cfg.get("output_dir") if eval_cfg else None
    if output_dir:
        return _repo_path(Path(output_dir))
    if config_path:
        p = Path(config_path)
        try:
            idx = p.parts.index("configs")
            rel = Path(*p.parts[idx + 1 :]).with_suffix("")
        except ValueError:
            rel = Path(p.stem)
        return _repo_path(Path(work_dir) / rel)
    return _repo_path(Path(work_dir))
