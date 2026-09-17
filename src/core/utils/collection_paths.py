"""One dataset, one directory: where collection data and its QC ledger live.

Recording, the console, and the dataset dashboard all point at the same native
LeRobot v2.1 directory per dataset — ``<data_root>/<collection_dir>``, where the
mounted task set's ``info.yaml`` names ``collection_dir``. Quality verdicts live
in that directory's ``meta/qc.jsonl``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.config import ConfigDict

DEFAULT_DATA_ROOT = "datasets/data_collection"
TASK_SETS_DIR = "task_sets"
ROBOT_DIR = "real_robot"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else _REPO_ROOT / resolved


def _task_set_directories(config: ConfigDict) -> list[Path]:
    configured = (config.get("collection") or {}).get("task_set_dir") or ""
    values = [configured] if isinstance(configured, (str, Path)) else list(configured)
    return [_resolve(str(value)) for value in values if str(value).strip()]


def data_root(config: ConfigDict, plan_root: Path | None = None) -> Path:
    """The one root holding ``task_sets/``, ``datasets/`` and ``assets/``."""
    storage = (config.get("collection") or {}).get("storage") or {}
    configured = str(storage.get("data_root", "") or "").strip()
    if configured:
        return _resolve(configured)
    # A mounted task set already states the root: <root>/task_sets/<set>.
    directories = [plan_root] if plan_root is not None else _task_set_directories(config)
    for directory in directories:
        if directory is None:
            continue
        if directory.name == TASK_SETS_DIR:
            return directory.parent
        if directory.parent.name == TASK_SETS_DIR:
            return directory.parent.parent
    return _resolve(DEFAULT_DATA_ROOT)


def runtime_dir(config: ConfigDict) -> Path:
    """Machine-local scratch space (logs and UI state), never synced."""
    storage = (config.get("collection") or {}).get("storage") or {}
    configured = str(storage.get("log_dir", "") or "").strip()
    return _resolve(configured or str(config.get("work_dir") or "work_dirs"))


def slot_state_path(config: ConfigDict, dataset: str) -> Path:
    """Where the console keeps one dataset's deferred/selected slot state."""
    return runtime_dir(config) / "collection_slots" / f"{dataset}.json"


def task_set_root(config: ConfigDict, dataset: str) -> Path | None:
    """Directory of the mounted task set named ``dataset``, or None."""
    for root in _task_set_directories(config):
        if root.name == dataset and (root / "tasks.csv").is_file():
            return root
        child = root / dataset
        if (child / "tasks.csv").is_file():
            return child
    return None


def dataset_location(
    config: ConfigDict, dataset: str, robot_type: str = "", plan_root: Path | None = None
) -> tuple[Path, str]:
    """Absolute directory and Hugging Face relative path for one dataset."""
    robot = robot_type or str((config.get("robot") or {}).get("type") or "")
    relative = f"datasets/{ROBOT_DIR}/{robot}/{dataset}"
    plan = Path(plan_root) if plan_root is not None else task_set_root(config, dataset)
    info_path = plan / "info.yaml" if plan is not None else None
    if info_path is not None and info_path.is_file():
        info = yaml.safe_load(info_path.read_text(encoding="utf-8-sig")) or {}
        if isinstance(info, dict) and str(info.get("collection_dir") or "").strip():
            relative = str(info["collection_dir"]).strip()
    path = Path(relative)
    if (
        not relative.startswith("datasets/")
        or path.is_absolute()
        or ".." in path.parts
        or path.name != dataset
    ):
        raise ValueError(f"collection_dir must be 'datasets/<...>/{dataset}', got {relative!r}")
    return data_root(config, plan) / path, relative
