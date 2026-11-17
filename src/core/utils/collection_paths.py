"""One dataset, one directory: where collection data and its QC ledger live.

Recording, the console, and the dataset dashboard all point at the same native
LeRobot v2.1 directory per dataset — ``<data_root>/<collection_dir>``, where the
mounted task set's ``info.yaml`` names ``collection_dir``. Quality verdicts live
in that directory's ``meta/qc.jsonl``.
"""

from __future__ import annotations

import socket
from pathlib import Path

from core.cfg import ConfigDict
from core.utils.scene_plan import load_plan_yaml

DEFAULT_DATA_ROOT = "datasets/data_collection"
TASK_SETS_DIR = "task_sets"
TASK_SET_VERSIONS = ("larybenchv2.v1.1", "larybenchv2.v1.2")
ROBOT_DIR = "real_robot"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else _REPO_ROOT / resolved


def _task_set_directories(config: ConfigDict) -> list[Path]:
    configured = (config.get("collection") or {}).get("task_set_dir") or ""
    values = [configured] if isinstance(configured, (str, Path)) else list(configured)
    return [_resolve(str(value)) for value in values if str(value).strip()]


def discover_task_sets(root: Path) -> dict[str, Path]:
    """Discover leaves directly below a root or its two current version folders."""
    if (root / "tasks.csv").is_file():
        return {root.name: root}
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    leaves = []
    for directory in directories:
        if (directory / "tasks.csv").is_file():
            leaves.append(directory)
        elif directory.name in TASK_SET_VERSIONS:
            leaves.extend(
                sorted(
                    child
                    for child in directory.iterdir()
                    if child.is_dir() and (child / "tasks.csv").is_file()
                )
            )
    result = {}
    for leaf in leaves:
        if leaf.name in result:
            raise ValueError(f"Duplicate task-set name {leaf.name}: {result[leaf.name]}, {leaf}")
        result[leaf.name] = leaf
    return result


def resolve_task_set(root: Path, dataset: str) -> Path | None:
    """Resolve a leaf name without searching historical or deeper directories."""
    candidates = [root] if root.name == dataset else []
    candidates.append(root / dataset)
    candidates.extend(root / version / dataset for version in TASK_SET_VERSIONS)
    matches = [path for path in candidates if (path / "tasks.csv").is_file()]
    if len(matches) > 1:
        raise ValueError(f"Duplicate task-set name {dataset}: {matches}")
    return matches[0] if matches else None


def data_root(config: ConfigDict, plan_root: Path | None = None) -> Path:
    """The one root holding ``task_sets/``, ``datasets/`` and ``assets/``."""
    storage = (config.get("collection") or {}).get("storage") or {}
    configured = str(storage.get("data_root", "") or "").strip()
    if configured:
        return _resolve(configured)
    # A task set may live directly below task_sets or one version folder.
    directories = [plan_root] if plan_root is not None else _task_set_directories(config)
    for directory in directories:
        if directory is None:
            continue
        for ancestor in (directory, directory.parent, directory.parent.parent):
            if ancestor.name == TASK_SETS_DIR:
                return ancestor.parent
    return _resolve(DEFAULT_DATA_ROOT)


def runtime_dir(config: ConfigDict) -> Path:
    """Machine-local scratch space (logs and UI state), never synced."""
    storage = (config.get("collection") or {}).get("storage") or {}
    configured = str(storage.get("log_dir", "") or "").strip()
    return _resolve(configured or str(config.get("work_dir") or "work_dirs"))


def slot_state_path(config: ConfigDict, dataset: str) -> Path:
    """Keep each host's console cursor separate on a shared runtime directory."""
    return runtime_dir(config) / "collection_slots" / socket.gethostname() / f"{dataset}.json"


def task_set_root(config: ConfigDict, dataset: str) -> Path | None:
    """Directory of the mounted task set named ``dataset``, or None."""
    for root in _task_set_directories(config):
        if root.name != dataset and (root / "tasks.csv").is_file():
            continue
        plan = resolve_task_set(root, dataset)
        if plan is not None:
            return plan
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
        info = load_plan_yaml(info_path)
        collection_dir = str(info.get("collection_dir") or "").strip()
        if collection_dir:
            relative = collection_dir
    path = Path(relative)
    if (
        not relative.startswith("datasets/")
        or path.is_absolute()
        or ".." in path.parts
        or path.name != dataset
    ):
        raise ValueError(f"collection_dir must be 'datasets/<...>/{dataset}', got {relative!r}")
    return data_root(config, plan) / path, relative
