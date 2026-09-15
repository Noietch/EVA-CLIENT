"""Publish and fetch complete task-set directories through Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import os
import shutil


def dataset_repo_path(api, repo_id: str, dataset_name: str, revision: str) -> str:
    suffix = f"/{dataset_name}/meta/episodes.jsonl"
    matches = [path.removesuffix("/meta/episodes.jsonl")
               for path in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
               if path.startswith("datasets/") and path.endswith(suffix)]
    if len(matches) > 1:
        raise ValueError(f"Multiple remote datasets named {dataset_name}")
    if not matches:
        raise FileNotFoundError(f"HF dataset not found: {dataset_name}")
    return matches[0]


def _config(project_root: Path, storage: dict[str, Any] | None = None) -> dict[str, Any]:
    if storage is not None:
        value = storage.get("huggingface") or {}
        if isinstance(value, dict):
            return value
    path = project_root / "configs/local/huggingface.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face config is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Hugging Face config must be a mapping")
    return value


def _apply_proxy(cfg: dict[str, Any]) -> None:
    proxy = str(cfg.get("proxy", "")).strip()
    if proxy:
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.setdefault(name, proxy)


def publish_task_set(project_root: Path, task_set_dir: Path, storage: dict[str, Any] | None = None) -> dict[str, str]:
    from huggingface_hub import HfApi

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token = str(cfg.get("token", "")).strip()
    repo_id = str(cfg.get("repo_id", "")).strip()
    if not token or not repo_id:
        raise ValueError("Hugging Face config requires repo_id and token")
    task_set_dir = task_set_dir.resolve()
    required = ("info.yaml", "layout.yaml", "scene.csv", "tasks.csv")
    missing = [name for name in required if not (task_set_dir / name).is_file()]
    if missing:
        raise ValueError(f"task-set is missing: {', '.join(missing)}")
    commit = HfApi(token=token).upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(task_set_dir),
        path_in_repo=f"task_sets/{task_set_dir.name}",
        commit_message=f"Publish task set {task_set_dir.name}", token=token,
    )
    return {"repo_id": repo_id, "task_set": task_set_dir.name, "revision": commit.oid}


def fetch_task_set(project_root: Path, task_set: str, destination: Path, storage: dict[str, Any] | None = None) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token = str(cfg.get("token", "")).strip()
    repo_id = str(cfg.get("repo_id", "")).strip()
    if not token or not repo_id or not task_set or Path(task_set).name != task_set:
        raise ValueError("invalid Hugging Face task-set configuration")
    revision = str(cfg.get("revision", "main")).strip() or "main"
    cache = destination / ".hf_task_sets_cache"
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                      allow_patterns=[f"task_sets/{task_set}/*"],
                      local_dir=str(cache.resolve()), token=token)
    source = cache / "task_sets" / task_set
    target = destination / task_set
    if not (source / "tasks.csv").is_file():
        raise FileNotFoundError(f"task set was not found in HF repo: {task_set}")
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            (target / item.name).write_bytes(item.read_bytes())
    return {"repo_id": repo_id, "task_set": task_set, "revision": revision}


def publish_assets(project_root: Path, assets_dir: Path) -> dict[str, str]:
    from huggingface_hub import HfApi
    cfg = _config(project_root)
    _apply_proxy(cfg)
    token = str(cfg.get("token", "")).strip()
    repo_id = str(cfg.get("repo_id", "")).strip()
    if not token or not repo_id or not assets_dir.is_dir():
        raise ValueError("Hugging Face config or assets directory is invalid")
    commit = HfApi(token=token).upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(assets_dir.resolve()),
        path_in_repo="assets", commit_message="Update dataset assets", token=token,
    )
    return {"repo_id": repo_id, "revision": commit.oid, "files": str(sum(1 for _ in assets_dir.rglob("*")))}


def fetch_assets(project_root: Path, destination: Path, storage: dict[str, Any] | None = None) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token = str(cfg.get("token", "")).strip()
    repo_id = str(cfg.get("repo_id", "")).strip()
    if not token or not repo_id:
        raise ValueError("Hugging Face config requires repo_id and token")
    revision = str(cfg.get("revision", "main")).strip() or "main"
    cache = destination / ".hf_assets_cache"
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                      allow_patterns=["assets/*"], local_dir=str(cache.resolve()), token=token)
    source = cache / "assets"
    if not source.is_dir():
        raise FileNotFoundError("assets were not found in HF repo")
    target = destination / "assets"
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target_item = target / item.name
        if item.is_dir():
            shutil.copytree(item, target_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target_item)
    return {"repo_id": repo_id, "revision": revision}


def publish_dataset(project_root: Path, dataset_dir: Path, dataset_name: str, storage: dict[str, Any] | None = None) -> dict[str, str]:
    from huggingface_hub import HfApi
    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token, repo_id = str(cfg.get("token", "")).strip(), str(cfg.get("repo_id", "")).strip()
    if not token or not repo_id or not dataset_dir.is_dir() or Path(dataset_name).name != dataset_name:
        raise ValueError("invalid Hugging Face dataset configuration")
    if not (dataset_dir / "meta/episodes.jsonl").is_file():
        raise FileNotFoundError("Dataset has no saved episodes")
    api = HfApi(token=token)
    revision = str(cfg.get("revision", "main"))
    try:
        remote_path = dataset_repo_path(api, repo_id, dataset_name, revision)
    except FileNotFoundError:
        remote_path = f"datasets/{dataset_name}"
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(dataset_dir.resolve()),
        path_in_repo=remote_path, commit_message=f"Update dataset {dataset_name}", token=token,
        revision=revision,
    )
    return {"repo_id": repo_id, "dataset": dataset_name, "revision": commit.oid}


def fetch_dataset(project_root: Path, dataset_name: str, destination: Path, storage: dict[str, Any] | None = None) -> dict[str, str]:
    from huggingface_hub import HfApi, snapshot_download
    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token, repo_id = str(cfg.get("token", "")).strip(), str(cfg.get("repo_id", "")).strip()
    if not token or not repo_id or Path(dataset_name).name != dataset_name:
        raise ValueError("invalid Hugging Face dataset configuration")
    revision = str(cfg.get("revision", "main")).strip() or "main"
    api = HfApi(token=token)
    revision = api.repo_info(repo_id, repo_type="dataset", revision=revision).sha
    remote_path = dataset_repo_path(api, repo_id, dataset_name, revision)
    cache = destination / ".hf_dataset_cache"
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                      allow_patterns=[f"{remote_path}/**"],
                      local_dir=str(cache.resolve()), token=token, max_workers=4)
    source = cache / remote_path
    if not (source / "meta/episodes.jsonl").is_file():
        raise FileNotFoundError(f"Downloaded dataset metadata missing: {dataset_name}")
    target = destination / dataset_name
    target.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if item.is_file():
            output = target / item.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            if item.relative_to(source).as_posix() == "meta/qc.jsonl" and output.exists():
                continue
            shutil.copyfile(item, output)
    return {"repo_id": repo_id, "dataset": dataset_name, "revision": revision}


def publish_qc(project_root: Path, dataset_dir: Path, dataset_name: str, storage: dict[str, Any] | None = None) -> dict[str, str]:
    from huggingface_hub import HfApi
    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token, repo_id = str(cfg.get("token", "")).strip(), str(cfg.get("repo_id", "")).strip()
    qc_path = dataset_dir / "meta" / "qc.jsonl"
    if not token or not repo_id or not qc_path.is_file():
        raise FileNotFoundError("QC file or Hugging Face configuration is missing")
    commit = HfApi(token=token).upload_file(
        path_or_fileobj=str(qc_path), path_in_repo=f"datasets/{dataset_name}/meta/qc.jsonl",
        repo_id=repo_id, repo_type="dataset", commit_message=f"Update QC {dataset_name}", token=token,
    )
    return {"repo_id": repo_id, "dataset": dataset_name, "revision": commit.oid}


def fetch_qc(project_root: Path, dataset_name: str, dataset_dir: Path, storage: dict[str, Any] | None = None) -> dict[str, str]:
    from huggingface_hub import hf_hub_download
    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token, repo_id = str(cfg.get("token", "")).strip(), str(cfg.get("repo_id", "")).strip()
    if not token or not repo_id or Path(dataset_name).name != dataset_name:
        raise ValueError("invalid Hugging Face QC configuration")
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                           filename=f"datasets/{dataset_name}/meta/qc.jsonl", revision=str(cfg.get("revision", "main")), token=token)
    target = dataset_dir / "meta" / "qc.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(path).read_bytes())
    return {"repo_id": repo_id, "dataset": dataset_name, "path": str(target)}
