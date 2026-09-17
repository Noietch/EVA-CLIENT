"""Publish and fetch complete task-set directories through Hugging Face Hub."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from tools.datasets.hf_qc import merge_qc, write_qc


def dataset_repo_path(
    api, repo_id: str, dataset_name: str, revision: str, expected_path: str | None = None
) -> str:
    suffix = f"/{dataset_name}/meta/episodes.jsonl"
    matches = [
        path.removesuffix("/meta/episodes.jsonl")
        for path in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if path.startswith("datasets/") and path.endswith(suffix)
    ]
    if expected_path:
        if expected_path in matches:
            return expected_path
        raise FileNotFoundError(f"HF dataset not found at configured path: {expected_path}")
    matches = [
        path
        for path in matches
        if not any(part.startswith(".hf_") or part == ".cache" for part in Path(path).parts)
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple remote datasets named {dataset_name}")
    if not matches:
        raise FileNotFoundError(f"HF dataset not found: {dataset_name}")
    return matches[0]


def qc_repo_path(remote_files: set[str], dataset_name: str, dataset_path: str) -> str | None:
    """Find QC at the dataset path or the older top-level set path."""
    for candidate in (dataset_path, f"datasets/{dataset_name}"):
        if f"{candidate}/meta/qc.jsonl" in remote_files:
            return candidate
    return None


def _config(project_root: Path, storage: dict[str, Any] | None = None) -> dict[str, Any]:
    if storage is not None:
        value = storage.get("huggingface")
        if isinstance(value, dict) and value.get("repo_id"):
            return value
    path = project_root / "configs/local/huggingface.yaml"
    if not path.is_file():
        return {"repo_id": "Noietch/data_collection"}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Hugging Face config must be a mapping")
    return value


def _apply_proxy(cfg: dict[str, Any]) -> None:
    proxy = str(cfg.get("proxy", "")).strip()
    if proxy:
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.setdefault(name, proxy)


def publish_task_set(
    project_root: Path, task_set_dir: Path, storage: dict[str, Any] | None = None
) -> dict[str, str]:
    from huggingface_hub import HfApi

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token = str(cfg.get("token", "")).strip() or None
    repo_id = str(cfg.get("repo_id", "")).strip()
    if not repo_id:
        raise ValueError("Hugging Face config requires repo_id and token")
    task_set_dir = task_set_dir.resolve()
    required = ("info.yaml", "layout.yaml", "scene.csv", "tasks.csv")
    missing = [name for name in required if not (task_set_dir / name).is_file()]
    if missing:
        raise ValueError(f"task-set is missing: {', '.join(missing)}")
    commit = HfApi(token=token).upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(task_set_dir),
        path_in_repo=f"task_sets/{task_set_dir.name}",
        commit_message=f"Publish task set {task_set_dir.name}",
        token=token,
    )
    return {"repo_id": repo_id, "task_set": task_set_dir.name, "revision": commit.oid}


def fetch_task_set(
    project_root: Path, task_set: str, destination: Path, storage: dict[str, Any] | None = None
) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token = str(cfg.get("token", "")).strip() or None
    repo_id = str(cfg.get("repo_id", "")).strip()
    if not repo_id or not task_set or Path(task_set).name != task_set:
        raise ValueError("invalid Hugging Face task-set configuration")
    revision = str(cfg.get("revision", "main")).strip() or "main"
    with tempfile.TemporaryDirectory(prefix="hf-task-set-") as temporary:
        cache = Path(temporary)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"task_sets/{task_set}/*"],
            local_dir=str(cache),
            token=token,
        )
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
    token = str(cfg.get("token", "")).strip() or None
    repo_id = str(cfg.get("repo_id", "")).strip()
    if not repo_id or not assets_dir.is_dir():
        raise ValueError("Hugging Face config or assets directory is invalid")
    commit = HfApi(token=token).upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(assets_dir.resolve()),
        path_in_repo="assets",
        commit_message="Update dataset assets",
        token=token,
    )
    return {
        "repo_id": repo_id,
        "revision": commit.oid,
        "files": str(sum(1 for _ in assets_dir.rglob("*"))),
    }


def fetch_assets(
    project_root: Path, task_set_dir: Path, assets_dir: Path, storage: dict[str, Any] | None = None
) -> dict[str, str]:
    """Fetch only photos referenced by the selected task set's scenes."""
    from huggingface_hub import HfApi, hf_hub_download

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token, repo_id = str(cfg.get("token", "")).strip() or None, str(cfg.get("repo_id", "")).strip()
    if not repo_id or not (task_set_dir / "scene.csv").is_file():
        raise ValueError("invalid Hugging Face assets configuration")
    object_ids: set[str] = set()
    with (task_set_dir / "scene.csv").open(encoding="utf-8-sig", newline="") as handle:
        for scene in csv.DictReader(handle):
            for placement in json.loads(scene.get("placements") or "[]"):
                object_id = str(placement.get("object_id") or "").strip()
                if object_id:
                    object_ids.add(object_id)
    if not object_ids:
        raise FileNotFoundError(f"No asset references in task set: {task_set_dir.name}")
    revision = str(cfg.get("revision", "main")).strip() or "main"
    api = HfApi(token=token)
    revision = api.repo_info(repo_id, repo_type="dataset", revision=revision).sha
    remote_files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    if "assets/objects.csv" not in remote_files:
        raise FileNotFoundError("HF assets catalog is missing")
    catalog_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename="assets/objects.csv",
        token=token,
    )
    photo_dirs: set[str] = set()
    with Path(catalog_path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("object_id") or "").strip() in object_ids:
                photo_dir = str(row.get("photo_dir") or "").strip()
                if photo_dir and Path(photo_dir).name == photo_dir:
                    photo_dirs.add(photo_dir)
    if not photo_dirs:
        raise FileNotFoundError(f"No assets found for task set: {task_set_dir.name}")
    selected_files = [
        path
        for path in remote_files
        if any(path.startswith(f"assets/object_photos/{directory}/") for directory in photo_dirs)
    ]
    if not selected_files:
        raise FileNotFoundError(f"No asset photos published for task set: {task_set_dir.name}")
    files = 0
    for filename in selected_files:
        source = hf_hub_download(
            repo_id=repo_id, repo_type="dataset", revision=revision, filename=filename, token=token
        )
        output = assets_dir / Path(filename).relative_to("assets")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        files += 1
    return {
        "repo_id": repo_id,
        "revision": revision,
        "files": str(files),
        "set": task_set_dir.name,
        "objects": str(len(object_ids)),
        "path": str(assets_dir),
    }


def publish_dataset(
    project_root: Path,
    dataset_dir: Path,
    dataset_name: str,
    storage: dict[str, Any] | None = None,
    *,
    new_remote_path: str | None = None,
    include_qc: bool = True,
) -> dict[str, str]:
    from huggingface_hub import HfApi

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token, repo_id = str(cfg.get("token", "")).strip() or None, str(cfg.get("repo_id", "")).strip()
    if not repo_id or not dataset_dir.is_dir() or Path(dataset_name).name != dataset_name:
        raise ValueError("invalid Hugging Face dataset configuration")
    if not (dataset_dir / "meta/episodes.jsonl").is_file():
        raise FileNotFoundError("Dataset has no saved episodes")
    api = HfApi(token=token)
    revision = str(cfg.get("revision", "main"))
    if new_remote_path and (
        not new_remote_path.startswith("datasets/")
        or Path(new_remote_path).name != dataset_name
        or ".." in Path(new_remote_path).parts
    ):
        raise ValueError("invalid remote dataset path for selected set")
    try:
        remote_path = dataset_repo_path(api, repo_id, dataset_name, revision, new_remote_path)
    except FileNotFoundError:
        remote_path = new_remote_path or f"datasets/{dataset_name}"
    qc_path = dataset_dir / "meta/qc.jsonl"
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir.resolve()),
        path_in_repo=remote_path,
        commit_message=f"Update dataset {dataset_name}",
        token=token,
        revision=revision,
        **({"ignore_patterns": ["meta/qc.jsonl"]} if not include_qc or qc_path.is_file() else {}),
    )
    if include_qc and qc_path.is_file():
        return publish_qc(
            project_root, dataset_dir, dataset_name, storage, expected_path=remote_path
        )
    return {"repo_id": repo_id, "dataset": dataset_name, "revision": commit.oid}


def fetch_dataset(
    project_root: Path,
    dataset_name: str,
    destination: Path,
    storage: dict[str, Any] | None = None,
    *,
    expected_path: str | None = None,
) -> dict[str, str]:
    from huggingface_hub import HfApi, snapshot_download

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token, repo_id = str(cfg.get("token", "")).strip() or None, str(cfg.get("repo_id", "")).strip()
    if not repo_id or Path(dataset_name).name != dataset_name:
        raise ValueError("invalid Hugging Face dataset configuration")
    revision = str(cfg.get("revision", "main")).strip() or "main"
    api = HfApi(token=token)
    revision = api.repo_info(repo_id, repo_type="dataset", revision=revision).sha
    remote_path = dataset_repo_path(api, repo_id, dataset_name, revision, expected_path)
    with tempfile.TemporaryDirectory(prefix="hf-dataset-") as temporary:
        cache = Path(temporary)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{remote_path}/**"],
            local_dir=str(cache),
            token=token,
            max_workers=4,
        )
        source = cache / remote_path
        if not (source / "meta/episodes.jsonl").is_file():
            raise FileNotFoundError(f"Downloaded dataset metadata missing: {dataset_name}")
        target = destination / dataset_name
        target.mkdir(parents=True, exist_ok=True)
        for item in source.rglob("*"):
            if item.is_file():
                output = target / item.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                if item.relative_to(source).as_posix() == "meta/qc.jsonl":
                    local = output.read_bytes() if output.is_file() else b""
                    merged = merge_qc(local, item.read_bytes())
                    if merged != local:
                        write_qc(output, merged)
                    continue
                shutil.copyfile(item, output)
    return {"repo_id": repo_id, "dataset": dataset_name, "revision": revision}


def publish_qc(
    project_root: Path,
    dataset_dir: Path,
    dataset_name: str,
    storage: dict[str, Any] | None = None,
    *,
    expected_path: str | None = None,
) -> dict[str, str]:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token, repo_id = str(cfg.get("token", "")).strip() or None, str(cfg.get("repo_id", "")).strip()
    qc_path = dataset_dir / "meta" / "qc.jsonl"
    if not repo_id:
        raise ValueError("Hugging Face config requires repo_id")
    if not qc_path.is_file():
        raise FileNotFoundError(f"QC file is missing: {qc_path}")
    remote_path = expected_path or f"datasets/{dataset_name}"
    if (
        not remote_path.startswith("datasets/")
        or Path(remote_path).name != dataset_name
        or ".." in Path(remote_path).parts
    ):
        raise ValueError("invalid remote QC path for selected set")
    api = HfApi(token=token)
    branch = str(cfg.get("revision", "main")).strip() or "main"
    revision = api.repo_info(repo_id, repo_type="dataset", revision=branch).sha
    remote_files = set(api.list_repo_files(repo_id, repo_type="dataset", revision=revision))
    published_path = qc_repo_path(remote_files, dataset_name, remote_path)
    path_in_repo = f"{remote_path}/meta/qc.jsonl"
    remote = b""
    if published_path:
        remote_file = hf_hub_download(
            repo_id,
            filename=f"{published_path}/meta/qc.jsonl",
            repo_type="dataset",
            revision=revision,
            token=token,
        )
        remote = Path(remote_file).read_bytes()
    local = qc_path.read_bytes()
    merged = merge_qc(local, remote)
    if merged != remote or published_path != remote_path:
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            revision=branch,
            parent_commit=revision,
            operations=[CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=merged)],
            commit_message=f"Merge QC {dataset_name}",
            token=token,
        )
        revision = commit.oid
    if merged != local:
        if qc_path.read_bytes() != local:
            raise ValueError(f"Local QC changed during upload; retry: {qc_path}")
        write_qc(qc_path, merged)
    return {
        "repo_id": repo_id,
        "dataset": dataset_name,
        "revision": revision,
        "path": path_in_repo,
    }


def fetch_qc(
    project_root: Path,
    dataset_name: str,
    dataset_dir: Path,
    storage: dict[str, Any] | None = None,
    *,
    expected_path: str | None = None,
) -> dict[str, str]:
    from huggingface_hub import HfApi, hf_hub_download

    cfg = _config(project_root, storage)
    _apply_proxy(cfg)
    token, repo_id = str(cfg.get("token", "")).strip() or None, str(cfg.get("repo_id", "")).strip()
    if not repo_id or Path(dataset_name).name != dataset_name:
        raise ValueError("invalid Hugging Face QC configuration")
    api = HfApi(token=token)
    revision = str(cfg.get("revision", "main")).strip() or "main"
    revision = api.repo_info(repo_id, repo_type="dataset", revision=revision).sha
    remote_path = dataset_repo_path(api, repo_id, dataset_name, revision, expected_path)
    remote_files = set(api.list_repo_files(repo_id, repo_type="dataset", revision=revision))
    qc_path = qc_repo_path(remote_files, dataset_name, remote_path)
    if qc_path is not None:
        qc_filename = f"{qc_path}/meta/qc.jsonl"
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=qc_filename,
            revision=revision,
            token=token,
        )
        target = dataset_dir / "meta" / "qc.jsonl"
        local = target.read_bytes() if target.is_file() else b""
        merged = merge_qc(local, Path(path).read_bytes())
        if merged != local:
            write_qc(target, merged)
        return {
            "repo_id": repo_id,
            "dataset": dataset_name,
            "path": str(target),
            "source": "qc.jsonl",
        }

    # Older datasets keep quality metadata in episodes.jsonl instead of qc.jsonl.
    episodes_filename = f"{remote_path}/meta/episodes.jsonl"
    target = dataset_dir / "meta" / "episodes.jsonl"
    if not target.is_file():
        return {
            "repo_id": repo_id,
            "dataset": dataset_name,
            "source": "none",
            "message": f"QC is not published for {dataset_name}",
        }
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=episodes_filename,
        revision=revision,
        token=token,
    )
    remote_rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    remote_by_episode = {int(row["episode_index"]): row for row in remote_rows}
    local_rows = [
        json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    fields = ("quality", "quality_issues", "qc_verdict", "qc_note", "qc_reason")
    updated = 0
    for row in local_rows:
        remote = remote_by_episode.get(int(row.get("episode_index", -1)))
        if remote is None:
            continue
        available = {key: remote[key] for key in fields if key in remote}
        if available:
            row.update(available)
            updated += 1
    if not updated:
        return {
            "repo_id": repo_id,
            "dataset": dataset_name,
            "source": "none",
            "message": f"No matching QC metadata is published for {dataset_name}",
        }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=".episodes-qc-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in local_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)
    return {
        "repo_id": repo_id,
        "dataset": dataset_name,
        "path": str(target),
        "source": "episodes.jsonl",
        "episodes": str(updated),
    }
