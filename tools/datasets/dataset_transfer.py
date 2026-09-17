"""Background dataset operations: task-set transfers, Hugging Face comparisons,
and automatic QC scans.

Rows are task sets: one row per ``<collection_root>/task_sets/<set>`` plan
directory, mirroring the EVA-CLIENT console dataset manager.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from tools.datasets.auto_qc import DatasetAutoQc
from tools.datasets.collection import PlanCatalog
from tools.datasets.hf_compare import (
    compare_files,
    compare_qc,
    data_files,
    is_dataset_content,
    qc_entries,
    qc_summary,
    read_jsonl,
)
from tools.datasets.hf_task_sets import (
    TASK_FILES,
    _apply_proxy,
    _config,
    dataset_repo_path,
    fetch_qc,
    fetch_task_set,
    publish_dataset,
    publish_qc,
    publish_task_set,
)

ACTIONS = {
    "refresh",
    "verify",
    "auto_qc",
    "upload_data",
    "upload_qc",
    "download_qc",
    "upload_task",
    "download_task",
}

# Remote reads use immutable HF revisions; only local reads need leases.
ACCESS = {
    "refresh": {"local_data": "read", "local_qc": "read", "local_task": "read"},
    "verify": {"local_data": "read", "local_qc": "read", "local_task": "read"},
    "auto_qc": {"local_data": "read", "local_qc": "write"},
    "upload_data": {"local_data": "read", "remote_data": "write"},
    "upload_qc": {"local_qc": "read", "local_data": "read", "remote_qc": "write"},
    "download_qc": {"local_qc": "write", "remote_qc": "read"},
    "upload_task": {"local_task": "read", "remote_task": "write"},
    "download_task": {"local_task": "write"},
}

# Automatic QC is CPU-bound video decoding; keep a dataset scan on one worker
# and cap the datasets scanned in parallel.
SCAN_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 4))


class DatasetTransfer:
    """Run dataset operations: Hugging Face transfers, comparisons, QC scans."""

    def __init__(self, project_root: Path, catalog: PlanCatalog):
        self.project_root = project_root
        self.catalog = catalog
        self.lock = threading.RLock()
        self.remote: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}
        self.cancellations: dict[str, threading.Event] = {}
        self.leases: dict[tuple[str, str], tuple[str, dict]] = {}
        self.generations: dict[str, int] = {}
        self.condition = threading.Condition(self.lock)

    def rows(self) -> list[dict[str, Any]]:
        summaries = {row["batch_id"]: row for row in self.catalog.state()["all_batches"]}
        return [
            {
                "name": name,
                "robot": summaries[name]["robot_type"],
                "total": summaries[name]["target_episodes"],
                "collected": summaries[name]["collected"],
                "accept": summaries[name]["passed"],
                "fail": summaries[name]["failed"],
                "unreviewed": summaries[name]["unreviewed"],
            }
            for name in self.catalog.batch_ids()
        ]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            jobs = copy.deepcopy(list(self.jobs.values()))
            for job in jobs:
                elapsed = job.get("finished", time.time()) - job["started"]
                job["elapsed"] = elapsed
                job["eta"] = (
                    elapsed / job["completed"] * (job["total"] - job["completed"])
                    if job["completed"] and job["state"] == "running"
                    else None
                )
            return {
                "remote": copy.deepcopy(self.remote),
                "jobs": jobs,
                "job": jobs[-1] if jobs else None,
            }

    def start(self, action: str, names: list[str]) -> str:
        if action not in ACTIONS:
            raise ValueError("Unknown dataset action")
        targets = [self._target(name) for name in names]
        with self.lock:
            if sum(job["state"] == "running" for job in self.jobs.values()) >= 8:
                raise ValueError("Eight dataset jobs are already active")
            for old_id in list(self.jobs):
                if len(self.jobs) < 32:
                    break
                if self.jobs[old_id]["state"] != "running":
                    self.jobs.pop(old_id)
                    self.cancellations.pop(old_id)
            job_id = uuid.uuid4().hex
            self.cancellations[job_id] = threading.Event()
            self.jobs[job_id] = {
                "id": job_id,
                "action": action,
                "state": "running",
                "started": time.time(),
                "total": len(targets),
                "completed": 0,
                "current": "",
                "results": [],
                "detail": "",
                "waiting": False,
            }
        runner = self._run_scan if action == "auto_qc" else self._run
        threading.Thread(target=runner, args=(job_id, targets), daemon=True).start()
        return job_id

    def stop(self, job_id: str) -> None:
        with self.lock:
            if job_id not in self.cancellations:
                raise ValueError("Unknown dataset job")
            self.cancellations[job_id].set()
            self.jobs[job_id]["stop_requested"] = True
            self.condition.notify_all()

    def _target(self, name: str) -> dict[str, Any]:
        plan = self.catalog._batch_root(name)
        info = yaml.safe_load((plan / "info.yaml").read_text(encoding="utf-8-sig")) or {}
        dataset_dir = self.catalog._source_dataset_dir(name, info)
        relative = dataset_dir.relative_to(self.catalog.collection_root).as_posix()
        return {
            "name": name,
            "plan": plan,
            "dataset_dir": dataset_dir,
            # Local collection paths are not the remote layout; inspection
            # resolves the real prefix from the repository when this is None.
            "remote_path": relative if relative.startswith("datasets/") else None,
        }

    def _acquire(self, job_id: str, name: str, action: str) -> bool:
        access = ACCESS[action]
        with self.condition:
            while not self.cancellations[job_id].is_set():
                conflicts = sorted(
                    {
                        resource
                        for other_name, other_access in self.leases.values()
                        if other_name == name
                        for resource, mode in access.items()
                        if resource in other_access and "write" in (mode, other_access[resource])
                    }
                )
                if not conflicts:
                    self.leases[(job_id, name)] = (name, access)
                    if "write" in access.values():
                        self.generations[name] = self.generations.get(name, 0) + 1
                        self.remote.pop(name, None)
                    self.jobs[job_id].update(waiting=False, waiting_resources=[])
                    return True
                self.jobs[job_id].update(waiting=True, waiting_resources=conflicts)
                self.condition.wait(timeout=0.1)
            return False

    def _release(self, job_id: str, name: str) -> None:
        with self.condition:
            _, access = self.leases.pop((job_id, name))
            if "write" in access.values():
                self.generations[name] = self.generations.get(name, 0) + 1
                self.remote.pop(name, None)
            self.condition.notify_all()

    def _run(self, job_id: str, targets: list[dict]) -> None:
        """Run the targets one after another."""
        action = self.jobs[job_id]["action"]
        for target in targets:
            if self.cancellations[job_id].is_set():
                break
            self._process(job_id, action, target)
        self._finish(job_id)

    def _run_scan(self, job_id: str, targets: list[dict]) -> None:
        """Scan datasets in parallel; one worker owns one dataset."""
        action = self.jobs[job_id]["action"]
        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
            list(pool.map(partial(self._process, job_id, action), targets))
        self._finish(job_id)

    def _finish(self, job_id: str) -> None:
        with self.lock:
            self.jobs[job_id].update(
                state="stopped" if self.cancellations[job_id].is_set() else "done",
                current="",
                detail="",
                waiting=False,
                finished=time.time(),
            )

    def _process(self, job_id: str, action: str, target: dict) -> None:
        """Run one dataset operation and record its outcome on the job."""
        job = self.jobs[job_id]
        name = target["name"]
        with self.lock:
            job.update(current=name, detail="", waiting=False, waiting_resources=[])
        if not self._acquire(job_id, name, action):
            return
        try:
            with self.lock:
                job["waiting"] = False
            if action in {"refresh", "verify"}:
                self._refresh(job, target, name)
                detail = ""
            elif action == "auto_qc":
                detail = self._auto_qc(job_id, target, name)
            else:
                self._transfer(action, target, name)
                with self.lock:
                    self.remote.pop(name, None)
                detail = ""
            outcome = {"dataset": name, "ok": True, "detail": detail}
        except Exception as exc:  # per-set failure is reported in the job result
            outcome = {"dataset": name, "ok": False, "error": str(exc)}
            with self.lock:
                self.remote.setdefault(name, {}).update(error=str(exc))
        finally:
            self._release(job_id, name)
        with self.lock:
            job["results"].append(outcome)
            job["completed"] += 1

    def _auto_qc(self, job_id: str, target: dict, name: str) -> str:
        plan = yaml.safe_load((target["plan"] / "info.yaml").read_text(encoding="utf-8-sig"))
        robot_type = str((plan or {}).get("robot_type", ""))

        def report(position: int, total: int) -> None:
            with self.lock:
                self.jobs[job_id]["detail"] = f"{name} · {position}/{total}"

        scan = DatasetAutoQc(
            self.catalog, target["dataset_dir"], robot_type, self.cancellations[job_id]
        )
        result = scan.run(report)
        self.catalog._invalidate_plan_cache(name)
        return result["summary"]

    def _refresh(self, job: dict, target: dict, name: str) -> None:
        with self.lock:
            generation = self.generations.get(name, 0)
            inspection_started = time.time()

        def progress(index, total, filename):
            with self.lock:
                job["detail"] = f"{index}/{total} · {filename}"

        result = self._inspect(target, progress)
        with self.lock:
            writing = any(n == name and "write" in a.values() for n, a in self.leases.values())
            if generation != self.generations.get(name, 0) or writing:
                result["stale"] = True
                result.pop("verification", None)
            result["inspection_started"] = inspection_started
            if self.remote.get(name, {}).get("inspection_started", 0) <= inspection_started:
                self.remote[name] = result

    def _transfer(self, action: str, target: dict, name: str) -> None:
        if action == "download_qc":
            # The remote may only hand down QC, never dataset content.
            fetch_qc(
                self.project_root,
                name,
                target["dataset_dir"],
                expected_path=target["remote_path"],
            )
        elif action == "upload_data":
            publish_dataset(
                self.project_root,
                target["dataset_dir"],
                name,
                new_remote_path=target["remote_path"],
                include_qc=False,
            )
        elif action == "upload_task":
            publish_task_set(self.project_root, target["plan"])
        elif action == "download_task":
            fetch_task_set(self.project_root, name, target["plan"].parent)
        elif action == "upload_qc":
            qc_root = target["dataset_dir"]
            if (qc_root / "meta/qc.jsonl").is_file():
                publish_qc(self.project_root, qc_root, name, expected_path=target["remote_path"])
            else:
                entries = qc_entries(read_jsonl(qc_root / "meta/episodes.jsonl"), [])
                if not entries:
                    raise ValueError("No local episodes or QC to upload")
                with tempfile.TemporaryDirectory(prefix="eva-qc-upload-") as directory:
                    root = Path(directory)
                    (root / "meta").mkdir()
                    (root / "meta/qc.jsonl").write_text(
                        "".join(
                            json.dumps({"episode_index": int(index), **row}, ensure_ascii=False)
                            + "\n"
                            for index, row in entries.items()
                        )
                    )
                    publish_qc(self.project_root, root, name, expected_path=target["remote_path"])

    def _inspect(self, target: dict, progress=lambda *_: None) -> dict:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        cfg = _config(self.project_root)
        _apply_proxy(cfg)
        token = cfg.get("token") or None
        repo = cfg["repo_id"]
        api = HfApi(token=token)
        revision = api.repo_info(
            repo, repo_type="dataset", revision=cfg.get("revision") or "main"
        ).sha

        def tree(prefix):
            try:
                return {
                    f.path[len(prefix) + 1 :]: f
                    for f in api.list_repo_tree(
                        repo,
                        path_in_repo=prefix,
                        repo_type="dataset",
                        revision=revision,
                        recursive=True,
                    )
                    if hasattr(f, "blob_id")
                }
            except EntryNotFoundError:
                return {}

        def read_rows(prefix, files, name):
            if name not in files:
                return []
            file = hf_hub_download(
                repo, f"{prefix}/{name}", repo_type="dataset", revision=revision, token=token
            )
            return [
                json.loads(line) for line in Path(file).read_text().splitlines() if line.strip()
            ]

        prefix = target["remote_path"] or dataset_repo_path(api, repo, target["name"], revision)
        files = tree(prefix)
        plan_prefix = f"task_sets/{target['name']}"
        plan_files = tree(plan_prefix)
        episodes = read_rows(prefix, files, "meta/episodes.jsonl")
        qc = read_rows(prefix, files, "meta/qc.jsonl")
        task_count = None
        if "tasks.csv" in plan_files:
            task_file = hf_hub_download(
                repo,
                f"{plan_prefix}/tasks.csv",
                repo_type="dataset",
                revision=revision,
                token=token,
            )
            with Path(task_file).open(encoding="utf-8-sig", newline="") as stream:
                task_count = sum(
                    bool(row.get("task_id", "").strip()) for row in csv.DictReader(stream)
                )
        result = {
            "checked_at": time.time(),
            "revision": revision,
            "episodes": len(episodes),
            "task_count": task_count,
            "qc": qc_summary(episodes, qc),
            "files": len(files),
            "bytes": sum(f.size for f in files.values()),
            "data_exists": "meta/episodes.jsonl" in files,
            "task_exists": all(n in plan_files for n in TASK_FILES),
            "qc_exists": "meta/qc.jsonl" in files,
        }
        remote_data = {n: f for n, f in files.items() if is_dataset_content(n)}
        qc_root = target["dataset_dir"]
        result["verification"] = {
            "data": compare_files(data_files(target["dataset_dir"]), remote_data, progress),
            "task": compare_files(
                {n: target["plan"] / n for n in TASK_FILES if (target["plan"] / n).is_file()},
                {n: f for n, f in plan_files.items() if n in TASK_FILES},
                progress,
            ),
            "qc": compare_qc(
                qc_entries(
                    read_jsonl(qc_root / "meta/episodes.jsonl"),
                    read_jsonl(qc_root / "meta/qc.jsonl"),
                ),
                qc_entries(episodes, qc),
            ),
        }
        for kind, present in (("data", result["data_exists"]), ("task", result["task_exists"])):
            if not present and result["verification"][kind]["state"] == "same":
                result["verification"][kind]["state"] = "absent"
        return result
