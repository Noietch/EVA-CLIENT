"""Background dataset transfers and read-only HF content comparisons."""

from __future__ import annotations

import copy
import csv
import json
import threading
import time
import uuid
from pathlib import Path

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
    fetch_qc,
    fetch_task_set,
    publish_dataset,
    publish_qc,
    publish_task_set,
)

# Remote reads use immutable HF revisions; only local reads need leases.
ACCESS = {
    "refresh": {"local_data": "read", "local_qc": "read", "local_task": "read"},
    "verify": {"local_data": "read", "local_qc": "read", "local_task": "read"},
    "upload_data": {"local_data": "read", "remote_data": "write"},
    "upload_qc": {"local_qc": "read", "local_data": "read", "remote_qc": "write"},
    "download_qc": {"local_qc": "write", "remote_qc": "read"},
    "upload_task": {"local_task": "read", "remote_task": "write"},
    "download_task": {"local_task": "write"},
}


class DatasetManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.rows = {}
        self.jobs = {}
        self.cancellations = {}
        self.leases = {}
        self.generations = {}
        self.condition = threading.Condition(self.lock)

    def stop(self, job_id):
        with self.lock:
            if job_id not in self.cancellations:
                raise ValueError("Unknown dataset job")
            self.cancellations[job_id].set()
            self.jobs[job_id]["stop_requested"] = True
            self.condition.notify_all()

    def _acquire(self, job_id, name, action):
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
                    self.leases[job_id] = (name, access)
                    if "write" in access.values():
                        self.generations[name] = self.generations.get(name, 0) + 1
                        self.rows.pop(name, None)
                    self.jobs[job_id].update(waiting=False, waiting_resources=[])
                    return True
                self.jobs[job_id].update(waiting=True, waiting_resources=conflicts)
                self.condition.wait(timeout=0.1)
            return False

    def _release(self, job_id):
        with self.condition:
            name, access = self.leases.pop(job_id)
            if "write" in access.values():
                self.generations[name] = self.generations.get(name, 0) + 1
                self.rows.pop(name, None)
            self.condition.notify_all()

    def snapshot(self):
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
                "remote": copy.deepcopy(self.rows),
                "jobs": jobs,
                "job": jobs[-1] if jobs else None,
            }

    def start(self, action, targets, project_root, storage):
        if action not in {
            "refresh",
            "verify",
            "upload_data",
            "upload_qc",
            "download_qc",
            "upload_task",
            "download_task",
        }:
            raise ValueError("Unknown dataset action")
        with self.lock:
            if sum(j["state"] == "running" for j in self.jobs.values()) >= 8:
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
        threading.Thread(
            target=self._run,
            args=(job_id, action, targets, project_root, copy.deepcopy(storage)),
            daemon=True,
        ).start()
        return job_id

    def _run(self, job_id, action, targets, project_root, storage):
        job = self.jobs[job_id]
        cancelled = self.cancellations[job_id]
        for target in targets:
            if cancelled.is_set():
                break
            name = target["name"]
            with self.lock:
                job.update(current=name, detail="", waiting=False, waiting_resources=[])
            if not self._acquire(job_id, name, action):
                break
            try:
                if cancelled.is_set():
                    break
                with self.lock:
                    job["waiting"] = False
                if action in {"refresh", "verify"}:
                    with self.lock:
                        generation = self.generations.get(name, 0)
                        inspection_started = time.time()

                    def progress(index, total, filename):
                        with self.lock:
                            job["detail"] = f"{index}/{total} · {filename}"

                    result = self._inspect(target, project_root, storage, True, progress)
                    with self.lock:
                        writing = any(
                            n == name and "write" in a.values() for n, a in self.leases.values()
                        )
                        if generation != self.generations.get(name, 0) or writing:
                            result["stale"] = True
                            result.pop("verification", None)
                        result["inspection_started"] = inspection_started
                        if (
                            self.rows.get(name, {}).get("inspection_started", 0)
                            <= inspection_started
                        ):
                            self.rows[name] = result
                elif action == "upload_data":
                    publish_dataset(
                        project_root,
                        target["dataset_dir"],
                        name,
                        storage,
                        new_remote_path=target["remote_path"],
                        include_qc=False,
                    )
                elif action == "download_qc":
                    # The remote may only hand down QC, never dataset content.
                    fetch_qc(
                        project_root,
                        name,
                        target["dataset_dir"],
                        storage,
                        expected_path=target["remote_path"],
                    )
                elif action == "upload_task":
                    publish_task_set(project_root, target["plan"], storage)
                elif action == "download_task":
                    fetch_task_set(project_root, name, target["plan"].parent, storage)
                elif action == "upload_qc":
                    dataset_dir = target["dataset_dir"]
                    if not (dataset_dir / "meta/qc.jsonl").is_file():
                        raise ValueError(f"No QC ledger to upload: {dataset_dir}/meta/qc.jsonl")
                    publish_qc(
                        project_root,
                        dataset_dir,
                        name,
                        storage,
                        expected_path=target["remote_path"],
                    )
                if action not in {"refresh", "verify"}:
                    with self.lock:
                        self.rows.pop(name, None)
                outcome = {"dataset": name, "ok": True}
            except Exception as exc:
                outcome = {"dataset": name, "ok": False, "error": str(exc)}
                with self.lock:
                    self.rows.setdefault(name, {}).update(error=str(exc))
            finally:
                self._release(job_id)
            with self.lock:
                job["results"].append(outcome)
                job["completed"] += 1
        with self.lock:
            job.update(
                state="stopped" if cancelled.is_set() else "done",
                current="",
                detail="",
                waiting=False,
                finished=time.time(),
            )

    def _inspect(self, target, project_root, storage, verify, progress=lambda *_: None):
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        cfg = _config(project_root, storage)
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

        prefix = target["remote_path"]
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
        if verify:
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
