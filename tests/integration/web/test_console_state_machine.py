"""Console UI: state-machine gating and error paths.

The frontend disables buttons, but the backend is the real gatekeeper. These tests
hit the gates directly (wrong order, missing prerequisites) and assert the backend
refuses cleanly — no crash, correct ``last_error``, no spurious state change.
"""

from __future__ import annotations

import json
import queue
import signal
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlencode

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import robots  # noqa: F401  (registers ROBOT_REGISTRY incl. ur5e for single-arm UI tests)
from core.app import handlers
from core.app import run as app
from core.app.console import server as console_server
from core.app.console.server import ConsoleContext
from core.app.handlers import utils as handlers_utils
from core.app.state import (
    OutputTarget,
    RuntimeState,
    SessionMode,
    SessionState,
    SessionStatus,
)
from core.config import ConfigDict
from core.utils.dataset_upload import (
    DatasetUploadPlan,
    DatasetUploadResult,
)
from core.utils.upload_plan import (
    DirectoryUploadPlan,
    RemoteFilePlan,
    UploadFilePlan,
)
from tests.integration.web._harness import MULTI_STRATEGY, console_config, serve_console
from tools.conversion import DatasetExportSummary, QualityExportProgress

pytestmark = pytest.mark.integration


def test_run_before_setup_is_rejected(console):
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": "pick up the cup"})

    console.do("/api/run")  # no setup yet
    st = console.status()
    assert st["session_status"] == "unset"
    assert st["last_error"] == "Setup is required before running the next step"


@pytest.mark.parametrize(
    "console", [console_config(supported_inference_strategies=MULTI_STRATEGY)], indirect=True
)
def test_multi_strategy_setup_requires_explicit_choice(console):
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": "pick up the cup"})

    # With >1 strategy, setup must not auto-pick — it must surface the choice list.
    console.do("/api/setup")
    st = console.status()
    assert st["is_setup_done"] is False
    assert st["last_error"] == "Select inference strategy first: 1.sync  2.rtc"

    # After an explicit pick, setup succeeds.
    console.do("/api/select_strategy", {"strategy": "rtc"})
    assert console.runtime.selected_inference_strategy_key == "rtc"
    console.do("/api/setup")
    assert console.status()["is_setup_done"] is True


@pytest.mark.parametrize(
    "console",
    [
        console_config(
            eval=ConfigDict(
                cli_mode="sim",
                inference_strategy="sync",
                reset_after_each_trial=False,
                skip_warmup_after_first=False,
                checkpoints=(),
                tasks=(ConfigDict(prompt_en="pick up the cup"),),
            )
        )
    ],
    indirect=True,
)
def test_eval_cancel_discards_active_episode_and_is_idempotent(console):
    class _EvalLogger:
        is_evaluation = True

        def __init__(self) -> None:
            self.has_active_episode = False
            self.cancelled = 0

        def start_episode(self, task: str) -> None:
            del task
            self.has_active_episode = True

        def set_episode_meta(self, **fields) -> None:
            del fields

        def cancel_episode(self, reason: str) -> None:
            assert reason == "external eval cancel"
            self.cancelled += 1
            self.has_active_episode = False

        def status_snapshot(
            self,
            task: str,
            *,
            include_history: bool = True,
            collection_dataset: str | None = None,
        ) -> dict:
            del task
            del include_history, collection_dataset
            return {
                "active": self.has_active_episode,
                "saving": False,
                "queued_jobs": 0,
                "episodes": [],
                "queue": [],
            }

    prompt = "pick up the cup"
    eval_logger = _EvalLogger()
    console.runtime.episode_logger = eval_logger
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": prompt})
    console.do("/api/setup")
    console.runtime.web_phase = "ready"
    console.do(
        "/api/eval_start",
        {"clip_id": "clip-cancel", "prompt": prompt, "trial": 1},
    )
    assert console.status()["eval_recorder"]["active"] is True

    console.do("/api/eval_cancel")

    status = console.status()
    assert console.session.status is SessionStatus.READY
    assert status["web_phase"] == "ready"
    assert status["clip_id"] is None
    assert status["cell"] is None
    assert status["eval_recorder"]["active"] is False
    assert status["eval_recorder"]["saving"] is False
    assert status["eval_recorder"]["queued_jobs"] == 0
    assert "episodes" not in status["eval_recorder"]
    assert console.runtime.needs_pre_start_reset is True

    console.do("/api/eval_cancel")
    assert console.status()["eval_recorder"]["active"] is False
    assert eval_logger.cancelled == 1


def test_collection_slots_api_filters_independently_and_selects_one_cursor(tmp_path, monkeypatch):
    scene_plan = {
        "scenes": [
            {"scene_id": "SC-A", "placements": [{"name": "绿色杯子"}]},
            {"scene_id": "SC-B", "placements": [{"name": "白色盘子"}]},
        ],
        "tasks": [
            {
                "task_id": "TASK-CUP",
                "prompt_en": "pick up cup",
                "prompt_zh": "拿起杯子",
                "scene_ids": ["SC-A", "SC-B"],
                "scene_epsiodes_count": [2, 1],
            },
            {
                "task_id": "TASK-PLATE",
                "prompt_en": "place cup",
                "prompt_zh": "放置杯子",
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
            },
        ],
    }
    monkeypatch.setattr(console_server, "_load_scene_plan", lambda _config: scene_plan)

    class _Logger:
        has_active_episode = False

        def status_snapshot(self, task, *, include_history=True, collection_dataset=None):
            assert include_history is False
            assert task == "pick up cup"
            assert collection_dataset == "cup_set"
            return {"dataset_dir": str(tmp_path), "queue": []}

    with serve_console(console_config()) as h:
        h.runtime.episode_logger = cast(Any, _Logger())
        idle = h.get("/api/collection_slots?dataset=cup_set")
        all_slots = h.get("/api/collection_slots?dataset=cup_set&all=1")
        no_match = h.get("/api/collection_slots?dataset=cup_set&scene=SC-B&task=TASK-PLATE")
        selected = h.post(
            "/api/select_collection_slot",
            {"dataset": "cup_set", "slot_id": "TASK-PLATE:SC-A:0"},
        )
        skipped = h.post(
            "/api/skip_collection_slot",
            {"dataset": "cup_set", "slot_id": "TASK-PLATE:SC-A:0"},
        )
        status = h.status()

    assert idle.status == 200
    assert idle.json["slots"] == []
    assert idle.json["active"]["slot_id"] == "TASK-CUP:SC-A:0"
    assert all_slots.json["filtered_total"] == 4
    assert [slot["slot_id"] for slot in all_slots.json["slots"]] == [
        "TASK-CUP:SC-A:0",
        "TASK-CUP:SC-A:1",
        "TASK-PLATE:SC-A:0",
        "TASK-CUP:SC-B:0",
    ]
    assert no_match.json["filtered_total"] == 0
    assert selected.json["active"]["slot_id"] == "TASK-PLATE:SC-A:0"
    assert skipped.json["active"]["slot_id"] == "TASK-CUP:SC-A:0"
    assert status["collection_slot_id"] == "TASK-CUP:SC-A:0"


class _CollectDatasetLogger:
    def __init__(self, dataset_dir: Path, *, has_active_episode: bool = False) -> None:
        self._dataset_dir = dataset_dir
        self.has_active_episode = has_active_episode

    def status_snapshot(
        self,
        task: str,
        *,
        include_history: bool = True,
        collection_dataset: str | None = None,
    ) -> dict[str, str]:
        assert include_history is False
        assert task == "pick up cup"
        del collection_dataset
        return {"dataset_dir": str(self._dataset_dir)}


def _set_collect_dataset_logger(
    harness: Any,
    dataset_dir: Path,
    *,
    has_active_episode: bool = False,
    collection_set: str = "cup_set",
) -> None:
    harness.runtime.episode_logger = cast(
        Any,
        _CollectDatasetLogger(dataset_dir, has_active_episode=has_active_episode),
    )
    harness.session.selected_collect_task = "pick up cup"
    harness.session.selected_collect_set = collection_set


def _wait_for_quality_job(
    harness: Any,
    endpoint: str,
    job_id: str,
    *,
    terminal_states: set[str] | None = None,
) -> Any:
    terminal = terminal_states or {"completed", "failed"}
    deadline = time.monotonic() + 2
    while True:
        status = harness.get(f"{endpoint}?job_id={job_id}")
        if status.json["state"] in terminal:
            return status
        assert time.monotonic() < deadline
        time.sleep(0.01)


def _dataset_upload_plan(
    local_dir: Path,
    specs,
    actions: list[tuple[str, int, str]],
    remote_only: list[tuple[str, str]] | None = None,
):
    spec = specs[0]
    return DatasetUploadPlan(
        local_dir=str(local_dir.resolve()),
        backends=(
            (
                spec,
                DirectoryUploadPlan(
                    local_dir=str(local_dir.resolve()),
                    remote_dir=spec.target,
                    destination="robot@upload.example.com",
                    remote_exists=bool(remote_only)
                    or any(action != "new" for _, _, action in actions),
                    files=tuple(
                        UploadFilePlan(path, size, "0" * 32, action)
                        for path, size, action in actions
                    ),
                    remote_only_files=tuple(
                        RemoteFilePlan(path, digest) for path, digest in (remote_only or [])
                    ),
                ),
            ),
        ),
    )


def _write_quality_split_marker(
    accepted_dir: Path,
    source_dir: Path,
    *,
    dataset_format: str | None,
) -> None:
    (accepted_dir / "meta").mkdir(parents=True)
    marker = {
        "subset": "accepted",
        "source_dir": str(source_dir),
        "source_episode_indices": [0, 2],
    }
    if dataset_format is not None:
        marker["dataset_format"] = dataset_format
    (accepted_dir / "meta" / "quality_split.json").write_text(json.dumps(marker))


def _register_completed_quality_export(
    ctx: ConsoleContext,
    source_dir: Path,
    accepted_dir: Path,
    *,
    dataset_format: str,
) -> None:
    ctx.quality_export_jobs["completed-export"] = console_server._QualityExportJob(
        job_id="completed-export",
        source_dir=str(source_dir.resolve()),
        accepted_dir=str(accepted_dir.resolve()),
        rejected_dir=str((accepted_dir.parent / "rejected").resolve()),
        dataset_format=dataset_format,
        state="completed",
    )


def test_collect_quality_export_uses_active_task_dataset(tmp_path, monkeypatch):
    source = tmp_path / "pick_up_cup"
    source.mkdir()
    calls = []

    def export(
        source_dir,
        accepted_dir,
        rejected_dir,
        *,
        dataset_format,
        replace_existing,
        progress_callback,
    ):
        assert dataset_format == "lerobot_v21"
        assert replace_existing is True
        calls.append((source_dir, accepted_dir, rejected_dir))
        progress_callback(QualityExportProgress(1, 3, "accepted", 0))
        progress_callback(QualityExportProgress(3, 3, "rejected", 2))
        return DatasetExportSummary(
            source_dir=str(source_dir),
            accepted_dir=str(accepted_dir),
            rejected_dir=str(rejected_dir),
            dataset_format=dataset_format,
            source_episodes=3,
            accepted_episodes=2,
            rejected_episodes=1,
            accepted_frames=20,
            rejected_frames=10,
            rejected_source_indices=(1,),
        )

    monkeypatch.setattr(console_server, "export_dataset_by_quality", export)
    with serve_console(console_config()) as h:
        _set_collect_dataset_logger(h, source)
        response = h.post(
            "/api/collect_quality_export",
            {"task": "pick up cup", "dataset_format": "lerobot_v21"},
        )
        assert response.status == 202
        assert response.json["dataset_format"] == "lerobot_v21"
        status = _wait_for_quality_job(h, "/api/collect_quality_export", response.json["job_id"])

    assert status.status == 200
    assert status.json["state"] == "completed"
    assert status.json["dataset_format"] == "lerobot_v21"
    assert status.json["progress"] == 1.0
    assert status.json["episodes_completed"] == 3
    assert status.json["accepted_episodes"] == 2
    assert status.json["rejected_episodes"] == 1
    assert calls[0][0] == source.resolve()
    export_root = source.with_name("pick_up_cup_export") / "lerobot_v21"
    assert calls[0][1] == export_root / "accepted"
    assert calls[0][2] == export_root / "rejected"


def test_collect_quality_upload_scan_skips_matching_files_and_never_uses_rejected_export(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "legacy_local_name" / "raw"
    source.mkdir(parents=True)
    accepted = source.parent / "export" / "lerobot_v21" / "accepted"
    rejected = source.parent / "export" / "lerobot_v21" / "rejected"
    _write_quality_split_marker(accepted, source, dataset_format="lerobot_v21")
    rejected.mkdir(parents=True)
    calls = []
    existing_remote_dir = "/datasets/arx_x5/cup_set"

    def scan(local_dir, specs):
        calls.append((local_dir, specs))
        assert local_dir == accepted.resolve()
        assert local_dir != rejected.resolve()
        return _dataset_upload_plan(
            local_dir,
            specs,
            [("meta/info.json", 20, "same"), ("data.bin", 200, "same")],
        )

    monkeypatch.setattr(console_server, "scan_dataset_directory", scan)
    config = console_config()
    config.collection.storage.sftp = ConfigDict(
        host="upload.example.com",
        port=22,
        user="robot",
        identity_file=str(tmp_path / "key"),
        remote_dir="/datasets/arx_x5",
    )
    with serve_console(config) as h:
        _set_collect_dataset_logger(h, source)
        _register_completed_quality_export(
            h.runtime.console_ctx, source, accepted, dataset_format="lerobot_v21"
        )

        first = h.post(
            "/api/collect_quality_upload",
            {"task": "pick up cup", "dataset_format": "lerobot_v21"},
        )
        assert first.status == 202
        status = _wait_for_quality_job(
            h,
            "/api/collect_quality_upload",
            first.json["job_id"],
            terminal_states={"ready", "failed"},
        )

    assert status.status == 200
    assert status.json["state"] == "ready"
    assert status.json["remote_dir"] == existing_remote_dir
    assert status.json["files_total"] == 0
    assert status.json["bytes_total"] == 0
    assert status.json["local_files"] == 2
    assert status.json["files_skipped"] == 2
    assert len(calls) == 1
    assert calls[0][0] == accepted.resolve()
    assert calls[0][1][0].target == "/datasets/arx_x5/cup_set"
    receipt = json.loads((accepted.parent / "upload_receipt.json").read_text())
    assert receipt["source_episode_indices"] == [0, 2]


def test_collect_quality_upload_serializes_confirmed_plans(tmp_path, monkeypatch):
    source = tmp_path / "legacy_local_name" / "raw"
    source.mkdir(parents=True)
    accepted = source.parent / "export" / "lerobot_v21" / "accepted"
    _write_quality_split_marker(accepted, source, dataset_format="lerobot_v21")
    upload_started = threading.Event()
    upload_finished = threading.Event()

    def scan(local_dir, specs):
        return _dataset_upload_plan(local_dir, specs, [("data.bin", 4, "new")])

    def upload(local_dir, specs, *, plan, progress_callback):
        upload_started.set()
        assert upload_finished.wait(timeout=5)
        return DatasetUploadResult(
            local_dir=str(local_dir),
            remote_dir=specs[0].target,
            destination="sftp",
            files=1,
            bytes=4,
        )

    monkeypatch.setattr(console_server, "scan_dataset_directory", scan)
    monkeypatch.setattr(console_server, "upload_dataset_directory", upload)
    config = console_config()
    config.collection.storage.sftp = ConfigDict(
        host="upload.example.com",
        port=22,
        user="robot",
        identity_file=str(tmp_path / "key"),
        remote_dir="/datasets/arx_x5",
    )
    body = {"task": "pick up cup", "dataset_format": "lerobot_v21"}
    with serve_console(config) as h:
        _set_collect_dataset_logger(h, source)
        _register_completed_quality_export(
            h.runtime.console_ctx, source, accepted, dataset_format="lerobot_v21"
        )
        first = h.post("/api/collect_quality_upload", body)
        first_status = _wait_for_quality_job(
            h,
            "/api/collect_quality_upload",
            first.json["job_id"],
            terminal_states={"ready", "failed"},
        )
        second = h.post("/api/collect_quality_upload", body)
        second_status = _wait_for_quality_job(
            h,
            "/api/collect_quality_upload",
            second.json["job_id"],
            terminal_states={"ready", "failed"},
        )
        assert first_status.json["state"] == "ready"
        assert second_status.json["state"] == "ready"

        confirmed = h.post(
            "/api/collect_quality_upload",
            {**body, "confirmed": True, "plan_id": first.json["job_id"]},
        )
        assert confirmed.status == 202
        assert upload_started.wait(timeout=2)
        duplicate = h.post(
            "/api/collect_quality_upload",
            {**body, "confirmed": True, "plan_id": second.json["job_id"]},
        )
        upload_finished.set()
        completed = _wait_for_quality_job(h, "/api/collect_quality_upload", first.json["job_id"])

    assert duplicate.status == 409
    assert duplicate.json["error"] == "dataset upload is already running"
    assert completed.json["state"] == "completed"


def test_collect_quality_upload_requires_export_and_sanitizes_invalid_marker(tmp_path):
    source = tmp_path / "pick_up_cup"
    source.mkdir()
    accepted = source.with_name("pick_up_cup_export") / "lerobot_v21" / "accepted"

    with serve_console(console_config()) as h:
        _set_collect_dataset_logger(h, source)
        body = {"task": "pick up cup", "dataset_format": "lerobot_v21"}

        missing_export = h.post("/api/collect_quality_upload", body)
        assert missing_export.status == 409
        assert missing_export.json["error"] == "export the current dataset before upload"

        _register_completed_quality_export(
            h.runtime.console_ctx, source, accepted, dataset_format="lerobot_v21"
        )
        invalid_marker = h.post("/api/collect_quality_upload", body)

    assert invalid_marker.status == 400
    assert invalid_marker.json == {"ok": False, "error": "invalid accepted export"}
    assert str(tmp_path) not in str(invalid_marker.json)


def test_dashboard_upload_rejects_source_outside_discovered_roots(tmp_path):
    work_root = tmp_path / "dashboard-work"
    work_root.mkdir()
    outside = tmp_path / "outside" / "raw"
    outside.mkdir(parents=True)

    with serve_console(console_config(work_dir=str(work_root))) as h:
        response = h.post(
            "/api/dashboard_upload",
            {"source_dir": str(outside), "dataset_format": "lerobot_v21"},
        )

    assert response.status == 404
    assert response.json == {"ok": False, "error": "accepted export is unavailable"}


def test_review_episode_returns_series_without_mutating_runtime(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "work_dirs" / "review"
    data_dir = dataset_dir / "data" / "chunk-000"
    meta_dir = dataset_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    monkeypatch.setattr(handlers_utils, "_REPO_ROOT", repo_root)

    with serve_console(console_config(robot_type="ur5e")) as h:
        dim = h.runtime.robot.total_action_dim
        info = {
            "total_episodes": 1,
            "fps": 10,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [dim]},
                "action": {"dtype": "float32", "shape": [dim]},
                "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3]},
            },
        }
        (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "length": 2, "tasks": ["review"]}) + "\n",
            encoding="utf-8",
        )
        pq.write_table(
            pa.table(
                {
                    "observation.state": [
                        np.zeros(dim, dtype=np.float32).tolist(),
                        np.ones(dim, dtype=np.float32).tolist(),
                    ],
                    "action": [
                        np.zeros(dim, dtype=np.float32).tolist(),
                        np.ones(dim, dtype=np.float32).tolist(),
                    ],
                    "timestamp": [0.0, 0.1],
                    "frame_index": [0, 1],
                    "episode_index": [0, 0],
                    "index": [0, 1],
                    "task_index": [0, 0],
                }
            ),
            data_dir / "episode_000000.parquet",
        )

        sentinel_qpos = np.full((1, dim), 7.0, dtype=np.float32)
        h.runtime.collection_replay_qpos = sentinel_qpos
        h.runtime.collection_replay_episode = 9
        h.runtime.collection_replay_started = 123.0

        resp = h.post("/api/review_episode", {"dataset_dir": "work_dirs/review", "episode": 0})

        assert resp.status == 200
        assert resp.json["ok"] is True
        assert resp.json["frames"] == 2
        assert resp.json["timestamp"] == pytest.approx([0.0, 0.1])
        assert len(resp.json["state"]) == 2
        assert len(resp.json["action"]) == 2
        assert h.runtime.collection_replay_qpos is sentinel_qpos
        assert h.runtime.collection_replay_episode == 9
        assert h.runtime.collection_replay_started == 123.0
        assert h.runtime.replay_source is None


def test_review_transforms_are_addressed_by_episode_without_runtime_mount(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "work_dirs" / "review"
    data_dir = dataset_dir / "data" / "chunk-000"
    meta_dir = dataset_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    monkeypatch.setattr(handlers_utils, "_REPO_ROOT", repo_root)

    class _Scene:
        def __init__(self):
            self.calls = []

        def all_transforms_blob(self, qpos: np.ndarray) -> bytes:
            self.calls.append(qpos.copy())
            return b"EVAXFRM1" + qpos.astype(np.float32).tobytes()

    with serve_console(console_config(robot_type="ur5e")) as h:
        dim = h.runtime.robot.total_action_dim
        (meta_dir / "info.json").write_text(
            json.dumps(
                {
                    "total_episodes": 1,
                    "fps": 10,
                    "features": {
                        "observation.state": {"dtype": "float32", "shape": [dim]},
                        "action": {"dtype": "float32", "shape": [dim]},
                    },
                }
            ),
            encoding="utf-8",
        )
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "length": 2, "tasks": ["review"]}) + "\n",
            encoding="utf-8",
        )
        pq.write_table(
            pa.table(
                {
                    "observation.state": [
                        np.zeros(dim, dtype=np.float32).tolist(),
                        np.ones(dim, dtype=np.float32).tolist(),
                    ],
                    "action": [
                        np.zeros(dim, dtype=np.float32).tolist(),
                        np.ones(dim, dtype=np.float32).tolist(),
                    ],
                    "timestamp": [0.0, 0.1],
                    "frame_index": [0, 1],
                    "episode_index": [0, 0],
                    "index": [0, 1],
                    "task_index": [0, 0],
                }
            ),
            data_dir / "episode_000000.parquet",
        )
        sentinel_qpos = np.ones((1, dim), dtype=np.float32)
        h.runtime.collection_replay_qpos = sentinel_qpos
        scene = _Scene()
        h.runtime.console_ctx.scene = scene

        query = urlencode({"dataset_dir": "work_dirs/review", "episode": 0})
        resp = h.get(f"/api/review_transforms?{query}")

        assert resp.status == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.raw[:8] == b"EVAXFRM1"
        assert h.runtime.collection_replay_qpos is sentinel_qpos
        assert h.runtime.replay_source is None

        chunk_query = urlencode(
            {"dataset_dir": "work_dirs/review", "episode": 0, "start": 1, "count": 1}
        )
        chunk = h.get(f"/api/review_transforms?{chunk_query}")
        assert chunk.status == 200
        assert chunk.headers["x-eva-transform-start"] == "1"
        assert chunk.headers["x-eva-transform-count"] == "1"
        assert chunk.headers["x-eva-transform-total"] == "2"
        assert scene.calls[-1].shape[0] == 1


def test_collect_start_api_requires_activation_and_preserves_scene_metadata():
    with serve_console(console_config()) as h:
        h.runtime.transport.supports_collection = lambda: True
        resp = h.post("/api/collect_start")
        assert resp.json["ok"] is False
        assert h.status()["collection_teleop_armed"] is False

        h.post("/api/tab_switch", {"tab": "collect"})
        assert h.status()["collection_teleop_armed"] is False
        resp = h.post("/api/collect_start")
        assert resp.json["ok"] is False

        h.do("/api/collect_arm", {"enabled": True})
        assert h.status()["collection_teleop_armed"] is True
        resp = h.post(
            "/api/collect_start",
            {"scene_id": "SC-002", "scene_round": 2, "random_seed": 91},
        )
        command = h.runtime.command_queue.get_nowait()
        status = h.status()

    assert resp.json["ok"] is True
    assert command == "web:collect_start"
    assert status["collection_scene_id"] == "SC-002"
    assert status["collection_scene_round"] == 2
    assert status["collection_random_seed"] == 91


def test_collect_start_rejects_a_slot_that_is_no_longer_active(monkeypatch):
    monkeypatch.setattr(
        console_server,
        "_collection_slots_snapshot",
        lambda _ctx, _dataset: {"active": None},
    )
    with serve_console(console_config()) as h:
        h.runtime.transport.supports_collection = lambda: True
        h.post("/api/tab_switch", {"tab": "collect"})
        h.do("/api/collect_arm", {"enabled": True})
        h.session.selected_collect_set = "cup_set"
        h.session.collection_scene_id = "SC-001"
        h.session.collection_scene_round = 2
        h.session.collection_slot_id = "TASK-CUP:SC-001:2"
        h.session.collection_task_id = "TASK-CUP"

        response = h.post("/api/collect_start")

    assert response.status == 409
    assert response.json["error"] == "collection slot is no longer active"
    assert h.runtime.command_queue.empty()


def test_setup_fails_cleanly_when_policy_unreachable():
    """A real (non-mock) policy server that isn't listening must fail setup with a
    clear error, not hang or crash — the operator needs to know the server is down.
    Console setup uses fail_fast (one attempt, no 30s retry block)."""
    from tests.integration.web._harness import console_config, serve_console

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()

    config = console_config(
        policy=ConfigDict(type="openpi", host="127.0.0.1", port=dead_port),
        supported_inference_strategies={"sync": {"type": "BaseInferStrategy", "args": {}}},
    )
    with serve_console(config) as h:
        h.do("/api/select_mode", {"mode": "sim"})
        h.do("/api/select_task", {"task": "pick up the cup"})
        h.do("/api/setup")
        st = h.status()
        assert st["is_setup_done"] is False
        assert "Setup failed" in st["last_error"]


def _eva_loop_thread_count() -> int:
    return sum(1 for t in threading.enumerate() if "eva-" in t.name and "loop" in t.name)


@pytest.mark.parametrize(
    "console", [console_config(supported_inference_strategies=MULTI_STRATEGY)], indirect=True
)
def test_rapid_reselection_leaks_no_loop_threads(console):
    """Alternating mode/strategy reselection must not accumulate strategy loop threads.

    Each select_* tears down the prior infer_strategy (reset_infer_strategy ->
    stop_loop), so the live 'eva-*-loop' count must stay at the baseline regardless of
    how many times the operator flips between rtc and sync mid-session.
    """
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "real"})
    console.do("/api/select_strategy", {"strategy": "rtc"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    console.do("/api/setup")
    console.do("/api/run")  # starts the rtc continuous loop thread
    assert console.session.status is SessionStatus.RUNNING

    baseline = _eva_loop_thread_count()

    for i in range(30):
        mode = "real" if i % 2 == 0 else "sim"
        strategy = "rtc" if i % 2 == 0 else "sync"
        console.do("/api/select_mode", {"mode": mode})
        console.do("/api/select_strategy", {"strategy": strategy})
        console.do("/api/select_task", {"task": "pick up the cup"})
        console.do("/api/setup")

    console.do("/api/halt")
    time.sleep(0.2)  # let any in-flight join settle
    # +1 tolerance for a single loop thread mid-join (joins have a 5s timeout).
    assert _eva_loop_thread_count() <= baseline + 1


def test_halt_mid_chunk_aborts_real_publish_to_ready(console):
    """A pre-queued web:halt must abort an in-flight real publish early, land on READY,
    and clear interrupt_requested so the next run is not pre-aborted."""
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "real"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    console.do("/api/setup")
    s = console.session
    assert s.is_setup_done is True

    console.runtime.command_queue.put("web:halt")
    assert handlers.ensure_action_chunk(console.config, console.runtime, s)
    ok = handlers.publish_action_chunk(
        console.config,
        console.runtime,
        s.action_chunk[s.chunk_index :],
        OutputTarget.REAL,
        session=s,
        advance_session_chunk_index=True,
    )

    assert ok is False
    assert s.chunk_index < console.config.policy.backend_options["chunk_size"]  # stopped early
    assert s.status is SessionStatus.READY
    assert s.interrupt_requested is False


def test_manual_halt_aborts_staged_real_publish():
    command_queue: queue.Queue[str] = queue.Queue()

    class _Rate:
        def __init__(self) -> None:
            self.sleeps = 0

        def sleep(self) -> None:
            assert session.manual_publish_active is True
            self.sleeps += 1
            if self.sleeps == 1:
                command_queue.put("web:halt")

    class _Transport:
        def __init__(self) -> None:
            self.qpos = np.zeros(2, dtype=np.float32)
            self.published: list[tuple[str, np.ndarray]] = []
            self.rate = _Rate()

        def get_latest_qpos(self) -> np.ndarray:
            return self.qpos.copy()

        def publish_action(self, action: np.ndarray, target: str = "real") -> None:
            sent = np.asarray(action, dtype=np.float32).copy()
            self.published.append((target, sent))
            if target == "real":
                self.qpos = sent

        def create_rate(self, _hz: float) -> _Rate:
            return self.rate

    transport = _Transport()
    robot = SimpleNamespace(gripper_mask=(False, False))
    runtime = SimpleNamespace(
        robot=robot,
        transport=transport,
        command_queue=command_queue,
        prompt_ready=None,
        web_phase="idle",
    )
    target = np.asarray([handlers.MANUAL_MAX_QPOS_STEP * 3, 0.0], dtype=np.float32)
    session = SessionState(
        mode=SessionMode.MANUAL,
        status=SessionStatus.UNSET,
        manual_qpos=target,
        manual_real_qpos=np.zeros(2, dtype=np.float32),
    )
    config = ConfigDict(inference_cfg=ConfigDict(publish_rate=100))

    app.handle_command("web:manual_send", config, cast(RuntimeState, runtime), session)

    assert len(transport.published) == 1
    assert session.interrupt_requested is False
    assert session.manual_publish_active is False
    np.testing.assert_allclose(session.manual_real_qpos, transport.published[-1][1])
    assert not np.allclose(session.manual_real_qpos, session.manual_qpos)


def test_repeated_setup_run_stop_cycles_never_pin_running(console):
    """setup->run->publish->stop, ~15x. step_index is reset by run_setup (via
    reset_session_progress), so it must read 0 right after each /api/setup. The final
    status must settle on READY, never stuck RUNNING."""
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "real"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    s = console.session

    for _ in range(15):
        console.do("/api/setup")
        assert s.is_setup_done is True
        assert s.status is SessionStatus.READY
        assert s.step_index == 0

        console.do("/api/run")
        assert s.status is SessionStatus.RUNNING
        for _ in range(3):
            handlers.publish_next_action(console.config, console.runtime, s)

        console.do("/api/halt")
        assert s.status is SessionStatus.READY

    assert s.status is SessionStatus.READY


def test_eval_stop_during_running_motion_is_requeued_finalized_and_left_ready():
    command_queue: queue.Queue[str] = queue.Queue()

    class _Logger:
        def __init__(self) -> None:
            self.ended = 0

        def end_episode(self) -> None:
            self.ended += 1

    logger_obj = _Logger()
    finished_capture = []
    runtime = SimpleNamespace(
        web_phase="running",
        command_queue=command_queue,
        episode_logger=logger_obj,
        collection_capture_runner=None,
        transport=SimpleNamespace(finish_collection_capture=lambda: finished_capture.append(True)),
        needs_pre_start_reset=False,
    )
    session = SessionState(
        mode=SessionMode.REAL,
        status=SessionStatus.RUNNING,
        is_setup_done=True,
        selected_task="pick up the cup",
    )
    config = ConfigDict(eval=ConfigDict(reset_after_each_trial=False))

    interrupted = handlers.handle_motion_command(
        "web:stop", config, cast(RuntimeState, runtime), session
    )
    handlers.consume_motion_interrupt(session, SessionStatus.READY)

    assert interrupted is True
    assert list(command_queue.queue) == ["web:stop"]
    assert runtime.web_phase == "stopping"
    assert not (runtime.web_phase == "running" and session.status is not SessionStatus.RUNNING)

    app.handle_command(command_queue.get_nowait(), config, cast(RuntimeState, runtime), session)

    assert runtime.web_phase == "ready"
    assert runtime.needs_pre_start_reset is True
    assert logger_obj.ended == 1
    assert finished_capture == [True]


def test_requeued_motion_interrupt_returns_to_main_loop(console):
    console.do("/api/select_mode", {"mode": "real"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    s = console.session
    q = console.runtime.command_queue
    assert q is not None
    q.put("web:select_mode:sim")

    def raise_timeout(signum, frame):
        del signum, frame
        raise TimeoutError("poll_motion_commands did not return")

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    try:
        interrupted = handlers.poll_motion_commands(console.config, console.runtime, s)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

    assert interrupted is True
    assert s.interrupt_requested is True
    assert list(q.queue) == ["web:select_mode:sim"]
