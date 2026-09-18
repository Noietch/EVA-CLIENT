"""Console UI: state-machine gating and error paths.

The frontend disables buttons, but the backend is the real gatekeeper. These tests
hit the gates directly (wrong order, missing prerequisites) and assert the backend
refuses cleanly — no crash, correct ``last_error``, no spurious state change.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import robots  # noqa: F401  (registers ROBOT_REGISTRY incl. ur5e for single-arm UI tests)
from core.app import handlers
from core.app.console import server as console_server
from core.app.console.server import ConsoleContext
from core.app.handlers import utils as handlers_utils
from core.app.state import SessionStatus
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
from tests.integration._harness import MULTI_STRATEGY, console_config, serve_console
from tools.conversion import DatasetExportProgress, DatasetExportSummary

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


def test_completed_slot_retake_survives_polling_and_releases_after_save(tmp_path, monkeypatch):
    monkeypatch.setattr(
        console_server, "_load_scene_plan", lambda *_args: {"scenes": [], "tasks": []}
    )
    episodes = [
        {"slot_id": "INLINE-0:0", "episode_index": 0, "status": "saved", "quality": "green"}
    ]
    monkeypatch.setattr(
        console_server, "load_episode_history", lambda *_args, **_kwargs: {"episodes": episodes}
    )
    queue = []

    class Logger:
        has_active_episode = False

        def status_snapshot(self, *args, **kwargs):
            return {"dataset_dir": str(tmp_path), "queue": queue}

    with serve_console(console_config()) as h:
        h.runtime.episode_logger = cast(Any, Logger())
        body = {"dataset": "cup_set", "slot_id": "INLINE-0:0"}
        selected = h.post("/api/select_collection_slot", body)
        assert selected.status == 200
        assert selected.json["active"]["repair"]
        for _ in range(2):
            snapshot = h.get("/api/collection_slots?dataset=cup_set&all=1").json
            assert snapshot["active"]["slot_id"] == "INLINE-0:0"
            assert snapshot["slots"][0]["episode"]["episode_index"] == 0

        h.runtime.collection_teleop_armed = True
        h.runtime.console_ctx.active_tab = "collect"
        assert h.post("/api/collect_start", {}).json["ok"]

        for status in ("queued", "saving"):
            queue[:] = [{"slot_id": "INLINE-0:0", "episode_index": 1, "status": status}]
            snapshot = h.get("/api/collection_slots?dataset=cup_set&all=1").json
            assert snapshot["slots"][0]["state"] == "saving"
            assert snapshot["active"]["slot_id"] == "INLINE-0:1"
            assert h.post("/api/select_collection_slot", body).status == 409

        queue.clear()
        episodes.append({**episodes[0], "episode_index": 1})
        snapshot = h.get("/api/collection_slots?dataset=cup_set&all=1").json
        assert snapshot["slots"][0]["state"] == "complete"
        assert snapshot["slots"][0]["episode"]["episode_index"] == 1
        assert snapshot["active"]["slot_id"] == "INLINE-0:1"

        # A later failed QC result must replace the earlier green attempt.
        episodes.append({**episodes[0], "episode_index": 2, "quality": "red"})
        snapshot = h.get("/api/collection_slots?dataset=cup_set&all=1").json
        assert snapshot["slots"][0]["episode"]["episode_index"] == 2
        assert snapshot["slots"][0]["episode"]["quality"] == "red"
        assert snapshot["counts"]["complete"] == 0
        assert snapshot["slots"][0]["state"] == "rejected"
        assert snapshot["active"]["slot_id"] == "INLINE-0:1"
        assert h.post("/api/select_collection_slot", body).json["active"]["repair"]
        assert (
            h.get("/api/collection_slots?dataset=cup_set&all=1").json["active"]["slot_id"]
            == "INLINE-0:0"
        )


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


def _write_export_dataset(
    output_dir: Path, source_dir: Path, episodes: tuple[int, ...] = (0, 2)
) -> None:
    """A converted copy plus the source it was made from."""
    rows = "".join(json.dumps({"episode_index": index}) + "\n" for index in episodes)
    for directory in (output_dir, source_dir):
        (directory / "meta").mkdir(parents=True, exist_ok=True)
        (directory / "meta" / "episodes.jsonl").write_text(rows)


def _register_completed_quality_export(
    ctx: ConsoleContext,
    source_dir: Path,
    output_dir: Path,
    *,
    dataset_format: str,
) -> None:
    ctx.quality_export_jobs["completed-export"] = console_server._QualityExportJob(
        job_id="completed-export",
        source_dir=str(source_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        dataset_format=dataset_format,
        state="completed",
    )


def test_collect_quality_export_writes_one_dataset_copy(tmp_path, monkeypatch):
    source = tmp_path / "pick_up_cup"
    source.mkdir()
    calls = []

    def export(source_dir, output_dir, *, dataset_format, replace_existing, progress_callback):
        assert dataset_format == "lerobot_v3"
        assert replace_existing is True
        calls.append((source_dir, output_dir))
        progress_callback(DatasetExportProgress(1, 3, 0))
        progress_callback(DatasetExportProgress(3, 3, 2))
        return DatasetExportSummary(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            dataset_format=dataset_format,
            episodes=3,
        )

    monkeypatch.setattr(console_server, "export_dataset", export)
    with serve_console(console_config()) as h:
        _set_collect_dataset_logger(h, source)
        response = h.post(
            "/api/collect_quality_export",
            {"task": "pick up cup", "dataset": "cup_set", "dataset_format": "lerobot_v3"},
        )
        assert response.status == 202
        assert response.json["dataset_format"] == "lerobot_v3"
        status = _wait_for_quality_job(h, "/api/collect_quality_export", response.json["job_id"])

    assert status.status == 200
    assert status.json["state"] == "completed"
    assert status.json["progress"] == 1.0
    assert status.json["episodes"] == 3
    assert calls[0][0] == source.resolve()
    assert calls[0][1] == source.with_name(f"{source.name}_lerobot_v3")


def test_collect_quality_upload_scan_skips_matching_files_and_never_uses_rejected_export(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "cup_set"
    source.mkdir(parents=True)
    exported = source.with_name(f"{source.name}_lerobot_v3")
    _write_export_dataset(exported, source)
    calls = []
    existing_remote_dir = "/datasets/arx_x5/cup_set"

    def scan(local_dir, specs):
        calls.append((local_dir, specs))
        assert local_dir == exported.resolve()
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
            h.runtime.console_ctx, source, exported, dataset_format="lerobot_v3"
        )

        first = h.post(
            "/api/collect_quality_upload",
            {"task": "pick up cup", "dataset_format": "lerobot_v3"},
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
    assert calls[0][0] == exported.resolve()
    assert calls[0][1][0].target == "/datasets/arx_x5/cup_set"
    receipt = json.loads(exported.with_name(f"{exported.name}_upload_receipt.json").read_text())
    assert receipt["source_dir"] == str(source.resolve())
    assert receipt["source_episode_indices"] == [0, 2]


def test_collect_quality_upload_serializes_confirmed_plans(tmp_path, monkeypatch):
    source = tmp_path / "cup_set"
    source.mkdir(parents=True)
    exported = source.with_name(f"{source.name}_lerobot_v3")
    _write_export_dataset(exported, source)
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
    body = {"task": "pick up cup", "dataset_format": "lerobot_v3"}
    with serve_console(config) as h:
        _set_collect_dataset_logger(h, source)
        _register_completed_quality_export(
            h.runtime.console_ctx, source, exported, dataset_format="lerobot_v3"
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


def test_a_retake_replaces_what_the_console_shows_for_its_slot(tmp_path, monkeypatch):
    """A retake overwrites its slot's episode at the same index, path and URL.

    The console must project the new take and stop trusting a cached copy of the
    one it replaced, or the retake looks like it never saved.
    """
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "work_dirs" / "review"
    data_dir = dataset_dir / "data" / "chunk-000"
    meta_dir = dataset_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    monkeypatch.setattr(handlers_utils, "_REPO_ROOT", repo_root)

    with serve_console(console_config(robot_type="ur5e")) as h:
        dim = h.runtime.robot.total_action_dim
        camera = h.runtime.robot.observation_schema.cameras[0].observation_key

        def write_take(frames: int) -> None:
            (meta_dir / "info.json").write_text(
                json.dumps(
                    {
                        "total_episodes": 1,
                        "fps": 10,
                        "features": {
                            "observation.state": {"dtype": "float32", "shape": [dim]},
                            "action": {"dtype": "float32", "shape": [dim]},
                            camera: {"dtype": "video", "shape": [16, 16, 3]},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (meta_dir / "episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "length": frames, "tasks": ["review"]}) + "\n",
                encoding="utf-8",
            )
            values = [np.full(dim, index, dtype=np.float32).tolist() for index in range(frames)]
            pq.write_table(
                pa.table(
                    {
                        "observation.state": values,
                        "action": values,
                        "timestamp": [index / 10 for index in range(frames)],
                        "frame_index": list(range(frames)),
                        "episode_index": [0] * frames,
                        "index": list(range(frames)),
                        "task_index": [0] * frames,
                    }
                ),
                data_dir / "episode_000000.parquet",
            )
            video = dataset_dir / "videos" / "chunk-000" / camera / "episode_000000.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(str(video), fps=10, codec="libx264", macro_block_size=1) as out:
                for index in range(frames):
                    out.append_data(np.full((16, 16, 3), index * 20, dtype=np.uint8))

        write_take(2)
        episode_body = {"dataset_dir": "work_dirs/review", "episode": 0}
        first = h.post("/api/review_episode", episode_body)
        assert first.status == 200
        assert first.json["frames"] == 2
        video_query = urlencode({"dataset_dir": "work_dirs/review", "episode": 0, "cam": camera})
        assert h.get(f"/api/replay_video?{video_query}").headers["cache-control"] == "no-cache"

        write_take(3)
        second = h.post("/api/review_episode", episode_body)
        assert second.status == 200
        assert second.json["frames"] == 3


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
            {"scene_id": "SC-002", "scene_round": 2},
        )
        command = h.runtime.command_queue.get_nowait()
        status = h.status()

    assert resp.json["ok"] is True
    assert command == "web:collect_start"
    assert status["collection_scene_id"] == "SC-002"
    assert status["collection_scene_round"] == 2


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
