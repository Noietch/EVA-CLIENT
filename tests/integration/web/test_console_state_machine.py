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
from _harness import MULTI_STRATEGY, console_config, serve_console

import robots  # noqa: F401  (registers ROBOT_REGISTRY incl. ur5e for single-arm UI tests)
from core.app import handlers
from core.app import run as app
from core.app.console import server as console_server
from core.app.console.server import ConsoleContext, _serialize_manual_scene, _serialize_scene
from core.app.handlers import utils as handlers_utils
from core.app.state import (
    OutputTarget,
    RuntimeState,
    SessionMode,
    SessionState,
    SessionStatus,
)
from core.config import ConfigDict


def test_run_before_setup_is_rejected(console):
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": "pick up the cup"})

    console.do("/api/run")  # no setup yet
    st = console.status()
    assert st["session_status"] == "unset"
    assert st["last_error"] == "Setup is required before running the next step"


def test_setup_without_prompt_or_mode_is_rejected(console):
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    # no prompt selected
    console.do("/api/setup")
    assert console.status()["is_setup_done"] is False


@pytest.mark.parametrize("console", [console_config(robot_type="ur5e")], indirect=True)
def test_single_gripper_config_exposes_one_gripper_control(console):
    controls = console.get("/api/config").json["gripper_controls"]
    assert controls == [{"side": "l", "label": "GRIPPER", "group": "arm"}]


@pytest.mark.parametrize(
    "console",
    [console_config(robot=dict(type="r1_lite", gripper_open=100.0, gripper_close=0.0))],
    indirect=True,
)
def test_r1lite_manual_qpos_limits_expose_full_gripper_range(console):
    limits = console.get("/api/config").json["manual_qpos_limits"]

    assert len(limits) == 14
    assert limits[0] == {"min": -3.2, "max": 3.2, "step": 0.005}
    assert limits[6] == {"min": 0.0, "max": 100.0, "step": 0.1}
    assert limits[13] == {"min": 0.0, "max": 100.0, "step": 0.1}


def test_reselecting_mode_clears_completed_setup(console):
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    console.do("/api/setup")
    assert console.status()["is_setup_done"] is True

    # Changing mode invalidates the prepared state — setup must be redone.
    console.do("/api/select_mode", {"mode": "real"})
    st = console.status()
    assert st["is_setup_done"] is False
    assert st["session_status"] == "unset"


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


def test_single_strategy_is_auto_selected_on_setup(console):
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    console.do("/api/setup")
    # Exactly one strategy configured -> chosen implicitly, no error.
    assert console.runtime.selected_inference_strategy_key == "sync"
    assert console.status()["is_setup_done"] is True


def test_tab_switch_clears_collection_replay(console):
    console.session.mode = SessionMode.COLLECT
    console.session.last_error = "Setup failed"
    console.runtime.collection_replay_qpos = np.zeros((2, 14), dtype=float)
    console.runtime.collection_replay_episode = 0

    console.post("/api/tab_switch", {"tab": "replay"})
    assert console.session.last_error == ""
    assert console.runtime.collection_replay_qpos is None
    assert console.runtime.collection_replay_episode is None

    console.pump()

    assert console.session.mode is SessionMode.SELECT
    assert console.session.last_error == ""
    assert console.runtime.collection_replay_qpos is None
    assert console.runtime.collection_replay_episode is None


def test_tab_switch_away_from_collect_stops_armed_teleop():
    calls = []

    class _Transport:
        def set_hil_relay_enabled(self, enabled):
            calls.append(f"set_hil_relay_enabled:{enabled}")

        def stop_collection(self):
            calls.append("stop_collection")

    runtime = SimpleNamespace(
        replay_source=None,
        collection_replay_qpos=None,
        collection_replay_episode=None,
        collection_teleop_active=True,
        transport=_Transport(),
        episode_logger=None,
        infer_strategy=None,
        policy=None,
        ik_solver=None,
        web_phase="idle",
        rollout_intervention_active=False,
        rl_active=False,
    )
    session = SessionState(mode=SessionMode.COLLECT, status=SessionStatus.READY)
    config = ConfigDict(collection=ConfigDict(gate=ConfigDict(enabled=False)))

    app.handle_command("web:tab_switch:manual", config, cast(RuntimeState, runtime), session)

    assert calls == ["set_hil_relay_enabled:False", "stop_collection"]
    assert runtime.collection_teleop_active is False
    assert session.mode is SessionMode.SELECT


@pytest.mark.parametrize(
    "console",
    [console_config(eval=ConfigDict(cli_mode="real", tasks=(), checkpoints=()))],
    indirect=True,
)
def test_tab_switch_to_eval_restores_config_mode(console):
    # An eval run is fixed to the config's cli_mode; the generic SELECT reset would make
    # START reject with "select mode first" since EVAL never exposes a mode picker.
    console.session.mode = SessionMode.SELECT
    console.post("/api/tab_switch", {"tab": "eval"})
    console.pump()
    assert console.session.mode is SessionMode.REAL


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
                tasks=(
                    ConfigDict(prompt_en="pick up the cup"),
                    ConfigDict(prompt_en="pour soybean"),
                ),
            )
        )
    ],
    indirect=True,
)
def test_eval_stop_can_switch_prompt_and_start_next_rollout(console):
    first_prompt = "pick up the cup"
    second_prompt = "pour soybean"
    console.do("/api/connect")
    console.do("/api/select_mode", {"mode": "sim"})
    console.do("/api/select_task", {"task": first_prompt})
    console.do("/api/setup")
    console.runtime.web_phase = "ready"

    console.do(
        "/api/eval_start",
        {"clip_id": "clip-1", "prompt": first_prompt, "trial": 1},
    )
    for _ in range(5):
        handlers.publish_next_action(console.config, console.runtime, console.session)
    console.do("/api/eval_stop")
    assert console.session.status is SessionStatus.READY
    assert console.session.step_index == 5
    assert console.runtime.needs_pre_start_reset is True

    console.do("/api/select_task", {"task": second_prompt})
    console.do(
        "/api/eval_start",
        {"clip_id": "clip-2", "prompt": second_prompt, "trial": 1},
    )

    assert console.status()["session_status"] == "running"
    assert console.session.selected_task == second_prompt
    assert console.session.step_index == 0
    assert console.runtime.needs_pre_start_reset is False


def test_collect_loop_uses_publish_rate_without_dataset_fps():
    config = console_config(
        publish_rate=10,
        collection=ConfigDict(
            enabled=True,
            schema=ConfigDict(),
        ),
    )
    assert (
        app._target_loop_rate_hz(config, cast(RuntimeState, SimpleNamespace(replay_source=None)))
        == 10
    )


def test_replay_loop_uses_replay_fps_on_live_transport():
    config = console_config(publish_rate=50)
    runtime = SimpleNamespace(
        replay_source=object(),
        replay_fps=12,
        transport=SimpleNamespace(is_offline=lambda: False),
    )
    assert app._target_loop_rate_hz(config, cast(RuntimeState, runtime)) == 12


def test_config_exposes_resolved_collection_dataset_dir():
    config = console_config(
        collection=ConfigDict(
            storage=ConfigDict(log_dir="work_dirs/collection/ur5e"),
            schema=ConfigDict(columns={"qpos": "observation.qpos"}),
        ),
    )

    with serve_console(config) as h:
        collection = h.get("/api/config").json["collection"]

    repo_root = Path(__file__).resolve().parents[3]
    assert collection["dataset_dir"] == str(repo_root / "work_dirs" / "collection" / "ur5e")


def test_collect_status_follows_selected_collect_task():
    class _Logger:
        def __init__(self):
            self.tasks = []

        def status_snapshot(self, task):
            self.tasks.append(task)
            return {
                "dataset_dir": f"/datasets/{task.replace(' ', '_')}",
                "episodes": [{"episode_index": 0, "status": "saved"}],
                "queue": [],
            }

    with serve_console(console_config()) as h:
        logger = _Logger()
        h.runtime.episode_logger = logger

        h.post("/api/select_collect_task", {"task": "pick up cup"})
        collect = h.status()["collect"]

    assert logger.tasks[-1] == "pick up cup"
    assert collect["dataset_dir"] == "/datasets/pick_up_cup"
    assert collect["episodes"] == [{"episode_index": 0, "status": "saved"}]


@pytest.mark.parametrize("verdict", ["pass", "fail", ""])
def test_collect_qc_mark_routes_to_active_collection_logger(verdict):
    calls = []

    class _Logger:
        def mark_collection_qc(self, task, episode, qc_verdict, note):
            calls.append((task, episode, qc_verdict, note))
            return True

    with serve_console(console_config()) as h:
        h.runtime.episode_logger = cast(Any, _Logger())
        h.session.selected_collect_task = "pick up cup"

        resp = h.post(
            "/api/collect_qc_mark",
            {
                "task": "pick up cup",
                "episode": 3,
                "verdict": verdict,
                "note": "checked",
            },
        )

    assert resp.status == 200
    assert resp.json == {"ok": True, "episode": 3, "verdict": verdict}
    assert calls == [("pick up cup", 3, verdict, "checked")]


def test_collect_qc_mark_rejects_unsupported_verdict():
    with serve_console(console_config()) as h:
        resp = h.post(
            "/api/collect_qc_mark",
            {"episode": 0, "verdict": "approve", "note": ""},
        )

    assert resp.status == 400
    assert resp.json == {"ok": False, "error": "unsupported QC verdict: approve"}


@pytest.mark.parametrize("episode", [None, -1, 1.5, "1.5", "²", {}])
def test_collect_qc_mark_rejects_invalid_episode(episode):
    with serve_console(console_config()) as h:
        resp = h.post(
            "/api/collect_qc_mark",
            {"task": "unset", "episode": episode, "verdict": "pass", "note": ""},
        )

    assert resp.status == 400
    assert resp.json == {"ok": False, "error": "collection episode must be a non-negative integer"}


def test_collect_qc_mark_rejects_when_collection_logger_is_unavailable():
    with serve_console(console_config()) as h:
        resp = h.post(
            "/api/collect_qc_mark",
            {"task": "pick up cup", "episode": 0, "verdict": "pass", "note": ""},
        )

    assert resp.status == 409
    assert resp.json == {"ok": False, "error": "collection recording is unavailable"}


def test_collect_qc_mark_rejects_episode_that_is_not_saved():
    class _Logger:
        def mark_collection_qc(self, task, episode, verdict, note):
            return False

    with serve_console(console_config()) as h:
        h.runtime.episode_logger = cast(Any, _Logger())
        h.session.selected_collect_task = "pick up cup"

        resp = h.post(
            "/api/collect_qc_mark",
            {
                "task": "pick up cup",
                "episode": 4,
                "verdict": "fail",
                "note": "pending",
            },
        )

    assert resp.status == 409
    assert resp.json == {"ok": False, "error": "collection episode is not saved"}


def test_collect_qc_mark_rejects_review_from_previous_collection_task():
    calls = []

    class _Logger:
        def mark_collection_qc(self, task, episode, verdict, note):
            calls.append((task, episode, verdict, note))
            return True

    with serve_console(console_config()) as h:
        h.runtime.episode_logger = cast(Any, _Logger())
        h.session.selected_collect_task = "place cup"

        resp = h.post(
            "/api/collect_qc_mark",
            {
                "task": "pick up cup",
                "episode": 0,
                "verdict": "pass",
                "note": "reviewed before task switch",
            },
        )

    assert resp.status == 409
    assert resp.json == {"ok": False, "error": "collection review task is no longer active"}
    assert calls == []


def test_status_exposes_replay_action_key_for_series_cache():
    with serve_console(console_config()) as h:
        h.runtime.replay_action_key = "action_eef"

        assert h.status()["replay_action_key"] == "action_eef"


def test_inspect_dataset_resolves_relative_dataset_dir(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "work_dirs" / "collection" / "ur5e"
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    info = {
        "total_episodes": 3,
        "features": {
            "observation.qpos": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(handlers_utils, "_REPO_ROOT", repo_root)
    monkeypatch.chdir(elsewhere)

    with serve_console(console_config()) as h:
        resp = h.post("/api/inspect_dataset", {"dataset_dir": "work_dirs/collection/ur5e"})

    assert resp.json["ok"] is True
    assert resp.json["n_episodes"] == 3
    assert resp.json["keys"]["state"]["default"] == "observation.qpos"


def test_load_replay_dataset_mounts_episode_before_response(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "work_dirs" / "replay"
    data_dir = dataset_dir / "data" / "chunk-000"
    meta_dir = dataset_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    monkeypatch.setattr(handlers_utils, "_REPO_ROOT", repo_root)

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
                        "observation.images.cam_high": {
                            "dtype": "video",
                            "shape": [480, 640, 3],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "length": 2, "tasks": ["replay"]}) + "\n",
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

        resp = h.post(
            "/api/load_replay_dataset",
            {
                "dataset_dir": "work_dirs/replay",
                "episode": 0,
                "keys": {"state": "observation.state", "action": "action"},
                "video_keys": {"cam_high": "observation.images.cam_high"},
                "action_mode": "joint",
            },
        )

        assert resp.status == 200
        assert resp.json["ok"] is True
        assert resp.json["episode"] == 0
        assert resp.json["frames"] == 2
        assert resp.json["dataset_dir"] == str(dataset_dir)
        assert resp.json["video_keys"] == {"cam_high": "observation.images.cam_high"}
        assert h.runtime.replay_source is not None
        assert h.runtime.command_queue.empty()


def test_replay_video_resolves_relative_dataset_episode_video(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "work_dirs" / "collection" / "ur5e"
    meta_dir = dataset_dir / "meta"
    video_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.cam_high"
    meta_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    info = {
        "total_episodes": 8,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/{episode_file}.mp4",
        "features": {
            "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 7, "file_stem": "prompt_a_ep_000007"}) + "\n",
        encoding="utf-8",
    )
    payload = b"episode-7-video"
    (video_dir / "prompt_a_ep_000007.mp4").write_bytes(payload)
    monkeypatch.setattr(handlers_utils, "_REPO_ROOT", repo_root)

    path = "/api/replay_video?dataset_dir=work_dirs/collection/ur5e&episode=7&cam=cam_high"
    with serve_console(console_config()) as h:
        resp = h.get(path)
        suffix = h.get(path, headers={"Range": "bytes=-5"})

    assert resp.status == 200
    assert resp.headers["content-type"] == "video/mp4"
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.headers["cache-control"] == "public, max-age=3600"
    assert resp.raw == payload
    assert suffix.status == 206
    assert suffix.headers["content-range"] == (
        f"bytes {len(payload) - 5}-{len(payload) - 1}/{len(payload)}"
    )
    assert suffix.raw == payload[-5:]


def test_replay_poster_resolves_relative_dataset_episode_video(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "work_dirs" / "collection" / "ur5e"
    meta_dir = dataset_dir / "meta"
    video_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.cam_high"
    meta_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    info = {
        "total_episodes": 8,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/{episode_file}.mp4",
        "features": {
            "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 7, "file_stem": "prompt_a_ep_000007"}) + "\n",
        encoding="utf-8",
    )
    video_path = video_dir / "prompt_a_ep_000007.mp4"
    video_path.write_bytes(b"episode-7-video")
    monkeypatch.setattr(handlers_utils, "_REPO_ROOT", repo_root)

    class FakeCapture:
        def __init__(self, path: str) -> None:
            self.path = path

        def read(self):
            assert self.path == str(video_path)
            return True, np.zeros((2, 3, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr(console_server.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(
        console_server.cv2,
        "imencode",
        lambda ext, frame, args: (True, np.frombuffer(b"jpeg-frame", dtype=np.uint8)),
    )

    path = "/api/replay_poster?dataset_dir=work_dirs/collection/ur5e&episode=7&cam=cam_high"
    with serve_console(console_config()) as h:
        resp = h.get(path)

    assert resp.status == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.headers["cache-control"] == "public, max-age=3600"
    assert resp.raw == b"jpeg-frame"


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


def test_client_trace_is_logged_without_enqueuing_command(console, caplog):
    with caplog.at_level("INFO", logger="core.app.console.server"):
        resp = console.post(
            "/api/client_trace",
            {
                "client_id": "browser-a",
                "seq": 7,
                "event": "review.video.error",
                "details": {"episode": 3, "camera": "cam_high"},
            },
        )

    assert resp.status == 200
    assert resp.json == {"ok": True}
    assert console.runtime.command_queue.empty()
    assert "[UI_TRACE]" in caplog.text
    assert "client=browser-a" in caplog.text
    assert "seq=7" in caplog.text
    assert "event=review.video.error" in caplog.text
    assert '"episode":3' in caplog.text


def test_scene_poll_returns_cached_frame_while_batch_transforms_are_loading(console):
    cached = {"available": True, "arms": {"cached": {}}}
    ctx = console_server.ConsoleRequestHandler.ctx
    ctx.scene_cache = cached
    ctx.scene_batch_loading = threading.Event()
    ctx.scene_batch_loading.set()

    class ExplodingScene:
        def transforms(self, _qpos):
            raise AssertionError("live scene must not contend with batch transforms")

    ctx.scene = ExplodingScene()

    resp = console.get("/api/scene")

    assert resp.status == 200
    assert resp.json == cached


def test_native_transform_batch_uses_isolated_worker(monkeypatch):
    qpos = np.zeros((3, 14), dtype=np.float32)
    calls = []

    class Future:
        def result(self, timeout):
            assert timeout == 120
            return b"worker-transform-blob"

    class Executor:
        def submit(self, fn, *args):
            calls.append((fn, args))
            return Future()

    class NativeScene:
        def all_transforms_blob(self, _qpos):
            raise AssertionError("native scene must not compute the batch in the web process")

    monkeypatch.setattr(console_server, "_get_transform_executor", lambda: Executor())
    monkeypatch.setattr(console_server, "_is_native_urdf_scene", lambda _scene: True)

    raw = console_server._compute_transform_blob(NativeScene(), qpos, "r1_lite", 100.0, 0.0)

    assert raw == b"worker-transform-blob"
    assert len(calls) == 1
    assert calls[0][1][0:3] == ("r1_lite", 100.0, 0.0)
    np.testing.assert_array_equal(calls[0][1][3], qpos)


def test_transform_worker_prewarm_is_non_blocking(monkeypatch):
    submitted = []

    class Future:
        def add_done_callback(self, callback):
            submitted.append(callback)

    class Executor:
        def submit(self, fn, *args):
            submitted.append((fn, args))
            return Future()

    monkeypatch.setattr(console_server, "_get_transform_executor", lambda: Executor())
    config = SimpleNamespace(
        robot=SimpleNamespace(type="r1_lite", gripper_open=100.0, gripper_close=0.0)
    )
    runtime = SimpleNamespace(robot=SimpleNamespace(total_action_dim=14))

    console_server._prewarm_transform_worker(config, runtime)

    assert len(submitted) == 2
    assert submitted[0][1][0:3] == ("r1_lite", 100.0, 0.0)
    assert submitted[0][1][3].shape == (0, 14)


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
        console_server.ConsoleRequestHandler.ctx.scene = scene

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


def test_rollout_review_is_stateless_local_playback(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "work_dirs" / "review"
    data_dir = dataset_dir / "data" / "chunk-000"
    meta_dir = dataset_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    monkeypatch.setattr(handlers_utils, "_REPO_ROOT", repo_root)

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
        resp = h.post(
            "/api/review_episode",
            {"dataset_dir": "work_dirs/review", "episode": 0},
        )

        assert resp.status == 200
        assert resp.json["ok"] is True
        assert resp.json["frames"] == 2
        assert h.runtime.collection_replay_qpos is sentinel_qpos
        assert h.runtime.collection_replay_started == 0.0


def test_collection_replay_status_stops_playing_at_last_frame():
    with serve_console(console_config()) as h:
        h.runtime.collection_replay_qpos = np.zeros((2, 3), dtype=np.float32)
        h.runtime.collection_replay_episode = 0
        h.runtime.collection_replay_fps = 10
        h.runtime.collection_replay_started = time.monotonic()

        replay = h.status()["collection_replay"]
        assert replay["active"] is True
        assert replay["playing"] is True
        assert replay["frame_index"] == 0

        h.runtime.collection_replay_started = time.monotonic() - 1.0
        replay = h.status()["collection_replay"]
        assert replay["active"] is True
        assert replay["playing"] is False
        assert replay["frame_index"] == 1


def test_collect_step_collection_capture_is_background_only():
    class _Transport:
        def __init__(self):
            self.acquired = 0

        def acquire_collection_raw(self):
            self.acquired += 1
            return object()

    class _Logger:
        is_collection_enabled = True

        def ingest_collection_snapshot(self, snapshot):
            raise AssertionError("collection-enabled collect_step must not ingest")

    transport = _Transport()
    runtime = SimpleNamespace(episode_logger=_Logger(), transport=transport)

    assert not handlers.collect_step(console_config(), cast(RuntimeState, runtime))
    assert transport.acquired == 0


def test_collect_start_runs_collection_capture_in_background():
    snapshots = [object(), object(), object()]
    ingested = []

    class _Transport:
        def supports_collection(self):
            return True

        def start_collection(self):
            return None

        def reset_hil_control(self):
            return None

        def set_hil_relay_enabled(self, enabled):
            return None

        def clear_collection_backlog(self):
            return 42.0

        def acquire_collection_raw(self):
            if snapshots:
                return snapshots.pop(0)
            return None

    class _Logger:
        is_collection_enabled = True
        has_active_episode = True

        def is_queue_full(self):
            return False

        def start_episode(self, *, task, collection_min_capture_time=None):
            return None

        def ingest_collection_snapshot(self, snapshot):
            ingested.append(snapshot)

        def end_episode(self):
            return True

    runtime = SimpleNamespace(
        episode_logger=_Logger(),
        transport=_Transport(),
        collection_replay_qpos=None,
        collection_replay_episode=None,
        collection_teleop_armed=True,
        collection_teleop_active=False,
        last_collection_timestamp=None,
    )
    session = SessionState(selected_collect_task="pick")
    config = console_config(inference_cfg=ConfigDict(publish_rate=200))

    assert handlers.collect_start(config, cast(RuntimeState, runtime), session)
    deadline = time.monotonic() + 1.0
    while len(ingested) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    handlers.collect_stop(config, cast(RuntimeState, runtime), session)

    assert len(ingested) == 3


def test_collect_start_clears_backlog_and_passes_cutoff_to_logger():
    calls = []

    class _Transport:
        def supports_collection(self):
            return True

        def start_collection(self):
            calls.append("start_collection")

        def reset_hil_control(self):
            calls.append("reset_hil_control")

        def set_hil_relay_enabled(self, enabled):
            calls.append(("set_hil_relay_enabled", enabled))

        def clear_collection_backlog(self):
            calls.append("clear_collection_backlog")
            return 42.0

        def acquire_collection_raw(self):
            return None

    class _Logger:
        is_collection_enabled = True

        def is_queue_full(self):
            return False

        def start_episode(self, *, task, collection_min_capture_time=None):
            calls.append(("start_episode", task, collection_min_capture_time))

    runtime = SimpleNamespace(
        episode_logger=_Logger(),
        transport=_Transport(),
        collection_replay_qpos=np.zeros((1, 2), dtype=np.float32),
        collection_replay_episode=7,
        collection_teleop_armed=True,
        collection_teleop_active=False,
        last_collection_timestamp=None,
    )
    session = SessionState(selected_collect_task="pick")
    config = console_config(inference_cfg=ConfigDict(publish_rate=200))

    assert handlers.collect_start(config, cast(RuntimeState, runtime), session)
    try:
        assert calls == [
            "reset_hil_control",
            ("set_hil_relay_enabled", True),
            "start_collection",
            "clear_collection_backlog",
            ("start_episode", "pick", 42.0),
        ]
        assert session.status is SessionStatus.RUNNING
        assert runtime.collection_replay_qpos is None
        assert runtime.collection_replay_episode is None
    finally:
        runtime.collection_capture_runner.stop()
        runtime.collection_capture_runner = None


def test_collect_start_requires_activation_gate():
    calls = []

    class _Transport:
        def supports_collection(self):
            return True

        def start_collection(self):
            calls.append("start_collection")

    class _Logger:
        is_collection_enabled = True

        def is_queue_full(self):
            return False

        def start_episode(self, *, task, collection_min_capture_time=None):
            calls.append("start_episode")

    runtime = SimpleNamespace(
        episode_logger=_Logger(),
        transport=_Transport(),
        collection_replay_qpos=None,
        collection_replay_episode=None,
        collection_teleop_armed=False,
        collection_teleop_active=False,
    )
    session = SessionState(selected_collect_task="pick")
    config = ConfigDict(collection=ConfigDict(gate=ConfigDict(enabled=False)))

    assert not handlers.collect_start(config, cast(RuntimeState, runtime), session)
    assert calls == []
    assert session.last_error == "Collection teleop requires COLLECT activation"


def test_collect_start_api_requires_collect_tab_activation():
    with serve_console(console_config()) as h:
        resp = h.post("/api/collect_start")
        assert resp.json["ok"] is False
        assert h.status()["collection_teleop_armed"] is False

        h.post("/api/tab_switch", {"tab": "collect"})
        assert h.status()["collection_teleop_armed"] is False
        resp = h.post("/api/collect_start")
        assert resp.json["ok"] is False

        h.post("/api/tab_switch", {"tab": "collect", "collect_teleop_armed": True})
        assert h.status()["collection_teleop_armed"] is True


def test_unknown_route_returns_404(console):
    assert console.get("/api/does_not_exist").status == 404
    assert console.post("/api/also_missing").status == 404


def test_setup_fails_cleanly_when_policy_unreachable():
    """A real (non-mock) policy server that isn't listening must fail setup with a
    clear error, not hang or crash — the operator needs to know the server is down.
    Console setup uses fail_fast (one attempt, no 30s retry block)."""
    from _harness import console_config, serve_console

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


# --- lifecycle: thread leaks, counter resets, halt, run cycles ---


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


def test_step_sim_chunk_updates_preview_for_each_action():
    actions = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )
    session = SessionState(
        mode=SessionMode.STEP,
        status=SessionStatus.READY,
        is_setup_done=True,
        selected_task="pick up the cup",
        action_chunk=actions.copy(),
    )

    class _Transport:
        def __init__(self) -> None:
            self.published: list[tuple[str, np.ndarray]] = []
            self.preview_snapshots: list[np.ndarray] = []

        def publish_action(self, action: np.ndarray, target: str = "real") -> None:
            self.published.append((target, np.asarray(action, dtype=np.float32).copy()))

        def create_rate(self, _hz: float):
            transport = self

            class _Rate:
                def sleep(self) -> None:
                    transport.preview_snapshots.append(session.sim_preview_qpos.copy())

            return _Rate()

    transport = _Transport()
    runtime = SimpleNamespace(transport=transport)
    config = ConfigDict(inference_cfg=ConfigDict(publish_rate=10))

    assert handlers.run_one_chunk_on_sim(config, cast(RuntimeState, runtime), session)

    assert [target for target, _ in transport.published] == ["sim"] * len(actions)
    np.testing.assert_allclose(np.asarray(transport.preview_snapshots), actions)
    np.testing.assert_allclose(session.pending_real_chunk, actions)
    np.testing.assert_allclose(session.sim_preview_qpos, actions[-1])
    assert session.chunk_index == len(actions)
    assert session.status is SessionStatus.READY


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


def test_manual_send_publishes_gripper_target_without_joint_step_scaling():
    class _Rate:
        def sleep(self) -> None:
            pass

    class _Transport:
        def __init__(self) -> None:
            self.qpos = np.zeros(14, dtype=np.float32)
            self.published: list[tuple[str, np.ndarray]] = []

        def get_latest_qpos(self) -> np.ndarray:
            return self.qpos.copy()

        def publish_action(self, action: np.ndarray, target: str = "real") -> None:
            sent = np.asarray(action, dtype=np.float32).copy()
            self.published.append((target, sent))
            if target == "real":
                self.qpos = sent

        def create_rate(self, _hz: float) -> _Rate:
            return _Rate()

    transport = _Transport()
    robot = SimpleNamespace(
        gripper_mask=tuple(i in {6, 13} for i in range(14)),
    )
    runtime = SimpleNamespace(
        robot=robot,
        transport=transport,
        command_queue=queue.Queue(),
        prompt_ready=None,
        web_phase="idle",
    )
    target = np.zeros(14, dtype=np.float32)
    target[6] = 100.0
    session = SessionState(
        mode=SessionMode.MANUAL,
        status=SessionStatus.UNSET,
        manual_qpos=target,
        manual_real_qpos=np.zeros(14, dtype=np.float32),
    )
    config = ConfigDict(inference_cfg=ConfigDict(publish_rate=100))

    app.handle_command("web:manual_send", config, cast(RuntimeState, runtime), session)

    assert len(transport.published) == 1
    assert transport.published[-1][0] == "real"
    assert transport.published[-1][1][6] == pytest.approx(100.0)
    np.testing.assert_allclose(session.manual_real_qpos, target)


def test_manual_send_status_returns_to_idle_after_publish(console):
    console.session.mode = SessionMode.MANUAL
    console.session.manual_qpos = np.asarray(console.runtime.robot.initial_qpos, dtype=np.float32)

    resp = console.post("/api/manual_send")

    assert resp.json == {"ok": True}
    assert console.status()["manual_publish_active"] is True
    console.pump()
    assert console.status()["manual_publish_active"] is False


def test_manual_scene_payloads_keep_eased_mesh_updates():
    class _Scene:
        def transforms(self, _qpos: np.ndarray | None) -> dict:
            return {"body": {}}

    class _Transport:
        def get_latest_qpos(self) -> np.ndarray:
            return np.asarray([0.25, -0.25], dtype=np.float32)

    runtime = SimpleNamespace(
        transport=_Transport(),
        robot=SimpleNamespace(initial_qpos=np.zeros(2, dtype=np.float32)),
        replay_source=None,
        collection_replay_qpos=None,
        collection_replay_started=0.0,
        collection_replay_fps=10,
    )
    session = SessionState(
        mode=SessionMode.MANUAL,
        manual_qpos=np.asarray([0.5, -0.5], dtype=np.float32),
    )
    ctx = ConsoleContext(
        config=ConfigDict(),
        runtime=cast(RuntimeState, runtime),
        session=session,
        scene=_Scene(),
    )

    assert "instant" not in _serialize_scene(ctx)
    assert "instant" not in _serialize_manual_scene(ctx)


def test_halt_keeps_active_episode_open_for_continue():
    class _Logger:
        def __init__(self) -> None:
            self.ended = 0

        def end_episode(self) -> None:
            self.ended += 1

    logger_obj = _Logger()
    runtime = SimpleNamespace(
        web_phase="idle",
        replay_source=None,
        episode_logger=logger_obj,
        rollout_episode_logger=None,
    )
    session = SessionState(
        mode=SessionMode.SIM,
        status=SessionStatus.RUNNING,
        is_setup_done=True,
        selected_task="pick up the cup",
        step_index=5,
    )

    app.handle_command("web:halt", ConfigDict(), cast(RuntimeState, runtime), session)

    assert session.status is SessionStatus.READY
    assert logger_obj.ended == 0


def test_run_clears_collection_replay_preview():
    runtime = SimpleNamespace(
        web_phase="idle",
        replay_source=None,
        infer_strategy=None,
        collection_replay_qpos=np.zeros((2, 14), dtype=float),
        collection_replay_episode=1,
    )
    session = SessionState(
        mode=SessionMode.SIM,
        status=SessionStatus.READY,
        is_setup_done=True,
        selected_task="pick up the cup",
        step_index=5,
    )

    app.handle_command("web:run", ConfigDict(), cast(RuntimeState, runtime), session)

    assert runtime.collection_replay_qpos is None
    assert runtime.collection_replay_episode is None


def test_run_uses_rollout_logger_without_opening_collection_logger(monkeypatch):
    calls = []
    rollout_logger = object()
    runtime = SimpleNamespace(
        rollout_episode_logger=None,
        replay_source=None,
    )
    session = SessionState(
        mode=SessionMode.REAL,
        status=SessionStatus.READY,
        is_setup_done=True,
        selected_task="pick up the cup",
    )
    monkeypatch.setattr(app, "_select_only_inference_strategy_if_needed", lambda *args: True)
    monkeypatch.setattr(app, "run_warmup_and_start", lambda *args: True)
    monkeypatch.setattr(app, "is_replay", lambda runtime: False)

    def _begin_rollout(config, runtime, session):
        calls.append("rollout")
        runtime.rollout_episode_logger = rollout_logger

    monkeypatch.setattr(app, "begin_rollout_save_episode", _begin_rollout)
    monkeypatch.setattr(app, "start_episode", lambda *args: calls.append("collection"))

    app._dispatch_run(ConfigDict(), cast(RuntimeState, runtime), session)

    assert calls == ["rollout"]
    assert session.status is SessionStatus.RUNNING


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


def test_eval_start_skips_setup_when_already_warmed(monkeypatch):
    """SETUP (web:warmup) homes + validates; a following RUN (web:start) must NOT re-home
    when nothing since left the arm un-reset — otherwise the SETUP key buys nothing."""
    setup_calls = {"n": 0}

    def _fake_setup(config, runtime, session, fail_fast=False):
        setup_calls["n"] += 1
        session.is_setup_done = True
        session.status = SessionStatus.READY
        return True

    def _fake_dispatch(config, runtime, session):
        session.status = SessionStatus.RUNNING

    monkeypatch.setattr(app, "run_setup", _fake_setup)
    monkeypatch.setattr(app, "_dispatch_run", _fake_dispatch)

    runtime = SimpleNamespace(
        web_phase="ready",
        replay_source=None,
        episode_logger=None,
        current_cell={"prompt": "pick up the cup", "trial": 1},
        current_clip_id=None,
        needs_pre_start_reset=False,
        infer_strategy=None,
        policy=None,
        ik_solver=None,
    )
    session = SessionState(
        mode=SessionMode.REAL,
        status=SessionStatus.READY,
        is_setup_done=True,  # already warmed by a prior SETUP
        selected_task="pick up the cup",
    )

    app.handle_command("web:start", ConfigDict(), cast(RuntimeState, runtime), session)

    assert setup_calls["n"] == 0  # skipped the redundant home + validate
    assert session.status is SessionStatus.RUNNING


def test_eval_start_runs_setup_when_not_warmed(monkeypatch):
    """The skip is gated: a RUN with no prior SETUP or an explicit deferred reset
    must still run setup so the home trajectory is sent before publishing."""
    setup_calls = {"n": 0}

    def _fake_setup(config, runtime, session, fail_fast=False):
        setup_calls["n"] += 1
        session.is_setup_done = True
        session.status = SessionStatus.READY
        return True

    monkeypatch.setattr(app, "run_setup", _fake_setup)
    monkeypatch.setattr(
        app, "_dispatch_run", lambda *a: setattr(a[2], "status", SessionStatus.RUNNING)
    )

    runtime = SimpleNamespace(
        web_phase="ready",
        replay_source=None,
        episode_logger=None,
        current_cell={"prompt": "pick up the cup", "trial": 1},
        current_clip_id=None,
        needs_pre_start_reset=True,
    )
    session = SessionState(
        mode=SessionMode.REAL,
        status=SessionStatus.READY,
        is_setup_done=True,
        selected_task="pick up the cup",
    )

    app.handle_command("web:start", ConfigDict(), cast(RuntimeState, runtime), session)

    assert setup_calls["n"] == 1


def test_eval_start_binds_cell_prompt_before_task_gate(monkeypatch):
    def _fake_setup(config, runtime, session, fail_fast=False):
        session.is_setup_done = True
        session.status = SessionStatus.READY
        return True

    monkeypatch.setattr(app, "run_setup", _fake_setup)
    monkeypatch.setattr(
        app, "_dispatch_run", lambda *a: setattr(a[2], "status", SessionStatus.RUNNING)
    )

    runtime = SimpleNamespace(
        web_phase="ready",
        replay_source=None,
        episode_logger=None,
        current_cell={"prompt": "pick up the cup", "trial": 1},
        current_clip_id=None,
        needs_pre_start_reset=False,
        infer_strategy=None,
        policy=None,
        ik_solver=None,
    )
    session = SessionState(
        mode=SessionMode.REAL,
        status=SessionStatus.READY,
        is_setup_done=True,
        selected_task=None,
    )

    app.handle_command("web:start", ConfigDict(), cast(RuntimeState, runtime), session)

    assert session.selected_task == "pick up the cup"
    assert session.status is SessionStatus.RUNNING


def test_eval_stop_during_running_motion_is_requeued_finalized_and_left_ready():
    command_queue: queue.Queue[str] = queue.Queue()

    class _Logger:
        def __init__(self) -> None:
            self.ended = 0

        def end_episode(self) -> None:
            self.ended += 1

    logger_obj = _Logger()
    runtime = SimpleNamespace(
        web_phase="running",
        command_queue=command_queue,
        episode_logger=logger_obj,
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


def test_interrupting_verb_during_reset_is_requeued_not_dropped(console):
    """A motion-interrupting verb seen mid-reset must abort the motion AND be re-queued
    (not silently dropped) so the main loop applies it once the reset unwinds.

    Driven through handle_motion_command (the unit poll_motion_commands invokes per
    action) rather than reset_with_setup, so this test stays focused on the
    interrupt/re-queue contract. consume_motion_interrupt then clears
    interrupt_requested before the reset would return, so it reads False after.
    """
    console.do("/api/select_mode", {"mode": "real"})
    console.do("/api/select_task", {"task": "pick up the cup"})
    s = console.session

    interrupted = handlers.handle_motion_command(
        "web:select_mode:sim", console.config, console.runtime, s
    )
    assert interrupted is True
    assert s.interrupt_requested is True
    assert "web:select_mode:sim" in list(console.runtime.command_queue.queue)

    handlers.consume_motion_interrupt(s, SessionStatus.UNSET)
    assert s.interrupt_requested is False
    assert s.status is SessionStatus.UNSET

    console.pump()
    assert s.mode.value == "sim"


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


def test_sync_wait_can_ignore_gripper_tracking_error():
    class _Rate:
        def __init__(self):
            self.sleeps = 0

        def sleep(self):
            self.sleeps += 1

    class _Transport:
        def __init__(self):
            self.rate = _Rate()

        def create_rate(self, publish_rate):
            del publish_rate
            return self.rate

        def get_latest_qpos(self):
            return np.asarray([1.0, 0.444], dtype=np.float32)

    transport = _Transport()
    runtime = SimpleNamespace(
        transport=transport,
        robot=SimpleNamespace(gripper_mask=(False, True)),
    )

    reached = handlers._wait_until_reached(
        cast(RuntimeState, runtime),
        np.asarray([1.0, 0.0], dtype=np.float32),
        publish_rate=10,
        threshold=0.05,
        max_ticks=5,
        ignore_gripper_error=True,
    )

    assert reached is True
    assert transport.rate.sleeps == 0
