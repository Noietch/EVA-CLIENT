from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from core.app import handlers
from core.app.local_history import LocalHistoryStore
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def _config(tmp_path):
    config = console_config()
    config.console.session_history = True
    config.console.history_dir = str(tmp_path / ".eva-client")
    config.transport.image_height = 32
    config.transport.image_width = 32
    config.inference_cfg.setup_warmup_chunks = 0
    config.rollout.storage.enabled = True
    config.rollout.storage.log_dir = str(tmp_path / "legacy-rollouts")
    config.rollout.storage.async_save = False
    config.rollout.storage.fps = 10
    return config


def _prepare(h, mode="real"):
    h.do("/api/select_mode", {"mode": mode})
    h.do("/api/select_task", {"task": "pick up the cup"})
    h.do("/api/select_strategy", {"strategy": "sync"})
    h.do("/api/setup")
    assert h.status()["is_setup_done"], h.status()["last_error"]


def _publish(h, count=8):
    for _ in range(count):
        handlers.publish_next_action(h.runtime.active_config or h.config, h.runtime, h.session)
    time.sleep(0.03)


def test_stop_and_continue_stay_in_one_run_until_new_run_is_selected(tmp_path):
    config = _config(tmp_path)
    with serve_console(config) as h:
        assert h.get("/api/local_sessions").json["sessions"] == []
        assert not Path(config.console.history_dir).exists()
        _prepare(h)
        h.do("/api/run")
        _publish(h, 3)
        assert h.runtime.rollout_episode_logger is None
        h.do("/api/halt")
        sessions = h.get("/api/local_sessions").json["sessions"]
        assert len(sessions) == 1
        session_id = sessions[0]["id"]
        first = h.get(f"/api/local_session?id={session_id}").json["session"]["rollouts"][0]
        assert first["prompt"] == "pick up the cup"
        assert first["steps"] == 3
        assert first["record"] is False
        assert first["status"] == "stopped"
        assert not (Path(config.console.history_dir) / "rollouts").exists()
        h.do("/api/run")
        _publish(h, 2)
        h.do("/api/halt")
        saved = h.get("/api/local_sessions").json["sessions"]
        assert len(saved) == 1
        assert all(item["rollouts"] == 1 for item in saved)
        assert saved[0]["runs"][0]["id"] == first["id"]
        assert saved[0]["runs"][0]["steps"] == 5
        h.do("/api/run")
        _publish(h, 2)
        h.do("/api/halt")
        saved = LocalHistoryStore(config.console.history_dir).list()
        assert len(saved) == 1
        assert saved[0]["runs"][0]["steps"] == 7
        qpos = h.runtime.transport.get_latest_qpos().copy()
        assert h.do("/api/new_debug_run").status == 200
        assert h.status()["selected_task"] is None
        assert h.status()["local_history"]["current_rollout_id"] is None
        assert h.status()["cli_mode"] == "real"
        assert h.status()["selected_strategy"] == "sync"
        np.testing.assert_array_equal(h.runtime.transport.get_latest_qpos(), qpos)
        assert len(LocalHistoryStore(config.console.history_dir).list()) == 1
        _prepare(h)
        h.do("/api/run")
        _publish(h, 2)
        h.do("/api/halt")
        saved = LocalHistoryStore(config.console.history_dir).list()
        assert len(saved) == 2
        assert len({row["runs"][0]["id"] for row in saved}) == 2
        assert not (tmp_path / "legacy-rollouts").exists()


def test_recorded_rollout_is_saved_under_home_store_and_review_is_read_only(tmp_path):
    config = _config(tmp_path)
    with serve_console(config) as h:
        assert h.do("/api/debug_record", {"enabled": True}).status == 200
        _prepare(h)
        h.do("/api/run")
        _publish(h, 12)
        assert h.post("/api/debug_record", {"enabled": False}).status == 409
        assert (
            h.post(
                "/api/delete_local_run",
                {
                    "session_id": h.runtime.local_session_id,
                    "rollout_id": h.runtime.local_rollout_id,
                },
            ).status
            == 409
        )
        h.do("/api/halt")
        h.runtime.rollout_episode_logger.finalize()
        summaries = h.get("/api/local_sessions").json["sessions"]
        data = h.get("/api/local_session?id=" + summaries[0]["id"]).json["session"]
        record = data["rollouts"][0]
        assert record["record_status"] == "saved", record
        assert record["playable"] is True
        assert Path(record["dataset_dir"]).is_relative_to(Path(config.console.history_dir))
        qpos = h.runtime.transport.get_latest_qpos().copy()
        response = h.post(
            "/api/local_replay", {"session_id": data["id"], "rollout_id": record["id"]}
        )
        assert response.status == 200, response.json
        assert response.json["frames"] > 0
        assert response.json["video_keys"]
        np.testing.assert_array_equal(h.runtime.transport.get_latest_qpos(), qpos)
        assert h.runtime.command_queue.empty()
        payload = {"session_id": data["id"], "rollout_id": record["id"]}
        assert h.post("/api/delete_local_run", payload).status == 200
        assert h.get("/api/local_sessions").json["sessions"] == []
        assert h.post("/api/local_replay", payload).status == 404
        assert h.post("/api/restore_local_run", payload).status == 200
        assert h.get("/api/local_sessions").json["sessions"][0]["recorded"] == 1
        np.testing.assert_array_equal(h.runtime.transport.get_latest_qpos(), qpos)
        assert h.runtime.command_queue.empty()


def test_approve_automatically_previews_next_chunk_and_keeps_one_run_until_stop(tmp_path):
    config = _config(tmp_path)
    with serve_console(config) as h:
        _prepare(h, mode="step")
        h.do("/api/step_infer")
        rollout_id = h.runtime.local_rollout_id
        session_id = h.runtime.local_session_id
        version = h.status()["step_preview_version"]
        first_size = len(h.session.pending_real_chunk)
        h.do("/api/step_commit")
        assert h.session.step_index == first_size
        assert h.session.pending_real_chunk is not None
        assert h.status()["step_preview_version"] == version + 1
        assert h.status()["step_operation"] == ""
        assert h.runtime.local_rollout_id == rollout_id
        second_size = len(h.session.pending_real_chunk)
        h.do("/api/step_commit")
        assert h.session.step_index == first_size + second_size
        assert h.runtime.local_rollout_id == rollout_id
        assert h.get("/api/local_sessions").json["sessions"][0]["rollouts"] == 1
        h.do("/api/halt")
        assert h.runtime.local_rollout_id is None
        assert h.session.pending_real_chunk is None
        record = h.get("/api/local_session?id=" + session_id).json["session"]["rollouts"][0]
        assert record["status"] == "stopped"
        assert record["steps"] == first_size + second_size


def test_invalid_or_unrecorded_history_cannot_be_replayed(tmp_path):
    config = _config(tmp_path)
    with serve_console(config) as h:
        assert h.get("/api/local_session?id=../../etc/passwd").status == 404
        assert h.post("/api/debug_record", {"enabled": "yes"}).status == 400
        store = LocalHistoryStore(config.console.history_dir)
        session, run = store.begin(None, prompt="metadata only", model="a", record=False)
        response = h.post("/api/local_replay", {"session_id": session, "rollout_id": run})
        assert response.status == 404
        assert "no saved recording" in response.json["error"]


def test_changing_approval_mode_closes_the_preview_history(tmp_path):
    with serve_console(_config(tmp_path)) as h:
        _prepare(h, mode="step")
        h.do("/api/step_infer")
        session_id = h.runtime.local_session_id
        h.do("/api/select_mode", {"mode": "real"})
        assert h.runtime.local_rollout_id is None
        record = h.get(f"/api/local_session?id={session_id}").json["session"]["rollouts"][0]
        assert record["status"] == "stopped"
        assert record["steps"] == 0


def test_unwritable_history_prevents_a_new_execution(tmp_path):
    config = _config(tmp_path)
    Path(config.console.history_dir).write_text("not a directory")
    with serve_console(config) as h:
        _prepare(h)
        h.do("/api/run")
        assert h.session.status.value != "running"
        assert "Unable to save local session" in h.session.last_error
        assert h.runtime.local_rollout_id is None


def test_stop_queued_after_approval_prevents_the_following_request(tmp_path, monkeypatch):
    from core.app import run as app

    with serve_console(_config(tmp_path)) as h:
        _prepare(h, mode="step")
        h.do("/api/step_infer")
        version = h.runtime.step_preview_version

        def commit_and_stop(_config, runtime, session):
            session.step_index += len(session.pending_real_chunk)
            session.pending_real_chunk = None
            runtime.command_queue.put("web:halt")
            return True

        monkeypatch.setattr(app, "run_pending_chunk_on_real", commit_and_stop)
        h.do("/api/step_commit")
        assert h.session.pending_real_chunk is None
        assert h.runtime.step_preview_version == version
        assert h.runtime.local_rollout_id is None
        assert h.runtime.step_operation == ""


def test_following_preview_error_ends_run_without_executing_more_actions(tmp_path, monkeypatch):
    from core.app import run as app

    with serve_console(_config(tmp_path)) as h:
        _prepare(h, mode="step")
        h.do("/api/step_infer")
        session_id = h.runtime.local_session_id
        approved_size = len(h.session.pending_real_chunk)
        version = h.runtime.step_preview_version

        def fail_preview(_config, _runtime, session):
            session.last_error = "Model unavailable"
            return False

        monkeypatch.setattr(app, "run_one_chunk_on_sim", fail_preview)
        h.do("/api/step_commit")
        assert h.session.step_index == approved_size
        assert h.session.pending_real_chunk is None
        assert h.runtime.step_preview_version == version
        assert h.runtime.local_rollout_id is None
        assert h.runtime.step_operation == ""
        record = h.get("/api/local_session?id=" + session_id).json["session"]["rollouts"][0]
        assert record["status"] == "error"


def test_other_workspaces_keep_the_existing_single_chunk_approval(tmp_path):
    with serve_console(_config(tmp_path)) as h:
        _prepare(h, mode="step")
        h.do("/api/step_infer")
        version = h.runtime.step_preview_version
        h.runtime.console_ctx.active_tab = "replay"
        h.do("/api/step_commit")
        assert h.session.pending_real_chunk is None
        assert h.runtime.step_preview_version == version


def test_finished_run_can_be_deleted_and_restored_while_another_run_is_active(tmp_path):
    config = _config(tmp_path)
    with serve_console(config) as h:
        store = LocalHistoryStore(config.console.history_dir)
        previous, saved = store.begin(None, prompt="past run", model="a", record=False)
        store.update(previous, saved, status="stopped")
        _prepare(h)
        h.do("/api/run")
        active = h.runtime.local_rollout_id
        payload = {"session_id": previous, "rollout_id": saved}
        assert h.post("/api/delete_local_run", payload).status == 200
        assert h.post("/api/delete_local_run", payload).status == 200
        assert h.post("/api/restore_local_run", payload).status == 200
        assert h.runtime.local_rollout_id == active
        assert h.session.status.value == "running"
        assert (
            h.post(
                "/api/delete_local_run",
                {
                    "session_id": h.runtime.local_session_id,
                    "rollout_id": active,
                },
            ).status
            == 409
        )
        h.do("/api/halt")


def test_continued_recording_appends_episodes_without_overwriting_prior_frames(tmp_path):
    import hashlib
    import json

    config = _config(tmp_path)
    with serve_console(config) as h:
        h.do("/api/debug_record", {"enabled": True})
        _prepare(h)
        h.do("/api/run")
        first_id = h.runtime.local_rollout_id
        _publish(h, 8)
        assert h.post("/api/new_debug_run").status == 409
        h.do("/api/halt")
        directory = Path(h.runtime.debug_record_dataset_dir)
        original = {
            p: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*.parquet")
        }
        assert original
        h.do("/api/run")
        assert h.runtime.local_rollout_id == first_id
        _publish(h, 5)
        h.do("/api/halt")
        rows = [
            json.loads(line)
            for line in (directory / "meta/episodes.jsonl").read_text().splitlines()
        ]
        assert [row["episode_index"] for row in rows] == [0, 1]
        assert all(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for path, digest in original.items()
        )
        saved = h.get("/api/local_sessions").json["sessions"]
        assert len(saved) == 1
        assert saved[0]["runs"][0]["steps"] == 13
        assert h.status()["local_history"]["current_rollout_id"] == first_id
        assert not h.status()["local_history"]["saving"]


def test_changing_prompt_continues_current_run_without_homing(tmp_path):
    config = _config(tmp_path)
    with serve_console(config) as h:
        _prepare(h)
        h.do("/api/run")
        _publish(h, 3)
        h.do("/api/halt")
        current = h.status()["local_history"]["current_rollout_id"]
        qpos = h.runtime.transport.get_latest_qpos().copy()
        h.do("/api/select_task", {"task": "move it farther left"})
        assert h.status()["is_setup_done"]
        assert h.status()["step_index"] == 3
        np.testing.assert_array_equal(h.runtime.transport.get_latest_qpos(), qpos)
        h.do("/api/run")
        assert h.runtime.local_rollout_id == current
        _publish(h, 2)
        h.do("/api/halt")
        assert h.get("/api/local_sessions").json["sessions"][0]["runs"][0]["steps"] == 5


def test_live_permission_switch_preserves_setup_progress_and_current_run(tmp_path, monkeypatch):
    config = _config(tmp_path)
    with serve_console(config) as h:
        _prepare(h)
        h.do("/api/run")
        _publish(h, 3)
        current = h.runtime.local_current_rollout_id
        task = h.session.selected_task
        reset_calls = []
        monkeypatch.setattr(h.runtime.policy, "reset", lambda: reset_calls.append(1))
        before = h.runtime.transport.get_latest_qpos().copy()
        h.do("/api/select_mode", {"mode": "step"})
        assert h.status()["cli_mode"] == "step"
        assert h.status()["is_setup_done"]
        assert h.status()["step_index"] == 3
        assert h.status()["selected_task"] == task
        assert h.session.status.value == "ready"
        assert h.session.pending_real_chunk is None
        assert h.runtime.local_current_rollout_id == current
        assert reset_calls == []
        np.testing.assert_array_equal(h.runtime.transport.get_latest_qpos(), before)
        h.do("/api/step_infer")
        assert h.runtime.local_rollout_id == current
        assert h.session.pending_real_chunk is not None
        preview = h.session.pending_real_chunk.copy()
        h.do("/api/select_mode", {"mode": "real"})
        assert h.status()["is_setup_done"]
        assert h.status()["step_index"] == 3
        assert h.session.status.value == "ready"
        assert h.session.pending_real_chunk is None
        assert reset_calls == []
        h.do("/api/run")
        assert h.runtime.local_rollout_id == current
        _publish(h, 2)
        h.do("/api/halt")
        rows = h.get("/api/local_sessions").json["sessions"]
        assert len(rows) == 1
        assert rows[0]["runs"][0]["steps"] == 5
        assert len(preview) > 0


def test_permission_switch_interrupts_an_approved_chunk_without_automatic_resume(
    tmp_path, monkeypatch
):
    with serve_console(_config(tmp_path)) as h:
        _prepare(h, mode="step")
        h.do("/api/step_infer")
        current = h.runtime.local_current_rollout_id
        published = []
        original = h.runtime.transport.publish_action

        def publish(action, target="real"):
            original(action, target)
            if target == "real":
                published.append(action.copy())
                if len(published) == 1:
                    h.runtime.command_queue.put("web:select_mode:real")

        monkeypatch.setattr(h.runtime.transport, "publish_action", publish)
        h.do("/api/step_commit")
        assert len(published) == 1
        assert h.status()["cli_mode"] == "real"
        assert h.status()["is_setup_done"]
        assert h.status()["step_index"] == 1
        assert h.session.status.value == "ready"
        assert h.session.pending_real_chunk is None
        assert h.runtime.local_current_rollout_id == current
        assert len(h.get("/api/local_sessions").json["sessions"]) == 1


def test_permission_switch_interrupts_preview_without_publishing_to_robot(tmp_path, monkeypatch):
    with serve_console(_config(tmp_path)) as h:
        _prepare(h, mode="step")
        published = []
        original = h.runtime.transport.publish_action

        def publish(action, target="real"):
            published.append(target)
            original(action, target)
            if len(published) == 1:
                h.runtime.command_queue.put("web:select_mode:real")

        monkeypatch.setattr(h.runtime.transport, "publish_action", publish)
        h.do("/api/step_infer")
        assert published == ["sim"]
        assert h.status()["cli_mode"] == "real"
        assert h.status()["is_setup_done"]
        assert h.status()["step_index"] == 0
        assert h.session.status.value == "ready"
        assert h.session.pending_real_chunk is None
        row = h.get("/api/local_sessions").json["sessions"][0]["runs"][0]
        assert row["status"] == "interrupted"


def test_selecting_current_permission_does_not_interrupt_execution(tmp_path):
    from core.app.handlers.control import handle_motion_command

    with serve_console(_config(tmp_path)) as h:
        _prepare(h)
        h.do("/api/run")
        current = h.runtime.local_rollout_id
        assert not handle_motion_command("web:select_mode:real", h.config, h.runtime, h.session)
        assert h.runtime.command_queue.empty()
        h.do("/api/select_mode", {"mode": "real"})
        assert h.session.status.value == "running"
        assert h.status()["is_setup_done"]
        assert h.runtime.local_rollout_id == current
        h.do("/api/halt")
