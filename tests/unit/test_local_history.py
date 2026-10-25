from __future__ import annotations

import json

import pytest

from core.app.local_history import LocalHistoryStore

pytestmark = pytest.mark.unit


def test_sessions_persist_multiple_executions_without_recording_data(tmp_path):
    store = LocalHistoryStore(tmp_path / ".eva-client")
    assert store.list() == []
    assert not store.root.exists()
    session, first = store.begin(None, prompt="pick a cup", model="model-a", record=False)
    store.update(session, first, status="completed", steps=4)
    _, second = store.begin(session, prompt="place the cup", model="model-b", record=False)
    store.update(session, second, status="stopped", steps=5)
    restored = LocalHistoryStore(store.root)
    data = restored.read(session)
    assert [row["prompt"] for row in data["rollouts"]] == ["pick a cup", "place the cup"]
    assert restored.list()[0]["rollouts"] == 2
    assert all(not row["playable"] for row in data["rollouts"])
    assert not (store.root / "rollouts").exists()


def test_recording_files_and_interrupted_process_are_reflected_in_history(tmp_path, monkeypatch):
    store = LocalHistoryStore(tmp_path)
    session, rollout = store.begin(None, prompt="test", model="a", record=True)
    monkeypatch.setattr(
        "core.app.local_history.os.kill", lambda *_: (_ for _ in ()).throw(ProcessLookupError())
    )
    assert store.read(session)["rollouts"][0]["status"] == "interrupted"
    metadata = store.dataset_path(rollout) / "meta/episodes.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"episode_index": 0}) + "\n")
    record = store.read(session)["rollouts"][0]
    assert record["playable"] is True
    assert record["record_status"] == "saved"


@pytest.mark.parametrize("value", ["../outside", "", "x" * 32, "/tmp/history"])
def test_history_ids_cannot_select_paths_outside_the_store(tmp_path, value):
    store = LocalHistoryStore(tmp_path)
    with pytest.raises(ValueError):
        store.read(value)
    with pytest.raises(ValueError):
        store.dataset_path(value)


def test_deleted_recording_can_be_restored_from_local_trash(tmp_path):
    store = LocalHistoryStore(tmp_path)
    session, run = store.begin(None, prompt="recorded", model="a", record=True)
    store.update(session, run, status="completed")
    metadata = store.dataset_path(run) / "meta/episodes.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"episode_index": 0}\n')
    store.set_deleted(session, run, True)
    assert store.list() == []
    assert store.read(session)["rollouts"] == []
    assert not store.dataset_path(run).exists()
    assert (store.root / "trash" / run / "meta/episodes.jsonl").is_file()
    store.set_deleted(session, run, False)
    assert store.read(session)["rollouts"][0]["playable"] is True
    assert metadata.is_file()


def test_deleting_one_run_keeps_other_runs_and_blocks_an_active_run(tmp_path):
    store = LocalHistoryStore(tmp_path)
    session, first = store.begin(None, prompt="one", model="a", record=False)
    with pytest.raises(ValueError, match="finish"):
        store.set_deleted(session, first, True)
    store.update(session, first, status="stopped")
    _, second = store.begin(session, prompt="two", model="a", record=False)
    store.update(session, second, status="stopped")
    store.set_deleted(session, first, True)
    assert [run["id"] for run in store.read(session)["rollouts"]] == [second]


def test_metadata_write_failure_restores_the_recording_location(tmp_path, monkeypatch):
    store = LocalHistoryStore(tmp_path)
    session, run = store.begin(None, prompt="keep", model="a", record=True)
    store.update(session, run, status="completed")
    metadata = store.dataset_path(run) / "meta/episodes.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"episode_index": 0}\n')

    def fail_write(_data):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write", fail_write)
    with pytest.raises(OSError, match="disk full"):
        store.set_deleted(session, run, True)
    assert metadata.is_file()
    assert store.read(session)["rollouts"][0]["playable"] is True


def test_delete_and_restore_are_safe_to_retry(tmp_path):
    store = LocalHistoryStore(tmp_path)
    session, run = store.begin(None, prompt="finished", model="a", record=False)
    store.update(session, run, status="stopped")
    store.set_deleted(session, run, True)
    store.set_deleted(session, run, True)
    assert store.list() == []
    store.set_deleted(session, run, False)
    store.set_deleted(session, run, False)
    assert len(store.read(session)["rollouts"]) == 1


def test_saved_episode_does_not_hide_a_later_save_and_resume_waits_for_it(tmp_path):
    store = LocalHistoryStore(tmp_path)
    session, run = store.begin(None, prompt="continue", model="a", record=True)
    metadata = store.dataset_path(run) / "meta/episodes.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"episode_index": 0}\n')
    store.update(session, run, status="stopped", steps=8, record_status="saving")
    assert store.read(session)["rollouts"][0]["record_status"] == "saving"
    with pytest.raises(ValueError, match="still saving"):
        store.resume(session, run, record=True)
    store.update(session, run, record_status="saved")
    assert store.resume(session, run, record=True) == 8
    row = store.read(session)["rollouts"][0]
    assert row["status"] == "running"
    assert row["record_status"] == "recording"
    assert row["continuations"] == 1
