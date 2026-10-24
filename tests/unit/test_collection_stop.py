from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from core.app.console import server as console_server
from core.app.console.collection_slots import (
    CollectionSlotState,
    build_collection_slots,
    collection_slot_status,
    load_slot_state,
    select_collection_slot,
)
from core.app.console.server import ConsoleContext
from core.app.handlers.recording import collect_cancel, collect_stop
from core.app.state import SessionState, SessionStatus
from core.config import ConfigDict
from core.utils.collection_paths import slot_state_path

pytestmark = pytest.mark.unit


def test_cancel_releases_client_episode_when_sim_has_exited() -> None:
    calls = []

    class Logger:
        is_collection_enabled = True

        def cancel_sim_recording(self):
            raise RuntimeError("SIM connection refused")

        def cancel_episode(self, reason):
            calls.append(reason)

    session = SessionState(status=SessionStatus.RUNNING, collection_slot_id="slot-A")
    runtime = SimpleNamespace(
        episode_logger=Logger(),
        collection_capture_runner=None,
        transport=SimpleNamespace(finish_collection_capture=lambda: None),
    )
    collect_cancel(runtime, session)
    assert calls == ["user cancel"]
    assert session.status == SessionStatus.READY
    assert session.collection_slot_id == "slot-A"
    assert "SIM connection refused" in session.last_error


def test_current_red_advances_forward_and_can_be_manually_retaken():
    slots = build_collection_slots(
        ConfigDict(collection=ConfigDict(tasks={"set": [("task", 2)]})), {}, "set"
    )
    episode = {"episode_index": 7, "slot_id": slots[0].slot_id, "quality": "red", "status": "saved"}
    rows, active, counts = collection_slot_status(
        slots, [episode], [], CollectionSlotState([], slots[0].slot_id)
    )
    assert rows[0]["state"] == "rejected"
    assert counts["rejected"] == 1
    assert active["slot_id"] == slots[1].slot_id
    _, active, _ = collection_slot_status(
        slots[:1], [episode], [], CollectionSlotState([], slots[0].slot_id)
    )
    assert active is None
    _, active, _ = collection_slot_status(
        slots, [episode], [], CollectionSlotState([], slots[0].slot_id, 7, True)
    )
    assert active["slot_id"] == slots[0].slot_id
    assert active["repair"]


@pytest.mark.parametrize("failed_save", [True])
def test_automatic_red_retake_survives_sync_until_new_capture(tmp_path, monkeypatch, failed_save):
    slots = build_collection_slots(
        ConfigDict(collection=ConfigDict(tasks={"set": [("task", 3)]})), {}, "set"
    )
    episodes = [{"episode_index": 0, "slot_id": slots[0].slot_id, "quality": "green"}]
    queue = []
    if failed_save:
        queue.append({"slot_id": slots[1].slot_id, "status": "failed"})
    else:
        episodes.append({"episode_index": 1, "slot_id": slots[1].slot_id, "quality": "red"})
    for episode in episodes:
        episode["status"] = "saved"
    state = CollectionSlotState([], slots[0].slot_id)

    def snapshot(_ctx, _dataset):
        _, active, _ = collection_slot_status(slots, episodes, queue, state)
        return {"active": active, "dataset_dir": str(tmp_path), "slot_state": state}

    monkeypatch.setattr(console_server, "_collection_slots_snapshot", snapshot)
    session = SessionState()
    session.selected_collect_set = "set"
    ctx = SimpleNamespace(
        session=session,
        collection_selection_lock=threading.RLock(),
        config=ConfigDict(collection={"storage": {"log_dir": str(tmp_path)}}),
        runtime=SimpleNamespace(
            episode_logger=SimpleNamespace(has_active_episode=False), active_config=None
        ),
    )
    state_path = slot_state_path(ctx.config, "set")
    for _ in range(2):
        active = console_server._sync_collection_slot_session(ctx)
        assert active["slot_id"] == slots[1].slot_id
        state = load_slot_state(state_path)

    queue.append({"slot_id": slots[1].slot_id, "status": "saving"})
    assert console_server._sync_collection_slot_session(ctx)["slot_id"] == slots[2].slot_id


def test_saved_retake_advances_despite_reused_episode_index(tmp_path, monkeypatch):
    slots = build_collection_slots(
        ConfigDict(collection=ConfigDict(tasks={"set": [("task", 2)]})), {}, "set"
    )
    episodes = [
        {"episode_index": index, "slot_id": slot.slot_id, "status": "saved", "qc_verdict": "fail"}
        for index, slot in enumerate(slots)
    ]
    config = ConfigDict(collection={"storage": {"log_dir": str(tmp_path)}})
    state_path = slot_state_path(config, "set")
    select_collection_slot(
        state_path, slots[0].slot_id, CollectionSlotState([]), episode_index=0, manual=True
    )

    def snapshot(_ctx, _dataset):
        state = load_slot_state(state_path)
        _, active, _ = collection_slot_status(slots, episodes, [], state)
        return {"active": active, "slot_state": state}

    monkeypatch.setattr(console_server, "_collection_slots_snapshot", snapshot)
    session = SessionState()
    session.selected_collect_set = "set"
    session.collection_slot_id = slots[0].slot_id
    ctx = SimpleNamespace(
        session=session,
        collection_selection_lock=threading.RLock(),
        config=config,
        runtime=SimpleNamespace(
            episode_logger=SimpleNamespace(has_active_episode=False), active_config=None
        ),
        collection_save_revision=0,
        completed_collection_dataset="",
        completed_collection_slot_id="",
    )

    assert console_server._sync_collection_slot_session(ctx)["slot_id"] == slots[0].slot_id
    episodes[0]["qc_verdict"] = "unreviewed"
    assert (
        console_server._sync_collection_slot_session(ctx, slots[0].slot_id)["slot_id"]
        == slots[1].slot_id
    )
    assert session.collection_slot_id == slots[1].slot_id
    assert load_slot_state(state_path).selected_slot_id == slots[1].slot_id
    assert ctx.collection_save_revision == 1
    assert ctx.completed_collection_dataset == "set"
    assert ctx.completed_collection_slot_id == slots[0].slot_id


class _CollectionLogger:
    is_collection_enabled = True

    def __init__(self, saved: bool) -> None:
        self.saved = saved
        self.wait_calls = 0
        self._last_save_job = None

    def set_save_completion_callback(self, _callback) -> None:
        return None

    def end_episode(self) -> bool:
        return self.saved

    def wait_for_saves(self, **_kwargs) -> bool:
        self.wait_calls += 1
        return True


def _runtime(saved: bool, synced: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        episode_logger=_CollectionLogger(saved),
        collection_capture_runner=None,
        transport=SimpleNamespace(finish_collection_capture=lambda: None),
        console_ctx=SimpleNamespace(sync_collection_slot=lambda slot_id: synced.append(slot_id)),
    )


def test_collect_stop_advances_slot_after_episode_is_queued() -> None:
    synced: list[str] = []
    session = SessionState(status=SessionStatus.RUNNING)
    session.collection_slot_id = "slot-1"
    runtime = _runtime(True, synced)

    config = SimpleNamespace(collection=ConfigDict(sim_recording_url=""))
    assert collect_stop(config, runtime, session)

    assert session.status is SessionStatus.READY
    assert synced == ["slot-1"]
    assert runtime.episode_logger.wait_calls == 1


def test_collect_stop_keeps_slot_when_episode_has_no_frames() -> None:
    synced: list[str] = []
    session = SessionState(status=SessionStatus.RUNNING)
    runtime = _runtime(False, synced)
    runtime.transport.collection_diagnostics = lambda: ""

    config = SimpleNamespace(collection=ConfigDict(sim_recording_url=""))
    assert not collect_stop(config, runtime, session)

    assert session.status is SessionStatus.READY
    assert synced == []
    assert "0 frames" in session.last_error


def test_sim_collect_stop_queues_without_waiting_and_keeps_two_jobs_separate() -> None:
    class Job:
        def __init__(self, slot_id: str, episode_index: int) -> None:
            self.episode_meta = {"slot_id": slot_id}
            self.episode_index = episode_index
            self.status = "queued"
            self.error = ""

    class Logger:
        is_collection_enabled = True

        def __init__(self) -> None:
            self.callback = None
            self.jobs: list[Job] = []
            self._last_save_job = None

        def set_save_completion_callback(self, callback) -> None:
            self.callback = callback

        def stop_sim_recording(self) -> None:
            return None

        def end_episode(self) -> bool:
            slot_id = "slot-A" if not self.jobs else "slot-B"
            job = Job(slot_id, len(self.jobs))
            self.jobs.append(job)
            self._last_save_job = job
            return True

        def wait_for_saves(self, **_kwargs) -> bool:
            raise AssertionError("SIM collection must not wait for background saves")

    logger_obj = Logger()
    session = SessionState(
        status=SessionStatus.RUNNING,
        selected_collect_set="dataset",
        collection_slot_id="slot-A",
    )
    sync_calls = 0

    ctx = ConsoleContext(
        config=ConfigDict(collection=ConfigDict(sim_recording_url="http://sim")),
        runtime=SimpleNamespace(episode_logger=logger_obj, active_config=None),
        session=session,
    )

    def sync_collection_slot() -> dict[str, str]:
        nonlocal sync_calls
        sync_calls += 1
        session.collection_slot_id = "slot-B" if sync_calls == 1 else "slot-C"
        return {"slot_id": session.collection_slot_id}

    ctx.sync_collection_slot = sync_collection_slot

    runtime = SimpleNamespace(
        episode_logger=logger_obj,
        collection_capture_runner=None,
        transport=SimpleNamespace(
            finish_collection_capture=lambda: None,
            collection_diagnostics=lambda: "",
        ),
        console_ctx=ctx,
    )
    config = SimpleNamespace(collection=ConfigDict(sim_recording_url="http://sim"))

    assert collect_stop(config, runtime, session)
    assert collect_stop(config, runtime, session)
    assert sync_calls == 2
    pending = ctx.collection_save_snapshot("dataset")
    assert len(pending) == 2
    assert {record["slot_id"] for record in pending} == {"slot-A", "slot-B"}
    assert all(record["dataset"] == "dataset" for record in pending)
    assert ctx.collection_save_revision == 0

    logger_obj.jobs[1].status = "failed"
    logger_obj.jobs[1].error = "background write failed"
    logger_obj.callback(logger_obj.jobs[1])
    pending = ctx.collection_save_snapshot("dataset")
    assert len(pending) == 2
    failed = next(record for record in pending if record["slot_id"] == "slot-B")
    assert failed["status"] == "failed"
    assert failed["error"] == "background write failed"

    logger_obj.jobs[0].status = "saved"
    logger_obj.callback(logger_obj.jobs[0])
    assert ctx.collection_save_revision == 1
    assert [record["slot_id"] for record in ctx.collection_save_snapshot("dataset")] == ["slot-B"]


def test_sim_collect_stop_does_not_advance_when_raw_stop_fails() -> None:
    class Logger:
        is_collection_enabled = True

        def stop_sim_recording(self) -> None:
            raise RuntimeError("footer missing")

        def cancel_sim_recording(self) -> None:
            return None

        def cancel_episode(self, _reason: str) -> None:
            return None

    synced: list[str] = []
    session = SessionState(status=SessionStatus.RUNNING, collection_slot_id="slot-A")
    runtime = SimpleNamespace(
        episode_logger=Logger(),
        collection_capture_runner=None,
        transport=SimpleNamespace(finish_collection_capture=lambda: None),
        console_ctx=SimpleNamespace(sync_collection_slot=lambda: synced.append("advanced")),
    )
    config = SimpleNamespace(collection=ConfigDict(sim_recording_url="http://sim"))

    assert not collect_stop(config, runtime, session)
    assert synced == []
    assert "footer missing" in session.last_error


def test_collection_slots_projects_next_candidate_while_current_sim_records(tmp_path, monkeypatch):
    config = ConfigDict(
        collection=ConfigDict(
            tasks={"dataset": [("task", 2)]},
            storage=ConfigDict(data_root=str(tmp_path), log_dir=str(tmp_path)),
        )
    )

    class Logger:
        is_collection_enabled = True
        has_active_episode = True

        def status_snapshot(self, *_args, **_kwargs):
            return {"dataset_dir": str(tmp_path), "queue": []}

    session = SessionState(
        selected_collect_set="dataset",
        collection_slot_id="INLINE-0:DEFAULT:0",
    )
    runtime = SimpleNamespace(episode_logger=Logger(), active_config=None)
    ctx = ConsoleContext(config=config, runtime=runtime, session=session)
    monkeypatch.setattr(
        console_server, "_load_scene_plan", lambda *_args: {"scenes": [], "tasks": []}
    )

    snapshot = console_server._collection_slots_snapshot(ctx, "dataset")

    assert snapshot["active"]["slot_id"] == "INLINE-0:DEFAULT:0"
    assert snapshot["next_candidate"]["slot_id"] == "INLINE-0:DEFAULT:1"
    assert snapshot["pending_saves"] == []


def test_collection_slots_reuses_unbounded_candidate_after_projection(tmp_path, monkeypatch):
    config = ConfigDict(
        collection=ConfigDict(
            tasks={"dataset": [("task", -1)]},
            storage=ConfigDict(data_root=str(tmp_path), log_dir=str(tmp_path)),
        )
    )

    class Logger:
        is_collection_enabled = True
        has_active_episode = True

        def status_snapshot(self, *_args, **_kwargs):
            return {"dataset_dir": str(tmp_path), "queue": []}

    session = SessionState(
        selected_collect_set="dataset",
        collection_slot_id="INLINE-0:DEFAULT:0",
    )
    ctx = ConsoleContext(
        config=config,
        runtime=SimpleNamespace(episode_logger=Logger(), active_config=None),
        session=session,
    )
    monkeypatch.setattr(
        console_server, "_load_scene_plan", lambda *_args: {"scenes": [], "tasks": []}
    )

    snapshot = console_server._collection_slots_snapshot(ctx, "dataset")

    assert snapshot["next_candidate"]["slot_id"] == "INLINE-0:DEFAULT:0"


def test_collection_slots_prioritize_scene_then_layout_ordered_task_object() -> None:
    config = ConfigDict(
        collection=ConfigDict(
            tasks={
                "set": [
                    ("right task", 1),
                    ("left task", 1),
                    ("top task", 1),
                    ("other scene", 1),
                ]
            }
        )
    )
    scene_plan = {
        "positions": [
            {"position_id": "P1", "x": 0, "y": 0},
            {"position_id": "P2", "x": 250, "y": 0},
            {"position_id": "P3", "x": 0, "y": 250},
        ],
        "scenes": [
            {
                "scene_id": "SC-A",
                "placements": [
                    {"object_id": "OBJ-R", "position_ids": ["P2"]},
                    {"object_id": "OBJ-L", "position_ids": ["P1"]},
                    {"object_id": "OBJ-T", "position_ids": ["P3"]},
                    {"object_id": "OBJ-FIXED", "position_ids": ["P3"]},
                ],
            },
            {
                "scene_id": "SC-B",
                "placements": [{"object_id": "OBJ-B", "position_ids": ["P1"]}],
            },
        ],
        "tasks": [
            {
                "task_id": "TASK-R",
                "prompt_en": "right task",
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
                "operation_object_ids": ["OBJ-FIXED", "OBJ-R"],
            },
            {
                "task_id": "TASK-L",
                "prompt_en": "left task",
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
                "operation_object_ids": ["OBJ-FIXED", "OBJ-L"],
            },
            {
                "task_id": "TASK-T",
                "prompt_en": "top task",
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
                "operation_object_ids": ["OBJ-FIXED", "OBJ-T"],
            },
            {
                "task_id": "TASK-B",
                "prompt_en": "other scene",
                "scene_ids": ["SC-B"],
                "scene_epsiodes_count": [1],
                "operation_object_ids": ["OBJ-B"],
            },
        ],
    }

    slots = build_collection_slots(config, scene_plan, "set")

    assert [slot.slot_id for slot in slots] == [
        "TASK-L:SC-A:0",
        "TASK-R:SC-A:0",
        "TASK-T:SC-A:0",
        "TASK-B:SC-B:0",
    ]
