from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.app.console import server as console_server
from core.app.console.collection_slots import (
    CollectionSlot,
    CollectionSlotState,
    build_collection_slots,
    collection_slot_status,
    load_slot_state,
)
from core.app.handlers.recording import collect_stop
from core.app.state import SessionState, SessionStatus
from core.config import ConfigDict

pytestmark = pytest.mark.unit


def test_auto_red_stays_red_and_requires_manual_retake():
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


class _CollectionLogger:
    is_collection_enabled = True

    def __init__(self, saved: bool) -> None:
        self.saved = saved

    def end_episode(self) -> bool:
        return self.saved


def _runtime(saved: bool, synced: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        episode_logger=_CollectionLogger(saved),
        collection_capture_runner=None,
        transport=SimpleNamespace(finish_collection_capture=lambda: None),
        console_ctx=SimpleNamespace(sync_collection_slot=lambda: synced.append("next")),
    )


def test_collect_stop_advances_slot_after_episode_is_queued() -> None:
    synced: list[str] = []
    session = SessionState(status=SessionStatus.RUNNING)

    assert collect_stop(SimpleNamespace(), _runtime(True, synced), session)

    assert session.status is SessionStatus.READY
    assert synced == ["next"]


def test_collect_stop_keeps_slot_when_episode_has_no_frames() -> None:
    synced: list[str] = []
    session = SessionState(status=SessionStatus.RUNNING)
    runtime = _runtime(False, synced)
    runtime.transport.collection_diagnostics = lambda: ""

    assert not collect_stop(SimpleNamespace(), runtime, session)

    assert session.status is SessionStatus.READY
    assert synced == []
    assert "0 frames" in session.last_error


def test_backend_slot_sync_updates_session_and_persisted_cursor(tmp_path, monkeypatch) -> None:
    session = SessionState()
    session.selected_collect_set = "cup_set"
    session.collection_slot_id = "TASK-CUP:SC-A:0"
    active = {
        "dataset": "cup_set",
        "task": "pick up cup",
        "task_index": 0,
        "task_id": "TASK-CUP",
        "scene_id": "SC-A",
        "round_index": 1,
        "slot_id": "TASK-CUP:SC-A:1",
    }
    snapshot = {
        "active": active,
        "dataset_dir": str(tmp_path),
        "slot_state": CollectionSlotState([], "TASK-CUP:SC-A:0"),
    }
    monkeypatch.setattr(
        console_server,
        "_collection_slots_snapshot",
        lambda _ctx, dataset: snapshot if dataset == "cup_set" else None,
    )
    ctx = SimpleNamespace(
        session=session,
        runtime=SimpleNamespace(episode_logger=SimpleNamespace(has_active_episode=False)),
    )

    assert console_server._sync_collection_slot_session(ctx) == active

    assert session.collection_slot_id == "TASK-CUP:SC-A:1"
    assert session.collection_scene_round == 1
    assert load_slot_state(tmp_path).selected_slot_id == "TASK-CUP:SC-A:1"


def test_saving_slot_stays_busy_while_cursor_advances() -> None:
    slots = [
        CollectionSlot(
            slot_id=f"TASK-CUP:SC-A:{round_index}",
            ordinal=round_index,
            dataset="cup_set",
            task_index=0,
            task_id="TASK-CUP",
            task="pick up cup",
            task_zh="拿起杯子",
            scene_id="SC-A",
            scene_label="Scene A",
            round_index=round_index,
            round_total=2,
        )
        for round_index in range(2)
    ]
    rows, active, _counts = collection_slot_status(
        slots,
        [],
        [{"slot_id": slots[0].slot_id, "status": "saving"}],
        CollectionSlotState([], slots[0].slot_id),
    )

    assert rows[0]["state"] == "saving"
    assert active is not None
    assert active["slot_id"] == slots[1].slot_id
    assert active["state"] == "active"
