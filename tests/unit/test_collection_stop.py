from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.app.console import server as console_server
from core.app.console.collection_slots import (
    CollectionSlotState,
    build_collection_slots,
    collection_slot_status,
    load_slot_state,
)
from core.app.handlers.recording import collect_stop
from core.app.state import SessionState, SessionStatus
from core.config import ConfigDict

pytestmark = pytest.mark.unit


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
        config=ConfigDict(collection={"storage": {"log_dir": str(tmp_path)}}),
        runtime=SimpleNamespace(
            episode_logger=SimpleNamespace(has_active_episode=False), active_config=None
        ),
    )
    state_path = tmp_path / "collection_slots" / "set.json"
    for _ in range(2):
        active = console_server._sync_collection_slot_session(ctx)
        assert active["slot_id"] == slots[1].slot_id
        state = load_slot_state(state_path)

    queue.append({"slot_id": slots[1].slot_id, "status": "saving"})
    assert console_server._sync_collection_slot_session(ctx)["slot_id"] == slots[2].slot_id


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
