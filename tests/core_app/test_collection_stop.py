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


def test_collection_slots_fall_back_to_legacy_operation_object_names() -> None:
    config = ConfigDict(
        collection=ConfigDict(tasks={"set": [("right task", 1), ("left task", 1)]})
    )
    scene_plan = {
        "positions": [
            {"position_id": "P1", "x": 0, "y": 0},
            {"position_id": "P2", "x": 250, "y": 0},
        ],
        "scenes": [
            {
                "scene_id": "SC-A",
                "placements": [
                    {"object_id": "OBJ-R", "name": "右侧物体", "position_ids": ["P2"]},
                    {"object_id": "OBJ-L", "name": "左侧物体", "position_ids": ["P1"]},
                ],
            }
        ],
        "tasks": [
            {
                "task_id": "TASK-R",
                "prompt_en": "right task",
                "operation_object": "右侧物体",
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
            },
            {
                "task_id": "TASK-L",
                "prompt_en": "left task",
                "operation_object": "左侧物体",
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
            },
        ],
    }

    slots = build_collection_slots(config, scene_plan, "set")

    assert [slot.task_id for slot in slots] == ["TASK-L", "TASK-R"]


def test_collection_slots_put_left_hand_before_right_hand_at_same_position() -> None:
    right_prompt = "use the right arm to move the object"
    left_prompt = "use the left arm to move the object"
    config = ConfigDict(
        collection=ConfigDict(tasks={"set": [(right_prompt, 1), (left_prompt, 1)]})
    )
    scene_plan = {
        "positions": [{"position_id": "P1", "x": 0, "y": 0}],
        "scenes": [
            {
                "scene_id": "SC-A",
                "placements": [{"object_id": "OBJ", "position_ids": ["P1"]}],
            }
        ],
        "tasks": [
            {
                "task_id": "TASK-R",
                "prompt_en": right_prompt,
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
                "operation_object_ids": ["OBJ"],
            },
            {
                "task_id": "TASK-L",
                "prompt_en": left_prompt,
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
                "operation_object_ids": ["OBJ"],
            },
        ],
    }

    slots = build_collection_slots(config, scene_plan, "set")

    assert [slot.task_id for slot in slots] == ["TASK-L", "TASK-R"]


def test_collection_slots_prioritize_hand_before_object_position() -> None:
    right_prompt = "use the right arm to move the left object"
    left_prompt = "use the left arm to move the right object"
    config = ConfigDict(
        collection=ConfigDict(tasks={"set": [(right_prompt, 1), (left_prompt, 1)]})
    )
    scene_plan = {
        "positions": [
            {"position_id": "P1", "x": 0, "y": 0},
            {"position_id": "P2", "x": 250, "y": 0},
        ],
        "scenes": [
            {
                "scene_id": "SC-A",
                "placements": [
                    {"object_id": "OBJ-R", "position_ids": ["P2"]},
                    {"object_id": "OBJ-L", "position_ids": ["P1"]},
                ],
            }
        ],
        "tasks": [
            {
                "task_id": "TASK-R",
                "prompt_en": right_prompt,
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
                "operation_object_ids": ["OBJ-R"],
            },
            {
                "task_id": "TASK-L",
                "prompt_en": left_prompt,
                "scene_ids": ["SC-A"],
                "scene_epsiodes_count": [1],
                "operation_object_ids": ["OBJ-L"],
            },
        ],
    }

    slots = build_collection_slots(config, scene_plan, "set")

    assert [slot.task_id for slot in slots] == ["TASK-L", "TASK-R"]
