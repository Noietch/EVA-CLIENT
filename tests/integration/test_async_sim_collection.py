from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.app.console.server import ConsoleContext
from core.app.state import SessionState
from core.cfg import ConfigDict
from core.recorder.episode import EpisodeLogger
from robots.base import ActuatorGroup, ObservationSchema, Robot

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("complete_before_register", [False, True])
def test_async_failure_preserves_reason_and_final_index(complete_before_register: bool) -> None:
    ctx = ConsoleContext(
        config=ConfigDict(), runtime=SimpleNamespace(), session=SessionState(),
    )
    job = SimpleNamespace(
        episode_meta={"slot_id": "slot-A"}, episode_index=1, status="queued", error="",
    )
    if not complete_before_register:
        ctx.register_collection_save_job(job, "dataset")
    job.episode_index = 2
    job.status = "failed"
    job.error = "dataset commit failed"
    ctx.collection_save_job_finished(job)
    if complete_before_register:
        ctx.register_collection_save_job(job, "dataset")
    assert ctx.collection_save_snapshot("dataset") == [{
        "dataset": "dataset", "slot_id": "slot-A", "episode_index": 2,
        "status": "failed", "error": "dataset commit failed",
    }]
    assert ctx.collection_save_revision == 0


def _robot() -> Robot:
    return Robot(
        name="fake_arm",
        actuator_groups=(ActuatorGroup("arm", 2, ("j0", "j1")),),
        initial_qpos=np.zeros(2, dtype=np.float32),
        observation_schema=ObservationSchema(cameras=(), state_composition=("arm",)),
    )


def test_two_sim_save_jobs_keep_payloads_and_completion_state_isolated(tmp_path: Path) -> None:
    collection = ConfigDict(
        sim_recording_url="",
        storage=ConfigDict(
            data_root=str(tmp_path), log_dir=str(tmp_path), fps=30, save_queue_max=2
        ),
        schema=ConfigDict(
            robot_type="fake_arm",
            min_episode_frames=1,
            max_frame_dt_factor=3.0,
            arms={},
            cameras={},
            columns={
                "qpos": "observations.state.qpos",
                "action_qpos": "action.qpos",
            },
        ),
        tasks={"dataset": [("task", -1)]},
    )
    logger = EpisodeLogger(
        tmp_path / "runtime",
        _robot(),
        fps=30,
        dataset_keys=ConfigDict(video_keys={}),
        collection=collection,
        async_save=True,
        save_queue_max=2,
    )
    session = SessionState(selected_collect_set="dataset")
    ctx = ConsoleContext(
        config=collection,
        runtime=SimpleNamespace(episode_logger=logger, active_config=None),
        session=session,
    )
    logger.set_save_completion_callback(ctx.collection_save_job_finished)
    entered = threading.Event()
    release = threading.Event()
    written_slots: list[str] = []

    def write_job(job) -> None:
        written_slots.append(str(job.episode_meta["slot_id"]))
        entered.set()
        assert release.wait(5)

    logger._write_job = write_job

    def queue_sim_job(slot_id: str):
        logger.start_episode("task", collection_dataset="dataset")
        logger.set_episode_meta(slot_id=slot_id)
        logger._collection_writer._sim_execution = object()
        assert logger.end_episode()
        return logger._last_save_job

    job_a = queue_sim_job("slot-A")
    assert entered.wait(5)
    job_b = queue_sim_job("slot-B")
    ctx.register_collection_save_job(job_a, "dataset")
    ctx.register_collection_save_job(job_b, "dataset")

    assert job_a.collection_payload is not job_b.collection_payload
    assert job_b.status == "queued"
    assert ctx.collection_save_snapshot("dataset")[1]["slot_id"] == "slot-B"

    release.set()
    assert logger.wait_for_saves(timeout=5, fail_on_error=True)
    assert written_slots == ["slot-A", "slot-B"]
    assert ctx.collection_save_revision == 2
    assert ctx.collection_save_snapshot("dataset") == []
