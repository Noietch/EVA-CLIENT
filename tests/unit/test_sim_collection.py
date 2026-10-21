import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import numpy as np
import pytest
from mcap.writer import Writer

import core.recorder.collection as collection_module
import core.recorder.sim_collection as sim_collection
from core.recorder.collection import CollectionEpisodeWriter
from core.recorder.sim_collection import (
    SimRecordingClient,
    SimRecordingError,
    SimRecordingHandle,
    create_sim_recording_handle,
    read_sim_execution,
)
from core.types import Observation

pytestmark = pytest.mark.unit


def _state(qpos: list[float]) -> dict[str, object]:
    return {"qpos": qpos, "scene_state": {}, "objects": {}, "liquids": {}}


def _write_execution(
    path: Path,
    *,
    steps: list[dict],
    footer: dict | None = None,
    initial_qpos: list[float] | None = None,
    topic: str = "/sim_execution",
    encoding: str = "json",
    sequences: list[int] | None = None,
) -> None:
    records = [
        {
            "type": "header",
            "schema_version": 1,
            "episode_id": "episode-1",
            "sim_metadata": {"layout": "test"},
            "control_dt": 0.1,
            "physics_dt": 0.01,
            "initial_state": _state(initial_qpos or [1.0, 2.0]),
        },
        *steps,
    ]
    if footer is not None:
        records.append(footer)
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schema_id = writer.register_schema("sim_execution", "json", b"")
        channel_id = writer.register_channel(topic, encoding, schema_id)
        for message_number, record in enumerate(records):
            writer.add_message(
                channel_id,
                log_time=message_number,
                publish_time=message_number,
                sequence=message_number if sequences is None else sequences[message_number],
                data=json.dumps(record).encode("utf-8"),
            )
        writer.finish()


def _step(step_id: int, *, sim_time: float | None = None) -> dict:
    payload = {
        "type": "step",
        "step_id": step_id,
        "source_frame_id": 100 + step_id,
        "action": [5.0 + step_id, 6.0 + step_id],
        "state": _state([3.0 + step_id, 4.0 + step_id]),
    }
    if sim_time is not None:
        payload["sim_time"] = sim_time
    return payload


def test_sim_reader_uses_cached_frames_and_initial_state_for_frame_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "episode.mcap"
    _write_execution(
        path,
        steps=[
            _step(0, sim_time=0.1),
            _step(1, sim_time=0.2),
        ],
        footer={"type": "footer", "steps": 2, "complete": True},
    )

    record = read_sim_execution(path, "episode-1", expected_steps=2)
    assert len(record.frames) == 2
    monkeypatch.setattr(
        sim_collection,
        "_iter_mcap_payloads",
        lambda _path: pytest.fail("iter_frames reparsed the already validated MCAP"),
    )
    frames = list(record.iter_frames(expected_dim=2))

    np.testing.assert_allclose(frames[0].state_qpos, [1.0, 2.0])
    np.testing.assert_allclose(frames[0].action_qpos, [5.0, 6.0])
    np.testing.assert_allclose(frames[1].state_qpos, [3.0, 4.0])
    np.testing.assert_allclose(frames[1].action_qpos, [6.0, 7.0])
    assert [frame.sim_time for frame in frames] == [0.0, 0.1]


def test_cached_sim_reader_preserves_final_dimension_validation(tmp_path: Path) -> None:
    path = tmp_path / "bad-final-state.mcap"
    bad_step = _step(0, sim_time=0.1)
    bad_step["state"] = _state([3.0])
    _write_execution(
        path,
        steps=[bad_step],
        footer={"type": "footer", "steps": 1, "complete": True},
    )

    record = read_sim_execution(path, "episode-1", expected_steps=1)
    with pytest.raises(SimRecordingError, match="state.qpos"):
        list(record.iter_frames(expected_dim=2))


def test_cached_sim_reader_keeps_initial_state_dimension_contract(tmp_path: Path) -> None:
    path = tmp_path / "bad-initial-state.mcap"
    records = [_step(0, sim_time=0.1), _step(1, sim_time=0.2)]
    _write_execution(
        path,
        steps=records,
        footer={"type": "footer", "steps": 2, "complete": True},
        initial_qpos=[1.0],
    )

    record = read_sim_execution(path, "episode-1", expected_steps=2)
    frames = list(record.iter_frames(expected_dim=2))
    assert frames[0].state_qpos.tolist() == [1.0]


def test_sim_reader_rejects_mcap_integrity_channel_and_footer_errors(tmp_path: Path) -> None:
    sequence_gap = tmp_path / "message-sequence-gap.mcap"
    _write_execution(
        sequence_gap,
        steps=[_step(0, sim_time=0.1)],
        footer={"type": "footer", "steps": 1, "complete": True},
        sequences=[0, 2, 2],
    )
    with pytest.raises(SimRecordingError, match="message sequence gap"):
        read_sim_execution(sequence_gap, "episode-1", expected_steps=1)

    bad_topic = tmp_path / "bad-topic.mcap"
    _write_execution(
        bad_topic,
        steps=[_step(0, sim_time=0.1)],
        footer={"type": "footer", "steps": 1, "complete": True},
        topic="/other",
    )
    with pytest.raises(SimRecordingError, match="unexpected MCAP topic"):
        read_sim_execution(bad_topic, "episode-1", expected_steps=1)

    bad_encoding = tmp_path / "bad-encoding.mcap"
    _write_execution(
        bad_encoding,
        steps=[_step(0, sim_time=0.1)],
        footer={"type": "footer", "steps": 1, "complete": True},
        encoding="cdr",
    )
    with pytest.raises(SimRecordingError, match="unexpected MCAP encoding"):
        read_sim_execution(bad_encoding, "episode-1", expected_steps=1)

    bad_crc = tmp_path / "bad-crc.mcap"
    _write_execution(
        bad_crc,
        steps=[_step(0, sim_time=0.1)],
        footer={"type": "footer", "steps": 1, "complete": True},
    )
    corrupted = bytearray(bad_crc.read_bytes())
    crc_probe = len(corrupted) // 4
    corrupted[crc_probe] ^= 1
    bad_crc.write_bytes(corrupted)
    with pytest.raises(SimRecordingError, match="invalid MCAP"):
        read_sim_execution(bad_crc, "episode-1", expected_steps=1)

    bad_footer = tmp_path / "bad-footer-count.mcap"
    _write_execution(
        bad_footer,
        steps=[_step(0, sim_time=0.1)],
        footer={"type": "footer", "steps": 2, "complete": True},
    )
    with pytest.raises(SimRecordingError, match="footer step count"):
        read_sim_execution(bad_footer, "episode-1", expected_steps=1)


def test_collection_fk_batches_state_and_action_together() -> None:
    class _Solver:
        def __init__(self) -> None:
            self.chunks: list[np.ndarray] = []

        def fk_chunk(self, qpos: np.ndarray) -> np.ndarray:
            self.chunks.append(qpos.copy())
            result = np.zeros((len(qpos), 8), dtype=np.float32)
            result[:, :2] = qpos[:, :2]
            return result

    solver = _Solver()
    robot = SimpleNamespace(total_action_dim=2, arm_groups=(object(),))
    writer = object.__new__(CollectionEpisodeWriter)
    writer._schema = SimpleNamespace(arms={"arm": "arm"})
    writer._logger = SimpleNamespace(
        _robot=robot,
        _fps=30.0,
        _eef_reference_frame=None,
    )
    writer._client_fk_solver = solver
    frames = [
        Observation(
            images={},
            state_qpos=np.asarray([index, index + 1], dtype=np.float32),
            action_qpos=np.asarray([index + 10, index + 11], dtype=np.float32),
        )
        for index in range(50)
    ]

    writer._derive_eef(frames)

    assert len(solver.chunks) == 1
    assert solver.chunks[0].shape == (128, 2)
    np.testing.assert_array_equal(solver.chunks[0][:50], [frame.state_qpos for frame in frames])
    np.testing.assert_array_equal(solver.chunks[0][50:100], [frame.action_qpos for frame in frames])
    np.testing.assert_array_equal(frames[0].state_eef[:2], [0.0, 1.0])
    np.testing.assert_array_equal(frames[-1].action_eef[:2], [59.0, 60.0])


def test_sim_reader_rejects_missing_time_and_sequence_gap(tmp_path: Path) -> None:
    missing_time = tmp_path / "missing-time.mcap"
    _write_execution(
        missing_time,
        steps=[_step(0)],
        footer={"type": "footer", "steps": 1, "complete": True},
    )
    with pytest.raises(SimRecordingError, match="sim_time"):
        read_sim_execution(missing_time, "episode-1", expected_steps=1)

    gap = tmp_path / "gap.mcap"
    _write_execution(
        gap,
        steps=[_step(1, sim_time=0.1)],
        footer={"type": "footer", "steps": 1, "complete": True},
    )
    with pytest.raises(SimRecordingError, match="sequence gap"):
        read_sim_execution(gap, "episode-1", expected_steps=1)


def test_sim_reader_rejects_incomplete_footer(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.mcap"
    _write_execution(path, steps=[_step(0, sim_time=0.1)])

    with pytest.raises(SimRecordingError, match="incomplete"):
        read_sim_execution(path, "episode-1", expected_steps=1)


def test_sim_reader_rejects_truncated_mcap(tmp_path: Path) -> None:
    path = tmp_path / "truncated.mcap"
    _write_execution(
        path,
        steps=[_step(0, sim_time=0.1)],
        footer={"type": "footer", "steps": 1, "complete": True},
    )
    path.write_bytes(path.read_bytes()[:-32])

    with pytest.raises(SimRecordingError, match="MCAP"):
        read_sim_execution(path, "episode-1", expected_steps=1)


def test_scene_metadata_accepts_canonical_composite_task_id() -> None:
    scene_id = "v12_eval_v12_case_m059_001"
    task_id = "v12_eval_v12_EV1C-ACT02"
    sim_collection._check_scene_metadata(
        {
            "task_id": f"{task_id}__{scene_id}",
            "scene_id": scene_id,
            "metadata": {
                "task_item": {"task_item_id": f"{task_id}__{scene_id}"},
                "layout": {"layout_index": 0, "uid": "layout-0"},
            },
        },
        {
            "task_id": task_id,
            "scene_id": scene_id,
            "sim_metadata": {
                "task_item": {"task_item_id": f"{task_id}__{scene_id}"}
            },
        },
    )


def test_scene_metadata_rejects_different_composite_task_id() -> None:
    scene_id = "v12_eval_v12_case_m059_001"
    with pytest.raises(SimRecordingError, match="task_id"):
        sim_collection._check_scene_metadata(
            {
                "task_id": f"v12_eval_v12_EV1C-ACT03__{scene_id}",
                "scene_id": scene_id,
            },
            {
                "task_id": "v12_eval_v12_EV1C-ACT02",
                "scene_id": scene_id,
                "sim_metadata": {
                    "task_item": {
                        "task_item_id": f"v12_eval_v12_EV1C-ACT02__{scene_id}"
                    }
                },
            },
        )


def test_sim_recording_path_is_absolute_and_exclusive(tmp_path: Path) -> None:
    handle = create_sim_recording_handle(tmp_path, scene_revision=3)

    assert handle.path.is_absolute()
    assert handle.path == (tmp_path / "raw" / f"{handle.episode_id}.mcap").resolve()
    assert not handle.path.exists()
    handle.path.open("x", encoding="utf-8").close()


def test_sim_rpc_failure_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(sim_collection._SIM_OPENER, "open", fail)
    handle = SimRecordingHandle("episode-1", tmp_path / "episode.mcap", 4)

    with pytest.raises(SimRecordingError, match="RPC failed"):
        SimRecordingClient("http://127.0.0.1:18093").stop(handle)


def test_cancel_rpc_failure_releases_local_handle_and_preserves_raw(tmp_path: Path) -> None:
    handle = SimRecordingHandle("episode-1", tmp_path / "episode.mcap", 4)
    handle.path.write_bytes(b"incomplete raw")

    class _Client:
        def cancel(self, _handle):
            raise SimRecordingError("cancel unavailable")

    writer = object.__new__(CollectionEpisodeWriter)
    writer._sim_client = _Client()
    writer._sim_handle = handle

    with pytest.raises(SimRecordingError, match="cancel unavailable"):
        writer.cancel_sim_recording()
    assert writer._sim_handle is None
    assert handle.path.read_bytes() == b"incomplete raw"


def test_start_cancel_failure_keeps_uncertain_sim_handle(tmp_path: Path) -> None:
    class _Client:
        def resolve_scene_revision(self, _metadata):
            return 4

        def start(self, _handle, _metadata):
            raise SimRecordingError("start unavailable")

        def cancel(self, _handle):
            raise SimRecordingError("cancel unavailable")

    writer = object.__new__(CollectionEpisodeWriter)
    writer._sim_client = _Client()
    writer._sim_handle = None

    with pytest.raises(RuntimeError, match="cancel failed"):
        writer.start_sim_recording(tmp_path, {})
    assert writer._sim_handle is not None


def test_sim_progress_uses_cached_status_steps_and_surfaces_rpc_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handle = SimRecordingHandle("episode-1", tmp_path / "episode.mcap", 4)
    calls = 0

    class _Client:
        def recording_steps(self, _episode_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise SimRecordingError("progress unavailable")
            return 7

    writer = object.__new__(CollectionEpisodeWriter)
    writer._sim_client = _Client()
    writer._sim_handle = handle
    writer._sim_status_steps = 0
    writer._sim_status_checked_at = 0.0
    writer._sim_status_error = ""
    writer._records = []
    writer._received_snapshots = 0
    writer._skipped_before_start = 0
    writer._state_lock = threading.Lock()
    clock = iter((1.0, 1.1, 1.6))
    monkeypatch.setattr(collection_module.time, "monotonic", lambda: next(clock))

    assert writer.frame_counts()["received"] == 7
    assert writer.frame_counts()["received"] == 7
    assert calls == 1
    progress = writer.frame_counts()
    assert progress["recorded"] == 7
    assert calls == 2
    assert progress["sim_recording_error"] == "progress unavailable"
