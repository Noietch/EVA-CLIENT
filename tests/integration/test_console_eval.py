"""Console EVAL/RESULT tabs: config serialization + result persistence contract.

The eval state machine (start/stop) needs a live policy + the main loop, which is
out of scope offline. What this test pins down is the part that survives a reload:
the clip_id-keyed result store the EVAL tab writes and the RESULT tab reads back
(with the episode-index join).
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from core.app import run as app
from core.app.console import server
from core.app.handlers.recording import eval_episode_dir
from core.app.state import SessionStatus
from core.config import ConfigDict, load_config
from core.recorder.episode import EpisodeLogger
from core.types import Observation
from tests.integration._harness import console_config, serve_console

pytestmark = pytest.mark.integration


def _eval_config():
    eval_cfg = ConfigDict(
        trials_per_prompt=3,
        cli_mode="real",
        checkpoints=(),
        tasks=(
            ConfigDict(
                prompt_en="pick up the block",
                prompt_zh="pick up the block",
                milestones=(("grasp", "grasp"), ("place", "place")),
            ),
            ConfigDict(
                prompt_en="pick up the cup",
                prompt_zh="pick up the cup",
                milestones=(("grasp", "grasp"), ("place", "place")),
            ),
        ),
    )
    cfg = console_config()
    cfg.eval = eval_cfg
    return cfg


def test_results_dedup_latest_episode_per_clip(tmp_path):
    # A re-test reuses the clip_id on a later episode; the latest episode row wins.
    with serve_console(_eval_config(), tmp_path) as h:
        ep_meta = tmp_path / "episodes" / "raw" / "meta"
        ep_meta.mkdir(parents=True)
        (ep_meta / "episodes.jsonl").write_text(
            '{"episode_index": 0, "clip_id": "c1", "trial": 1, "score": 1}\n'
            '{"episode_index": 3, "clip_id": "c1", "trial": 1, "score": 4}\n',
            encoding="utf-8",
        )
        recs = h.get("/api/results").json["records"]
        assert len(recs) == 1  # one row per clip
        assert recs[0]["episode_index"] == 3  # latest episode wins
        assert recs[0]["score"] == 4


def _scene_plan(root, task_id, prompt, photo):
    root.mkdir(parents=True)
    (root / "tasks.csv").write_text(
        "task_id,prompt_en,scene_ids,scene_epsiodes_count,total_epsiodes_count\n"
        f"{task_id},{prompt},SC-1,1,1\n"
    )
    placements = json.dumps([{"object_id": "CUP", "position_ids": ["P1"]}])
    with (root / "scene.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_id", "placements"])
        writer.writerow(["SC-1", placements])
    (root / "layout.yaml").write_text("sampling_points:\n- {position_id: P1, x: 0, y: 0}\n")
    (root / "objects.csv").write_text("object_id,name_en,photo_dir\nCUP,cup,cup\n")
    photos = root / "object_photos/cup"
    photos.mkdir(parents=True)
    (photos / "cup.png").write_bytes(photo)


def test_eval_scene_plan_and_photos_use_the_evaluation_task_set(tmp_path):
    eval_root = tmp_path / "eval/batch"
    collect_root = tmp_path / "collect/batch"
    _scene_plan(eval_root, "EVAL-CUP", "pick up the cup", b"eval-photo")
    _scene_plan(collect_root, "COLLECT-CUP", "collect the cup", b"collect-photo")
    cfg = _eval_config()
    cfg.eval.task_set_dir = [str(eval_root)]
    cfg.eval.tasks[1].task_id = "EVAL-CUP"
    cfg.eval.tasks[1].task_set = "batch"
    cfg.collection.task_set_dir = str(collect_root)
    with serve_console(cfg) as h:
        config = h.get("/api/config").json
        assert config["eval"]["scene_plan_sets"] == ["batch"]
        assert config["eval"]["tasks"][1]["task_id"] == "EVAL-CUP"
        assert config["eval"]["tasks"][1]["task_set"] == "batch"
        response = h.get("/api/scene_plan?scope=eval&set=batch")
        assert response.status == 200
        assert response.json["tasks"][0]["task_id"] == "EVAL-CUP"
        assert response.json["scenes"][0]["placements"][0]["position_id"] == "P1"
        photo_url = response.json["objects"][0]["photo_url"]
        assert "scope=eval" in photo_url
        assert h.get(photo_url).raw == b"eval-photo"
        collection = h.get("/api/scene_plan?set=batch")
        assert collection.json["tasks"][0]["task_id"] == "COLLECT-CUP"
        assert h.get(collection.json["objects"][0]["photo_url"]).raw == b"collect-photo"
        assert h.get("/api/scene_plan?scope=eval&set=unknown").status == 400
        assert h.get("/api/scene_plan?scope=unknown").status == 400


def test_eval_scene_plan_requires_explicit_evaluation_configuration(tmp_path):
    _scene_plan(tmp_path / "batch", "CUP", "pick up the cup", b"photo")
    cfg = _eval_config()
    cfg.collection.task_set_dir = str(tmp_path / "batch")
    cfg.collection.storage.data_root = str(tmp_path)
    with serve_console(cfg) as h:
        response = h.get("/api/scene_plan?scope=eval")
        assert response.status == 400
        config = h.get("/api/config").json
        assert config["scene_plan_sets"]["eval"] == []
        assert config["eval"]["scene_plan_sets"] == []


def test_dashboard_and_eval_can_select_an_unmounted_evaluation_task_set(tmp_path):
    train_root = tmp_path / "task_sets/larybenchv2.v1.2/train"
    eval_root = tmp_path / "task_sets/larybenchv2.v1.2/eval"
    _scene_plan(train_root, "TRAIN-CUP", "collect the cup", b"train-photo")
    _scene_plan(eval_root, "EVAL-CUP", "pick up the cup", b"eval-photo")
    cfg = _eval_config()
    cfg.collection.storage.data_root = str(tmp_path)
    cfg.eval.task_set_dir = str(tmp_path / "task_sets")
    cfg.collection.task_set_dir = [str(train_root)]
    cfg.collection.tasks = {"train": [("collect the cup", 1)]}
    with serve_console(cfg) as h:
        config = h.get("/api/config").json
        assert config["scene_plan_sets"]["collect"] == ["train"]
        assert set(config["scene_plan_sets"]["eval"]) == {
            "larybenchv2.v1.2/train",
            "larybenchv2.v1.2/eval",
        }
        assert config["eval"]["scene_plan_sets"] == config["scene_plan_sets"]["eval"]
        response = h.get("/api/scene_plan?scope=eval&set=larybenchv2.v1.2%2Feval")
        assert response.status == 200
        assert response.json["tasks"][0]["task_id"] == "EVAL-CUP"
        assert h.get(response.json["objects"][0]["photo_url"]).raw == b"eval-photo"


def test_version_task_set_identity_survives_catalog_growth(tmp_path):
    older = tmp_path / "task_sets/larybenchv2.v1.1/eval"
    _scene_plan(older, "CUP", "pick up the cup", b"old-photo")
    cfg = _eval_config()
    cfg.collection.storage.data_root = str(tmp_path)
    cfg.eval.task_set_dir = str(tmp_path / "task_sets")
    cfg.dashboard = ConfigDict(raw_roots=[str(tmp_path / "run")])
    with serve_console(cfg, tmp_path / "run") as h:
        original = h.get("/api/config").json["scene_plan_sets"]["eval"]
        assert original == ["larybenchv2.v1.1/eval"]
        snapshot = h.get("/api/scene_plan?scope=eval&set=larybenchv2.v1.1%2Feval").json
        meta = tmp_path / "run/model/episodes/raw/meta"
        meta.mkdir(parents=True)
        record = {
            "episode_index": 0,
            "clip_id": "before-version-added",
            "task_set": original[0],
            "task_id": "CUP",
            "scene_id": "SC-1",
            "scene_map": snapshot,
            "length": 30,
        }
        (meta / "episodes.jsonl").write_text(json.dumps(record) + "\n")
        _scene_plan(tmp_path / "task_sets/larybenchv2.v1.2/eval", "CUP", "new cup", b"new")
        current = h.get("/api/config").json["scene_plan_sets"]["eval"]
        assert current == ["larybenchv2.v1.1/eval", "larybenchv2.v1.2/eval"]
        assert h.get("/api/scene_plan?scope=eval&set=larybenchv2.v1.1%2Feval").status == 200
        dashboard = h.get(
            "/api/dashboard?scene_scope=eval&task_set=larybenchv2.v1.1%2Feval"
            "&task_id=CUP&scene_id=SC-1"
        ).json
        assert dashboard["views"]["eval"]["episodes"] == 1
        assert h.get("/api/dashboard").json["views"]["eval"]["episodes"] == 1
        results = h.get("/api/results_all").json
        rows = next(row["records"] for row in results["models"] if row["model_name"] == "model")
        assert rows[0]["scene_map"] == snapshot


def test_sim_eval_preset_has_valid_empty_model_and_task_choices():
    cfg = load_config("configs/03_evaluation/dual_yam_eva_sim_eval.py")
    assert cfg.eval.checkpoints == []
    assert cfg.eval.tasks == []
    cfg.transport.type = "debug"
    cfg.policy.type = "mock"
    with serve_console(cfg) as h:
        response = h.get("/api/config")
        assert response.status == 200
        assert response.json["eval"]["tasks"] == []


def test_scene_catalog_and_photos_do_not_change_with_checkpoint_config(tmp_path):
    base = tmp_path / "base/task_sets"
    active = tmp_path / "checkpoint/task_sets"
    for version in ("larybenchv2.v1.1", "larybenchv2.v1.2"):
        _scene_plan(base / version / "eval", "CUP", "pick up the cup", version.encode())
    _scene_plan(active / "other", "OTHER", "other prompt", b"wrong photo")
    cfg = _eval_config()
    cfg.collection.storage.data_root = str(base.parent)
    cfg.eval.task_set_dir = str(base)
    with serve_console(cfg) as h:
        changed = _eval_config()
        changed.collection.storage.data_root = str(active.parent)
        h.runtime.active_config = changed
        response = h.get("/api/config")
        names = response.json["eval"]["scene_plan_sets"]
        assert names == ["larybenchv2.v1.1/eval", "larybenchv2.v1.2/eval"]
        assert response.json["scene_plan_sets"]["eval"] == names
        for name in names:
            plan = h.get("/api/scene_plan?scope=eval&set=" + name).json
            assert plan["task_set"] == name
            assert h.get(plan["objects"][0]["photo_url"]).raw == name.split("/")[0].encode()


def test_optional_scene_discovery_error_is_reported_without_blocking_console(tmp_path, monkeypatch):
    cfg = _eval_config()

    def unavailable(_config):
        raise PermissionError("scene task-set mount unavailable")

    monkeypatch.setattr(server, "evaluation_task_sets", unavailable)
    with serve_console(cfg) as h:
        response = h.get("/api/config")
        assert response.status == 200
        assert response.json["scene_plan_sets"]["eval"] == []
        assert "mount unavailable" in response.json["scene_plan_errors"]["eval"]
        assert h.get("/api/scene_plan?scope=eval").status == 400


def test_eval_scene_start_records_identity_and_score_binds_to_the_saved_attempt(
    tmp_path, monkeypatch
):
    plan_root = tmp_path / "plans/eval-cups"
    _scene_plan(plan_root, "CUP", "pick up the cup", b"cup")
    cfg = _eval_config()
    cfg.eval.task_set_dir = str(plan_root)
    cfg.eval.reset_after_each_trial = False
    with serve_console(cfg, tmp_path / "run") as h:
        recorder = EpisodeLogger(
            eval_episode_dir(h.runtime.console_ctx.output_dir, "model"),
            h.runtime.robot,
            fps=10,
            dataset_keys=cfg.transport.dataset_keys,
            eval_mode=True,
        )
        h.runtime.episode_logger = recorder

        def dispatch(_config, runtime, session):
            recorder.start_episode(session.selected_task)
            state = np.zeros(runtime.robot.total_action_dim, dtype=np.float32)
            for frame in range(3):
                recorder.record_step(
                    Observation(state_qpos=state, images={}), state, timestamp=10.0 + frame / 10
                )
            session.status = SessionStatus.RUNNING

        monkeypatch.setattr(app, "_dispatch_run", dispatch)
        monkeypatch.setattr(app, "run_setup", lambda *_args: True)
        h.runtime.web_phase = "ready"
        body = dict(
            clip_id="scene-attempt",
            prompt="pick up the cup",
            trial=1,
            task_set="eval-cups",
            task_id="CUP",
            scene_id="SC-1",
            model_name="model",
        )
        assert h.do("/api/eval_start", body).status == 200
        assert recorder._episode_meta["scene_id"] == "SC-1"
        assert h.status()["cell"]["task_set"] == "eval-cups"
        assert h.post("/api/eval_start", {**body, "clip_id": "wrong"}).status == 409
        assert h.runtime.current_clip_id == "scene-attempt"
        assert h.do("/api/eval_stop").status == 200
        recorder.wait_for_saves(fail_on_error=True)
        record = h.get("/api/results").json["records"][0]
        assert record["task_id"] == "CUP"
        assert record["scene_map"]["scenes"][0]["scene_id"] == "SC-1"
        score = {**body, "score": 1, "max_score": 1}
        assert h.post("/api/score", {**score, "scene_id": "SC-2"}).status == 409
        assert h.post("/api/score", {**score, "model_name": "other"}).status == 409
        assert h.post("/api/score", score).status == 200
        assert h.get("/api/results").json["records"][0]["score"] == 1
        for invalid in (
            {"scene_id": "SC-2"},
            {"sim_metadata": {"scene": {"scene_id": "SC-2"}}},
            {"prompt": "unrelated task"},
        ):
            assert h.post("/api/eval_start", {**body, **invalid}).status == 400
            assert h.runtime.current_clip_id is None
        assert h.runtime.command_queue.empty()


def test_results_only_include_configured_models(tmp_path):
    for model in ("selected", "unconfigured"):
        meta = tmp_path / model / "episodes/raw/meta"
        meta.mkdir(parents=True)
        (meta / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "clip_id": model, "length": 30}) + "\n"
        )
    cfg = _eval_config()
    cfg.eval.result_models = ["selected", "selected"]
    with serve_console(cfg, tmp_path) as console:
        models = console.get("/api/results_all").json["models"]
    assert [entry["model_name"] for entry in models].count("selected") == 1
    assert "unconfigured" not in [entry["model_name"] for entry in models]
    assert next(entry for entry in models if entry["model_name"] == "selected")["records"]
    cfg.eval = None
    with serve_console(cfg, tmp_path) as console:
        assert console.get("/api/results_all").json["models"] == []


def test_dashboard_requires_explicit_roots_even_when_work_and_output_have_records(tmp_path):
    root = tmp_path / "work"
    meta = root / "old_model/episodes/raw/meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "clip_id": "old", "length": 30}) + "\n"
    )
    cfg = _eval_config()
    cfg.work_dir = str(root)
    cfg.collection.storage.data_root = str(root)
    with serve_console(cfg, root) as console:
        payload = console.get("/api/dashboard").json
        assert payload["sources"] == []
        assert payload["upload_candidates"] == []
        cfg.dashboard = ConfigDict(raw_roots=str(root))
        payload = console.get("/api/dashboard").json
        assert len(payload["sources"]) == 1
        assert payload["views"]["eval"]["episodes"] == 1
