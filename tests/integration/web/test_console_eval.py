"""Console EVAL/RESULT tabs: config serialization + result persistence contract.

The eval state machine (start/stop) needs a live policy + the main loop, which is
out of scope offline. What these tests pin down is the part that survives a reload:
the flat-prompt eval section in /api/config, and the clip_id-keyed result store the
EVAL tab writes and the RESULT tab reads back (with the episode-index join).
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

from _harness import WebHarness, build_runtime, console_config, free_port

from core.app.console.episode_preview import EpisodePreview
from core.app.console.server import (
    ConsoleContext,
    ConsoleRequestHandler,
    ThreadingHTTPServer,
)
from core.config import ConfigDict


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


def _serve_eval_console(config, output_dir: str):
    @contextlib.contextmanager
    def _serve():
        runtime, session = build_runtime(config)
        port = free_port()
        ctx = ConsoleContext(
            config=config,
            runtime=runtime,
            session=session,
            obs_reader=runtime.transport.create_observation_reader(),
            scene=None,
            output_dir=output_dir,
            preview=EpisodePreview(config, Path(output_dir)),
        )
        ConsoleRequestHandler.ctx = ctx
        server = ThreadingHTTPServer(("127.0.0.1", port), ConsoleRequestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield WebHarness(config, runtime, session, port)
        finally:
            server.shutdown()
            server.server_close()
            runtime.transport.close()

    return _serve()


def test_config_serves_flat_eval_section(tmp_path):
    with _serve_eval_console(_eval_config(), str(tmp_path)) as h:
        cfg = h.get("/api/config").json
        ev = cfg["eval"]
        assert [p["prompt_en"] for p in ev["tasks"]] == ["pick up the block", "pick up the cup"]
        assert ev["trials_per_prompt"] == 3
        assert [m["id"] for m in ev["tasks"][0]["milestones"]] == ["grasp", "place"]
        assert "default_milestones" not in ev
        # No scene/pos/coord leak into the flat schema.
        assert "scenes" not in ev
        assert all("pos" not in p and "coord" not in p for p in ev["tasks"])


def test_results_read_from_episodes_jsonl(tmp_path):
    # New contract: scores live in the lerobot dataset's meta/episodes.jsonl (no separate
    # results.jsonl). /api/results reads those rows back, so a re-launch auto-resumes.
    with _serve_eval_console(_eval_config(), str(tmp_path)) as h:
        assert h.get("/api/results").json["records"] == []

        ep_meta = tmp_path / "episodes" / "meta"
        ep_meta.mkdir(parents=True)
        (ep_meta / "episodes.jsonl").write_text(
            '{"episode_index": 7, "clip_id": "c1", "prompt": "pick up the block", '
            '"trial": 1, "score": 2, "max_score": 4, "milestones": {"grasp": true}, '
            '"note": "ok"}\n',
            encoding="utf-8",
        )

        recs = h.get("/api/results").json["records"]
        assert len(recs) == 1
        assert recs[0]["episode_index"] == 7
        assert recs[0]["clip_id"] == "c1"
        assert recs[0]["score"] == 2
        assert recs[0]["note"] == "ok"


def test_results_dedup_latest_episode_per_clip(tmp_path):
    # A re-test reuses the clip_id on a later episode; the latest episode row wins.
    with _serve_eval_console(_eval_config(), str(tmp_path)) as h:
        ep_meta = tmp_path / "episodes" / "meta"
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
