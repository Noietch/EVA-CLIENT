"""End-to-end eval rollout flow over the REAL console server + handlers.

Offline (debug transport + mock policy), drives bootstrap -> run -> stop -> score and
asserts the new contract:

  * recording roots at a PER-MODEL dataset  <output_dir>/<model_name>/episodes  (req2)
  * score/note/milestones land ONLY in that dataset's meta/episodes.jsonl + parquet +
    info.json — no results.jsonl is written                                     (req4)
  * /api/results reads those rows back (so a re-launch auto-resumes)             (req3)
  * /api/score is REJECTED (409) for a trial with no recorded episode            (req1)

Run: PYTHONPATH=src python tests/manual/verify_eval_flow_e2e.py
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
import threading
import types
from pathlib import Path

_av = types.ModuleType("av")
_av.__path__ = []
_log = types.ModuleType("av.logging")
_log.set_level = lambda *a, **k: None  # type: ignore[reportAttributeAccessIssue]
for _n in ("PANIC", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "CRITICAL"):
    setattr(_log, _n, 0)
_av.logging = _log  # type: ignore[reportAttributeAccessIssue]
sys.modules["av"] = _av
sys.modules["av.logging"] = _log

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration" / "web"))

from _harness import WebHarness, build_runtime, free_port

from core.app.console.episode_preview import EpisodePreview
from core.app.console.server import (
    ConsoleContext,
    ConsoleRequestHandler,
    ThreadingHTTPServer,
)
from core.app.handlers import (
    enable_default_recording,
    eval_episode_dir,
    eval_model_name,
    maybe_build_episode_logger,
    publish_next_action,
)
from core.config import ConfigDict, load_config


def _offline(config):
    # Swap the real zmq transport + openpi policy for the offline debug/mock pair, keep
    # everything else (robot, eval section, single-strategy) intact.
    return dataclasses.replace(
        config,
        transport=ConfigDict(type="debug"),
        policy=ConfigDict(type="mock", backend_options=ConfigDict(dict(chunk_size=8))),
        publish_rate=1000,
        supported_inference_strategies={"async": {"type": "sync", "args": {"execute_horizon": 5}}},
    )


def main() -> int:
    cfg = _offline(load_config("configs/03_evaluation/arx_r5_eval.py"))
    assert cfg.eval is not None
    # Each checkpoint carries its own ConfigDict; make those offline too so bootstrap's
    # connect() uses the mock policy instead of dialing port 9000.
    cfg = dataclasses.replace(
        cfg,
        eval=dataclasses.replace(
            cfg.eval,
            checkpoints=tuple(
                dataclasses.replace(c, config=_offline(c.config)) for c in cfg.eval.checkpoints
            ),
        ),
    )
    out = Path(tempfile.mkdtemp(prefix="evalflow_"))
    # No ffmpeg/pyav backend offline -> record state/action only (the metadata flow under
    # test is video-independent).
    cfg = dataclasses.replace(cfg, log=dataclasses.replace(cfg.log, save_video=False))
    cfg = enable_default_recording(cfg, str(out))
    runtime, session = build_runtime(cfg)
    runtime.eval_output_dir = str(out)
    maybe_build_episode_logger(cfg, runtime)

    port = free_port()
    ctx = ConsoleContext(
        config=cfg,
        runtime=runtime,
        session=session,
        obs_reader=runtime.transport.create_observation_reader(),
        scene=None,
        output_dir=str(out),
        preview=EpisodePreview(cfg, out),
    )
    ConsoleRequestHandler.ctx = ctx
    server = ThreadingHTTPServer(("127.0.0.1", port), ConsoleRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    h = WebHarness(cfg, runtime, session, port)
    ok = True
    try:
        h.runtime.command_queue.put("web:bootstrap")
        h.pump()
        model = eval_model_name(cfg, runtime)
        ds = eval_episode_dir(str(out), model)
        print(f"model_name={model}")
        print(f"per-model dataset={ds.relative_to(out)}  exists={ds.exists()}")
        ok &= model == "arx_r5_openpi_baseline"

        assert cfg.eval is not None
        prompt = cfg.eval.prompts[0].prompt_en

        # req1: scoring a trial that never ran is rejected.
        gate = h.do(
            "/api/score",
            {"clip_id": "eval-never", "score": 2, "max_score": 4, "milestones": {}, "note": ""},
        )
        print(f"req1 gate: score-before-run status={gate.status} (expect 409)")
        ok &= gate.status == 409

        # Select prompt + RUN (records an episode), then STOP (flushes it).
        h.do("/api/select_task", {"task": prompt})
        clip = "eval-flow-1"
        h.do("/api/eval_start", {"clip_id": clip, "prompt": prompt, "trial": 1})
        # Drive a few publish steps so the episode has frames.
        for _ in range(6):
            publish_next_action(runtime.active_config or cfg, runtime, session)
        h.do("/api/eval_stop")
        h.pump()

        epjsonl = ds / "meta" / "episodes.jsonl"
        rows = (
            [json.loads(ln) for ln in epjsonl.read_text().splitlines()] if epjsonl.exists() else []
        )
        print(
            f"episodes.jsonl rows={len(rows)} first_clip={rows[0].get('clip_id') if rows else None}"
        )
        ran_ok = len(rows) == 1 and rows[0].get("clip_id") == clip
        ok &= ran_ok

        # Score it — should land in the dataset, NOT a results.jsonl.
        sc = h.do(
            "/api/score",
            {
                "clip_id": clip,
                "prompt": prompt,
                "trial": 1,
                "score": 3,
                "max_score": 4,
                "milestones": {"approach": True, "grasp": True, "lift": True, "place": False},
                "note": "solid run",
            },
        )
        print(f"score status={sc.status} body={sc.json}")
        ok &= sc.status == 200

        row = [json.loads(ln) for ln in epjsonl.read_text().splitlines()][0]
        print(
            f"scored row: score={row.get('score')} note={row.get('note')!r}"
            f" milestones={row.get('milestones')}"
        )
        ok &= row.get("score") == 3 and row.get("note") == "solid run"

        info = json.loads((ds / "meta" / "info.json").read_text())
        print(
            f"info.json eval present={'eval' in info}"
            f" scored={info.get('eval', {}).get('scored_episodes')}"
        )
        ok &= info.get("eval", {}).get("scored_episodes") == 1

        # No results.jsonl anywhere under the eval tree.
        stray = list(out.rglob("results.jsonl"))
        print(f"stray results.jsonl files={[str(p.relative_to(out)) for p in stray]} (expect none)")
        ok &= len(stray) == 0

        # req3: /api/results reads the scored row back from the dataset.
        res = h.get("/api/results").json
        recs = res.get("records", [])
        print(
            f"/api/results: model={res.get('model_name')} records={len(recs)}"
            f" score={recs[0].get('score') if recs else None}"
        )
        ok &= (
            res.get("model_name") == "arx_r5_openpi_baseline"
            and len(recs) == 1
            and recs[0]["score"] == 3
            and recs[0]["episode_index"] == 0
        )
    finally:
        server.shutdown()
        server.server_close()
        runtime.transport.close()
        shutil.rmtree(out, ignore_errors=True)

    print("=" * 60)
    print("eval flow e2e:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
