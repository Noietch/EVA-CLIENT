"""req4 verification: eval scoring metadata is written into the lerobot dataset itself.

Records an eval episode (eval_mode=True), scores it via patch_episode_meta, and asserts
the score/milestones/note land in BOTH meta/episodes.jsonl AND the episode parquet's
eval_score / eval_max_score columns, and that meta/info.json declares the eval section +
eval feature columns. No separate results.jsonl is involved.

Run: PYTHONPATH=src python tests/manual/verify_eval_lerobot_meta.py
"""

from __future__ import annotations

import json
import shutil
import sys
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

import numpy as np
import pyarrow.parquet as pq

import robots  # noqa: F401
from core.config import load_config
from core.recorder.episode import EpisodeLogger
from core.registry import ROBOT_REGISTRY
from core.types import Observation

CFG = "configs/03_evaluation/arx_r5_eval.py"


def main() -> int:
    cfg = load_config(CFG)
    robot = ROBOT_REGISTRY.build(cfg.robot.type)
    ds = Path("work_dirs/_verify_eval_meta")
    if ds.exists():
        shutil.rmtree(ds)
    logger = EpisodeLogger(
        log_dir=ds,
        robot=robot,
        fps=cfg.log.fps,
        dataset_keys=cfg.transport.dataset_keys,
        convert_bgr_to_rgb=cfg.transport.convert_bgr_to_rgb,
        async_save=False,
        eval_mode=True,
    )
    dim = len(robot.initial_qpos)
    prompt = "pick up the block and place it on the plate."

    logger.start_episode(task=prompt)
    for t in range(12):
        obs = Observation(state_qpos=np.full(dim, 0.01 * t, np.float32), images={})
        logger.record_step(obs, np.full(dim, 0.02 * t, np.float32), timestamp=t / 30.0)
    logger.set_episode_meta(prompt=prompt, trial=1, clip_id="eval-meta-1")
    logger.end_episode()
    ep = 0

    ok = True

    # Before scoring: parquet has the eval columns seeded to defaults.
    pqpath = ds / "data" / "chunk-000" / "episode_000000.parquet"
    tbl = pq.read_table(str(pqpath))
    has_cols = "eval_score" in tbl.column_names and "eval_max_score" in tbl.column_names
    seeded = set(tbl.column("eval_score").to_pylist()) == {-1}
    print(f"parquet eval columns present={has_cols} seeded_default={seeded}")
    ok &= has_cols and seeded

    # Score it (mirrors server _eval_score -> patch_episode_meta).
    milestones = {"approach": True, "grasp": True, "lift": False, "place": False}
    logger.patch_episode_meta(
        ep,
        score=2,
        max_score=4,
        milestones=milestones,
        note="ok grasp",
        scored_at="2026-06-20T12:00:00",
    )

    # episodes.jsonl row carries all the scoring keys.
    rows = [json.loads(ln) for ln in (ds / "meta" / "episodes.jsonl").read_text().splitlines()]
    row = rows[0]
    jsonl_ok = (
        row["score"] == 2
        and row["max_score"] == 4
        and row["note"] == "ok grasp"
        and row["milestones"] == milestones
        and row["clip_id"] == "eval-meta-1"
    )
    print(
        f"episodes.jsonl row keys: score={row.get('score')} max={row.get('max_score')} "
        f"note={row.get('note')!r} milestones={row.get('milestones')}"
    )
    ok &= jsonl_ok

    # Parquet eval columns now reflect the score across all frames.
    tbl2 = pq.read_table(str(pqpath))
    sc = set(tbl2.column("eval_score").to_pylist())
    mx = set(tbl2.column("eval_max_score").to_pylist())
    pq_ok = sc == {2} and mx == {4}
    print(f"parquet after score: eval_score={sc} eval_max_score={mx}")
    ok &= pq_ok

    # info.json declares the eval section + eval feature columns.
    info = json.loads((ds / "meta" / "info.json").read_text())
    ev = info.get("eval", {})
    feat = info.get("features", {})
    info_ok = (
        "eval_score" in feat
        and "eval_max_score" in feat
        and ev.get("scored_episodes") == 1
        and ev.get("total_episodes") == 1
        and "score" in ev.get("episode_meta_keys", [])
        and "eval_score" in ev.get("parquet_columns", [])
    )
    print(f"info.json eval section: {json.dumps(ev)}")
    print(f"info.json features has eval cols: {'eval_score' in feat and 'eval_max_score' in feat}")
    ok &= info_ok

    # No separate results.jsonl was created by the recorder.
    no_results = not (ds / "results.jsonl").exists()
    print(f"no recorder-side results.jsonl: {no_results}")
    ok &= no_results

    print("=" * 60)
    print("req4 lerobot-meta verification:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
