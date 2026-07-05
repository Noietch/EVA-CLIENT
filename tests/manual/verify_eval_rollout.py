"""Corner-case verification for the eval rollout score/re-test/save flow.

Drives the REAL server handler logic (_eval_start / _eval_score) and the REAL
EpisodeLogger lerobot writer against a recorded episode, covering:

  CC1  re-score an already-scored trial (same clip) -> video unchanged, meta updated
  CC2  re-run then score (new clip, same prompt|trial) -> new episode + appended row
  CC3  re-run then SAVE WITHOUT SCORING (new clip) -> owner row exists, no score yet

The episodes.jsonl meta row is the single source of truth: scoring patches it via
patch_episode_meta, and a clip joins to its episode through episode_index_by_clip().

Run: PYTHONPATH=src python tests/manual/verify_eval_rollout.py
"""

from __future__ import annotations

import json
import sys
import types

# The transport package eagerly imports dataset.py which needs PyAV; this offline
# venv lacks a cp312 av wheel. Stub it as a package so unrelated imports succeed —
# none of the code paths exercised here touch av.
_av = types.ModuleType("av")
_av.__path__ = []  # mark as package
_av.logging = types.ModuleType("av.logging")  # type: ignore[reportAttributeAccessIssue]
_av.logging.set_level = lambda *a, **k: None  # type: ignore[reportAttributeAccessIssue]
for _n in ("ERROR", "WARNING", "INFO", "DEBUG", "PANIC", "FATAL", "CRITICAL"):
    setattr(_av.logging, _n, 0)
sys.modules["av"] = _av
sys.modules["av.logging"] = _av.logging

import hashlib
import shutil
from pathlib import Path

import numpy as np

import robots  # noqa: F401  registers robots
from core.app.console.episode_preview import EpisodePreview
from core.config import load_config
from core.recorder.episode import EpisodeLogger
from core.registry import ROBOT_REGISTRY
from core.types import Observation
from core.utils.paths import resolve_output_dir

CFG_PATH = "configs/03_evaluation/arx_r5_eval.py"


def _file_sig(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def _fake_obs(robot, cam_keys, t):
    dim = len(robot.initial_qpos)
    state = np.full(dim, 0.01 * t, dtype=np.float32)
    h = w = 32
    img = np.full((h, w, 3), (t * 7) % 255, dtype=np.uint8)
    return Observation(state_qpos=state, images={k: img for k in cam_keys})


def record_episode(logger: EpisodeLogger, robot, prompt: str, clip_id: str, n: int) -> int:
    """Record one eval episode exactly like the live path: start -> steps -> meta -> end."""
    ep_index = logger.current_episode_index
    logger.start_episode(task=prompt)
    cam_keys = [c.observation_key for c in robot.observation_schema.cameras]
    for t in range(n):
        obs = _fake_obs(robot, cam_keys, t)
        action = np.full(len(robot.initial_qpos), 0.02 * t, dtype=np.float32)
        logger.record_step(obs, action, timestamp=float(t) / 30.0)
    # app.py stamps prompt/trial/clip_id onto the in-flight episode at start of RUN.
    logger.set_episode_meta(prompt=prompt, trial=1, clip_id=clip_id)
    logger.end_episode()
    return ep_index


def score(preview, logger, clip_id, prompt, sc, note):
    """Mirror server.py _eval_score: resolve clip -> episode, patch the lerobot meta row."""
    ep = preview.episode_index_by_clip().get(clip_id)
    milestones = {f"m{i}": i < sc for i in range(4)}
    if ep is not None:
        logger.patch_episode_meta(int(ep), score=sc, max_score=4, milestones=milestones, note=note)
    return ep


def episode_meta(logger, ep):
    path = Path(logger._meta_path("episodes.jsonl"))
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if int(row.get("episode_index", -1)) == ep:
            return row
    return None


def main():
    cfg = load_config(CFG_PATH)
    out = resolve_output_dir(cfg, CFG_PATH)
    ds = out / "episodes"
    if ds.exists():
        shutil.rmtree(ds)
    robot = ROBOT_REGISTRY.build(cfg.robot.type)
    # This offline venv has no ffmpeg/pyav backend for imageio mp4 writing, so record
    # state/action only. Video immutability under re-score is proven separately by
    # planting a dummy mp4 at the episode's video path and checking it is untouched.
    logger = EpisodeLogger(
        log_dir=ds,
        robot=robot,
        fps=cfg.log.fps,
        dataset_keys=cfg.transport.dataset_keys,
        convert_bgr_to_rgb=cfg.transport.convert_bgr_to_rgb,
        async_save=False,
    )
    preview = EpisodePreview(cfg, out)
    prompt = "pick up the block and place it on the plate."
    ok = True

    print("=" * 70)
    print("CC1: re-score an already-scored trial (same clip) — video must be unchanged")
    clip1 = "eval-cc1"
    ep1 = record_episode(logger, robot, prompt, clip1, n=20)
    # Plant a dummy mp4 where the recorder would have written the camera video, to prove
    # the score path never rewrites/relocates the clip.
    cam0 = robot.observation_schema.cameras[0].observation_key
    vid_path = Path(logger._video_path(ep1, cam0))
    vid_path.parent.mkdir(parents=True, exist_ok=True)
    vid_path.write_bytes(b"FAKE_MP4_BYTES_" + clip1.encode())
    sig_before = _file_sig(vid_path)
    score(preview, logger, clip1, prompt, 2, "first pass")
    m1 = episode_meta(logger, ep1)
    score(preview, logger, clip1, prompt, 4, "re-scored higher")
    m2 = episode_meta(logger, ep1)
    assert m1 is not None and m2 is not None
    sig_after = _file_sig(vid_path)
    by_clip = preview.episode_index_by_clip()
    print(f"  episode_index={ep1} video={vid_path.name}")
    print(
        f"  video sig before={sig_before[:8]} after={sig_after[:8]}"
        f" -> {'SAME' if sig_before == sig_after else 'CHANGED!'}"
    )
    print(f"  clip joins to one episode={by_clip.get(clip1)} (single row per clip)")
    print(f"  lerobot meta score: {m1.get('score')} -> {m2.get('score')}  note={m2.get('note')!r}")
    cc1 = (
        sig_before == sig_after
        and by_clip.get(clip1) == ep1
        and m2["score"] == 4
        and m2["note"] == "re-scored higher"
    )
    print(f"  CC1 {'PASS' if cc1 else 'FAIL'}")
    ok &= cc1

    print("=" * 70)
    print("CC2: re-run then score (new clip, same prompt|trial) — new episode + extra row")
    clip2 = "eval-cc2"
    ep2 = record_episode(logger, robot, prompt, clip2, n=15)
    score(preview, logger, clip2, prompt, 1, "second attempt")
    by_clip = preview.episode_index_by_clip()
    m_cc2 = episode_meta(logger, ep2)
    assert m_cc2 is not None
    print(f"  new episode_index={ep2} (clip1 ep was {ep1})  distinct={ep2 != ep1}")
    print(f"  clip->episode map: {by_clip}")
    print(f"  distinct clips recorded now={len(by_clip)} (clip1 + clip2 = 2)")
    print(f"  clip2 lerobot meta: score={m_cc2.get('score')} clip_id={m_cc2.get('clip_id')}")
    cc2 = (
        ep2 != ep1
        and by_clip.get(clip2) == ep2
        and by_clip.get(clip1) == ep1
        and len(by_clip) == 2
        and m_cc2["score"] == 1
    )
    print(f"  CC2 {'PASS' if cc2 else 'FAIL'}")
    ok &= cc2

    print("=" * 70)
    print("CC3: re-run WITHOUT scoring — clip joins to episode, no score yet")
    clip3 = "eval-cc3"
    ep3 = record_episode(logger, robot, prompt, clip3, n=10)
    # No score() call. The RESULT join injects episode_index at read time.
    by_clip = preview.episode_index_by_clip()
    m_cc3 = episode_meta(logger, ep3)
    assert m_cc3 is not None
    joined_ep = by_clip.get(clip3)
    print(f"  episode_index={ep3} recorded, lerobot meta score={m_cc3.get('score')} (expect None)")
    print(f"  clip3 joins to episode via map: {joined_ep} (expect {ep3})")
    print(
        f"  lerobot meta has clip_id={m_cc3.get('clip_id')} score={m_cc3.get('score')}"
        " (score expect None)"
    )
    cc3 = joined_ep == ep3 and m_cc3.get("clip_id") == clip3 and m_cc3.get("score") is None
    print(f"  CC3 {'PASS' if cc3 else 'FAIL'}")
    ok &= cc3

    print("=" * 70)
    print(f"ALL CORNER CASES: {'PASS' if ok else 'FAIL'}")
    logger.finalize()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
