"""One four-state QC model shared by the console and the dataset tools.

A verdict recorded in ``meta/qc.jsonl`` always outranks the capture-time
``quality`` flag: an explicit ``pass`` keeps an episode the capture marked red,
and an explicit ``fail`` rejects one that recorded green. Capture quality only
decides while no verdict stands, so a red episode the reviewer has not judged
yet still reads as failed. An explicit human pass also overrides SIM replay
failures and pending reports; the replay result and reason remain unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def qc_state(
    verdict: Any,
    quality: Any,
    replay_verified: Any = None,
    replay_required: bool = False,
    qc_auto_verdict: Any = None,
) -> str:
    """Resolve one episode's recorded verdict and capture quality into its state.

    Returns ``unreviewed``, ``passed`` or ``failed``. An explicit pass overrides
    machine and replay failures. Without it, a required failed replay rejects
    the episode, and a machine failure remains effective.
    """
    verdict = str(verdict or "").strip().lower()
    if verdict == "pass":
        return "passed"
    state = {
        "unreviewed": "unreviewed",
        "fail": "failed",
    }.get(verdict, "failed" if str(quality or "").lower() == "red" else "unreviewed")
    if verdict != "fail" and qc_auto_verdict == "fail":
        state = "failed"
    if replay_required and replay_verified is False:
        return "failed"
    return state


def replay_report_path(dataset_dir: Path, episode: dict[str, Any]) -> Path | None:
    """Resolve the actual report for an episode with a SIM replay contract."""
    report_ref = str(episode.get("render_report") or "").strip()
    if not isinstance(episode.get("sim_metadata"), dict) and not report_ref:
        return None
    path = (
        Path(report_ref)
        if report_ref
        else Path("meta/render_reports/episode_{:06d}.json".format(int(episode["episode_index"])))
    )
    return path if path.is_absolute() else Path(dataset_dir) / path


def replay_report_signature(dataset_dir: Path, episodes: list[dict[str, Any]]) -> tuple[int, ...]:
    """Track each SIM report itself, including in-place edits and custom paths."""
    signature = []
    for episode in episodes:
        path = replay_report_path(dataset_dir, episode)
        if path is None:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            signature.extend((0, 0, 0))
        else:
            signature.extend((stat.st_ino, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def replay_validation(dataset_dir: Path, episode: dict[str, Any]) -> tuple[bool | None, str]:
    """Validate the SIM replay summary for one saved episode.

    ``(None, "")`` means this is an ordinary recording without a SIM replay contract;
    ``(None, "replay_report_pending")`` means a SIM report is still being produced.
    A SIM recording is identified by ``sim_metadata`` or ``render_report`` in its
    episode metadata. Human QC overrides are resolved separately by ``qc_state``.
    """
    report_path = replay_report_path(dataset_dir, episode)
    if report_path is None:
        return None, ""
    try:
        report_text = report_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "replay_report_pending"
    try:
        report = json.loads(report_text)
    except ValueError:
        return False, "replay_report_invalid"
    if not isinstance(report, dict):
        return False, "replay_report_invalid"
    evaluation = report.get("task_evaluation")
    if not isinstance(evaluation, dict):
        return False, "replay_evaluation_missing"
    if (
        "success" not in evaluation
        or "error" not in evaluation
        or evaluation["success"] is not True
        or evaluation["error"] is not None
    ):
        error = evaluation.get("error")
        reason = str((error.get("message") if isinstance(error, dict) else error) or "")
        result = evaluation.get("final_result")
        diagnostics = result.get("diagnostics") if isinstance(result, dict) else None
        if (
            isinstance(diagnostics, dict)
            and isinstance(diagnostics.get("joint"), str)
            and all(
                isinstance(diagnostics.get(key), (int, float))
                for key in ("target_position", "joint_position", "target_error")
            )
        ):
            joint_reason = (
                f"关节 {diagnostics['joint']}："
                f"目标 {diagnostics['target_position']:.5g}，"
                f"实际 {diagnostics['joint_position']:.5g}，"
                f"偏差 {diagnostics['target_error']:.5g}"
            )
            reason = f"{reason}；{joint_reason}" if reason else joint_reason
        return False, reason or "replay_summary_failed"
    return True, ""
