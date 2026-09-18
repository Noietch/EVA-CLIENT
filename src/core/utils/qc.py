"""One four-state QC model shared by the console and the dataset tools.

A verdict recorded in ``meta/qc.jsonl`` always outranks the capture-time
``quality`` flag: an explicit ``pass`` keeps an episode the capture marked red,
and an explicit ``fail`` rejects one that recorded green. Capture quality only
decides while no verdict stands, so a red episode the reviewer has not judged
yet still reads as failed.
"""

from __future__ import annotations

from typing import Any


def qc_state(verdict: Any, quality: Any) -> str:
    """Resolve one episode's recorded verdict and capture quality into its state.

    Returns ``unreviewed``, ``passed`` or ``failed``.
    """
    verdict = str(verdict or "").strip().lower()
    if verdict == "unreviewed":
        return "unreviewed"
    if verdict == "pass":
        return "passed"
    if verdict == "fail":
        return "failed"
    return "failed" if str(quality or "").lower() == "red" else "unreviewed"
