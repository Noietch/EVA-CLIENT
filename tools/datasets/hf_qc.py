"""Merge timestamped dataset QC records."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

LEGACY_QC_UPDATED_AT = "2026-09-15T00:00:00+08:00"


def _rows(content: bytes) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        episode = int(row["episode_index"])
        if episode in rows:
            raise ValueError(f"Duplicate QC episode_index: {episode}")
        row.setdefault("qc_updated_at", LEGACY_QC_UPDATED_AT)
        timestamp = datetime.fromisoformat(row["qc_updated_at"].replace("Z", "+00:00"))
        if timestamp.utcoffset() is None:
            raise ValueError(f"QC timestamp must have a timezone: episode {episode}")
        rows[episode] = row
    return rows


def merge_qc(local: bytes, remote: bytes) -> bytes:
    """Keep the newest complete QC row per episode; reject ambiguous ties."""
    rows = _rows(local)
    for episode, candidate in _rows(remote).items():
        existing = rows.get(episode)
        if existing is None:
            rows[episode] = candidate
            continue
        current_time = datetime.fromisoformat(existing["qc_updated_at"].replace("Z", "+00:00"))
        remote_time = datetime.fromisoformat(candidate["qc_updated_at"].replace("Z", "+00:00"))
        if remote_time > current_time:
            rows[episode] = candidate
        elif remote_time == current_time and candidate != existing:
            raise ValueError(f"Conflicting QC records at the same time: episode {episode}")
    return "".join(
        json.dumps(rows[episode], ensure_ascii=False) + "\n" for episode in sorted(rows)
    ).encode("utf-8")


def write_qc(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".qc-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    shutil.copymode(path, temporary)
    temporary.replace(path)
