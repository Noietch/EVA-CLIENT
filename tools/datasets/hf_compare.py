"""Compare local dataset files and QC against the Hugging Face remote tree.

Shared by the robot console and the dataset service, which run the same
inspection against the same repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.utils.qc import qc_state


def file_digest(path: Path, lfs: bool) -> str:
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_files(local: dict[str, Path], remote: dict, progress=lambda *_: None) -> dict:
    missing = sorted(set(remote) - set(local))
    extra = sorted(set(local) - set(remote))
    changed = []
    unknown = []
    common = sorted(set(local) & set(remote))
    for index, name in enumerate(common):
        item = remote[name]
        lfs = getattr(item, "lfs", None)
        expected = (
            (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))
            if lfs
            else item.blob_id
        )
        before = local[name].stat()
        if not expected:
            unknown.append(name)
        elif before.st_size != item.size or file_digest(local[name], bool(lfs)) != expected:
            changed.append(name)
        after = local[name].stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            unknown.append(name)
        progress(index + 1, len(common), name)
    return {
        "state": "unknown" if unknown else "different" if missing or extra or changed else "same",
        "missing_local": missing,
        "local_only": extra,
        "changed": changed,
        "unknown": unknown,
    }


def is_dataset_content(name: str) -> bool:
    path = Path(name)
    # QC, UI selection state and export reports are not recorded dataset content.
    return (
        path.parts[0] in {"data", "videos", "meta"}
        and name not in {"meta/qc.jsonl", "meta/collection_slots.json", "meta/quality_split.json"}
        and not any(part.startswith(".") for part in path.parts)
    )


def data_files(root: Path) -> dict[str, Path]:
    return {
        p.relative_to(root).as_posix(): p
        for folder in ("data", "videos", "meta")
        for p in (root / folder).rglob("*")
        if p.is_file() and is_dataset_content(p.relative_to(root).as_posix())
    }


def qc_summary(episodes: list[dict], qc: list[dict]) -> dict:
    overrides = {r["episode_index"]: r for r in qc}
    counts = {"accept": 0, "fail": 0, "unreviewed": 0}
    for episode in episodes:
        row = {**episode, **overrides.get(episode.get("episode_index"), {})}
        state = qc_state(row.get("qc_verdict"), row.get("quality"))
        counts[{"failed": "fail", "passed": "accept"}.get(state, "unreviewed")] += 1
    return counts


def read_jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.is_file()
        else []
    )


def qc_entries(episodes: list[dict], qc: list[dict]) -> dict:
    overrides = {r["episode_index"]: r for r in qc}
    merged = {r["episode_index"]: {**r, **overrides.get(r["episode_index"], {})} for r in episodes}
    for index, row in overrides.items():
        merged.setdefault(index, row)
    return {
        str(index): {key: row.get(key) or "" for key in ("qc_verdict", "qc_note", "qc_reason")}
        for index, row in merged.items()
    }


def compare_qc(local: dict, remote: dict) -> dict:
    missing = sorted(set(remote) - set(local))
    extra = sorted(set(local) - set(remote))
    changed = sorted(k for k in set(local) & set(remote) if local[k] != remote[k])
    return {
        "state": "different" if missing or extra or changed else "same",
        "missing_local": missing,
        "local_only": extra,
        "changed": changed,
        "unknown": [],
    }
