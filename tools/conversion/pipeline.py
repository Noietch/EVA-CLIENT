"""Convert one recorded dataset into one output dataset per format.

QC verdicts are not part of the conversion: the dataset carries them in
``meta/qc.jsonl``, so a consumer filters the exported copy itself.
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

from ._publish import publish_outputs

DatasetExporter = Callable[
    [Path, Path, Callable[[int, dict[str, Any]], None] | None],
    None,
]

_EXPORTER_IMPORTS: dict[str, tuple[str, str]] = {
    "lerobot_v3": ("tools.conversion.lerobot_v3", "export_lerobot_v3"),
    "hdf5": ("tools.conversion.hdf5", "export_hdf5"),
    "mcap": ("tools.conversion.mcap", "export_mcap"),
}
DATASET_EXPORT_FORMATS = tuple(_EXPORTER_IMPORTS)


@dataclasses.dataclass(frozen=True)
class DatasetExportProgress:
    episodes_completed: int
    episodes_total: int
    source_episode_index: int | None


@dataclasses.dataclass(frozen=True)
class DatasetExportSummary:
    source_dir: str
    output_dir: str
    dataset_format: str
    episodes: int


def export_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    dataset_format: str,
    replace_existing: bool = False,
    progress_callback: Callable[[DatasetExportProgress], None] | None = None,
) -> DatasetExportSummary:
    """Convert ``source_dir`` into ``output_dir`` in ``dataset_format``."""
    if dataset_format not in DATASET_EXPORT_FORMATS:
        expected = ", ".join(DATASET_EXPORT_FORMATS)
        raise ValueError(f"unsupported dataset format {dataset_format!r}; expected {expected}")
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("the output directory must not hold the source dataset")
    if output_dir.exists() and not replace_existing:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")

    episodes = sum(
        1
        for line in (source_dir / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    stage = stage_root / "dataset"
    try:
        _load_exporter(dataset_format)(source_dir, stage, _progress(progress_callback, episodes))
        publish_outputs(((output_dir, stage),), replace_existing=replace_existing)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    return DatasetExportSummary(str(source_dir), str(output_dir), dataset_format, episodes)


def _progress(
    progress_callback: Callable[[DatasetExportProgress], None] | None, total: int
) -> Callable[[int, dict[str, Any]], None] | None:
    if progress_callback is None:
        return None

    def update(completed: int, row: dict[str, Any]) -> None:
        progress_callback(
            DatasetExportProgress(
                episodes_completed=completed,
                episodes_total=total,
                source_episode_index=row.get("episode_index"),
            )
        )

    return update


def _load_exporter(dataset_format: str) -> DatasetExporter:
    module_name, function_name = _EXPORTER_IMPORTS[dataset_format]
    return getattr(import_module(module_name), function_name)


__all__ = [
    "DATASET_EXPORT_FORMATS",
    "DatasetExportProgress",
    "DatasetExportSummary",
    "export_dataset",
]
