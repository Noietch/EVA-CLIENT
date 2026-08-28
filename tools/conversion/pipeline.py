from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

from .native import QualityExportProgress as DatasetExportProgress
from .native import QualitySplitSummary, split_dataset_by_quality

DatasetExporter = Callable[
    [Path, Path, Callable[[int, dict[str, Any]], None] | None],
    None,
]

_EXPORTER_IMPORTS: dict[str, tuple[str, str] | None] = {
    "lerobot_v21": None,
    "lerobot_v3": ("tools.conversion.lerobot_v3", "export_lerobot_v3"),
    "hdf5": ("tools.conversion.hdf5", "export_hdf5"),
    "mcap": ("tools.conversion.mcap", "export_mcap"),
}
DATASET_EXPORT_FORMATS = tuple(_EXPORTER_IMPORTS)


@dataclasses.dataclass(frozen=True)
class DatasetExportSummary:
    source_dir: str
    accepted_dir: str
    rejected_dir: str
    dataset_format: str
    source_episodes: int
    accepted_episodes: int
    rejected_episodes: int
    accepted_frames: int
    rejected_frames: int
    rejected_source_indices: tuple[int, ...]


def export_dataset_by_quality(
    source_dir: Path,
    accepted_dir: Path,
    rejected_dir: Path,
    *,
    dataset_format: str,
    replace_existing: bool = False,
    progress_callback: Callable[[DatasetExportProgress], None] | None = None,
) -> DatasetExportSummary:
    if dataset_format not in DATASET_EXPORT_FORMATS:
        expected = ", ".join(DATASET_EXPORT_FORMATS)
        raise ValueError(f"unsupported dataset format {dataset_format!r}; expected {expected}")
    source_dir = Path(source_dir).resolve()
    accepted_dir = Path(accepted_dir).resolve()
    rejected_dir = Path(rejected_dir).resolve()
    if accepted_dir == rejected_dir or source_dir in {accepted_dir, rejected_dir}:
        raise ValueError("source, accepted, and rejected directories must be distinct")
    if dataset_format == "lerobot_v21":
        summary = split_dataset_by_quality(
            source_dir,
            accepted_dir,
            rejected_dir,
            replace_existing=replace_existing,
            progress_callback=_native_progress_callback(progress_callback),
        )
        return _summary(summary, dataset_format)

    # Prepare same-filesystem staging roots for atomic publication
    for output in (accepted_dir, rejected_dir):
        if output.exists() and not replace_existing:
            raise FileExistsError(f"output directory already exists: {output}")
        if output.exists() and not output.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    native_root = Path(tempfile.mkdtemp(prefix=".conversion.native.", dir=accepted_dir.parent))
    accepted_stage_root = Path(
        tempfile.mkdtemp(prefix=".conversion.accepted.", dir=accepted_dir.parent)
    )
    rejected_stage_root = Path(
        tempfile.mkdtemp(prefix=".conversion.rejected.", dir=rejected_dir.parent)
    )
    native_accepted = native_root / "accepted"
    native_rejected = native_root / "rejected"
    stage_accepted = accepted_stage_root / "dataset"
    stage_rejected = rejected_stage_root / "dataset"

    # Split once, convert both subsets, and publish them as one result
    try:
        native_summary = split_dataset_by_quality(
            source_dir,
            native_accepted,
            native_rejected,
            progress_callback=None,
        )
        if progress_callback is not None:
            progress_callback(DatasetExportProgress(0, native_summary.source_episodes, "", None))
        _convert_subset(
            native_accepted,
            stage_accepted,
            dataset_format,
            subset="accepted",
            offset=0,
            total=native_summary.source_episodes,
            progress_callback=progress_callback,
        )
        _convert_subset(
            native_rejected,
            stage_rejected,
            dataset_format,
            subset="rejected",
            offset=native_summary.accepted_episodes,
            total=native_summary.source_episodes,
            progress_callback=progress_callback,
        )
        _publish_pair(
            ((accepted_dir, stage_accepted), (rejected_dir, stage_rejected)),
            replace_existing=replace_existing,
        )
        return _summary(native_summary, dataset_format, accepted_dir, rejected_dir)
    finally:
        shutil.rmtree(native_root, ignore_errors=True)
        shutil.rmtree(accepted_stage_root, ignore_errors=True)
        shutil.rmtree(rejected_stage_root, ignore_errors=True)


def _convert_subset(
    source_dir: Path,
    output_dir: Path,
    dataset_format: str,
    *,
    subset: str,
    offset: int,
    total: int,
    progress_callback: Callable[[DatasetExportProgress], None] | None,
) -> None:
    output_dir.mkdir(parents=True)
    source_indices = json.loads((source_dir / "meta" / "quality_split.json").read_text())[
        "source_episode_indices"
    ]
    exporter = _load_exporter(dataset_format)

    def update(completed: int, row: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        local_index = int(row["episode_index"])
        progress_callback(
            DatasetExportProgress(
                episodes_completed=offset + completed,
                episodes_total=total,
                subset=subset,
                source_episode_index=int(source_indices[local_index]),
            )
        )

    # Convert the subset with the format-specific writer selected above.
    exporter(source_dir, output_dir, update)


def _publish_pair(
    outputs: tuple[tuple[Path, Path], tuple[Path, Path]],
    *,
    replace_existing: bool,
) -> None:
    backups: dict[Path, tuple[Path, Path]] = {}
    published: list[Path] = []
    try:
        for output, stage in outputs:
            if output.exists() and not replace_existing:
                raise FileExistsError(f"output directory already exists: {output}")
            if output.exists() and not output.is_dir():
                raise NotADirectoryError(f"output path is not a directory: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                backup_root = Path(
                    tempfile.mkdtemp(prefix=f".{output.name}.previous.", dir=output.parent)
                )
                backup = backup_root / output.name
                output.replace(backup)
                backups[output] = (backup_root, backup)
            stage.replace(output)
            published.append(output)
    except Exception:
        for output in reversed(published):
            shutil.rmtree(output, ignore_errors=True)
        for output, (_, backup) in backups.items():
            if backup.exists() and not output.exists():
                backup.replace(output)
        raise
    finally:
        for backup_root, _ in backups.values():
            shutil.rmtree(backup_root, ignore_errors=True)


def _summary(
    value: QualitySplitSummary,
    dataset_format: str,
    accepted_dir: Path | None = None,
    rejected_dir: Path | None = None,
) -> DatasetExportSummary:
    return DatasetExportSummary(
        source_dir=value.source_dir,
        accepted_dir=str(accepted_dir or value.accepted_dir),
        rejected_dir=str(rejected_dir or value.rejected_dir),
        dataset_format=dataset_format,
        source_episodes=value.source_episodes,
        accepted_episodes=value.accepted_episodes,
        rejected_episodes=value.rejected_episodes,
        accepted_frames=value.accepted_frames,
        rejected_frames=value.rejected_frames,
        rejected_source_indices=value.rejected_source_indices,
    )


def _native_progress_callback(
    progress_callback: Callable[[DatasetExportProgress], None] | None,
) -> Callable[[Any], None] | None:
    if progress_callback is None:
        return None

    def update(progress: Any) -> None:
        progress_callback(
            DatasetExportProgress(
                episodes_completed=int(progress.episodes_completed),
                episodes_total=int(progress.episodes_total),
                subset=str(progress.subset),
                source_episode_index=(
                    None
                    if progress.source_episode_index is None
                    else int(progress.source_episode_index)
                ),
            )
        )

    return update


def _load_exporter(
    dataset_format: str,
) -> DatasetExporter:
    exporter_import = _EXPORTER_IMPORTS[dataset_format]
    if exporter_import is None:
        raise ValueError(f"no exporter for dataset format {dataset_format!r}")
    module_name, function_name = exporter_import
    return getattr(import_module(module_name), function_name)


__all__ = [
    "DATASET_EXPORT_FORMATS",
    "DatasetExportProgress",
    "DatasetExportSummary",
    "export_dataset_by_quality",
]
