from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def publish_output_pair(
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
