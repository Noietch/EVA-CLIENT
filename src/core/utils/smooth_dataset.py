"""Offline smoothing helpers for collected LeRobot datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import savgol_filter

ARM_QPOS_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
QPOS_COLUMNS = ("observations.state.qpos", "action.qpos")


def smooth_arm_qpos(qpos: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    """Smooth arm joints in a 14D dual-arm qpos trajectory while preserving grippers.

    Args:
        qpos: Joint trajectory shaped [T, 14], with grippers at indices 6 and 13.
        window: Odd Savitzky-Golay window length in frames.
        polyorder: Polynomial order for the Savitzky-Golay filter.

    Returns:
        Smoothed qpos trajectory shaped [T, 14].
    """
    values = np.asarray(qpos, dtype=np.float64)
    if window % 2 != 1:
        raise ValueError("window must be odd")
    if window <= polyorder:
        raise ValueError("window must be greater than polyorder")
    if values.shape[0] < window:
        raise ValueError(f"qpos length {values.shape[0]} is shorter than window {window}")

    smoothed = values.copy()
    smoothed[:, ARM_QPOS_INDICES] = savgol_filter(
        values[:, ARM_QPOS_INDICES],
        window_length=window,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )
    smoothed[0] = values[0]
    smoothed[-1] = values[-1]
    return smoothed


def smooth_collection_dataset(
    source_dir: Path,
    output_dir: Path,
    window: int = 9,
    polyorder: int = 2,
) -> dict[str, Any]:
    """Copy a collection dataset and smooth arm qpos columns in every episode parquet.

    Args:
        source_dir: LeRobot dataset root containing data/chunk-000 episode parquets.
        output_dir: New dataset root to create. It must not already exist.
        window: Odd Savitzky-Golay window length in frames.
        polyorder: Polynomial order for the Savitzky-Golay filter.

    Returns:
        Summary dict with episode and frame counts.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")

    shutil.copytree(source_dir, output_dir)
    episode_paths = sorted((output_dir / "data" / "chunk-000").glob("episode_*.parquet"))
    total_frames = 0
    for episode_path in episode_paths:
        table = pq.read_table(episode_path)
        columns = {name: table[name] for name in table.column_names}
        for qpos_column in QPOS_COLUMNS:
            qpos = np.asarray(columns[qpos_column].to_pylist(), dtype=np.float64)
            columns[qpos_column] = pa.array(smooth_arm_qpos(qpos, window, polyorder).tolist())
        pq.write_table(pa.table(columns, schema=table.schema), episode_path)
        total_frames += table.num_rows

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "episodes": len(episode_paths),
        "frames": total_frames,
        "window": window,
        "polyorder": polyorder,
        "qpos_columns": list(QPOS_COLUMNS),
        "arm_qpos_indices": ARM_QPOS_INDICES.tolist(),
    }
    with (output_dir / "meta" / "smoothing.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary
