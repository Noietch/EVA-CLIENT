"""Smoke tests for the new PDF builder + calibration_poses coerce."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.calibration import CharucoBoardSpec, build_charuco_pdf

_REPO = Path(__file__).resolve().parents[3]
_DEFAULTS = _REPO / "configs" / "00_base" / "defaults.py"


def test_pdf_bytes_are_a_valid_pdf():
    pdf = build_charuco_pdf(CharucoBoardSpec(), paper="a4")
    assert pdf.startswith(b"%PDF-1.4")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 5_000


def test_pdf_letter_paper():
    pdf = build_charuco_pdf(CharucoBoardSpec(), paper="letter")
    assert pdf.startswith(b"%PDF-1.4")


def test_pdf_rejects_unknown_paper():
    import pytest

    with pytest.raises(ValueError, match="unknown paper size"):
        build_charuco_pdf(CharucoBoardSpec(), paper="tabloid")


def test_calibration_poses_default_empty(tmp_path: Path):
    from core.config import load_config
    import robots  # noqa: F401

    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text(f'_base_ = ["{_DEFAULTS}"]\n', encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.calibration_poses == ()


def test_calibration_poses_override(tmp_path: Path):
    from core.config import load_config
    import robots  # noqa: F401

    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text(
        f'_base_ = ["{_DEFAULTS}"]\n'
        "calibration_poses = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg.calibration_poses, tuple)
    assert len(cfg.calibration_poses) == 2
    assert isinstance(cfg.calibration_poses[0], np.ndarray)
    assert cfg.calibration_poses[0].dtype == np.float32
    assert cfg.calibration_poses[0].tolist() == [
        np.float32(0.1),
        np.float32(0.2),
        np.float32(0.3),
    ]
