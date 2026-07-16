"""Tests for the ``calibration`` config block and _coerce_calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import robots  # noqa: F401  (registers robots)
import transport  # noqa: F401  (registers transport backends)
from core.config import load_config
from robots.base import CameraCalibration

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_DEFAULTS = _CONFIGS_DIR / "00_base" / "defaults.py"
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "calibration_results.json"


def _write_config(path: Path, body: str) -> Path:
    """Same helper style as tests/config/test_config.py — inherit real defaults."""
    path.write_text(f"_base_ = [{str(_DEFAULTS)!r}]\n{body}", encoding="utf-8")
    return path


def test_load_without_calibration_keeps_empty_dict(tmp_path: Path) -> None:
    """A preset with no ``calibration`` override still gets the empty default block."""
    cfg = load_config(_write_config(tmp_path / "no_calib.py", ""))
    assert cfg.calibration.cameras == {}
    assert cfg.calibration.results_path is None


def test_existing_collection_preset_backward_compat() -> None:
    """A real collection preset that never mentions calibration must still load
    with an empty cameras dict (byte-identical vs pre-PR#1 behavior)."""
    preset = _CONFIGS_DIR / "02_collection" / "dual_agilex_piper.py"
    if not preset.exists():
        pytest.skip(f"{preset} not in this checkout")
    cfg = load_config(preset)
    assert cfg.calibration.cameras == {}
    assert cfg.calibration.results_path is None


def test_load_with_calibration_json_materializes_np_arrays(tmp_path: Path) -> None:
    """`results_path` -> raiden-schema json -> CameraCalibration instances w/ np arrays."""
    cfg = load_config(
        _write_config(
            tmp_path / "with_calib.py",
            f"calibration = dict(cameras={{}}, results_path={str(_FIXTURE)!r})",
        )
    )
    cams = cfg.calibration.cameras
    assert set(cams) == {"cam_high", "cam_left_wrist"}

    high = cams["cam_high"]
    assert isinstance(high, CameraCalibration)
    assert high.K.shape == (3, 3)
    assert high.K.dtype == np.float64
    assert high.K[0, 0] == 600.0
    assert high.dist.shape == (5,)
    assert high.attach_link == "left_gripper_link"
    assert high.T_cam_link is not None
    assert high.T_cam_link.shape == (4, 4)
    assert high.image_size == (640, 480)

    wrist = cams["cam_left_wrist"]
    assert isinstance(wrist, CameraCalibration)
    assert wrist.attach_link == ""
    assert wrist.T_cam_link is None


def test_inline_overlay_wins_field_by_field(tmp_path: Path) -> None:
    """Inline `calibration.cameras[<name>]` overlays the json entry field-by-field."""
    body = (
        f"calibration = dict(results_path={str(_FIXTURE)!r}, "
        "cameras={'cam_high': {'attach_link': 'overridden_link'}})"
    )
    cfg = load_config(_write_config(tmp_path / "overlay.py", body))
    high = cfg.calibration.cameras["cam_high"]
    assert high.attach_link == "overridden_link"
    # K/dist survive from the json — the overlay only mutated attach_link.
    assert high.K[0, 0] == 600.0
    assert high.dist.shape == (5,)


def test_missing_K_raises(tmp_path: Path) -> None:
    body = (
        "calibration = dict(cameras={"
        "'bad_cam': {'dist': [0.0, 0.0, 0.0, 0.0, 0.0]}"
        "})"
    )
    with pytest.raises(ValueError, match="missing required K/dist"):
        load_config(_write_config(tmp_path / "bad.py", body))
