"""Round-trip + atomicity tests for the raiden-schema writer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.calibration import (
    CameraCalibrationEntry,
    read_raiden_json,
    write_raiden_json,
)


def _entry() -> CameraCalibrationEntry:
    return CameraCalibrationEntry(
        K=np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
        dist=np.array([0.05, -0.10, 0.001, -0.001, 0.02]),
        attach_link="left_gripper_link",
        T_cam_link=np.eye(4),
        image_size=(640, 480),
    )


def test_write_then_read_roundtrip(tmp_path: Path):
    entries = {"cam_high": _entry(), "cam_left": CameraCalibrationEntry(
        K=_entry().K, dist=_entry().dist, attach_link="",
        T_cam_link=None, image_size=(1280, 720),
    )}
    dest = tmp_path / "calibration_results.json"
    write_raiden_json(dest, entries)
    read = read_raiden_json(dest)
    assert set(read) == {"cam_high", "cam_left"}
    assert np.allclose(read["cam_high"].K, entries["cam_high"].K)
    assert np.allclose(read["cam_high"].dist, entries["cam_high"].dist)
    assert read["cam_high"].attach_link == "left_gripper_link"
    assert read["cam_high"].image_size == (640, 480)
    assert read["cam_left"].T_cam_link is None
    assert read["cam_left"].image_size == (1280, 720)


def test_write_schema_matches_raiden(tmp_path: Path):
    dest = tmp_path / "calibration_results.json"
    write_raiden_json(dest, {"cam_high": _entry()})
    raw = json.loads(dest.read_text())
    assert set(raw.keys()) == {"cameras"}
    cam = raw["cameras"]["cam_high"]
    assert set(cam.keys()) == {"K", "dist", "attach_link", "T_cam_link", "image_size"}
    assert len(cam["K"]) == 3 and len(cam["K"][0]) == 3
    assert isinstance(cam["dist"], list)
    assert cam["image_size"] == [640, 480]


def test_write_is_atomic_on_failure(tmp_path: Path, monkeypatch):
    dest = tmp_path / "calibration_results.json"
    dest.write_text('{"cameras":{}}')  # pre-existing content
    pre = dest.read_text()

    def boom(*a, **kw):
        raise RuntimeError("simulated fs failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError, match="simulated fs failure"):
        write_raiden_json(dest, {"cam_high": _entry()})
    # Destination file must be untouched.
    assert dest.read_text() == pre
    # No stray tmp files.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".calibration_results")]
    assert leftovers == []
