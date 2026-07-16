"""Tests for tools.calibration / detector on synthetic ChArUco images."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest

from tools.calibration import (
    DEFAULT_DICT_NAMES,
    CharucoBoardSpec,
    build_cv_board,
    detect_board,
    draw_overlay,
    match_image_points,
)


def _spec() -> CharucoBoardSpec:
    return CharucoBoardSpec(dict_name="DICT_5X5_100", cols=5, rows=7,
                            square_length=0.03, marker_length=0.023)


def _board_image_bgr(spec: CharucoBoardSpec, w: int = 800, h: int = 1000) -> np.ndarray:
    board = build_cv_board(spec)
    img_gray = board.generateImage((w, h), marginSize=20, borderBits=1)
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


def test_build_cv_board_matches_spec():
    spec = _spec()
    board = build_cv_board(spec)
    assert board.getChessboardSize() == (spec.cols, spec.rows)
    assert board.getSquareLength() == pytest.approx(spec.square_length, rel=1e-5)
    assert board.getMarkerLength() == pytest.approx(spec.marker_length, rel=1e-5)


def test_unknown_dictionary_raises():
    with pytest.raises(ValueError, match="unknown ArUco dictionary"):
        build_cv_board(CharucoBoardSpec(dict_name="NOT_A_DICT"))


def test_board_spec_is_frozen():
    spec = _spec()
    with pytest.raises(FrozenInstanceError):
        spec.cols = 99  # type: ignore[misc]


def test_default_dict_names_include_common():
    assert "DICT_5X5_100" in DEFAULT_DICT_NAMES
    assert "DICT_6X6_250" in DEFAULT_DICT_NAMES


def test_detect_board_finds_corners_on_synthetic_image():
    spec = _spec()
    img = _board_image_bgr(spec)
    detection = detect_board(img, spec)
    assert detection.detected
    # ChArUco has (cols-1)*(rows-1) interior corners.
    assert detection.n_corners == (spec.cols - 1) * (spec.rows - 1)


def test_detect_board_empty_on_noise_image():
    spec = _spec()
    noise = (np.random.default_rng(0).integers(0, 255, size=(400, 400, 3)).astype(np.uint8))
    detection = detect_board(noise, spec)
    assert not detection.detected


def test_match_image_points_shapes():
    spec = _spec()
    img = _board_image_bgr(spec)
    detection = detect_board(img, spec)
    objp, imgp = match_image_points(spec, detection)
    assert objp.ndim == 2 and objp.shape[1] == 3
    assert imgp.ndim == 2 and imgp.shape[1] == 2
    assert objp.shape[0] == imgp.shape[0] == detection.n_corners


def test_draw_overlay_never_mutates_input():
    spec = _spec()
    img = _board_image_bgr(spec)
    detection = detect_board(img, spec)
    original = img.copy()
    copy = img.copy()
    draw_overlay(copy, detection)
    # We drew on the copy; the original passed in is untouched.
    assert np.array_equal(img, original)
    # And the copy is different (something was drawn on it).
    assert not np.array_equal(copy, original)
