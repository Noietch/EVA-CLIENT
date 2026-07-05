"""Tests for project_eef_uv — the EEF world -> camera pixel projection."""

from __future__ import annotations

import cv2
import numpy as np

from tools.calibration import project_eef_uv, rvec_tvec_to_se3


def test_projects_matches_cv2_projectpoints():
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    T_world_link = np.eye(4)  # link = world
    T_cam_link = rvec_tvec_to_se3(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5]))
    eef_world = np.array([0.0, 0.0, 1.0])
    uv = project_eef_uv(eef_world, T_world_link, T_cam_link, K, dist)
    # cv2 ground truth: point at (0, 0, 0.5) in cam frame projects to principal point.
    expected, _ = cv2.projectPoints(
        eef_world.reshape(1, 3).astype(np.float32),
        cv2.Rodrigues(T_cam_link[:3, :3])[0],
        T_cam_link[:3, 3],
        K,
        dist,
    )
    assert np.allclose(uv, expected.reshape(-1), atol=1e-3)


def test_projects_multiple_points():
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    T_world_link = np.eye(4)
    T_cam_link = rvec_tvec_to_se3(np.zeros(3), np.array([0.0, 0.0, 0.5]))
    eef_world = np.array([[0.0, 0.0, 1.0], [0.05, 0.05, 1.0]])
    uv = project_eef_uv(eef_world, T_world_link, T_cam_link, K, dist)
    assert uv.shape == (2, 2)
    assert not np.isnan(uv).any()


def test_point_behind_camera_is_nan():
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    T_world_link = np.eye(4)
    T_cam_link = np.eye(4)  # camera at origin, pointing +Z
    # Point behind the camera (Z_world negative in camera frame).
    eef_world = np.array([0.0, 0.0, -1.0])
    uv = project_eef_uv(eef_world, T_world_link, T_cam_link, K, dist)
    assert np.isnan(uv).all()


def test_link_transform_moves_the_camera():
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    # Camera at origin. Link at (0, 0, 1). Then T_cam_link places camera 0.5m ahead of link.
    T_world_link = rvec_tvec_to_se3(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    T_cam_link = rvec_tvec_to_se3(np.zeros(3), np.array([0.0, 0.0, 0.5]))
    # EEF at world origin -> in cam frame at z=0.5, x=y=0 -> principal point.
    eef_world = np.array([0.0, 0.0, 0.0])
    uv = project_eef_uv(eef_world, T_world_link, T_cam_link, K, dist)
    assert np.isnan(uv).all() or np.allclose(uv, [320.0, 240.0], atol=1.0)
