from __future__ import annotations

import dataclasses
from typing import Any

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class CameraSpec:
    name: str
    observation_key: str
    source: str
    width: int
    height: int
    fps: int
    rotation: int = 0


def parse_camera_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return np.rot90(image, k=3).copy()
    if rotation == 180:
        return np.rot90(image, k=2).copy()
    if rotation == 270:
        return np.rot90(image, k=1).copy()
    return image


def open_cameras(cameras: tuple[CameraSpec, ...]) -> dict[str, Any]:
    """Open configured OpenCV captures.

    Args:
        cameras: Camera specs with OpenCV source and capture properties.

    Returns:
        Mapping from camera name to an opened ``cv2.VideoCapture`` object.
    """
    captures = {}
    for camera in cameras:
        capture = cv2.VideoCapture(parse_camera_source(camera.source))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
        capture.set(cv2.CAP_PROP_FPS, camera.fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open camera {camera.name}: {camera.source}")
        captures[camera.name] = capture
    return captures


def read_camera_images(
    captures: dict[str, Any],
    cameras: tuple[CameraSpec, ...],
) -> dict[str, np.ndarray]:
    """Read one frame from every configured camera.

    Args:
        captures: Opened camera captures keyed by camera name.
        cameras: Camera specs matching the capture names.

    Returns:
        Images keyed by EVA observation camera key.
    """
    images = {}
    camera_by_name = {camera.name: camera for camera in cameras}
    for camera_name, capture in captures.items():
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame from camera {camera_name}")
        camera = camera_by_name[camera_name]
        images[camera.observation_key] = rotate_image(frame, camera.rotation)
    return images
