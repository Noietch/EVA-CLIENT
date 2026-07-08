"""Image resize helpers shared by app handlers and recorders."""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an image to fit a target size, preserving aspect ratio with padding.

    Args:
        image: Source image, HWC array.
        height: Target height.
        width: Target width.

    Returns:
        Padded resized HWC uint8 array.
    """
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    pil_image = Image.fromarray(image)
    resized = np.asarray(
        pil_image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR),
    )
    top = (height - new_h) // 2
    bottom = height - new_h - top
    left = (width - new_w) // 2
    right = width - new_w - left
    if resized.ndim == 2:
        padded = np.zeros((height, width), dtype=resized.dtype)
        padded[top : height - bottom, left : width - right] = resized
        return padded
    padded = np.zeros((height, width, resized.shape[2]), dtype=resized.dtype)
    padded[top : height - bottom, left : width - right, :] = resized
    return padded


def resize_direct(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an image directly to a target size without preserving aspect ratio.

    Args:
        image: Source image, HWC array.
        height: Target height.
        width: Target width.

    Returns:
        Directly resized HWC uint8 array.
    """
    pil_image = Image.fromarray(image)
    resized = np.asarray(
        pil_image.resize((width, height), resample=Image.Resampling.BILINEAR),
    )
    return resized
