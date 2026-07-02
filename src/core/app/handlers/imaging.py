"""Action processing utilities — chunk normalization, trajectory interpolation,
and image preprocessing (resize/pad/layout).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_action_chunk(actions: object) -> np.ndarray:
    """Normalize an action payload to a 2D float32 array.

    Args:
        actions: Action payload shaped [D] or [T, D].

    Returns:
        Normalized [T, D] float32 array.
    """
    # np.array (not asarray) to force a writable, owned copy: the policy socket may hand
    # back a read-only buffer (msgpack-backed) and downstream snaps gripper dims in place.
    chunk = np.array(actions, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[np.newaxis, :]
    if chunk.ndim != 2:
        raise ValueError(f"Expected actions to have 1 or 2 dims, got shape {chunk.shape}")
    return chunk


def build_linear_trajectory(current: np.ndarray, target: np.ndarray, steps: int) -> np.ndarray:
    """Linearly interpolate between two vectors.

    Args:
        current: Start vector.
        target: End vector (same shape as current).
        steps: Number of interpolation steps.

    Returns:
        Interpolated [steps, D] float32 array.
    """
    current_vector = np.asarray(current, dtype=np.float32)
    target_vector = np.asarray(target, dtype=np.float32)
    if current_vector.shape != target_vector.shape:
        raise ValueError(
            f"Linear trajectory shape mismatch: "
            f"current={current_vector.shape}, target={target_vector.shape}"
        )
    return np.linspace(current_vector, target_vector, num=max(2, steps), dtype=np.float32)


# --- Image utilities ---


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


def prepare_image(
    image: np.ndarray,
    convert_bgr_to_rgb: bool,
    height: int,
    width: int,
    resize_pad: bool = True,
    image_layout: str = "chw",
) -> np.ndarray:
    """Preprocess an image into the model's expected layout and size.

    Args:
        image: Source image array.
        convert_bgr_to_rgb: Swap channel order from BGR to RGB when True.
        height: Target height.
        width: Target width.
        resize_pad: Pad to preserve aspect ratio when True, else resize directly.
        image_layout: "chw" to transpose to channel-first, else channel-last.

    Returns:
        Processed uint8 array in the requested layout.
    """
    converted = np.asarray(image)
    if converted.dtype != np.uint8:
        converted = np.clip(converted, 0, 255).astype(np.uint8)
    if converted.ndim == 2:
        converted = np.repeat(converted[..., None], 3, axis=2)
    if convert_bgr_to_rgb and converted.ndim == 3 and converted.shape[2] == 3:
        converted = converted[:, :, ::-1]
    if resize_pad:
        converted = resize_with_pad(converted, height, width)
    else:
        converted = resize_direct(converted, height, width)
    if image_layout == "chw":
        return np.transpose(converted, (2, 0, 1))
    return converted
