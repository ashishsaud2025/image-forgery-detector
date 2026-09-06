"""Error Level Analysis (ELA) feature extraction.

Re-save an image at a known JPEG quality, then take the amplified
pixel-wise absolute difference to expose regions whose compression
history mismatches the rest of the image.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_QUALITIES = (70, 85, 95)
DEFAULT_SCALE = 15


def to_8bit(image: np.ndarray) -> np.ndarray:
    """Convert an arbitrary dtype image to a uint8 RGB array."""
    if image.dtype == np.uint8:
        arr = image
    elif image.dtype == np.uint16:
        arr = (image >> 8).astype(np.uint8)
    elif np.issubdtype(image.dtype, np.floating):
        arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr


def _recompress_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """Recompress a uint8 BGR image at the given JPEG quality."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG recompression failed")
    recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if recompressed is None:
        raise RuntimeError("JPEG recompression decode failed")
    if recompressed.shape != image.shape:
        recompressed = cv2.resize(recompressed, (image.shape[1], image.shape[0]))
    return recompressed


def ela_single_quality(
    image: np.ndarray, quality: int = DEFAULT_QUALITIES[1], scale: float = DEFAULT_SCALE
) -> np.ndarray:
    """Compute a single-quality ELA difference map (uint8, BGR, amplified)."""
    bgr = cv2.cvtColor(to_8bit(image), cv2.COLOR_RGB2BGR)
    recompressed = _recompress_jpeg(bgr, quality)
    diff = cv2.absdiff(bgr.astype(np.int16), recompressed.astype(np.int16))
    amplified = np.clip(diff * float(scale), 0, 255).astype(np.uint8)

    equalized = np.empty_like(amplified)
    for c in range(3):
        equalized[:, :, c] = cv2.equalizeHist(amplified[:, :, c])
    return equalized


def ela_error_map_single(
    image: np.ndarray, quality: int = DEFAULT_QUALITIES[1]
) -> np.ndarray:
    """Raw (unscaled) per-channel absolute error map as float."""
    bgr = cv2.cvtColor(to_8bit(image), cv2.COLOR_RGB2BGR)
    recompressed = _recompress_jpeg(bgr, quality)
    return np.abs(bgr.astype(np.float32) - recompressed.astype(np.float32))


def ela_features(
    image: np.ndarray,
    qualities: tuple[int, ...] = DEFAULT_QUALITIES,
    scale: float = DEFAULT_SCALE,
) -> np.ndarray:
    """Compute ELA maps across several qualities and stack them.

    Returns a float32 array of shape (H, W, n_qualities + 1): per-quality
    grayscale heatmaps plus an aggregated mean heatmap channel.
    """
    quality_maps: list[np.ndarray] = []
    for q in qualities:
        error = ela_error_map_single(image, quality=q)
        per_chan_mean = error.mean(axis=2, keepdims=True)
        rough_heat = np.clip(per_chan_mean * scale, 0, 255)
        quality_maps.append(rough_heat.astype(np.float32))

    stack = np.concatenate(quality_maps, axis=2)
    mean_map = stack.mean(axis=2, keepdims=True)
    return np.concatenate([stack, mean_map], axis=2).astype(np.float32)


def ela_heatmap_gray(image: np.ndarray, quality: int = DEFAULT_QUALITIES[1]) -> np.ndarray:
    """Single grayscale heatmap for the demo/overlay."""
    error = ela_error_map_single(image, quality=quality)
    gray = error.mean(axis=2)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)