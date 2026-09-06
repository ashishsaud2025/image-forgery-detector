"""Generic image-pair loading helpers."""

import os

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image as a uint8 RGB array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: str) -> np.ndarray:
    """Load a binary foreground/forgery mask. Returns uint8 {0, 255} mask."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not decode mask: {path}")
    return (mask > 0).astype(np.uint8) * 255


def load_image_pair(image_path: str, mask_path: str | None):
    """Load an (image, mask) pair. Mask optional; returns None for it."""
    image = load_image(image_path)
    mask = load_mask(mask_path) if mask_path else None
    if mask is not None and mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return image, mask


def list_images(root: str, extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")) -> list[str]:
    """Recursively list image files under root."""
    out = []
    if not root or not os.path.isdir(root):
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(extensions):
                out.append(os.path.join(dirpath, name))
    return sorted(out)