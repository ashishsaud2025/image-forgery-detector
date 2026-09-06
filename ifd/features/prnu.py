"""Reference-based PRNU fingerprint matching (stretch goal).

Implements the standard pipeline from Lukas, Fridrich & Goljan:

1. Estimate per-image noise residual with a wavelet-domain denoiser.
2. Average residuals (with contrast masking) over ~20-50 known-authentic
   images from a single camera device to obtain the reference K.
3. Correlate blocks of the suspect image's residual against K and flag
   low-correlation blocks as forged.

This requires multiple authentic images per camera (e.g. Dresden).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


# Denoising


def wavelet_denoise_residual(gray: np.ndarray) -> np.ndarray:
    """Mihalcik-style wavelet denoiser residual (calibrated, integer data)."""
    was_int = gray.dtype != np.float32
    if was_int:
        gray_f = gray.astype(np.float32)
    else:
        gray_f = gray

    try:
        import pywt

        coeffs = pywt.wavedecn(gray_f, wavelet="db4", level=4)
        sigma_est = _mad_estimator(gray_f)
        threshold = 2.0 * sigma_est * np.sqrt(2.0 * np.log(gray_f.size))
        for detail in coeffs[1:]:
            for key in list(detail.keys()):
                arr = detail[key]
                arr[np.abs(arr) < threshold] = 0.0
                detail[key] = arr
        denoised = pywt.waverecn(coeffs, wavelet="db4")
        # pywt.waverecn can overshoot odd dimensions by one pixel.
        if denoised.shape != gray_f.shape:
            denoised = denoised[: gray_f.shape[0], : gray_f.shape[1]]
    except Exception:  # pragma: no cover
        denoised = cv2.bilateralFilter(gray_f, d=9, sigmaColor=75, sigmaSpace=75)

    residual = gray_f - denoised
    if was_int:
        residual = np.clip(np.round(residual), 0, 255).astype(np.uint8)
    return residual


def _mad_estimator(image: np.ndarray) -> float:
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.median(np.abs(dx - dy)) / 0.6745 / np.sqrt(2.0))


def extract_single_residual(image_rgb: np.ndarray) -> np.ndarray:
    """Extract the PRNU-like noise residual of a single RGB image.

    Runs the denoiser independently on each channel and combines the
    colored residual (intensities). Returns float32 residual in [0,1].
    """
    from ..features.ela import to_8bit

    rgb = to_8bit(image_rgb)
    residual = np.zeros(rgb.shape, dtype=np.float32)
    for c in range(3):
        gray = rgb[:, :, c]
        residual[:, :, c] = wavelet_denoise_residual(gray).astype(np.float32) / 255.0
    # Intensity residual = mean across color channels.
    return residual.mean(axis=2).astype(np.float32)


def compute_photo_response_non_uniformity(images: list[np.ndarray]) -> np.ndarray:
    """Build a reference PRNU pattern K from multiple authentic images.

    Parameters
    ----------
    images : list of uint8 RGB arrays from a single camera device.

    Returns
    -------
    np.ndarray : float32 reference fingerprint.
    """
    if len(images) < 20:
        LOGGER.warning("Fewer than 20 images (%d); fingerprint will be noisy.", len(images))

    residuals = []
    for img in images:
        residual = extract_single_residual(img)
        # Zero-mean the residual before the sum (contrast/discontinuity masking).
        residual = residual - residual.mean()
        residuals.append(residual)

    n = len(residuals)
    ref = np.zeros_like(residuals[0])
    for r in residuals:
        ref += r
    ref /= max(n, 1)

    # Normalize to unit norm + zero mean for correlation-friendly use.
    ref = ref - ref.mean()
    norm = np.linalg.norm(ref)
    if norm > 0:
        ref /= norm
    return ref


# Correlation / forgery localization


def correlation_map(residual: np.ndarray, fingerprint: np.ndarray, block: int = 64) -> np.ndarray:
    """Normalized cross-correlation of the residual with the reference
    fingerprint, computed over sliding blocks.

    Returns a float map the same size as the input; each pixel holds the
    correlation value of its enclosing block (scaled for display as
    [-1, 1]).
    """
    res = residual.astype(np.float32)
    fp = fingerprint.astype(np.float32)

    # Local pixel norm of residual within the block window.
    core = np.full_like(res, np.nan)
    half = block // 2

    for y in range(0, res.shape[0], block):
        for x in range(0, res.shape[1], block):
            y0, y1 = max(y - half, 0), min(y + half, res.shape[0])
            x0, x1 = max(x - half, 0), min(x + half, res.shape[1])
            patch = res[y0:y1, x0:x1]
            fp_patch = fp[y0:y1, x0:x1]

            if patch.size == 0 or fp_patch.size == 0:
                continue

            pred0 = patch - patch.mean()
            pred1 = fp_patch - fp_patch.mean()
            denom = np.sqrt(
                (pred0 * pred0).sum() * (pred1 * pred1).sum()
            )
            if denom < 1e-12:
                continue
            core[y0:y1, x0:x1] = float((pred0 * pred1).sum() / denom)

    # Fill unfinished NaNs with global stats.
    known = core[~np.isnan(core)]
    if known.size:
        core = np.where(np.isnan(core), known.mean(), core)
    else:
        core = np.zeros_like(core)
    return core