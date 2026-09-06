"""Blind noise-residual forensics.

Approach 2.2 #1 from the spec: denoise the image, take the residual
(image - denoised), and estimate a local noise-variance map over a
sliding window. Regions with mismatched noise statistics (e.g. spliced-in
patches) stand out.

Optionally supports wavelet-domain denoising via PyWavelets.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 32
WAVELET = "db4"
LEVELS = 5


# Denoising
def denoise_wavelet(gray: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Approximate wavelet soft-threshold denoiser (a Mihalcik-style filter).

    Uses a generic Laplace-of-Gaussian estimator in the wavelet domain;
    falls back to a bilateral filter (bundled with OpenCV, always available)
    when the `pywt` import fails.
    """
    try:
        import pywt

        coeffs = pywt.wavedecn(gray, wavelet=WAVELET, level=LEVELS)
        # Multiplicative universal threshold on detail subbands.
        estimate = sigma * np.sqrt(2.0 * np.log(gray.size))
        for detail in coeffs[1:]:
            for key in list(detail.keys()):
                arr = detail[key]
                mask = np.abs(arr) < estimate
                arr[mask] = 0.0
                detail[key] = arr
        denoised = pywt.waverecn(coeffs, wavelet=WAVELET)
        # pywt.waverecn can overshoot odd dimensions by one pixel.
        if denoised.shape != gray.shape:
            denoised = denoised[: gray.shape[0], : gray.shape[1]]
        return denoised
    except Exception as exc:  # pragma: no cover - pywt availability
        LOGGER.warning("Wavelet denoiser unavailable (%s); using bilateral filter.", exc)
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def noise_residual(image: np.ndarray, denoiser="wavelet") -> np.ndarray:
    """Compute the per-pixel noise residual (image - denoised) on the
    luminance channel.

    Parameters
    ----------
    image : np.ndarray, uint8 RGB or BGR, or grayscale.
    denoiser : 'wavelet' | 'bilateral' | 'gaussian'

    Returns
    -------
    np.ndarray : float32 residual of shape (H, W).
    """
    from .ela import to_8bit

    rgb = to_8bit(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if denoiser == "bilateral":
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    elif denoiser == "gaussian":
        denoised = cv2.GaussianBlur(gray, (5, 5), 1.5)
    else:
        denoised = denoise_wavelet(gray)

    return gray - denoised.astype(np.float32)


# Local statistics


def local_noise_variance_map(
    residual: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE, step: int = 1
) -> np.ndarray:
    """Local noise-variance map computed on a sliding window.

    Returns a float32 map of the same spatial size as `residual` using a
    uniform filter over the squared residual. `step` allows stride < block
    size (default 1) so the map is dense.
    """
    squared = residual.astype(np.float64) ** 2
    kernel = np.ones((block_size, block_size)) / float(block_size * block_size)
    var_map = cv2.filter2D(squared, -1, kernel, borderType=cv2.BORDER_REFLECT)

    if step > 1:
        var_map = var_map[::step, ::step]  # downsample

    return var_map.astype(np.float32)


def noise_heatmap(
    image: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    denoiser: str = "wavelet",
    normalize: bool = True,
) -> np.ndarray:
    """Get a normalized uint8 noise-consistency heatmap for the demo.

    Higher values indicate higher local noise variance (i.e. potentially
    a mismatched/forged region).
    """
    residual = noise_residual(image, denoiser=denoiser)
    var_map = local_noise_variance_map(residual, block_size=block_size)

    # Log-scale to tame dynamic range.
    eps = 1e-8
    logmap = np.log1p(var_map / (np.percentile(var_map, 99) + eps))

    if normalize:
        out = cv2.normalize(logmap, None, 0, 255, cv2.NORM_MINMAX)
        return out.astype(np.uint8)
    return logmap.astype(np.float32)