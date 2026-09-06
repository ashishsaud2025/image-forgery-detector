"""End-to-end single-image detector: ELA + noise features -> verdict.

This is the top-level entry point used by scripts, the demo CLI, and the
evaluation harness. It owns the two feature branches and the fusion step.

Two fusion tiers:
  * heuristic  - threshold + blob detection (brittle baseline, documented FPs)
  * learned    - `BlockFeatureClassifier` (RandomForest over block stats)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .features import (
    DEFAULT_QUALITIES,
    ela_error_map_single,
    local_noise_variance_map,
    noise_residual,
    to_8bit,
)
from .fusion import (
    BlockFeatureClassifier,
    DetectionResult,
    heuristic_fusion,
    normalize_heat,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Tunables for the heuristic fusion tier."""

    ela_qualities: tuple[int, ...] = DEFAULT_QUALITIES
    ela_threshold: float = 0.55
    noise_threshold: float = 0.60
    min_area: int = 25
    combine: str = "union"
    suspect_fraction: float = 0.01


def build_feature_maps(image_rgb: np.ndarray, quality: int) -> dict[str, np.ndarray]:
    """Canonical raw feature maps shared by training and inference.

    Returns
    -------
    dict with:
      'ela'   : float32 (H, W) mean per-channel absolute ELA error
      'noise' : float32 (H, W) log1p(local noise variance, block 32)
      'gray'  : float32 (H, W) luminance
    """
    gray = to_8bit(image_rgb).astype(np.float32).mean(axis=2)

    err = ela_error_map_single(image_rgb, quality=quality)
    ela = err.mean(axis=2).astype(np.float32)

    residual = noise_residual(image_rgb, denoiser="wavelet")
    var_map = local_noise_variance_map(residual, block_size=32)
    noise = np.log1p(var_map).astype(np.float32)

    return {"ela": ela, "noise": noise, "gray": gray}


class Detector:
    """Runs the two feature branches and fuses them into a verdict.

    Examples
    --------
    >>> det = Detector()                          # heuristic tier
    >>> result = det.detect(image_rgb)

    >>> det = Detector(classifier=clf)            # learned tier
    >>> result = det.detect(image_rgb)
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        classifier: BlockFeatureClassifier | None = None,
    ):
        self.config = config or DetectorConfig()
        self.classifier = classifier

    # Feature branches

    def ela_heatmap(self, image: np.ndarray, quality: int) -> np.ndarray:
        """Normalized ELA heatmap (float32, [0,1])."""
        err = ela_error_map_single(image, quality=quality).mean(axis=2)
        return normalize_heat(err.astype(np.float32))

    def noise_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Normalized noise-variance heatmap (float32, [0,1])."""
        residual = noise_residual(image, denoiser="wavelet")
        var_map = np.log1p(local_noise_variance_map(residual, block_size=32))
        return normalize_heat(var_map.astype(np.float32))

    def feature_maps(self, image: np.ndarray, quality: int) -> dict[str, np.ndarray]:
        """Raw feature maps: {'ela', 'noise', 'gray'}."""
        return build_feature_maps(image, quality)

    # Full pipeline

    def detect(self, image: np.ndarray, ela_quality: int | None = None) -> DetectionResult:
        """Produce a full DetectionResult for a single image.

        Parameters
        ----------
        image : uint8 RGB, BGR or grayscale array.
        ela_quality : JPEG re-save quality for the ELA branch. If None, the
            middle of the configured sweep is used. The classic strong case
            is re-saving at the same quality the suspect host was last saved
            at.
        """
        rgb = to_8bit(np.asarray(image))
        if rgb.ndim == 2:
            import cv2

            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.ndim == 3 and rgb.shape[2] == 4:
            import cv2

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

        if self.classifier is not None:
            return self._detect_learned(rgb, ela_quality)

        q = ela_quality or self.config.ela_qualities[1]
        ela = self.ela_heatmap(rgb, q)
        noise = self.noise_heatmap(rgb)
        return heuristic_fusion(
            ela,
            noise,
            ela_threshold=self.config.ela_threshold,
            noise_threshold=self.config.noise_threshold,
            min_area=self.config.min_area,
            combine=self.config.combine,
            suspect_fraction=self.config.suspect_fraction,
        )

    def _detect_learned(self, rgb: np.ndarray, ela_quality: int | None) -> DetectionResult:
        assert self.classifier is not None
        q = ela_quality or self.config.ela_qualities[1]
        maps = build_feature_maps(rgb, q)

        feats, corners = self.classifier.extract(maps)
        proba = self.classifier.predict_proba(feats)
        clf_heat = self.classifier.heatmap_from_blocks(proba, corners, rgb.shape[:2])

        n = max(len(proba), 1)
        peak = float(proba.max())

        from .fusion import threshold_components

        # A forged patch shows up as a contiguous cluster of strongly-flagged
        # blocks. Spurious texture blocks (the documented FP mode) are
        # scattered and small; require a cluster equal to at least two
        # blocks in size before declaring tampering.
        block_px = self.classifier.block * self.classifier.block
        clus = threshold_components(clf_heat, 0.5, min_area=2 * block_px)
        mask = (clus > 0).astype(np.uint8) * 255
        has_big_cluster = bool(clus.any())

        is_tampered = has_big_cluster and peak >= 0.5
        big_area_frac = float(clus.sum()) / (rgb.shape[0] * rgb.shape[1])

        # Confidence blends peak block strength with the flagged area.
        confidence = float(
            np.clip(0.5 * peak + min(max(big_area_frac * 4.0, 0.0), 0.5), 0, 1)
        )

        # Tie heuristic branches back in for the report heatmap (learned tier
        # gets the most weight).
        ela = normalize_heat(maps["ela"])
        noise = normalize_heat(maps["noise"])
        clf_heat_norm = normalize_heat(clf_heat)
        fused = (0.5 * ela + 0.25 * noise + 1.0 * clf_heat_norm) / 1.75

        feature_maps = {
            "ela": ela,
            "noise": noise,
            "classifier": clf_heat_norm,
        }

        return DetectionResult(
            is_tampered=is_tampered,
            confidence=confidence,
            heatmap=fused.astype(np.float32),
            mask=mask,
            bboxes=_mask_bboxes(mask) if mask.any() else None,
            feature_maps=feature_maps,
        )


def _mask_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (y0, x0, y1, x1) per connected region."""
    import cv2

    n, labels, stats, cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    boxes = [
        (int(stats[i, 1]), int(stats[i, 0]), int(stats[i, 1] + stats[i, 3]), int(stats[i, 0] + stats[i, 2]))
        for i in range(1, n)
    ]
    return boxes