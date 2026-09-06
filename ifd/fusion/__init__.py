"""Fusion of ELA + noise features into a verdict and localization mask.

Two tiers:

1. `heuristic_fusion`: threshold + connected components per branch,
   fused via union/intersection (the spec's "MVP").
2. `BlockFeatureClassifier`: a scikit-learn classifier (RandomForest by
   default) trained on block-level statistics of [ELA map, noise map,
   intensity]. This is the low-dependency "learned fusion" tier; a CNN
   can substitute by consuming the same canonical feature maps from
   `ifd.detector.build_feature_maps`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class DetectionResult:
    """Output of a fused detector."""

    is_tampered: bool
    confidence: float              # 0..1 (probability of being tampered)
    heatmap: np.ndarray | None     # float32 (H, W) soft localization
    mask: np.ndarray | None        # uint8 {0,255} hard localization
    bboxes: list[tuple[int, int, int, int]] | None = None
    feature_maps: dict[str, np.ndarray] | None = None


def normalize_heat(feature: np.ndarray) -> np.ndarray:
    """Normalize a feature channel to [0, 1] using percentile clipping."""
    f = np.asarray(feature, dtype=np.float32)
    lo, hi = float(np.percentile(f, 1)), float(np.percentile(f, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(f, dtype=np.float32)
    return np.clip((f - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def threshold_components(
    heat: np.ndarray, threshold: float = 0.55, min_area: int = 25
) -> np.ndarray:
    """Binarize a heatmap and keep connected components above `min_area` px.

    Returns a **bool** array so it can be used directly as a numpy index /
    mask. A uint8 0/1 array would trigger fancy (integer) indexing when used
    to select from another array instead of boolean masking.
    """
    binary = heat > threshold
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    labeled, n = ndimage.label(binary)
    out = np.zeros_like(binary, dtype=bool)
    for i in range(1, n + 1):
        if int((labeled == i).sum()) >= min_area:
            out[labeled == i] = True
    return out


def _mask_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (y0, x0, y1, x1) of each connected region."""
    if mask.sum() == 0:
        return []
    labeled, n = ndimage.label(mask > 0)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if ys.size:
            boxes.append((int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())))
    return boxes


def heuristic_fusion(
    ela_map: np.ndarray,
    noise_map: np.ndarray,
    ela_threshold: float = 0.55,
    noise_threshold: float = 0.60,
    min_area: int = 25,
    combine: str = "union",
    suspect_fraction: float = 0.004,
) -> DetectionResult:
    """Combine the ELA and noise heatmaps without any learning.

    Parameters
    ----------
    ela_map, noise_map : float32 heatmaps (higher = more suspicious).
    combine : 'union' (either branch flags) | 'intersection' (both must).
    suspect_fraction : minimum fraction of pixels flagged for an
        image-level "tampered" verdict.

    Returns
    -------
    DetectionResult.
    """
    ela_heat = normalize_heat(ela_map)
    noise_heat = normalize_heat(noise_map)

    ela_bin = threshold_components(ela_heat, ela_threshold, min_area)
    noise_bin = threshold_components(noise_heat, noise_threshold, min_area)

    if combine == "intersection":
        fused_bin = (ela_bin & noise_bin)
        fused_heat = np.minimum(ela_heat, noise_heat)
    else:
        fused_bin = (ela_bin | noise_bin)
        fused_heat = (ela_heat + noise_heat) * 0.5

    mask = (fused_bin > 0).astype(np.uint8) * 255
    flagged_frac = float(fused_bin.mean())
    is_tampered = flagged_frac >= suspect_fraction

    # Confidence combines how much area was flagged and how strongly.
    # Cast to bool so this is a boolean selection, not uint8 fancy indexing.
    flagged = (ela_bin | noise_bin).astype(bool)
    strength = float(np.mean(fused_heat[flagged])) if flagged.any() else 0.0
    confidence = float(np.clip(0.5 * strength + min(flagged_frac * 20.0, 0.5), 0.0, 1.0))

    return DetectionResult(
        is_tampered=is_tampered,
        confidence=confidence,
        heatmap=fused_heat.astype(np.float32),
        mask=mask,
        bboxes=_mask_bboxes(mask),
        feature_maps={"ela": ela_heat, "noise": noise_heat},
    )


# Learned fusion: block-level classical classifier


def block_features(
    maps: list[np.ndarray],
    block: int = 64,
    max_blocks: int = 512,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Extract per-block statistics from one or more aligned maps.

    Each block yields (num_maps * 4) features: [mean, std, p95, max]
    per map. Returns (X, corners) where corners are (y, x) top-lefts.
    """
    shape = maps[0].shape[:2]
    for m in maps[1:]:
        assert m.shape[:2] == shape, "All input maps must be the same size"
    rng = rng or np.random.default_rng(0)

    corners = [
        (y, x)
        for y in range(0, shape[0] - block + 1, block)
        for x in range(0, shape[1] - block + 1, block)
    ]
    if len(corners) > max_blocks:
        chosen = rng.choice(len(corners), size=max_blocks, replace=False)
        corners = [corners[i] for i in chosen]

    feats = np.zeros((len(corners), len(maps) * 4), dtype=np.float32)
    for i, (y, x) in enumerate(corners):
        col = 0
        for m in maps:
            patch = m[y : y + block, x : x + block]
            feats[i, col : col + 4] = [
                float(patch.mean()),
                float(patch.std()),
                float(np.quantile(patch, 0.95)),
                float(patch.max()),
            ]
            col += 4
    return feats, corners


def blocks_overlap_mask(corners: list[tuple[int, int]], mask: np.ndarray, block: int) -> np.ndarray:
    """Label each block 1 if its region overlaps the ground-truth mask."""
    labels = np.zeros(len(corners), dtype=np.int64)
    binary = (mask > 0)
    for i, (y, x) in enumerate(corners):
        labels[i] = int(binary[y : y + block, x : x + block].any())
    return labels


class BlockFeatureClassifier:
    """RandomForest (swap in SVM/LogReg) over block-level features.

    Typical pipeline (see `scripts/train_blocks.py`):

        clf = BlockFeatureClassifier()
        for image, mask in labeled_samples:
            maps   = build_feature_maps(image, quality)
            feats, corners = clf.extract(maps)
            labels = clf.label_blocks(corners, mask)
        clf.fit(all_feats, all_labels)

    At inference, `Detector(classifier=clf).detect(image)` folds each
    block's tamper probability back into a heatmap.
    """

    def __init__(self, block: int = 64, max_blocks_per_image: int = 512, seed: int = 0):
        self.block = block
        self.max_blocks_per_image = max_blocks_per_image
        self.rng = np.random.default_rng(seed)

        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )

    def extract(
        self, maps: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Block statistics over [ela, noise, gray] feature maps.

        Returns (X of shape (n_blocks, 12), corners as (y, x) top-lefts).
        """
        return block_features(
            [maps["ela"], maps["noise"], maps["gray"]],
            block=self.block,
            max_blocks=self.max_blocks_per_image,
            rng=self.rng,
        )

    def label_blocks(self, corners: list[tuple[int, int]], mask: np.ndarray | None) -> np.ndarray:
        """Per-block labels (1 if the block overlaps the forgery mask).

        For authentic images pass mask=None to label everything 0.
        """
        if mask is None:
            return np.zeros(len(corners), dtype=np.int64)
        return blocks_overlap_mask(corners, mask, self.block)

    def fit(
        self,
        features_list: list[np.ndarray],
        labels_list: list[np.ndarray],
    ) -> "BlockFeatureClassifier":
        X = np.concatenate(features_list, axis=0)
        y = np.concatenate(labels_list, axis=0)
        self.model.fit(X, y)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1] if hasattr(self.model, "predict_proba") else self.model.predict(features)

    def heatmap_from_blocks(
        self, proba: np.ndarray, corners: list[tuple[int, int]], shape: tuple[int, int]
    ) -> np.ndarray:
        """Fold per-block probabilities back into a full-size heatmap."""
        heat = np.zeros(shape, dtype=np.float32)
        for (y, x), p in zip(corners, proba):
            heat[y : y + self.block, x : x + self.block] = np.maximum(
                heat[y : y + self.block, x : x + self.block], float(p)
            )
        return heat

# NOTE: no imports from ..detector here to avoid a circular import.
# Training scripts import `build_feature_maps` from ..detector directly.