"""Evaluation harness: image-level and pixel-level metrics.

Image-level: accuracy, precision/recall, F1, ROC-AUC.
Pixel-level: IoU / Dice between predicted mask and ground truth.

Everything is dataset-driven so metrics can be reported per-dataset
rather than as one aggregate number (per the spec).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..fusion import DetectionResult

LOGGER = logging.getLogger(__name__)


@dataclass
class PixelMetrics:
    """Pixel-level localization metrics on the positive class.

    Only meaningful for tampered images that have a ground-truth mask.
    """

    iou: float = 0.0
    dice: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    n_images: int = 0
    n_predicted: int = 0

    @classmethod
    def empty(cls) -> "PixelMetrics":
        return cls()


@dataclass
class ImageMetrics:
    """Image-level binary classification metrics."""

    n: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalReport:
    """Everything collected for one dataset split."""

    name: str
    images: ImageMetrics = field(default_factory=ImageMetrics)
    pixels: PixelMetrics = field(default_factory=PixelMetrics)
    confidences: list[float] = field(default_factory=list)
    labels: list[bool] = field(default_factory=list)

    @property
    def roc_auc(self) -> float:
        if len(set(self.labels)) < 2 or len(self.confidences) < 2:
            return float("nan")
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(self.labels, self.confidences))

    def summary(self, indent: str = "  ") -> str:
        lines = [
            f"=== {self.name} ===",
            f"{indent}images  n={self.images.n}  acc={self.images.accuracy:.3f}  "
            f"prec={self.images.precision:.3f}  rec={self.images.recall:.3f}  "
            f"F1={self.images.f1:.3f}  AUC={self.roc_auc:.3f}",
        ]
        if self.pixels.n_images:
            pm = self.pixels
            lines.append(
                f"{indent}pixels  n={pm.n_images}  IoU={pm.iou:.3f}  "
                f"Dice={pm.dice:.3f}  F1={pm.f1:.3f}  P={pm.precision:.3f}  R={pm.recall:.3f}"
            )
        return "\n".join(lines)


def image_metrics(
    predictions: list[DetectionResult], labels: list[bool]
) -> tuple[ImageMetrics, list[float], list[bool]]:
    """Accumulate image-level metrics from detector outputs."""
    m = ImageMetrics(n=len(predictions))
    confs: list[float] = []
    for pred, label in zip(predictions, labels):
        guess = pred.is_tampered
        if guess and label:
            m.tp += 1
        elif guess and not label:
            m.fp += 1
        elif not guess and not label:
            m.tn += 1
        else:
            m.fn += 1
        confs.append(float(pred.confidence))
    return m, confs, [bool(l) for l in labels]


def pixel_metrics(
    predictions: list[DetectionResult],
    masks: list[np.ndarray | None],
    labels: list[bool],
) -> PixelMetrics:
    """Pixel-level IoU / Dice on tampered images that have a mask.

    Predictions on *authentic* images (with no ground-truth forgery mask)
    are ignored for localization scoring but counted as an extra signal
    (they contribute no positives).
    """
    pm = PixelMetrics()
    all_ious: list[float] = []
    all_dices: list[float] = []
    all_precs: list[float] = []
    all_recalls: list[float] = []

    for pred, mask, label in zip(predictions, masks, labels):
        if not label or mask is None:
            continue
        pm.n_images += 1
        if pred.mask is None:
            continue
        pm.n_predicted += 1

        predicted = np.asarray(pred.mask > 0)
        truth = np.asarray(mask > 0)

        # Resize predicted to truth shape if needed.
        if predicted.shape != truth.shape:
            import cv2

            predicted = cv2.resize(
                predicted.astype(np.uint8),
                (truth.shape[1], truth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        inter = float(np.logical_and(predicted, truth).sum())
        union = float(np.logical_or(predicted, truth).sum())
        dice_num = 2.0 * inter
        dice_den = float(predicted.sum() + truth.sum())

        all_ious.append(inter / union if union > 0 else (1.0 if inter > 0 else 0.0))
        all_dices.append(dice_num / dice_den if dice_den > 0 else (1.0 if inter > 0 else 0.0))
        true_pos = inter
        all_precs.append(true_pos / predicted.sum() if predicted.sum() else 0.0)
        all_recalls.append(true_pos / truth.sum() if truth.sum() else 0.0)

    if all_ious:
        pm.iou = float(np.mean(all_ious))
        pm.dice = float(np.mean(all_dices))
        pm.precision = float(np.mean(all_precs))
        pm.recall = float(np.mean(all_recalls))
        pm.f1 = (
            2 * pm.precision * pm.recall / (pm.precision + pm.recall)
            if (pm.precision + pm.recall)
            else 0.0
        )
    return pm


def evaluate(
    name: str,
    predictions: list[DetectionResult],
    labels: list[bool],
    masks: list[np.ndarray | None],
) -> EvalReport:
    """Run both tiers of evaluation and return an aggregate report."""
    images, confs, lbls = image_metrics(predictions, labels)
    pixels = pixel_metrics(predictions, masks, labels)
    return EvalReport(
        name=name,
        images=images,
        pixels=pixels,
        confidences=confs,
        labels=lbls,
    )