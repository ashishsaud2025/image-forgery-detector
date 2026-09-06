"""End-to-end smoke test on automatically-generated splicing cases.

1. Generates a train/test set of (authentic | forged) photo pairs using
   the classic mixed-compression-history recipe (`scripts/make_synthetic.py`).
2. Trains the block-level learned classifier on the train split.
3. Runs the full `Detector` on held-out images and checks:
   - image-level verdicts are correct (no false positives / negatives)
   - the learned localization mask overlaps the ground truth
   - the heuristic-tier detector also runs (as the documented baseline)
4. Prints an `ifd.eval` report.

Usage: python scripts/smoke_test.py [--n-train 12] [--n-test 6]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from make_synthetic import make_pair

from ifd import Detector
from ifd.detector import build_feature_maps
from ifd.eval import evaluate
from ifd.fusion import BlockFeatureClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

SWEEP_QUALITIES = (70, 85, 95)


def run() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=12)
    p.add_argument("--n-test", type=int, default=6)
    p.add_argument("--qh", type=int, default=90)
    p.add_argument("--qp", type=int, default=50)
    args = p.parse_args()

    qh, qp = args.qh, args.qp

    # 1. Data
    train, test = [], []
    for i in range(args.n_train):
        auth, forged, mask = make_pair(i, qh, qp)
        train.append((auth, 0, None))
        train.append((forged, 1, mask))
    for i in range(args.n_train, args.n_train + args.n_test):
        auth, forged, mask = make_pair(i, qh, qp)
        test.append((auth, 0, None))
        test.append((forged, 1, mask))

    # 2. Train learned tier
    clf = BlockFeatureClassifier(block=64)
    rng = np.random.default_rng(0)
    feats_all, labels_all = [], []
    for i, (img, label, mask) in enumerate(train):
        q = int(rng.choice(SWEEP_QUALITIES))
        maps = build_feature_maps(img, q)
        feats, corners = clf.extract(maps)
        feats_all.append(feats)
        labels_all.append(clf.label_blocks(corners, mask if label else None))
    clf.fit(feats_all, labels_all)
    LOGGER.info("Trained on %d blocks", sum(f.shape[0] for f in feats_all))

    # 3. Detect held-out (default quality, as an analyst without
    #    provenance information would)
    det_learned = Detector(classifier=clf)
    preds, labels, masks_gt = [], [], []
    for img, label, mask in test:
        result = det_learned.detect(img)
        preds.append(result)
        labels.append(bool(label))
        masks_gt.append(mask)

    report = evaluate("smoke (learned)", preds, labels, masks_gt)
    print(report.summary())

    # 4. Heuristic baseline (documented-brittle)
    det_heuristic = Detector()
    preds_h, labels_h, masks_h = [], [], []
    for img, label, mask in test:
        preds_h.append(det_heuristic.detect(img))
        labels_h.append(bool(label))
        masks_h.append(mask)
    report_h = evaluate("smoke (heuristic)", preds_h, labels_h, masks_h)
    print(report_h.summary())

    # 5. Assertions
    ok = True

    img_acc = report.images.accuracy
    if img_acc < 1.0:
        LOGGER.warning("Learned image-level accuracy %.2f < 1.0", img_acc)
        ok = False
    if report.pixels.n_images == 0 or report.pixels.iou <= 0.0:
        LOGGER.warning("No/localization too weak: IoU=%.3f (n=%d)",
                       report.pixels.iou, report.pixels.n_images)
        ok = False

    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())