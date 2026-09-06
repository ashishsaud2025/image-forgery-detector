"""Train the block-level classifier (learned fusion tier).

Runs on synthetic mixed-history forgeries by default so the pipeline can
be validated end-to-end without downloading a dataset. To train on real
data, point --dataset-root at a CASIA/Columbia tree; any image with no
mask is treated as authentic background.

    python scripts/train_blocks.py --n-train 12 --n-test 6 --out models

Saves `models/block_rf.joblib` and prints image-level + pixel-level
metrics on the held-out test set using the evaluation harness.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ifd import Detector
from ifd.config import load as load_config
from ifd.detector import build_feature_maps
from ifd.eval import evaluate
from ifd.fusion import BlockFeatureClassifier

CFG = load_config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

SWEEP_QUALITIES = (70, 85, 95)


def _synthetic_split(n_train: int, n_test: int, qh: int, qp: int, seed: int = 0):
    """Return lists of (image, label, mask) for authentic + forged samples."""
    from make_synthetic import make_pair

    train, test = [], []
    for i in range(n_train):
        auth, forged, mask = make_pair(seed + i, qh, qp)
        train.append((auth, 0, None))
        train.append((forged, 1, mask))
    for i in range(n_train, n_train + n_test):
        auth, forged, mask = make_pair(seed + i, qh, qp)
        test.append((auth, 0, None))
        test.append((forged, 1, mask))
    return train, test


def _spaced(items: list, n: int) -> list:
    """Sample up to `n` items evenly across the whole list.

    `CasiaDataset` orders samples by file name, which clusters manipulation
    classes together (all CND/CNN/... first). Taking the head of the list
    over-weights the alphabetically-first classes (mostly hard copy-move
    cases, several without masks); spacing the selection keeps the sample
    representative of the full CASIA mix.
    """
    n = min(n, len(items))
    if n <= 0:
        return []
    idx = np.linspace(0, len(items) - 1, n, dtype=int)
    return [items[i] for i in idx]


def _dataset_split(root: str, max_per_class: int):
    from ifd.data import CasiaDataset

    ds = CasiaDataset(root)
    sample = ds.summary()
    LOGGER.info(sample)
    train, test = [], []
    for i, s in enumerate(_spaced(ds.tampered, 2 * max_per_class)):
        img = ds.load_image(s)
        mask = ds.load_mask(s) if s.mask_path else None
        if mask is not None and mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(
                mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        (test if i % 4 == 0 else train).append((img, 1, mask))
    for i, s in enumerate(_spaced(ds.authentic, max_per_class)):
        img = ds.load_image(s)
        (test if i % 4 == 0 else train).append((img, 0, None))
    return train, test


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=CFG.models, help="Output directory for the .joblib model.")
    p.add_argument("--n-train", type=int, default=12)
    p.add_argument("--n-test", type=int, default=6)
    p.add_argument("--qh", type=int, default=90, help="Host/final save quality for synthetic forgery.")
    p.add_argument("--qp", type=int, default=50, help="Patch donor quality.")
    p.add_argument("--quality", type=int, default=None,
                   help="Single fixed ELA re-save quality for training. Default: random per image "
                        "from the sweep %s (robust to unknown provenance)." % (SWEEP_QUALITIES,))
    p.add_argument("--dataset-root", default=(CFG.casia or None),
                   help="Real dataset root (CASIA/Columbia). Default from config.toml "
                        "[training] casia (= %r)." % (CFG.casia,))
    p.add_argument("--keep-synthetic", action="store_true",
                   help="Mix synthetic mixed-history pairs into the dataset training set "
                        "alongside --dataset-root, instead of replacing them.")
    p.add_argument(
        "--authentic-root",
        default=(CFG.authentic or None),
        help="Folder of known-authentic real photos to add as label=0 training "
             "samples. Default from config.toml [training] authentic (= %r). "
             "Mixing real texture into the synthetic-only training set "
             "stops the classifier from learning 'high texture = tampered'."
        % (CFG.authentic,),
    )
    p.add_argument("--real-block-budget", type=int, default=512,
                   help="Max blocks sampled per real authentic photo during training. "
                        "Use a large value (e.g. 2500) so dense texture regions are "
                        "covered; leave small only if training time matters.")
    p.add_argument("--block", type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    clf = BlockFeatureClassifier(block=args.block)
    rng = np.random.default_rng(0)

    if args.quality is not None:
        fixed_quality = args.quality
    else:
        fixed_quality = None

    def elq(i: int) -> int:
        return fixed_quality if fixed_quality is not None else int(rng.choice(SWEEP_QUALITIES))

    if args.dataset_root:
        train, test = _dataset_split(args.dataset_root, args.n_train)
        if args.keep_synthetic or CFG.keep_synthetic:
            syn_train, _ = _synthetic_split(args.n_train, 0, args.qh, args.qp, seed=0)
            train = syn_train + train
            LOGGER.info("Mixed %d synthetic pairs into dataset training set",
                        len(syn_train) // 2)
    else:
        train, test = _synthetic_split(args.n_train, args.n_test, args.qh, args.qp, seed=0)
        if args.keep_synthetic:
            LOGGER.warning("--keep-synthetic has no effect without --dataset-root")

    if args.authentic_root:
        from ifd.data.loaders import list_images
        from ifd.data.loaders import load_image as _load_image

        # Real photos get a (possibly much larger) block budget so dense
        # texture regions are actually seen as authentic during training.
        wide = BlockFeatureClassifier(block=args.block, max_blocks_per_image=args.real_block_budget)
        n_auth = 0
        for p in list_images(args.authentic_root):
            try:
                train.append((_load_image(p), 0, None, wide))
                n_auth += 1
            except Exception as exc:  # skip unreadable files silently
                LOGGER.warning("skipping real photo %s: %s", p, exc)
        LOGGER.info("Added %d real authentic photos from %s (block budget %d)",
                    n_auth, args.authentic_root, args.real_block_budget)

    # Extract block features + labels
    feats_all, labels_all = [], []
    for i, item in enumerate(train):
        img, label, mask, wide = (*item, None)[:4]
        extractor = wide or clf
        quality = elq(i)
        maps = build_feature_maps(img, quality)
        feats, corners = extractor.extract(maps)
        lbl = extractor.label_blocks(corners, mask if label else None)
        feats_all.append(feats)
        labels_all.append(lbl)
        if (i + 1) % 10 == 0:
            LOGGER.info("Extracted %d/%d train images", i + 1, len(train))

    clf.fit(feats_all, labels_all)
    n_forged = sum(int(lbl.sum()) for lbl in labels_all)
    LOGGER.info("Trained on %d blocks (%d forged-labeled)",
                sum(f.shape[0] for f in feats_all), n_forged)

    # Evaluate on held-out test set
    preds, labels_out, masks_out = [], [], []
    det = Detector(classifier=clf)
    for i, (img, label, mask) in enumerate(test):
        # Infer at the *default* quality (as an analyst with no provenance
        # information would), not at any qh the generator used.
        result = det.detect(img)
        preds.append(result)
        labels_out.append(bool(label))
        masks_out.append(mask)
        if (i + 1) % 10 == 0:
            LOGGER.info("Detected %d/%d test images", i + 1, len(test))

    report = evaluate("held-out", preds, labels_out, masks_out)
    print(report.summary())

    out_path = os.path.join(args.out, "block_rf.joblib")
    try:
        import joblib

        joblib.dump(clf, out_path)
        print(f"Saved model -> {out_path}")
    except ImportError:
        import pickle

        with open(os.path.join(args.out, "block_rf.pkl"), "wb") as fh:
            pickle.dump(clf, fh)
        print("joblib unavailable; saved pickle instead.")


if __name__ == "__main__":
    main()