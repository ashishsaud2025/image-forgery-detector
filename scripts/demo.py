"""Demo CLI: run the full pipeline on an image/dataset and print a report.

Usage:
    python scripts/demo.py --image path/to/image.jpg [--out dir]
    python scripts/demo.py --casia path/to/CASIA2 [--max-images 200]
    python scripts/demo.py --casia1 path/to/CASIAS1
    python scripts/demo.py --columbia path/to/Columbia

Saves an HTML report with side-by-side heatmaps, a localization overlay,
and (when a mask is available) pixel-level metrics.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ifd import Detector, DetectorConfig, data, eval as ifd_eval
from ifd.config import load as load_config
from ifd.fusion import DetectionResult

CFG = load_config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def build_visual_pages(
    image: np.ndarray,
    result: DetectionResult,
    mask: np.ndarray | None,
    out_path: str,
) -> None:
    """Write a small self-contained HTML page with the demo visuals."""
    import base64
    import cv2

    # Prepare panels.
    rgb = image
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ela = result.feature_maps["ela"] if result.feature_maps else None
    noise = result.feature_maps["noise"] if result.feature_maps else None
    heat = result.heatmap

    def gray_to_jet_cmap(arr):
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float32)
        a = a - a.min()
        if a.max() > 0:
            a = a / a.max()
        gray = (a * 255).astype(np.uint8)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Overlay heat on original.
    overlay = rgb_bgr.copy() if isinstance(rgb_bgr, np.ndarray) else None
    heat_rgb = gray_to_jet_cmap(heat) if overlay is not None else None
    if overlay is not None and heat_rgb is not None:
        alpha = 0.45
        overlay = cv2.addWeighted(overlay, 1 - alpha, heat_rgb, alpha, 0)
        if mask is not None:
            m = (mask > 0).astype(np.uint8)
            red = np.zeros_like(rgb_bgr)
            red[:, :, 2] = 200
            red = cv2.bitwise_and(red, red, mask=m)
            overlay = cv2.addWeighted(overlay, 0.85, red, 0.15, 0)

    def to_data_uri(arr: np.ndarray | None) -> str:
        if arr is None:
            arr = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", arr)
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

    panels = {
        "Original": to_data_uri(rgb_bgr),
        "ELA map": to_data_uri(gray_to_jet_cmap(ela)),
        "Noise map": to_data_uri(gray_to_jet_cmap(noise)),
        "Fused heat": to_data_uri(heat_rgb),
        "Overlay": to_data_uri(overlay),
    }
    if mask is not None:
        panels["Ground truth"] = to_data_uri((mask > 0).astype(np.uint8) * 255)

    html = [
        "<html><head><title>IFD Report</title>",
        "<style>body{font-family:Consolas,monospace;background:#111;color:#ddd;margin:24px}"
        "h1{color:#ffdd57}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}"
        ".cell{background:#1c1c1c;border:1px solid #333;padding:8px}.cell img{width:100%}"
        ".cell .t{font-size:12px;color:#aaa;margin-top:6px}"
        ".badge{padding:2px 10px;border-radius:9px;font-weight:bold}"
        ".bad.t{background:#3a1414;color:#ff7b7b}.bad.f{background:#143a14;color:#7bff7b}</style></head><body>",
        f'<h1>Image Forgery Detector</h1><p>Verdict: <span class="badge bad {"t" if result.is_tampered else "f"}">'
        f'{"TAMPERED" if result.is_tampered else "AUTHENTIC"}</span> '
        f'confidence={result.confidence:.3f}</p>',
        '<div class="grid">',
    ]
    for title, uri in panels.items():
        html.append(f'<div class="cell"><img src="{uri}"><div class="t">{title}</div></div>')
    html.append("</div></body></html>")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))


def _load_from_sample(s):
    return data.load_image(s.image_path)


def _load_mask_from_sample(s):
    mask_path = getattr(s, "mask_path", None)
    return data.load_mask(mask_path) if mask_path else None


def detect_dataset(
    samples,
    label_text: str,
    max_images: int,
    detector_factory=None,
    load_image=None,
    load_mask=None,
):
    """Run detection over sample objects, decoding each image on demand.

    `samples` is an iterable of objects carrying `.is_tampered` and (for
    CASIA) `.mask_path` (e.g. ``CasiaSample`` / ``ColumbiaSample``). Images
    and masks are loaded one at a time inside the loop, so at most one
    decoded image (plus its mask) is alive regardless of `--max-images`.
    """
    import cv2

    load_img = load_image or _load_from_sample
    load_truth = load_mask or _load_mask_from_sample
    det = (detector_factory or (lambda: Detector(DetectorConfig())))()
    preds: list[DetectionResult] = []
    labels: list[bool] = []
    masks: list[np.ndarray | None] = []
    used: list[str] = []
    for s in samples:
        if max_images and len(preds) >= max_images:
            break
        try:
            img = load_img(s)
            mask = load_truth(s)
            if mask is not None and mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(
                    mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
                )
        except Exception as exc:
            logging.warning("load failed for %s: %s", s, exc)
            continue
        r = det.detect(img)
        preds.append(r)
        labels.append(bool(s.is_tampered))
        masks.append(mask)
        used.append(getattr(s, "name", str(len(preds))))
    report = ifd_eval.evaluate(label_text, preds, labels, masks)
    print(report.summary())
    return preds, used


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="Single image to analyze.")
    p.add_argument("--out", default=CFG.reports, help="Output dir for HTML pages.")
    p.add_argument("--casia", help="Root of CASIA v2 dataset.")
    p.add_argument("--casia1", help="Root of CASIA v1 dataset.")
    p.add_argument("--columbia", help="Root of Columbia dataset.")
    p.add_argument("--max-images", type=int, default=0, help="Cap number of images per dataset (0 = all).")
    p.add_argument("--model", default=None, help="Path to a trained BlockFeatureClassifier (.joblib/.pkl).")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    classifier = None
    if args.model:
        try:
            import joblib

            classifier = joblib.load(args.model)
        except Exception as exc:
            import pickle

            with open(args.model, "rb") as fh:
                classifier = pickle.load(fh)
        logging.info("Loaded classifier from %s", args.model)

    def make_detector():
        return Detector(DetectorConfig(), classifier=classifier)

    if args.image:
        img = data.load_image(args.image)
        det = make_detector()
        result = det.detect(img)
        print(f"Image {args.image}: {'TAMPERED' if result.is_tampered else 'AUTHENTIC'} "
              f"confidence={result.confidence:.3f}")
        page = os.path.join(args.out, "single.html")
        build_visual_pages(img, result, None, page)
        print(f"Wrote {page}")
        return

    if args.casia:
        ds = data.CasiaDataset(args.casia)
        print(ds.summary())
        tam, aut = ds.tampered, ds.authentic
        n = args.max_images // 2 if args.max_images else None
        samples = (tam[:n] if n else tam) + (aut[:n] if n else aut)
        detect_dataset(
            samples, "CASIA v2", args.max_images or 0, make_detector,
            load_image=ds.load_image, load_mask=ds.load_mask,
        )

    if args.casia1:
        ds = data.CasiaDataset(args.casia1)
        print(ds.summary())
        samples = ds.tampered
        detect_dataset(
            samples, "CASIA v1", args.max_images or 0, make_detector,
            load_image=ds.load_image, load_mask=ds.load_mask,
        )

    if args.columbia:
        ds = data.ColumbiaDataset(args.columbia)
        print(ds.summary())
        detect_dataset(ds.samples, "Columbia", args.max_images or 0, make_detector)


if __name__ == "__main__":
    main()
