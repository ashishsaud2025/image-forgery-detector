"""Generate synthetic spliced images that are detectable by construction.

Creates the *classic* ELA scenario the pipeline is designed for:

- The host photo is JPEG-encoded at a high quality (qh).
- The spliced patch comes from a *donor* photo JPEG-encoded at a much
  lower quality (qp), i.e. a different compression history.
- The final composite is re-saved at qh, so the host region is at a
  JPEG "fixpoint" (round-tripping at qh changes almost nothing) while the
  patch region still carries the strong quantization loss of qp.

Authentic counterparts are the host alone (under the same pipeline).

Also writes a ground-truth mask next to each forged image and a JPEG
version so the ELA branch has real compression history.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ifd.config import load as load_config

CFG = load_config()


def build_photo(rng: np.random.Generator, h: int = 512, w: int = 512) -> np.ndarray:
    """Synthesize a photolike RGB image: gradient sky + soft blobs + noise."""
    gy = np.linspace(30, 240, h, dtype=np.float32)[:, None]
    gx = np.linspace(20, 170, w, dtype=np.float32)[None, :]
    base = np.clip(gy + gx, 0, 255).astype(np.uint8)
    base = np.repeat(base[:, :, None], 3, axis=2)
    for _ in range(14):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        rr = int(rng.integers(18, 95))
        yy, xx = np.ogrid[:h, :w]
        disc = ((yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5 < rr
        base[disc] = rng.integers(40, 230, size=3)
    base = cv2.GaussianBlur(base, (7, 7), 0)
    base = np.clip(base.astype(np.float32) + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
    return base


def save_jpeg(image: np.ndarray, path: str, quality: int) -> None:
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])


def jpeg_encode_decode(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode and immediately decode at `quality` (gets a compression history)."""
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        save_jpeg(image, tmp, quality)
        return cv2.cvtColor(cv2.imread(tmp), cv2.COLOR_BGR2RGB)
    finally:
        os.remove(tmp)


def make_pair(
    seed: int,
    qh: int = 90,
    qp: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (authentic, forged, mask) triple sharing the same host."""
    rng = np.random.default_rng(seed)
    parent = build_photo(rng)
    donor = build_photo(rng)

    host = jpeg_encode_decode(parent, qh)
    patch_src = jpeg_encode_decode(donor, qp)

    m = cv2.getRotationMatrix2D(
        (patch_src.shape[1] / 2, patch_src.shape[0] / 2),
        rng.uniform(-20, 20),
        rng.uniform(0.9, 1.1),
    )
    patch = cv2.warpAffine(patch_src, m, (patch_src.shape[1], patch_src.shape[0]), borderMode=cv2.BORDER_REFLECT)

    y, x = int(rng.integers(40, 240)), int(rng.integers(40, 280))
    ph, pw = 160, 180

    forged = host.copy()
    forged[y : y + ph, x : x + pw] = patch[:ph, :pw]
    mask = np.zeros((host.shape[0], host.shape[1]), dtype=np.uint8)
    mask[y : y + ph, x : x + pw] = 255

    forged = jpeg_encode_decode(forged, qh)
    return host, forged, mask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=CFG.synthetic, help="Output directory.")
    p.add_argument("--n", type=int, default=1, help="Number of (auth|forged) pairs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qh", type=int, default=90, help="Host/final save quality.")
    p.add_argument("--qp", type=int, default=50, help="Patch donor quality (different history).")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        seed = args.seed + i
        auth, forged, mask = make_pair(seed, args.qh, args.qp)
        a_path = os.path.join(args.out, f"auth_{seed}.jpg")
        f_path = os.path.join(args.out, f"forged_{seed}.jpg")
        m_path = os.path.join(args.out, f"forged_{seed}_mask.png")
        save_jpeg(auth, a_path, args.qh)
        save_jpeg(forged, f_path, args.qh)
        cv2.imwrite(m_path, mask)
        print(f"seed={seed}: {a_path}, {f_path}, {m_path}")


if __name__ == "__main__":
    main()