"""CASIA dataset loader.

Loads CASIA v1/v2 images and pairs them with community-maintained
ground-truth masks (github.com/namtpham/casia2groundtruth and
casia1groundtruth). Handles the documented mask/image resolution mismatch
(i.e. the IML-Dataset-Corrections skip list) by excluding those files;
tampered images with no mask are indexed with `mask_path=None` and
loaders guard against that.

Naming inside a CASIA v2 directory looks like:
    Tp_D_N_A_s001_001.jpg          -> tampered
    Au_D_N_A_s001_01.jpg           -> authentic
    Tp_D_N_A_s001_001_gt.png       -> mask (community ground truth)

CASIA v1 tampered prefixes are `Tp_*` (tampering) and `Sp_*` (splicing).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .loaders import list_images
from .loaders import load_image as _load_image
from .loaders import load_mask as _load_mask


@dataclass
class CasiaSample:
    """A single image with its identity label and optional mask path."""

    image_path: str
    is_tampered: bool
    mask_path: str | None = None

    @property
    def name(self) -> str:
        return os.path.basename(self.image_path)


# Files documented as broken / misaligned in IML-Dataset-Corrections.
# Source: https://github.com/SunnyHaze/IML-Dataset-Corrections (synced 2026-09-05).
# Re-check that repo before assuming this list is complete or current.
KNOWN_BROKEN: frozenset[str] = frozenset(
    {
        # CASIA v2: mask/image resolution mismatches.
        "Tp_D_CNN_M_N_sec00011_cha00085_11227.jpg",
        "Tp_D_CRN_S_N_ani10191_ani10190_12437.jpg",
        "Tp_D_CRN_S_N_nat10130_pla00049_11524.jpg",
        "Tp_D_NND_M_B_nat20098_nat20073_01602.tif",
        "Tp_D_NRN_M_N_nat10134_nat00095_11912.jpg",
        "Tp_D_NRN_M_N_nat10134_nat10124_11913.jpg",
        "Tp_S_CRN_S_N_art00059_art00059_10508.tif",
        "Tp_S_NND_S_N_sec20064_sec20064_01654.tif",
        "Tp_S_NNN_S_N_art20077_art20077_01883.tif",
        "Tp_S_NNN_S_N_ind20037_ind20037_01778.tif",
        "Tp_S_NNN_S_N_sec00012_sec00012_11230.jpg",
        "Tp_S_NNN_S_N_sec00074_sec00074_00751.tif",
        "Tp_S_NRD_S_N_arc20079_arc20079_01719.tif",
        "Tp_S_NRD_S_N_pla20071_pla20071_01971.tif",
        "Tp_S_NRN_S_B_ind10002_ind10002_20010.jpg",
        "Tp_S_NRN_S_N_art20077_art20077_02316.tif",
        "Tp_S_NRN_S_N_pla20080_pla20080_01980.tif",
        # CASIA v1: documented as having no ground-truth mask at all.
        "Sp_D_NRN_A_cha0011_sec0011_0542.jpg",
    }
)


def skip_file(name: str) -> bool:
    """True if the file is a mask or known-broken and should be excluded."""
    if re.search(r"_gt\d*\.png$", name.lower()):
        return True
    if name.lower().endswith(("-gt.png", "_mask.png", "groundtruth.png")):
        return True
    return name in KNOWN_BROKEN


def is_tampered(filename: str) -> bool:
    """True for CASIA tampered prefixes: `Tp_*` and (v1) `Sp_*`."""
    return filename.lower().startswith(("t", "tp", "tampered", "sp"))


GT_SUFFIXES = ("", "1", "2", "3", "4", "5", "6", "7", "8", "9")


def _groundtruth_dirs(image_dir: str) -> list[str]:
    """Directories where a ground-truth mask might live for a CASIA image.

    1. next to the image itself (community convention: <base>_gt.png)
    2. a sibling '<dir>_gt' folder
    3. a sibling folder whose name contains 'groundtruth'
       (e.g. the official 'CASIA 2 Groundtruth' zip layout)
    """
    dirs = [image_dir]
    parent = os.path.dirname(image_dir)
    if parent and os.path.isdir(parent):
        dirs.append(os.path.join(parent, os.path.basename(image_dir) + "_gt"))
        dirs.append(os.path.join(parent, "groundtruth"))
        for name in os.listdir(parent):
            full = os.path.join(parent, name)
            if os.path.isdir(full) and "groundtruth" in name.lower():
                dirs.append(full)
    return dirs


def guess_mask_path(image_path: str) -> str | None:
    """Try to recover the ground-truth mask for a CASIA image.

    Some community masks carry a numbered suffix (`<base>_gt1.png`,
    `<base>_gt3.png`), so all variants are probed.
    """
    d = os.path.dirname(image_path)
    base, _ext = os.path.splitext(os.path.basename(image_path))
    for cand_dir in _groundtruth_dirs(d):
        for sfx in GT_SUFFIXES:
            c = os.path.join(cand_dir, f"{base}_gt{sfx}.png")
            if os.path.isfile(c):
                return c
    return None


def list_casia_samples(root: str) -> list[CasiaSample]:
    """Recursively index CASIA images and pair them with masks."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CASIA root not found: {root}")
    samples: list[CasiaSample] = []
    for path in list_images(root):
        name = os.path.basename(path)
        if skip_file(name):
            continue
        if name.lower().startswith("au"):
            tampered = False
        elif is_tampered(name):
            tampered = True
        else:
            continue
        mask = guess_mask_path(path) if tampered else None
        samples.append(CasiaSample(image_path=path, is_tampered=tampered, mask_path=mask))
    return samples


class CasiaDataset:
    """Lazy view over a CASIA directory tree."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.samples: list[CasiaSample] = list_casia_samples(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        for s in self.samples:
            yield s

    @property
    def tampered(self) -> list[CasiaSample]:
        return [s for s in self.samples if s.is_tampered]

    @property
    def authentic(self) -> list[CasiaSample]:
        return [s for s in self.samples if not s.is_tampered]

    def get_sample(self, index: int) -> CasiaSample:
        return self.samples[index]

    def load_image(self, sample: CasiaSample):
        return _load_image(sample.image_path)

    def load_mask(self, sample: CasiaSample):
        if sample.mask_path is None:
            return None
        return _load_mask(sample.mask_path)

    def summary(self) -> str:
        n_tam = len(self.tampered)
        n_auth = len(self.authentic)
        n_mask = sum(1 for s in self.tampered if s.mask_path)
        pct = (n_mask / n_tam * 100) if n_tam else 0.0
        return (
            f"CasiaDataset({os.path.basename(self.root) or self.root}):\n"
            f"  total={len(self.samples)} authentic={n_auth} tampered={n_tam}\n"
            f"  tampered with masks: {n_mask} ({pct:.0f}%)"
        )

    def write_cache_index(self, out_dir: str) -> str:
        """Dump a CSV index of image/mask/label for reproducibility."""
        import csv

        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "casia_index.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "mask", "label"])
            for s in self.samples:
                writer.writerow([s.image_path, s.mask_path or "", int(s.is_tampered)])
        return out