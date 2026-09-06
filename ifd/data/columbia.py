"""Columbia image-splicing dataset loader.

Columbia Uncompressed Image Splicing Detection includes authentic and
spliced image pairs. The directory layout is loose (each edition nests
differently); we detect "au"/"sp" (or "orig"/"spliced") markers in the
filename, falling back to directory clues.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .loaders import list_images


@dataclass
class ColumbiaSample:
    image_path: str
    is_tampered: bool


def _has_marker(path: str, markers: tuple[str, ...]) -> bool:
    low = os.path.basename(path).lower()
    return any(m in low for m in markers)


def list_columbia_samples(root: str) -> list[ColumbiaSample]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Columbia root not found: {root}")
    samples: list[ColumbiaSample] = []
    for path in list_images(root):
        low = os.path.basename(path).lower()
        if _has_marker(low, ("au_", "orig", "_authentic")):
            samples.append(ColumbiaSample(path, False))
        elif _has_marker(low, ("sp_", "spli", "_forged")):
            samples.append(ColumbiaSample(path, True))
        # Unknown naming: skip rather than guess wrong.
    return samples


class ColumbiaDataset:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.samples = list_columbia_samples(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def tampered(self) -> list[ColumbiaSample]:
        return [s for s in self.samples if s.is_tampered]

    @property
    def authentic(self) -> list[ColumbiaSample]:
        return [s for s in self.samples if not s.is_tampered]

    def summary(self) -> str:
        return (
            f"ColumbiaDataset({os.path.basename(self.root) or self.root}):\n"
            f"  total={len(self.samples)} authentic={len(self.authentic)} tampered={len(self.tampered)}"
        )