"""Single source of truth for paths and training data sources.

Reads config.toml at the repository root. Relative paths are resolved
against that root so every script behaves the same from any working
directory. Command-line flags still override these defaults.
"""
from __future__ import annotations

import dataclasses
import os
import tomllib
from pathlib import Path

DEFAULT_CONFIG = "config.toml"


def repo_root() -> Path:
    """Repository root: two levels up from this file (ifd/config.py)."""
    return Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Config:
    authentic: str
    casia: str
    keep_synthetic: bool
    models: str
    reports: str
    synthetic: str
    probes: str


def load(config_file: str | os.PathLike | None = None) -> Config:
    """Load train data sources and output paths from config.toml.

    Missing keys fall back to the shipped defaults. Empty strings for
    `authentic` / `casia` mean "not configured" for the caller.
    """
    root = repo_root()
    path = Path(config_file) if config_file else root / DEFAULT_CONFIG
    data = {}
    if path.is_file():
        with open(path, "rb") as fh:
            data = tomllib.load(fh)

    def resolve(section: str, key: str, default: str) -> str:
        value = data.get(section, {}).get(key, default)
        if not value:
            return ""
        pth = Path(value)
        return str(pth if pth.is_absolute() else root / pth)

    training = data.get("training", {})
    paths = data.get("paths", {})
    return Config(
        authentic=resolve("training", "authentic", "data/authentic"),
        casia=resolve("training", "casia", ""),
        keep_synthetic=bool(training.get("keep_synthetic", True)),
        models=resolve("paths", "models", "models"),
        reports=resolve("paths", "reports", "reports"),
        synthetic=resolve("paths", "synthetic", "synthetic"),
        probes=" ".join(str(root / p) for p in paths.get("probes", ["test.jpg", "test2.jpg"])),
    )