"""Shared utilities: batching, image I/O sugar, paths."""

import os


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path