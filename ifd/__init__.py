"""Image Forgery Detector (IFD): ELA + noise-fingerprint forensics.

Combined Error Level Analysis and blind noise-residual forensics pipeline
with a localization heatmap and an evaluation harness.
"""
from .detector import Detector, DetectorConfig, DetectionResult
from . import data
from . import features
from . import fusion
from . import eval

__version__ = "0.1.0"

__all__ = [
    "Detector",
    "DetectorConfig",
    "DetectionResult",
    "data",
    "features",
    "fusion",
    "eval",
    "__version__",
]