"""CandyEye — a lightweight, CPU-first object detector.

Public API::

    from candyeye import CandyEye, train

    model = CandyEye()                     # bundled yolo11n architecture
    results = model.train(data="dataset.yaml", epochs=100, pretrained=True)

    # ...or the function form:
    results = train(data="dataset.yaml", epochs=100)

Heavy submodules (torch, torchvision) are imported lazily so ``import
candyeye`` stays cheap.
"""
from __future__ import annotations

from candyeye.paths import (
    available_configs,
    cache_dir,
    default_config_path,
    default_weights_path,
    resolve_config,
)

__version__ = "0.1.0"

__all__ = [
    "CandyEye",
    "CandyEyePredictor",
    "train",
    "predict",
    "resolve_config",
    "default_config_path",
    "default_weights_path",
    "available_configs",
    "cache_dir",
    "__version__",
]


def __getattr__(name):
    if name == "CandyEye":
        from candyeye.core.candyeye import CandyEye
        return CandyEye
    if name == "train":
        from candyeye.training.trainer import train
        return train
    if name == "predict":
        from candyeye.inference.predict import predict
        return predict
    if name == "CandyEyePredictor":
        from candyeye.inference.predict import CandyEyePredictor
        return CandyEyePredictor
    raise AttributeError(f"module 'candyeye' has no attribute {name!r}")
