"""Resolve packaged resources (configs, default weights) for installed use.

Everything here is ``__file__``-relative, so it works the same whether CandyEye
is run from a checkout or from a wheel in ``site-packages``.  Users never need
to know the package root: ``CandyEye()`` and ``train()`` call into this module
to find the default architecture YAML and the bundled ``yolo11n`` weights.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT / "configs"
ASSETS_DIR = PACKAGE_ROOT / "assets"
DEFAULT_CONFIG = "yolo11.yaml"
DEFAULT_WEIGHTS = "yolo11n.pth"


def cache_dir() -> Path:
    """Per-user cache dir (overridable via ``$CANDYEYE_CACHE``)."""
    override = os.environ.get("CANDYEYE_CACHE")
    base = Path(override) if override else Path.home() / ".cache" / "candyeye"
    return base


def available_configs() -> dict[str, Path]:
    """Map bundled config stem -> path (``yolo11`` -> ``.../yolo11.yaml``)."""
    if not CONFIG_DIR.is_dir():
        return {}
    return {path.stem: path for path in sorted(CONFIG_DIR.glob("*.yaml"))}


def resolve_config(cfg=None) -> str | dict:
    """Normalise a model config argument.

    Accepts a dict (returned as-is), an existing YAML path, or a bundled config
    name such as ``"yolo11"`` / ``"yolo11.yaml"``.  ``None`` selects the default
    architecture.  This lets callers write ``CandyEye()`` or
    ``CandyEye("yolo11_exchange")`` instead of repo-root paths.
    """
    if isinstance(cfg, dict):
        return cfg
    if cfg is None:
        return str(CONFIG_DIR / DEFAULT_CONFIG)
    candidate = Path(cfg)
    if candidate.is_file():
        return str(candidate)
    configs = available_configs()
    name = candidate.name
    for key in (candidate.stem, name):
        if key in configs:
            return str(configs[key])
    raise FileNotFoundError(
        f"config {cfg!r} not found. Bundled configs: {sorted(configs)}. "
        "Pass a dict or an existing .yaml path to use your own architecture."
    )


def default_weights_path(name: str = "yolo11n") -> Path:
    """Path to the default pretrained weights.

    Resolution order: ``$CANDYEYE_WEIGHTS`` override, then the bundled
    ``assets/<name>.pth``, then a per-user cache download location.  The wheel
    ships the weights, so the bundled copy is normally what is returned.
    """
    override = os.environ.get("CANDYEYE_WEIGHTS")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"$CANDYEYE_WEIGHTS points to missing file: {path}")
        return path

    filename = name if name.endswith(".pth") else f"{name}.pth"
    bundled = ASSETS_DIR / filename
    if bundled.is_file():
        return bundled

    cached = cache_dir() / filename
    if cached.is_file():
        return cached
    raise FileNotFoundError(
        f"default weights {filename!r} not found (looked in {bundled} and {cached}). "
        "Set $CANDYEYE_WEIGHTS to an existing clean state_dict (.pth)."
    )


def default_config_path() -> Path:
    """Path to the bundled default architecture YAML."""
    return CONFIG_DIR / DEFAULT_CONFIG
