"""Resolve packaged resources (configs, default weights) for installed use.

Everything here is ``__file__``-relative, so it works the same whether CandyEye
is run from a checkout or from a wheel in ``site-packages``.  Users never need
to know the package root: ``CandyEye()`` and ``train()`` call into this module
to find the default architecture YAML and the default ``yolo11n`` weights.

The distribution ships the architecture configs but **not** the weights (they
are derived from Ultralytics' AGPL-3.0 checkpoints).  When no local copy exists,
:func:`default_weights_path` downloads the official checkpoint and converts it
to a clean state_dict in the per-user cache on first use.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT / "configs"
ASSETS_DIR = PACKAGE_ROOT / "assets"
DEFAULT_CONFIG = "yolo11.yaml"
DEFAULT_WEIGHTS = "yolo11n"

_ASSETS = "https://github.com/ultralytics/assets/releases/download"
_WEIGHTS_URL = f"{_ASSETS}/v8.3.0/{{name}}.pt"  # official Ultralytics checkpoints


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


def _download_file(url: str, destination: Path) -> None:
    """Stream ``url`` to ``destination`` (cleaning up on failure)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "candyeye"})
    try:
        with urllib.request.urlopen(request) as response, \
                open(destination, "wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def download_default_weights(name: str = DEFAULT_WEIGHTS) -> Path:
    """Fetch and convert the default weights into the cache; return the path.

    The official Ultralytics ``.pt`` is downloaded and re-saved as a clean
    state_dict of plain tensors (``<name>.pth``) so the rest of CandyEye never
    needs Ultralytics installed.  Unpickling the official checkpoint does need
    Ultralytics, available via ``pip install candyeye[weights]``.
    """
    cache = cache_dir()
    cached = cache / f"{name}.pth"
    if cached.is_file():
        return cached

    pt_path = cache / f"{name}.pt"
    if not pt_path.is_file():
        _download_file(_WEIGHTS_URL.format(name=name), pt_path)

    try:
        import ultralytics  # noqa: F401  (import side-effect unpickles its classes)
    except ImportError as exc:
        raise RuntimeError(
            "converting the official checkpoint needs Ultralytics once:\n"
            "  pip install candyeye[weights]\n"
            "or set $CANDYEYE_WEIGHTS to an existing clean state_dict (.pth)."
        ) from exc

    import torch

    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].state_dict()
    if not isinstance(state_dict, dict) or not all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        raise TypeError(f"unexpected checkpoint layout in {pt_path}")
    torch.save(state_dict, cached)
    print(f"Saved clean weights -> {cached}")
    return cached


def default_weights_path(name: str = DEFAULT_WEIGHTS, *, download: bool = True) -> Path:
    """Path to the default pretrained weights.

    Resolution order: ``$CANDYEYE_WEIGHTS`` override, then the bundled
    ``assets/<name>.pth`` (kept in the repo for offline dev/tests), then the
    per-user cache.  If none exist and ``download`` is true, the official
    checkpoint is fetched and converted into the cache.
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

    if download:
        return download_default_weights(Path(filename).stem)
    raise FileNotFoundError(
        f"default weights {filename!r} not found (looked in {bundled} and {cached}). "
        "Set $CANDYEYE_WEIGHTS to an existing clean state_dict (.pth)."
    )


def default_config_path() -> Path:
    """Path to the bundled default architecture YAML."""
    return CONFIG_DIR / DEFAULT_CONFIG
