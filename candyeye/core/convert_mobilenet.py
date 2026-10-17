"""Load torchvision MobileNetV3 ImageNet weights into a YAML-built model.

The ``mobilenetv3_*`` configs re-create the torchvision feature extractor so
the ImageNet state_dict remaps onto our layer indices nearly 1:1.  Two variants
are supported, selected by the config's ``mobilenet_variant`` key:

- ``small`` (``mobilenetv3_small.yaml``): stem + 9 inverted-residual blocks,
  layers 0..10 = torchvision ``features[0..9]`` — except layer 4, which re-uses
  ``features[3]`` (the /8 stage has only two blocks; a stride-1 repeat sits at
  index 4 so the head's P3 reference lands on a /8 map).  That source is loaded
  into *both* model layers 3 and 4.
- ``large`` (``mobilenetv3_large.yaml`` / ``_p2``): the full 16-block backbone,
  layers 0..15 = ``features[0..15]`` 1:1 (MobileNetV3-Large already has three
  /8 blocks and three /32 blocks, so no duplication is needed).

Small print that holds for both variants:

- the block sub-layout (``block.{0,1,2,3}``) and BatchNorm hyperparameters
  (``eps=1e-3, momentum=0.01``) match torchvision exactly.
- the head (and any SPPF/C2PSA layers) have no ImageNet counterpart and stay
  random-initialised, exactly like the YOLO pretrained path.
"""
from __future__ import annotations

import torch
from torch import nn

from candyeye.core.convert_yolo11 import load_weights

# torchvision feature index -> model layer index (list form: order within the
# YAML, duplicates allowed).
MOBILE_V3_SMALL_FEATURE_MAP: list[tuple[int, int]] = [
    (0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10),
]
MOBILE_V3_LARGE_FEATURE_MAP: list[tuple[int, int]] = [(i, i) for i in range(16)]

_VARIANT_MAPS: dict[str, list[tuple[int, int]]] = {
    "small": MOBILE_V3_SMALL_FEATURE_MAP,
    "large": MOBILE_V3_LARGE_FEATURE_MAP,
}


def _variant_map(variant: str) -> list[tuple[int, int]]:
    if variant not in _VARIANT_MAPS:
        raise ValueError(
            f"unknown MobileNetV3 variant {variant!r} (have {sorted(_VARIANT_MAPS)})")
    return _VARIANT_MAPS[variant]


# torchvision builder + weights enum per variant.
_TORCHVISION = {
    "small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
}


def imagenet_mobilenetv3_state(variant: str = "small") -> dict[str, torch.Tensor] | None:
    """Download (once) and return the torchvision ImageNet state_dict.

    Returns ``None`` if the weights are unavailable (e.g. offline first run);
    callers should skip ImageNet initialisation gracefully in that case.
    """
    fn_name, weights_name = _TORCHVISION[variant]
    try:
        import torchvision.models as tv_models

        builder = getattr(tv_models, fn_name)
        weights = getattr(tv_models.mobilenetv3, weights_name).IMAGENET1K_V1
        model = builder(weights=weights)
        return dict(model.state_dict())
    except Exception as exc:  # pragma: no cover - network/optional dependency
        print(f"ImageNet weights unavailable for MobileNetV3-{variant.title()}: {exc}")
        return None


def imagenet_mobilenetv3_small_state() -> dict[str, torch.Tensor] | None:
    """Backwards-compatible alias for the small variant."""
    return imagenet_mobilenetv3_state("small")


def imagenet_mobilenetv3_large_state() -> dict[str, torch.Tensor] | None:
    """Alias for the large variant's ImageNet state_dict."""
    return imagenet_mobilenetv3_state("large")


def remap_imagenet_backbone(
    state: dict[str, torch.Tensor], variant: str = "small",
) -> dict[str, torch.Tensor]:
    """Translate torchvision keys to ``model.<layer>.`` keys for our builder."""
    remapped: dict[str, torch.Tensor] = {}
    for feature, layer in _variant_map(variant):
        prefix = f"features.{feature}."
        target = f"model.{layer}."
        for key, value in state.items():
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if feature == 0:
                # stem: features.0.0.{conv} / features.0.1.{bn stats}
                rest = {"0": "conv", "1": "bn"}[rest.split(".")[0]] + rest[rest.index("."):]
            remapped[target + rest] = value
    return remapped


def mobilenet_variant(model: torch.nn.Module) -> str | None:
    """Return ``"small"``/``"large"`` if *model* has a MobileNetV3 backbone."""
    from candyeye.core.modules.mobilenet import MobileStem

    if not any(isinstance(m, MobileStem) for m in model.modules()):
        return None
    return getattr(model, "_mobilenet_variant", None) or "small"


def uses_mobilenet_backbone(model: torch.nn.Module) -> bool:
    """True if *model* was built from a MobileNetV3 YAML backbone."""
    return mobilenet_variant(model) is not None


def load_imagenet_backbone(
    model: torch.nn.Module, state: dict[str, torch.Tensor] | None = None,
    variant: str | None = None, verbose: bool = False,
) -> dict[str, list]:
    """Load ImageNet backbone weights into a ``mobilenetv3_*`` model.

    Returns the same report dict as :func:`load_weights`.  Repeated calls are
    no-ops (the first load is recorded on the model).
    """
    if getattr(model, "_imagenet_backbone", False):
        return {"loaded": [], "skipped": [], "unexpected": []}
    if variant is None:
        variant = mobilenet_variant(model) or "small"
    if state is None:
        state = imagenet_mobilenetv3_state(variant)
    if state is None:
        print("ImageNet backbone weights unavailable; skipping initialisation")
        return {"loaded": [], "skipped": [], "unexpected": []}
    report = load_weights(model, remap_imagenet_backbone(state, variant), verbose=verbose)
    model._imagenet_backbone = True
    return report


def backbone_parity(
    model: torch.nn.Module, state: dict[str, torch.Tensor] | None = None,
    variant: str | None = None, img: int = 128, atol: float = 1e-5,
) -> dict[int, float]:
    """Feed one random image and compare each backbone layer to a torchvision
    reference stack; returns {layer_index: max_abs_diff}.

    The reference stack mirrors the config's layer order, so every layer maps
    1:1 onto an identical ImageNet block and exact parity is expected.
    """
    if variant is None:
        variant = mobilenet_variant(model) or "small"
    if state is None:
        state = imagenet_mobilenetv3_state(variant)
    fn_name, _ = _TORCHVISION[variant]
    import torchvision.models as tv_models

    tv = getattr(tv_models, fn_name)(weights=None)
    tv.load_state_dict(state)
    feature_map = _variant_map(variant)
    features = tv.features
    reference = nn.Sequential(*(features[f] for f, _ in feature_map))

    x = torch.randn(1, 3, img, img)
    with torch.no_grad():
        was_training = model.training
        model.eval()
        tv.eval()
        our_out = x
        diffs: dict[int, float] = {}
        for i in range(len(feature_map)):
            our_out = model.model[i](our_out)  # backbone is a sequential chain
            diff = (our_out - reference[: i + 1](x)).abs().max().item()
            assert diff <= atol, (i, diff)
            diffs[i] = diff
        if was_training:
            model.train()
    return diffs
