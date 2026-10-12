"""Load torchvision MobileNetV3-Small ImageNet weights into a YAML-built model.

The ``mobilenetv3_small.yaml`` backbone re-creates the torchvision feature
extractor (stem + 9 inverted-residual blocks), so the ImageNet state_dict
remaps onto our layer indices nearly 1:1.  Small print:

- layers are indexed 0..10 with ``layer[i]`` = torchvision ``features[i]`` —
  except layer 4, which re-uses ``features[3]`` (the /8 stage has only two
  blocks; a stride-1 repeat sits at index 4 so the head's P3 reference lands
  on a /8 map).  That source is therefore loaded into *both* model layers
  3 and 4.  Forward parity is tested against a reference stack that repeats
  the same block, so every layer still matches its ImageNet counterpart
  exactly.
- the block sub-layout (``block.{0,1,2,3}``) and BatchNorm hyperparameters
  (``eps=1e-3, momentum=0.01``) match torchvision exactly.
- the head (``model.11..23``) has no ImageNet counterpart and stays
  random-initialised, exactly like the YOLO pretrained path.
"""
from __future__ import annotations

import torch
from torch import nn

from candyeye.core.convert_yolo11 import load_weights

# torchvision feature index -> model layer index (list form: order within the
# YAML, duplicates allowed).  features used: 0 (stem) .. 9.
MOBILE_V3_SMALL_FEATURE_MAP: list[tuple[int, int]] = [
    (0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10),
]


def imagenet_mobilenetv3_small_state() -> dict[str, torch.Tensor]:
    """Download (once) and return the torchvision ImageNet state_dict.

    Returns ``None`` if the weights are unavailable (e.g. offline first run);
    callers should skip ImageNet initialisation gracefully in that case.
    """
    try:
        from torchvision.models import mobilenet_v3_small
        from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        return dict(model.state_dict())
    except Exception as exc:  # pragma: no cover - network/optional dependency
        print(f"ImageNet weights unavailable for MobileNetV3-Small: {exc}")
        return None


def remap_imagenet_backbone(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate torchvision keys to ``model.<layer>.`` keys for our builder."""
    remapped: dict[str, torch.Tensor] = {}
    for feature, layer in MOBILE_V3_SMALL_FEATURE_MAP:
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


def uses_mobilenet_backbone(model: torch.nn.Module) -> bool:
    """True if *model* was built from the MobileNetV3-Small YAML backbone."""
    from candyeye.core.modules.mobilenet import MobileStem

    return any(isinstance(m, MobileStem) for m in model.modules())


def load_imagenet_backbone(
    model: torch.nn.Module, state: dict[str, torch.Tensor] | None = None,
    verbose: bool = False,
) -> dict[str, list]:
    """Load ImageNet backbone weights into a ``mobilenetv3_small`` model.

    Returns the same report dict as :func:`load_weights`.  Repeated calls are
    no-ops (the first load is recorded on the model).
    """
    if getattr(model, "_imagenet_backbone", False):
        return {"loaded": [], "skipped": [], "unexpected": []}
    if state is None:
        state = imagenet_mobilenetv3_small_state()
    report = load_weights(model, remap_imagenet_backbone(state), verbose=verbose)
    model._imagenet_backbone = True
    return report


def backbone_parity(
    model: torch.nn.Module, state: dict[str, torch.Tensor] | None = None,
    img: int = 128, atol: float = 1e-5,
) -> dict[int, float]:
    """Feed one random image and compare each backbone layer to a torchvision
    reference stack; returns {layer_index: max_abs_diff}.

    The reference stack repeats ``features[3]`` at position 3 (mirroring our
    device layer 4 = a stride-1 repeat of layer 3), so every layer 0..10 maps
    1:1 onto an identical ImageNet block and exact parity is expected.
    """
    if state is None:
        state = imagenet_mobilenetv3_small_state()
    from torchvision.models import mobilenet_v3_small

    tv = mobilenet_v3_small(weights=None)
    tv.load_state_dict(state)
    features = tv.features
    reference = nn.Sequential(*(features[f] for f, _ in MOBILE_V3_SMALL_FEATURE_MAP))

    x = torch.randn(1, 3, img, img)
    with torch.no_grad():
        was_training = model.training
        model.eval()
        tv.eval()
        our_out = x
        diffs: dict[int, float] = {}
        for i in range(0, 11):
            our_out = model.model[i](our_out)  # backbone is a sequential chain
            diff = (our_out - reference[: i + 1](x)).abs().max().item()
            assert diff <= atol, (i, diff)
            diffs[i] = diff
        if was_training:
            model.train()
    return diffs