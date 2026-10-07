"""YAML-driven model builder + forward graph for the YOLO11n detector.

The YAML in ``configs/yolo11.yaml`` is the single source of truth for the
architecture. The builder walks it exactly like the official implementation:

- each entry is ``[from, repeats, module, args]``
- ``from`` < 0  -> offset into the built list (-1 = previous layer)
- ``from`` >= 0 -> absolute layer index; a list = several layers (Concat / Detect)
- output channels are tracked alongside the built layers
- width : ``c2 = make_divisible(min(c2, max_channels) * width_mult, 8)``
- depth : ``repeats = max(round(repeats * depth_mult), 1)``

The result is an ``nn.Sequential`` indexed exactly like the official yolo11n
(0..23), so state_dict keys land at ``model.0.conv.weight``,
``model.23.cv3.1.0.1.conv.weight``, ... and Phase 2 can load the official
``yolo11n.pt`` 1:1 after re-initialising the head's class branch for nc=20.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from core.functions.layer_utils import make_divisible
from core.modules.blocks import Attention, Bottleneck, C2PSA, C3k, C3k2, PSABlock, SPPF
from core.modules.conv import Conv
from core.modules.detect import Detect


class Concat(nn.Module):
    """Concatenate several tensors along a dimension (neck merges)."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


_MODULE_MAP = {
    "Conv": Conv,
    "Bottleneck": Bottleneck,
    "C3k": C3k,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "Attention": Attention,
    "PSABlock": PSABlock,
    "C2PSA": C2PSA,
    "Concat": Concat,
    "Detect": Detect,
    "nn.Upsample": nn.Upsample,
    "nn.Conv2d": nn.Conv2d,
}

_CHANNEL_MODULES = {"Conv", "C3k2", "C2PSA", "SPPF", "C3k", "Bottleneck", "Attention", "PSABlock"}

# depth_mult, width_mult, max_channels
_SCALES = {"n": (0.50, 0.25, 1024)}


def parse_model(data: dict) -> nn.Sequential:
    """Expand the yaml layer table into an indexed nn.Sequential."""
    nc = data["nc"]
    scale = data.get("scale", "n")
    if scale not in _SCALES:
        raise ValueError(f"scale {scale!r} not supported (have {sorted(_SCALES)})")
    depth, width, max_channels = _SCALES[scale]

    x = data["backbone"] + data["head"]
    layers, ch = [], [3]  # ch[0] = image channels feeding layer 0

    for i, (f, n, m, args) in enumerate(x):
        args = [nc if a == "nc" else a for a in args]
        n = max(round(n * depth), 1)

        if m in _CHANNEL_MODULES:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m == "C3k2":
                args.insert(2, n)  # repeats slot inside the block
                n = 1
        elif m == "nn.Upsample":
            c2 = ch[f]
        elif m == "Concat":
            c2 = sum(ch[x] for x in f)
        elif m == "Detect":
            args.append([ch[x] for x in f])
            c2 = nc  # not referenced downstream; placeholder
        else:
            raise ValueError(f"unhandled module {m!r}")

        module_cls = _MODULE_MAP[m]
        if m == "Detect":
            module = module_cls(nc=args[0], ch=args[1])
        elif n > 1:
            module = nn.Sequential(*(copy.deepcopy(module_cls(*args)) for _ in range(n)))
        else:
            module = module_cls(*args)
        module.type = m
        module.i = i  # absolute index
        module.f = f  # from-list (int or list of ints)
        module.n = n  # repeats used

        layers.append(module)
        ch.append(c2)
    return nn.Sequential(*layers)


class YOLO(nn.Module):
    """The CandyEye detector: nn.Sequential graph + Detect head with strides."""

    def __init__(self, cfg: str = "configs/yolo11.yaml", nc: int | None = None, img_size: int = 128):
        super().__init__()
        as_dict = cfg if isinstance(cfg, dict) else None
        data = as_dict or self._load_yaml(cfg)
        if nc is not None:
            data["nc"] = nc
        self.model = parse_model(data)
        self.yaml = cfg
        if img_size % 32:
            raise ValueError(f"img_size must be divisible by 32, got {img_size}")
        self.stride = self._detect_stride(img_size)
        self.model[-1].stride = self.stride

    @staticmethod
    def _load_yaml(cfg: str) -> dict:
        import yaml

        with open(cfg) as fh:
            data = yaml.safe_load(fh)
        YOLO._validate_yaml(data)
        return data

    @staticmethod
    def _validate_yaml(data: dict) -> None:
        path = Path(data.get("path", "?"))
        for section in ("backbone", "head"):
            if section not in data:
                raise ValueError(f"yaml {path} missing section {section!r}")
            for entry in data[section]:
                if not (isinstance(entry, list) and len(entry) == 4):
                    raise ValueError(f"bad {section} entry {entry!r} (want [from, repeats, module, args])")
        if "nc" not in data:
            raise ValueError(f"yaml {path} missing nc")

    def _detect_stride(self, img_size: int) -> torch.Tensor:
        """Feed a blank image and infer P3/P4/P5 strides from output sizes."""
        with torch.no_grad():
            feats = self.model(torch.zeros(1, 3, img_size, img_size))
        if not isinstance(feats, (list, tuple)):
            return torch.ones(1)
        return torch.tensor([img_size / f.shape[-2] for f in feats])

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run the graph; returns the Detect training output (list of maps)."""
        y = []  # layer-output history, index == absolute layer index
        for m in self.model:
            f = m.f
            if isinstance(f, int):
                xi = x if f == -1 else y[f]
            else:
                xi = [x if j == -1 else y[j] for j in f]
            x = m(xi)
            y.append(x)
        return x