"""Convolution building block.

The smallest reusable unit of the YOLO-style architecture: a conv, its
batch-norm, and an activation, bundled as one nn.Module.

Connection to the other folder: we import `autopad` from
core/functions/layer_utils.py — stateless math lives in functions/,
stateful layers (anything with weights) live in modules/.
"""
from __future__ import annotations

import torch
from torch import nn

from core.functions.layer_utils import autopad


class Conv(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU, with automatic "same" padding.

    Can look up YOLOv5's `Conv` in models/common.py for a more detailed explanation of
    the design choices. This is a simplified version, with only the essential
    functionality needed for CandyEye.

    YUP v5 lol despite mimicking v11 architecture without its head, v5 is easier to understand
    Args Babyyy:
        c1: input channels
        c2: output channels
        k:  kernel size
        s:  stride
        p:  explicit padding (None = auto via autopad)
        g:  groups (g>1 gives depthwise conv, e.g. g == c1 == c2)
        d:  dilation
        act: True -> SiLU (default), False -> Identity (linear output),
             or pass an nn.Module directly (e.g. nn.LeakyReLU) to override.

    Layer order (conv -> bn -> act) matches YOLO exactly, which matters in
    Phase 2: state_dict keys then line up 1:1 with the official weights,
    so `model.0.conv.weight` in the checkpoint lands on `self.conv.weight`.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ):
        super().__init__()
        # bias=False because BatchNorm2d already has a learnable bias —
        # a second one would be redundant (standard practice).
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        # The three-way ternary: bool True -> SiLU, bool False -> Identity,
        # otherwise assume the caller passed an nn.Module instance.
        self.act = (
            nn.SiLU()
            if act is True
            else (nn.Identity() if act is False else act)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard NCHW tensor in -> NCHW tensor out."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Same result as forward(), but WITHOUT bn/act — only used after
        export/fusion has folded BN weights into the conv (Phase 6, ONNX
        export). Writing it now keeps the interface stable.
        """
        return self.act(self.conv(x))
