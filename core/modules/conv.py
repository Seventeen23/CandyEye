"""Convolution building block.

The smallest reusable unit in CandyEye: a conv, its
batch-norm, and an activation, bundled as one nn.Module.

Connection to the other folder: we import `autopad` from
core/functions/layer_utils.py — stateless math lives in functions/,
stateful layers (anything with weights) live in modules/.
"""
from __future__ import annotations

import math

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

    Layer order (conv -> bn -> act) matches the source architecture exactly, which matters in
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
        # Official ultralytics BatchNorm uses eps=1e-3 (not the torch default
        # 1e-5); eval inference must match, or every map drifts a little.
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
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

    def fuse(self) -> "Conv":
        """Fold BatchNorm into the conv: gamma/sqrt(var+eps) scales the weights,
        and the mean term becomes a bias. This is exactly the math the official
        ONNX export bakes in, so after `fuse()` our state_dict keys line up with
        the ONNX initializers (`model.0.conv.weight` + `.bias`, no `.bn.*`).
        """
        fused = torch.nn.utils.fusion.fuse_conv_bn_eval(self.conv, self.bn)
        self.conv = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            self.conv.kernel_size,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
            bias=True,
        )
        self.conv.weight.data.copy_(fused.weight.data)
        self.conv.bias.data.copy_(fused.bias.data)
        self.bn = nn.Identity()  # keep the attribute (loop-friendly), no params
        return self

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Same result as forward(), but WITHOUT bn/act — only used after
        export/fusion has folded BN weights into the conv (Phase 6, ONNX
        export). Writing it now keeps the interface stable.
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depthwise convolution: `groups == gcd(c1, c2)`, so for c1 == c2 each
    channel is filtered on its own. This is the MobileNet efficiency trick —
    a k×k depthwise conv costs ~1/k² of a normal one. Used by the Detect
    class branch (cv3) to keep the head cheap.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, d: int = 1,
                 act: bool | nn.Module = True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
