"""MobileNetV3-Small stem + inverted-residual blocks for the YAML builder.

These mirror ``torchvision.models.mobilenetv3`` (``Conv2dNormActivation`` +
``InvertedResidual`` + ``SElayer``) so that ImageNet weights can be remapped
1:1.  The key layout inside ``block``  is intentionally identical to
torchvision's:

- ``block.0`` expand 1x1 (+ BN + act), *omitted* when ``exp == c1``
- ``block.1`` depthwise ``k``x``k`` (groups = ``exp``) (+ BN + act)
- ``block.2`` SE layer (only when ``se``) — keys ``block.2.fc1/fc2``
- ``block.N`` project 1x1 (linear: BN only, no activation)

BatchNorm uses torchvision's ``eps=1e-3, momentum=0.01`` (NOT the YOLO
``Conv`` defaults) so the loaded ImageNet running stats keep their meaning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tv_make_divisible(v: float, divisor: int = 8) -> int:
    """torchvision's ``_make_divisible`` (adds the <90% correction step).

    The shared CandyEye ``make_divisible`` deliberately rounds to nearest for
    YOLO width scaling; torchvision instead bumps the result up when rounding
    would shrink a channel count by more than 10%.  The two differ for values
    like ``72 * 0.25 = 18`` (torchvision -> 24, nearest -> 16), which the SE
    squeeze widths in MobileNetV3-Large depend on.
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv_block(c1: int, c2: int, k: int, s: int, groups: int = 1,
                act=None) -> nn.Sequential:
    """Conv2dNormActivation equivalent: conv -> BN(eps=1e-3, mom=0.01) -> act."""
    layers = [nn.Conv2d(c1, c2, k, s, k // 2, groups=groups, bias=False),
              nn.BatchNorm2d(c2, eps=1e-3, momentum=0.01)]
    if act is not None:
        layers.append(act())
    return nn.Sequential(*layers)


class SE(nn.Module):
    """Squeeze-and-Excite (torchvision ``SElayer``, hardsigmoid scaling)."""

    def __init__(self, channels: int, squeeze: int):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, squeeze, 1, bias=True)
        self.fc2 = nn.Conv2d(squeeze, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = F.relu(self.fc1(scale))
        scale = F.hardsigmoid(self.fc2(scale))
        return x * scale


class MobileStem(nn.Module):
    """First MobileNetV3 conv (k=3, stride 2, hardswish) — keys ``conv/bn``."""

    def __init__(self, c1: int, c2: int = 16, k: int = 3, s: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.01)
        self.act = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MBConv(nn.Module):
    """MobileNetV3 inverted-residual block, torchvision-compatible.

    Args (from YAML): ``[c2, k, s, exp, se, act]`` where ``se`` is a bool and
    ``act`` is ``"RE"`` (ReLU) or ``"HS"`` (hardswish).  ``c1`` is injected by
    the YAML builder.  Channel counts are explicit (no width multiplier).
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1,
                 exp: int | None = None, se: bool = False, act: str = "RE"):
        super().__init__()
        if not (1 <= s <= 2):
            raise ValueError(f"MBConv stride must be 1 or 2, got {s}")
        self.use_res_connect = s == 1 and c1 == c2
        exp = c1 if exp is None else int(exp)
        act_layer = nn.Hardswish if str(act).upper().startswith("HS") else nn.ReLU

        block: list[nn.Module] = []
        if exp != c1:
            block.append(_conv_block(c1, exp, 1, 1, act=act_layer))          # expand
        block.append(_conv_block(exp, exp, k, s, groups=exp, act=act_layer))  # depthwise
        if se:
            block.append(SE(exp, _tv_make_divisible(exp * 0.25, 8)))
        block.append(_conv_block(exp, c2, 1, 1, act=None))                    # project
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_res_connect else out