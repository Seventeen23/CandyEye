"""MobileNetV3-Small backbone with a feature pyramid and detection head."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from core.modules.blocks import C3k2
from core.modules.conv import Conv
from core.modules.detect import Detect


class MobileNetV3SmallDetector(nn.Module):
    """Small VOC detector using MobileNetV3-Small features at strides 8/16/32.

    ImageNet initialization is optional because torchvision may need to
    download the checkpoint when it is not in the local torch cache.
    """
    def __init__(self, nc: int = 20, img_size: int = 128,
                 pretrained: bool = False):
        super().__init__()
        if img_size % 32:
            raise ValueError(f"img_size must be divisible by 32, got {img_size}")
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.backbone = backbone.features

        # MobileNet feature indices 3, 8, and 12 are at /8, /16, and /32.
        self.proj3 = Conv(24, 64, 1)
        self.proj4 = Conv(48, 128, 1)
        self.proj5 = Conv(576, 256, 1)
        self.fuse4 = C3k2(384, 128, n=1, c3k=False)
        self.fuse3 = C3k2(192, 64, n=1, c3k=False)
        self.down4 = Conv(64, 64, 3, 2)
        self.pan4 = C3k2(192, 128, n=1, c3k=False)
        self.down5 = Conv(128, 128, 3, 2)
        self.pan5 = C3k2(384, 256, n=1, c3k=False)

        detect = Detect(nc=nc, ch=(64, 128, 256))
        detect.stride = torch.tensor([8., 16., 32.])
        detect.bias_init()
        self.model = nn.ModuleList([detect])
        self.nc = nc
        self.img_size = img_size
        self.register_buffer("stride", torch.tensor([8., 16., 32.]))

    def forward(self, x: torch.Tensor, *, decode: bool | None = None):
        p3 = p4 = None
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 3:
                p3 = x
            elif i == 8:
                p4 = x
            elif i == 12:
                p5 = x

        p3, p4, p5 = self.proj3(p3), self.proj4(p4), self.proj5(p5)
        p4_td = self.fuse4(torch.cat((F.interpolate(p5, scale_factor=2,
                                                   mode="nearest"), p4), 1))
        p3_out = self.fuse3(torch.cat((F.interpolate(p4_td, scale_factor=2,
                                                    mode="nearest"), p3), 1))
        p4_out = self.pan4(torch.cat((self.down4(p3_out), p4_td), 1))
        p5_out = self.pan5(torch.cat((self.down5(p4_out), p5), 1))
        return self.model[-1]([p3_out, p4_out, p5_out], decode=decode)
