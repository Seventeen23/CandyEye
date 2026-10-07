"""Detection head: DFL projection + the decoupled, anchor-free Detect module.

This is the *parallel* part of the name. Each of the 3 feature maps (P3/P4/P5)
is fed through TWO independent sub-nets at once:

    cv2 -> box branch : 4 * reg_max channels (a distance distribution per side)
    cv3 -> class branch: nc channels (class logits)

and their outputs are concatenated per scale -> `no = nc + 4 * reg_max`.

One Detect instance serves all 3 scales; `ch` is the tuple of input channels.
- training mode: forward returns the raw per-scale maps  [(B, no, H, W), ...]
- eval mode:     `_inference` decodes boxes/strides -> (B, no, anchors)
No mask branch: bounding boxes only.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from core.modules.conv import Conv, DWConv


class DFL(nn.Module):
    """Distribution Focal Loss integral: reg_max bins -> 1 expected distance.

    A frozen (non-trainable) 1x1 conv whose weight is [0, 1, ..., reg_max-1].
    Applied to a softmax over the bins it computes the expectation of the
    predicted distance — turning a distribution into a single number.
    The 16 weights are the reason official params ≠ trainable params.
    """

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # batch, 4*reg_max, anchors
        return self.conv(
            x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        ).view(b, 4, a)


def make_anchors(feats: list[torch.Tensor], strides, grid_cell_offset: float = 0.5):
    """Anchor-center points (in cell units) + a matching stride tensor.

    For every scale we build the (x + 0.5, y + 0.5) grid and tag each point
    with its stride, then concatenate all 3 scales into one long list.
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:]
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor,
              xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """Decode (left, top, right, bottom) distances into boxes around anchors."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class Detect(nn.Module):
    """Decoupled anchor-free detection head (one instance for all 3 scales).

    Channel derivations (verified against official yolo11n):
        c2 = max(16, ch[0] // 4, 4 * reg_max)   # box-branch width  -> 64
        c3 = max(ch[0], min(nc, 100))           # class-branch width
    At n scale ch[0] = 64 (that's P3, the *smallest* map), so c2 = c3 = 64.
    """

    shape: tuple | None = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 20, reg_max: int = 16, ch: tuple = (64, 128, 256)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4  # outputs per anchor (84 at nc=20)
        self.stride = torch.zeros(self.nl)  # filled in by the model builder

        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))

        # box branch: plain 3x3 convs, ends in 4 * reg_max DFL channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * reg_max, 1))
            for x in ch
        )
        # class branch: depthwise-separable (cheap), ends in nc logits
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def forward(self, x: list[torch.Tensor], decode: bool | None = None):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if decode is False or (decode is None and self.training):
            return x
        return self._inference(x)

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode raw maps into (B, no, total_anchors): cxcywh + sigmoid cls."""
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.shape != shape:
            self.anchors, self.strides = (
                t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0),
                         xywh=True, dim=1) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Training-friendly bias init for box/cls heads (needs `stride` set).

        Box bias 1.0 makes initial distances sane; cls bias is set from the
        expected object density so early sigmoid outputs aren't all ~0.5.
        """
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
