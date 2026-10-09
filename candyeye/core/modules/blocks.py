"""Reusable convolution, CSP, and attention blocks used by CandyEye.

Bottleneck / C3k / C3k2       — CSP bottleneck stacks (the network body)
SPPF                          — spatial pyramid pooling (fast)
Attention / PSABlock / C2PSA  — position-sensitive attention (backbone tail)

Attribute names (cv1, cv2, cv3, m, ...) intentionally mirror ultralytics so
state_dict keys match the official checkpoint 1:1 (Phase 2 weight load).
"""
from __future__ import annotations

import torch
from torch import nn

from candyeye.core.modules.conv import Conv


class Bottleneck(nn.Module):
    """Standard residual bottleneck: cv1 squeeze -> cv2 expand, optional skip.

    Args:
        c1: input channels
        c2: output channels
        shortcut: add a residual connection if requested AND c1 == c2
        g: groups for the second conv
        k: kernel sizes (k[0] for cv1, k[1] for cv2)
        e: expansion factor -> hidden channels = int(c2 * e)
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 k: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden (bottleneck) channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    """CSP bottleneck with 3 convs; inner Bottlenecks use k=(k,k) and e=1.0.

    cv1 / cv2 are the two 1x1 CSP paths, cv3 merges them. Inner e=1.0 keeps
    hidden == c_, so the inner skip connection stays valid.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True,
                 g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    """C2f-style CSP stack whose n inner blocks are Bottlenecks (or C3k).

    cv1 splits the input into two halves of `c` channels; one half is chained
    through the n blocks; all (2 + n) maps are concatenated and merged by cv2.

    NOTE: inner blocks are built with their own default e=0.5 on `c`.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False,
                 e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c, 1, 1)
        self.cv2 = Conv((2 + n) * c, c2, 1)
        self.m = nn.ModuleList(
            C3k(c, c, 2, shortcut, g) if c3k else Bottleneck(c, c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast: pool 3x at pad=k//2, concat, merge.

    Max-pooling with a growing receptive field fuses 3 pyramid levels at the
    same resolution, so it captures multi-scale context without changing H×W.
    """

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3,
                 shortcut: bool = False):
        super().__init__()
        self.c_ = c1 // 2  # hidden channels (real ultralytics SPPF: cv1 uses SiLU)
        self.cv1 = Conv(c1, self.c_, 1, 1)
        self.cv2 = Conv(self.c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class Attention(nn.Module):
    """Multi-head self-attention over spatial positions.

    qkv / proj / pe are conv layers (1x1, 1x1, depthwise 3x3), not Linear, so
    everything stays NCHW and exports cleanly. `pe` adds a positional encoding
    to the values. `key_dim = head_dim * attn_ratio`.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2  # q + k + v packed into one conv
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # depthwise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).view(B, self.num_heads,
                               self.key_dim * 2 + self.head_dim, N)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) \
            + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    """Attention + feed-forward, each with its own residual connection."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4,
                 shortcut: bool = True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """Split -> PSABlock stack on one half -> concat -> merge. Requires c1 == c2."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2, "C2PSA needs c1 == c2 (residual shape)"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
              for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
