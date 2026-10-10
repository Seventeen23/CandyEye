"""Adaptive cross-scale feature exchange neck.

Treats P3/P4/P5 as interacting feature groups: neighboring scales exchange a
lightweight message instead of relying only on the fixed top-down/bottom-up
FPN path. How much of each neighbor's message is admitted is controlled by a
gate, available in three modes:

    none     fixed 0.5 mix (exchange with no gating; ablation control)
    static   sigmoid(learnable per-channel weight + bias)
    dynamic  sigmoid(1x1(ReLU(1x1(GAP([self, neighbor])))))  -- content-conditioned

The ``dynamic`` gate is the intended contribution: unlike BiFPN's fixed learned
scalar weights, its weights are computed at runtime from the features, so the
amount of cross-scale exchange adapts per input.

The module is deliberately cheap (bottlenecked depthwise messages, one iteration
by default) so it can run inside the CPU-only, 128px target envelope. Gates are
initialized near-closed so adding the neck starts close to identity and does not
destabilize a pretrained baseline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from candyeye.core.modules.conv import Conv, DWConv

_GATE_MODES = ("none", "static", "dynamic")


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Match a feature map to ``size`` (H, W): avg-pool down, nearest up."""
    if x.shape[-2:] == size:
        return x
    if x.shape[-1] > size[1]:
        return F.adaptive_avg_pool2d(x, size)
    return F.interpolate(x, size=size, mode="nearest")


class _Message(nn.Module):
    """Neighbor feature -> a message tensor at the target level's channels."""

    def __init__(self, c_in: int, c_out: int, width: int):
        super().__init__()
        self.proj = Conv(c_in, width, 1)
        self.dw = DWConv(width, width, 3)
        self.out = Conv(width, c_out, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.dw(self.proj(x)))


class _Gate(nn.Module):
    """Per-channel gate controlling how much of a message is admitted."""

    def __init__(self, c_self: int, c_other: int, mode: str,
                 bias_init: float = -4.0):
        super().__init__()
        if mode not in _GATE_MODES:
            raise ValueError(f"unknown exchange gate {mode!r} (have {_GATE_MODES})")
        self.mode = mode
        if mode == "static":
            self.weight = nn.Parameter(torch.zeros(1, c_self, 1, 1))
            self.bias = nn.Parameter(torch.full((1, c_self, 1, 1), bias_init))
        elif mode == "dynamic":
            hidden = max((c_self + c_other) // 4, 8)
            self.fc1 = nn.Conv2d(c_self + c_other, hidden, 1)
            self.fc2 = nn.Conv2d(hidden, c_self, 1)
            nn.init.zeros_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, bias_init)

    def forward(self, x_self: torch.Tensor, x_other: torch.Tensor):
        if self.mode == "none":
            return 0.5
        if self.mode == "static":
            return torch.sigmoid(self.weight + self.bias)
        pooled = F.adaptive_avg_pool2d(torch.cat((x_self, x_other), 1), 1)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))

    def close(self) -> None:
        """Force the gate ~0 (identity) — used for baseline-equivalence tests."""
        if self.mode == "static":
            with torch.no_grad():
                self.bias.fill_(-30.0)
        elif self.mode == "dynamic":
            with torch.no_grad():
                self.fc2.weight.zero_()
                self.fc2.bias.fill_(-30.0)


class ScaleExchange(nn.Module):
    """Bidirectional, gated message passing between adjacent pyramid levels.

    Input and output are lists of feature maps (typically 3 or 4 levels)
    with identical shapes/channels as produced by the neck, so the module drops in
    between an existing neck and the detection head.
    """

    def __init__(self, *args, gate: str | None = None, iters: int | None = None,
                 width: int | None = None, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected ScaleExchange arguments: {sorted(kwargs)}")
        # Accept both keyword form (MobileNet: channels..., gate=, iters=) and
        # the YAML-builder positional form (c1, c2, c3, gate, iters) for any
        # number of pyramid levels. A bare trailing int is a channel count, so
        # iters is only read positionally right after a positional gate.
        values = list(args)
        if gate is None and values and isinstance(values[-1], str):
            gate = values.pop(-1)
            if (iters is None and values
                    and isinstance(values[-1], int) and not isinstance(values[-1], bool)):
                iters = values.pop(-1)
        if len(values) < 2:
            raise ValueError(
                f"ScaleExchange expects at least 2 channel counts, got {len(values)}")
        self.channels = tuple(int(c) for c in values)
        gate = "none" if gate is None else str(gate)
        if gate not in _GATE_MODES:
            raise ValueError(f"unknown exchange gate {gate!r} (have {_GATE_MODES})")
        iters = 1 if iters is None else int(iters)
        if iters < 1:
            raise ValueError(f"iters must be >= 1, got {iters}")
        self.gate = gate
        self.iters = iters

        self.pairs = [(i, i + 1) for i in range(len(self.channels) - 1)]
        self.message_down = nn.ModuleList()
        self.message_up = nn.ModuleList()
        self.gate_down = nn.ModuleList()
        self.gate_up = nn.ModuleList()
        for a, b in self.pairs:
            ca, cb = self.channels[a], self.channels[b]
            wa = width or max(ca // 2, 16)
            wb = width or max(cb // 2, 16)
            # a <- b (upsample the coarser neighbor)
            self.message_down.append(_Message(cb, ca, wa))
            self.gate_down.append(_Gate(ca, cb, gate))
            # b <- a (downsample the finer neighbor)
            self.message_up.append(_Message(ca, cb, wb))
            self.gate_up.append(_Gate(cb, ca, gate))

    def _step(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        out = list(feats)
        for index, (a, b) in enumerate(self.pairs):
            xa, xb = feats[a], feats[b]
            xb_to_a = _resize(xb, xa.shape[-2:])
            xa_to_b = _resize(xa, xb.shape[-2:])
            out[a] = out[a] + self.gate_down[index](xa, xb_to_a) \
                * self.message_down[index](xb_to_a)
            out[b] = out[b] + self.gate_up[index](xb, xa_to_b) \
                * self.message_up[index](xa_to_b)
        return out

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) != len(self.channels):
            raise ValueError(f"ScaleExchange expects {len(self.channels)} feature maps, got {len(feats)}")
        out = list(feats)
        for _ in range(self.iters):
            out = self._step(out)
        return out

    def close_gates(self) -> "ScaleExchange":
        """Close every gate so the neck is (near) identity — for tests."""
        for gate in [*self.gate_down, *self.gate_up]:
            gate.close()
        return self
