"""Pure helpers shared by the model modules.

These are plain functions: no nn.Module, no weights, nothing to train.
Rule: modules may import functions, never the other way around — that keeps
this file trivially testable and free of circular imports.
"""
from __future__ import annotations


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Compute padding that keeps the feature map the SAME size ("same"
    padding) for a conv with kernel size k and dilation d.

    Why: Conv2d with stride 1 and padding=k//2 outputs the same H×W as the
    input, so the block author never has to do this arithmetic by hand.

    If p is given explicitly we just use it (lets callers opt out), which is
    what ultralytics' autopad does too.
    """
    if d > 1:
        # A dilated kernel covers more input pixels than its weight count
        # suggests: effective kernel = d * (k - 1) + 1
        # e.g. k=3, d=2 -> covers 5 pixels -> needs padding 2 to stay same-size
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def make_divisible(x: float, divisor: int = 8) -> int:
    """Round a channel count to the nearest multiple of `divisor`.

    Why: many backends (and our NCHW memory layout) run faster when channel
    counts are multiples of 8 — vectorized SIMD loads are 8-wide. Scaling a
    width multiplier (0.25, 0.5, ...) onto layer channels produces ugly
    numbers like 47; this rounds them to 48.

    Rounds to NEAREST (not always up) so width-scaling stays faithful to the
    original ratio, with a floor of one full divisor so we never go below it.
    """
    return max(divisor, int(x + divisor / 2) // divisor * divisor)
