"""Load the official YOLO11n weights (nc=80) into our model.

Reads the *clean* state_dict produced by ``scripts/bootstrap_weights.py``
(``weights/yolo11n.pth``, plain tensors only — no ultralytics dependency).
The loader is shape-checked: any tensor whose shape doesn't match is skipped
and reported, everything else is copied 1:1 (parameters *and* buffers such as
BatchNorm running statistics, so eval-mode inference is faithful).

Two load scenarios:
- ``YOLO(yaml, nc=80)``   — exact full load: 0 missing / 0 unexpected.
- ``YOLO(yaml, nc=20)``   — our VOC training target: the whole class branch
  ``model.23.cv3.*`` is nc/c3-dependent, so exactly ``EXPECTED_NC20_SKIPPED``
  (51 tensors, generated below from the known structure) are skipped and stay
  random-initialised.  The box branch (``cv2`` + ``dfl``) still loads 1:1, so
  box proposals remain the official ones.
"""
from __future__ import annotations

import torch
import onnx

WEIGHTS = "weights/yolo11n.pth"
ONNX_PATH = "tmp/yolo11n.onnx"  # official fp32 export (BN fused)

# The official checkpoint was trained with nc=80 (COCO), our training model
# uses nc=20 (VOC).  In Detect, c3 = max(ch[0], min(nc, 100)) pins the class
# branch's hidden width (80 -> 64) and its final Conv2d in/out (-> 20).
# Per scale that unmatchable part is:
#   cv3.<s>.0.1  Conv(x -> c3)      bn.weight/bias/mean/var + conv.weight
#   cv3.<s>.1.0  DWConv(c3 -> c3)    bn.*                          + conv.weight
#   cv3.<s>.1.1  Conv(c3 -> c3)      bn.*                          + conv.weight
#   cv3.<s>.2    Conv2d(c3 -> nc)    weight + bias
# = 17 tensors/scale x 3 scales = 51.  Everything else loads unchanged.
EXPECTED_NC20_SKIPPED = frozenset(
    [
        *(
            f"model.23.cv3.{s}.{a}.{b}.{comp}"
            for s in range(3)
            for (a, b) in ((0, 1), (1, 0), (1, 1))
            for comp in ("bn.bias", "bn.running_mean", "bn.running_var",
                         "bn.weight", "conv.weight")
        ),
        *(
            f"model.23.cv3.{s}.2.{comp}"
            for s in range(3)
            for comp in ("weight", "bias")
        ),
    ]
)


def load_official_state_dict(path: str = WEIGHTS) -> dict[str, torch.Tensor]:
    """Read the clean .pth (tensors only -> safe with torch.load defaults)."""
    sd = torch.load(path, map_location="cpu")
    if not (isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values())):
        raise TypeError(f"{path} is not a clean state_dict of tensors "
                        f"(re-run scripts/bootstrap_weights.py)")
    return sd


def load_weights(
    model: torch.nn.Module,
    official: dict[str, torch.Tensor],
    verbose: bool = False,
) -> dict[str, list]:
    """Shape-checked copy of *official* into *model*; nothing in-place original.

    Returns {"loaded": [...], "skipped": [(key, our_shape, off_shape)],
             "unexpected": [(key, off_shape)]}.  Buffers are copied too.
    """
    ours = model.state_dict()
    loaded, skipped, unexpected = [], [], []

    for key, value in official.items():  # first pass: classify
        if key not in ours:
            unexpected.append((key, tuple(value.shape)))
        elif ours[key].shape == value.shape:
            loaded.append(key)
        else:
            skipped.append((key, tuple(ours[key].shape), tuple(value.shape)))

    with torch.no_grad():  # in-place copy onto the model's existing tensors
        for key in loaded:
            ours[key].copy_(official[key])

    if verbose:
        print(f"loaded {len(loaded)} tensors, skipped {len(skipped)}, "
              f"unexpected {len(unexpected)}")
        for key, a, b in skipped:
            print(f"  skip {key}: ours {a} vs official {b}")
    return {"loaded": loaded, "skipped": skipped, "unexpected": unexpected}


# With BN fused away, the nc=20 class branch (c3 = min(nc, 100) -> 64 vs the
# official 80) differs in 4 tensors per scale x 3 scales = 24.  Per scale:
#   cv3.<s>.0.0.1  Conv(c3 -> c3 fixed 64<->80)     weight + bias
#   cv3.<s>.0.1.0  DWConv(c3 -> c3, depthwise)      weight + bias
#   cv3.<s>.0.1.1  Conv(c3 -> c3)                   weight + bias
#   cv3.<s>.2      Conv2d(c3 -> nc)                 weight + bias
EXPECTED_FUSED_NC20_SKIPPED = frozenset(
    [
        *(
            f"model.23.cv3.{s}.{a}.{b}.{comp}"
            for s in range(3)
            for (a, b) in ((0, 1), (1, 0), (1, 1))
            for comp in ("conv.weight", "conv.bias")
        ),
        *(
            f"model.23.cv3.{s}.2.{comp}"
            for s in range(3)
            for comp in ("weight", "bias")
        ),
    ]
)


def load_fused_from_onnx(path: str = ONNX_PATH) -> dict[str, torch.Tensor]:
    """Read the official ONNX graph initializers as a name -> fp32 Tensor dict.

    The graph's weights are BN-fused, so this only makes sense for a model
    that has been ``YOLO(...).fuse()``d — the key set then matches 1:1
    (``model.0.conv.weight`` + ``model.0.conv.bias``, no ``.bn.*``).
    """
    model = onnx.load(path)
    return {
        init.name: torch.tensor(onnx.numpy_helper.to_array(init), dtype=torch.float32)
        for init in model.graph.initializer
    }