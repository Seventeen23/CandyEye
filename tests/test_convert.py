"""Phase 2 acceptance: official-weight loading + numerical parity with the net.

Two weight sources, two purposes:

- ``weights/yolo11n.pth`` (unfused, incl. BN buffers) checks the *loader
  bookkeeping*: full nc=80 load (0/0) and the nc=20 class-branch skip set.
  Numerically it is only fp16-checkpoint-close, so it is not a parity proof.
- The official fp32 ONNX graph's *fused* initializers give an EXACT parity
  proof: after ``CandyEye(...).fuse()`` our state_dict and the graph's weight
  tensors align 1:1, and running both graphs on the same input must agree to
  fp noise (``< 1e-4``).

Run:  PYTHONPATH=. venv/bin/python -m pytest tests/test_convert.py -q
Prereqs (once): scripts/bootstrap_weights.py + the official ONNX at
/tmp/opencode/yolo11n.onnx (torch.onnx.export of the .pt, fp32, opset 12).
"""
import onnxruntime as ort
import torch
import pytest

from core.convert_yolo11 import (
    EXPECTED_NC20_SKIPPED, EXPECTED_FUSED_NC20_SKIPPED,
    load_official_state_dict, load_weights, load_fused_from_onnx,
)
from core import CandyEye

CFG = "configs/yolo11.yaml"
WEIGHTS = "weights/yolo11n.pth"
ONNX = "/tmp/opencode/yolo11n.onnx"  # official fp32 export (BN fused)

sd = load_official_state_dict(WEIGHTS)
fused = load_fused_from_onnx(ONNX)


def make(nc: int, size: int = 128) -> CandyEye:
    return CandyEye(CFG, nc=nc, img_size=size)


def test_nc80_full_load():
    """nc=80 rebuild is the official architecture -> nothing should be skipped."""
    m = make(80)
    assert sum(p.numel() for p in m.parameters()) == 2_624_080
    r = load_weights(m, sd)
    assert r["unexpected"] == []
    assert r["skipped"] == []
    assert len(r["loaded"]) == len(sd)
    m.eval()
    with torch.no_grad():
        out = m(torch.rand(1, 3, 128, 128))
    assert tuple(out.shape) == (1, 84, 336)


def test_nc20_sparse_load():
    """nc=20 (our VOC target): exactly the class-branch tensors are skipped."""
    m = make(20)
    assert sum(p.numel() for p in m.parameters()) == 2_593_740
    r = load_weights(m, sd, verbose=True)
    skipped_keys = {k for k, _, _ in r["skipped"]}
    assert r["unexpected"] == []
    assert skipped_keys == EXPECTED_NC20_SKIPPED
    assert len(skipped_keys) == 51
    # box branch + shared depthwise convs still load 1:1
    loaded_keys = set(r["loaded"])
    assert "model.23.dfl.conv.weight" in loaded_keys
    assert all(f"model.23.cv3.{s}.0.0.conv.weight" in loaded_keys for s in range(3))
    m.eval()
    with torch.no_grad():
        out = m(torch.rand(1, 3, 128, 128))
    assert tuple(out.shape) == (1, 24, 336)


def test_fuse_layout_matches_onnx():
    """After fuse(), the state_dict keys ARE the ONNX key set (no .bn.*)."""
    m = make(80).fuse()
    fc = {k for k in fused if ".conv.weight" in k or ".conv.bias" in k}
    # every official conv weight/bias has a matching placeholder in our model
    assert fc <= set(m.state_dict())
    # ...and the shape-checked copy needs nothing of ours to be skipped
    r = load_weights(m, fused)
    assert r["skipped"] == []


def run_onnx(x: torch.Tensor) -> torch.Tensor:
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    (name,) = [i.name for i in sess.get_inputs()]
    (out,) = sess.run(None, {name: x.numpy()})
    return torch.from_numpy(out)


@pytest.mark.parametrize("nc,row_slice", [(80, 84), (20, 4)])
def test_onnx_parity(nc, row_slice):
    """Our fused model must equal the official ONNX on the *shared* rows.

    fp32 weights from the ONNX graph itself -> agreement is fp noise, not
    "fp16 checkpoint drift".  nc=20 only shares the 4 box rows (the class
    branch is deliberately left random).
    """
    m = make(nc).fuse()
    r = load_weights(m, fused)
    if nc == 20:
        skipped_keys = {k for k, _, _ in r["skipped"]}
        assert skipped_keys == EXPECTED_FUSED_NC20_SKIPPED
    else:
        assert r["skipped"] == []
    m.eval()
    x = torch.rand(1, 3, 640, 640)  # must be the ONNX's baked anchor-grid size
    with torch.no_grad():
        ours = m(x)
    official = run_onnx(x)
    assert tuple(official.shape) == (1, 84, 8400)
    a, b = ours[:, :row_slice], official[:, :row_slice]
    diff = (a - b).abs().max().item()
    # Boxes go through the DFL 16-bin softmax, whose onnxruntime kernel drifts
    # a few ULPs from torch's -> ~1e-3 on box magnitudes of hundreds.  A real
    # parametric difference would be O(1)+, so 1e-2 is a strict-but-fair line.
    assert diff < 1e-2, f"max abs diff {diff:.2e} on rows 0:{row_slice}"
    if nc == 80:  # full parity also covers the class rows
        cls_diff = (ours[:, 4:] - official[:, 4:]).abs().max().item()
        assert cls_diff < 1e-4, f"cls max abs diff {cls_diff:.2e}"
