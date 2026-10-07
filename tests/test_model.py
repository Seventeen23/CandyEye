"""Phase 1 acceptance tests for the YOLO11n model built from the yaml.

Run:  PYTHONPATH=. venv/bin/python -m pytest tests/test_model.py -q
"""
import torch
import pytest

from core.yolo import YOLO

CFG = "configs/yolo11.yaml"


def make(**kw) -> YOLO:
    return YOLO(CFG, **kw)


def test_forward_shapes_and_strides():
    m = make()
    out = m(torch.randn(1, 3, 128, 128))
    shapes = [tuple(o.shape) for o in out]
    assert shapes == [(1, 84, 16, 16), (1, 84, 8, 8), (1, 84, 4, 4)]
    assert m.stride.tolist() == [8.0, 16.0, 32.0]


def test_param_count():
    assert sum(p.numel() for p in make().parameters()) == 2_593_740


def test_backward_gradients_flow():
    m = make()
    out = m(torch.randn(1, 3, 128, 128))
    loss = sum(o.float().mean() for o in out)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(bool(g.abs().sum() > 0) for g in grads)


def test_key_layout_matches_official_prefix():
    m = make()
    keys = list(m.state_dict())
    assert keys[0] == "model.0.conv.weight"
    assert keys[1] == "model.0.bn.weight"
    assert keys[-1] == "model.23.dfl.conv.weight"
    assert m.model[-1].dfl.conv.weight.numel() == 16
    assert m.model[-1].dfl.conv.weight.requires_grad is False


def test_imgsz_guard():
    with pytest.raises(ValueError):
        make(img_size=127)