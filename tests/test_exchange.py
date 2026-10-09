"""Tests for the adaptive cross-scale exchange neck."""
import pytest
import torch

from candyeye.core.candyeye import CandyEye
from candyeye.core.modules.exchange import ScaleExchange, _Gate

CFG = "configs/yolo11_exchange.yaml"


def _features(c1=8, c2=16, c3=32):
    return [torch.randn(2, c1, 16, 16), torch.randn(2, c2, 8, 8),
            torch.randn(2, c3, 4, 4)]


@pytest.mark.parametrize("gate", ["none", "static", "dynamic"])
@pytest.mark.parametrize("iters", [1, 2])
def test_shapes_are_preserved(gate, iters):
    module = ScaleExchange(8, 16, 32, gate=gate, iters=iters).eval()
    features = _features()
    out = module(features)
    assert len(out) == 3
    for result, source in zip(out, features):
        assert result.shape == source.shape


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="gate"):
        ScaleExchange(8, 16, 32, gate="bogus")
    with pytest.raises(ValueError, match="iters"):
        ScaleExchange(8, 16, 32, iters=0)
    with pytest.raises(ValueError, match="3 feature maps"):
        ScaleExchange(8, 16, 32)([torch.randn(1, 8, 16, 16)])


@pytest.mark.parametrize("gate", ["static", "dynamic"])
def test_closed_gates_make_the_neck_identity(gate):
    module = ScaleExchange(8, 16, 32, gate=gate).eval().close_gates()
    features = _features()
    with torch.no_grad():
        out = module(features)
    for result, source in zip(out, features):
        assert torch.allclose(result, source, atol=1e-4)


def test_dynamic_gate_is_content_conditioned():
    gate = _Gate(8, 8, "dynamic").eval()
    # At init fc2 is zeroed (gate starts closed); give it weights so the
    # content path is exercised.
    torch.nn.init.normal_(gate.fc2.weight, std=0.5)
    torch.nn.init.normal_(gate.fc2.bias, std=0.5)
    self_features = torch.randn(1, 8, 4, 4)
    with torch.no_grad():
        first = gate(self_features, torch.randn(1, 8, 4, 4))
        second = gate(self_features, torch.randn(1, 8, 4, 4))
    assert first.shape == (1, 8, 1, 1)
    assert not torch.allclose(first, second)


def test_static_gate_ignores_content():
    gate = _Gate(8, 8, "static").eval()
    self_features = torch.randn(1, 8, 4, 4)
    with torch.no_grad():
        first = gate(self_features, torch.randn(1, 8, 4, 4))
        second = gate(self_features, torch.randn(1, 8, 4, 4))
    assert torch.allclose(first, second)


def test_exchange_config_builds_and_forwards():
    model = CandyEye(CFG, nc=9, img_size=128).eval()
    with torch.no_grad():
        decoded = model(torch.randn(1, 3, 128, 128))
        raw = model(torch.randn(1, 3, 128, 128), decode=False)
    assert tuple(decoded.shape) == (1, 13, 336)
    assert [tuple(x.shape) for x in raw] == [
        (1, 73, 16, 16), (1, 73, 8, 8), (1, 73, 4, 4)
    ]
    assert isinstance(model.model[-2], ScaleExchange)
