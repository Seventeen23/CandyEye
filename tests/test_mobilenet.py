"""MobileNetV3-Small backbone (profile ``d``): build, ImageNet remap, parity.

The backbone is a non-YOLO feature extractor expressed in the YAML layer
table; the head is the stock YOLO11n neck copied verbatim.  These tests pin:

- the graph shape/stride contract the head relies on (P3=/8, P4=/16, P5=/32
  at layers 4/6/10, so the FPN concatenations line up),
- explicit channel counts (no YOLO width multiplier on the mobile blocks),
- 1:1 remapping of torchvision's ImageNet state_dict, and bit-exact forward
  parity against torchvision (needs the one-time weight download; skipped
  offline),
- bundled profile ``d`` + checkpoint -> predictor round-trip.
"""
import torch
import pytest

from candyeye.core import CandyEye
from candyeye.core.convert_mobilenet import (
    MOBILE_V3_SMALL_FEATURE_MAP, backbone_parity, imagenet_mobilenetv3_small_state,
    load_imagenet_backbone, remap_imagenet_backbone, uses_mobilenet_backbone,
)
from candyeye.core.modules.mobilenet import MBConv, MobileStem

CFG = "mobilenetv3_small"


def make(nc: int = 20, size: int = 128) -> CandyEye:
    return CandyEye(CFG, nc=nc, img_size=size)


def test_graph_shapes_strides_and_params():
    m = make()
    out = m(torch.randn(1, 3, 128, 128))
    assert [tuple(o.shape) for o in out] == [
        (1, 84, 16, 16), (1, 84, 8, 8), (1, 84, 4, 4)
    ]
    assert m.stride.tolist() == [8.0, 16.0, 32.0]
    assert sum(p.numel() for p in m.parameters()) == 1_436_692


def test_head_slots_land_on_p3_p4_p5_resolutions():
    m = make()
    # direct per-layer walk (backbone is a sequential chain)
    with torch.no_grad():
        a = torch.zeros(1, 3, 128, 128)
        sizes = {}
        for i in range(11):
            a = m.model[i](a)
            sizes[i] = a.shape[-1]
    assert sizes[4] == 16     # /8  P3
    assert sizes[6] == 8      # /16 P4
    assert sizes[10] == 4     # /32 P5
    assert sizes[4] == 2 * sizes[6] == 4 * sizes[10]


def test_modules_and_explicit_channels():
    m = make()
    assert isinstance(m.model[0], MobileStem)
    assert all(isinstance(m.model[i], MBConv) for i in range(1, 11))
    # channels are the torchvision plan verbatim (not width-scaled by 0.25)
    channel_of = {}
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        for i in range(11):
            x = m.model[i](x)
            channel_of[i] = x.shape[1]
    assert [channel_of[i] for i in range(11)] == [
        16, 16, 24, 24, 24, 40, 40, 40, 48, 48, 96
    ]
    # layer 4 is the stride-1 repeat of layer 3 (both residual)
    assert m.model[4].use_res_connect and m.model[3].use_res_connect


def test_profile_d_is_default_and_resolves():
    from candyeye.training.trainer import DEFAULT_PROFILE, PROFILES
    assert DEFAULT_PROFILE == "d"
    assert PROFILES["d"]["cfg"] == "mobilenetv3_small"


@pytest.fixture(scope="module")
def imagenet_state():
    state = imagenet_mobilenetv3_small_state()
    if state is None:
        pytest.skip("ImageNet weights unavailable (offline)")
    return state


def test_imagenet_remap_loads_every_backbone_tensor(imagenet_state):
    m = make(nc=9)
    report = load_imagenet_backbone(m, imagenet_state)
    assert report["unexpected"] == []
    assert report["skipped"] == []
    assert len(report["loaded"]) == 208
    # every remapped key targets a real model tensor
    for key in remap_imagenet_backbone(imagenet_state):
        assert key in m.state_dict()
    # idempotent: a second load is a no-op
    assert load_imagenet_backbone(m, imagenet_state)["loaded"] == []
    assert uses_mobilenet_backbone(m)


def test_backbone_forward_parity_with_torchvision(imagenet_state):
    m = make(nc=9)
    load_imagenet_backbone(m, imagenet_state)
    diffs = backbone_parity(m, imagenet_state, img=128, atol=1e-5)
    assert set(diffs) == set(range(11))
    assert all(v <= 1e-5 for v in diffs.values()), diffs


def test_feature_map_has_no_gaps():
    features = [f for f, _ in MOBILE_V3_SMALL_FEATURE_MAP]
    assert features == [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9]


def test_head_is_byte_identical_to_yolo11():
    import yaml
    from candyeye.paths import CONFIG_DIR

    with (CONFIG_DIR / "yolo11.yaml").open() as fh:
        base = yaml.safe_load(fh)
    with (CONFIG_DIR / f"{CFG}.yaml").open() as fh:
        mobile = yaml.safe_load(fh)
    assert mobile["head"] == base["head"]


def test_checkpoint_to_predictor_round_trip(tmp_path):
    from candyeye.inference.predict import CandyEyePredictor

    torch.manual_seed(0)
    model = make(nc=9)
    model.eval()
    payload = {
        "epoch": 1,
        "model": model.state_dict(),
        "config": {"model": {"cfg": CFG, "nc": 9, "img_size": 128}},
    }
    ckpt = tmp_path / "best.pt"
    torch.save(payload, ckpt)

    predictor = CandyEyePredictor(ckpt)
    assert predictor.nc == 9
    assert uses_mobilenet_backbone(predictor.model)
    import numpy as np

    result = predictor.predict(np.zeros((128, 128, 3), dtype=np.uint8))
    assert "detections" in result


# --- MobileNetV3-Large + P2 profiles (e/f/g) --------------------------------

LARGE_CFG = "mobilenetv3_large"


def test_large_graph_strides_and_params():
    m = CandyEye(LARGE_CFG, nc=20, img_size=128)
    assert m.stride.tolist() == [8.0, 16.0, 32.0]
    assert sum(p.numel() for p in m.parameters()) == 4_101_372
    # full-depth backbone: 16 torchvision blocks between stem and SPPF
    assert isinstance(m.model[0], MobileStem)
    assert all(isinstance(m.model[i], MBConv) for i in range(1, 16))


def test_large_p2_has_four_heads():
    m = CandyEye("mobilenetv3_large_p2", nc=20, img_size=128)
    assert m.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert sum(p.numel() for p in m.parameters()) == 4_144_544


def test_small_p2_has_four_heads():
    m = CandyEye("mobilenetv3_small_p2", nc=20, img_size=128)
    assert m.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert sum(p.numel() for p in m.parameters()) == 1_584_856


def test_profiles_e_f_g_registered():
    from candyeye.training.trainer import PROFILES
    assert PROFILES["e"]["cfg"] == "mobilenetv3_small_p2"
    assert PROFILES["f"]["cfg"] == "mobilenetv3_large_p2"
    assert PROFILES["g"]["cfg"] == "mobilenetv3_large"


def test_large_variant_and_feature_map():
    from candyeye.core.convert_mobilenet import (
        MOBILE_V3_LARGE_FEATURE_MAP, mobilenet_variant,
    )
    assert MOBILE_V3_LARGE_FEATURE_MAP == [(i, i) for i in range(16)]
    assert mobilenet_variant(CandyEye(LARGE_CFG, nc=9, img_size=128)) == "large"
    assert mobilenet_variant(CandyEye("mobilenetv3_small", nc=9, img_size=128)) == "small"


@pytest.fixture(scope="module")
def imagenet_large_state():
    from candyeye.core.convert_mobilenet import imagenet_mobilenetv3_large_state
    state = imagenet_mobilenetv3_large_state()
    if state is None:
        pytest.skip("ImageNet weights unavailable (offline)")
    return state


def test_large_imagenet_remap_and_parity(imagenet_large_state):
    m = CandyEye(LARGE_CFG, nc=9, img_size=128)
    report = load_imagenet_backbone(m, imagenet_large_state)
    assert report["unexpected"] == []
    assert report["skipped"] == []
    assert len(report["loaded"]) == 302
    diffs = backbone_parity(m, imagenet_large_state, img=128, atol=1e-5)
    assert set(diffs) == set(range(16))
    assert all(v <= 1e-5 for v in diffs.values()), diffs
