"""Training schedule and MobileNet detector contract checks."""
import pytest
import torch

from core.backbone_mobilenet import MobileNetV3SmallDetector
from core import CandyEye
from training.loss import DetectionLoss
from training.trainer import lr_factor


def test_warmup_cosine_schedule_boundaries():
    assert lr_factor(0, 10, warmup_epochs=3) == pytest.approx(.1)
    assert lr_factor(3, 10, warmup_epochs=3) == pytest.approx(1.)
    assert lr_factor(10, 10, warmup_epochs=3) == pytest.approx(.01)
    assert lr_factor(2, 2, warmup_epochs=3) == pytest.approx(1.)


def test_mobilenet_detector_outputs_and_loss():
    model = MobileNetV3SmallDetector(nc=20, img_size=64, pretrained=False)
    model.train()
    features = model(torch.randn(1, 3, 64, 64))
    assert [tuple(x.shape) for x in features] == [
        (1, 84, 8, 8), (1, 84, 4, 4), (1, 84, 2, 2)]
    loss = DetectionLoss(model)(features, torch.tensor([[0, 2, .5, .5, .4, .4]]))
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_yolo_train_dispatches_to_high_level_api(monkeypatch, tmp_path):
    import training.trainer as trainer
    from core import CandyEye as PublicCandyEye
    from core import train as train_entry

    assert PublicCandyEye.__name__ == "CandyEye"

    seen = {}

    def fake_train(model, **kwargs):
        seen.update(kwargs)
        return {"save_dir": tmp_path}

    monkeypatch.setattr(trainer, "train_model", fake_train)
    model = CandyEye("configs/yolo11.yaml", img_size=64)
    result = model.train(data="configs/default.yaml", epochs=7, imgsz=96,
                         batch=8, patience=4)

    assert result["save_dir"] == tmp_path
    assert seen["epochs"] == 7
    assert seen["imgsz"] == 96
    assert seen["batch"] == 8
    assert seen["patience"] == 4
    assert model.train(False) is model
    assert not model.training

    result = train_entry(model="configs/yolo11.yaml", data="configs/default.yaml",
                         epochs=5, imgsz=64)
    assert result["save_dir"] == tmp_path
    assert seen["epochs"] == 5
