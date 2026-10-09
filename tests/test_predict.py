"""Letterbox robustness and checkpoint img_size handling for inference."""
import numpy as np
import torch

from candyeye.core import CandyEye
from candyeye.inference.predict import CandyEyePredictor, letterbox


def test_letterbox_survives_extreme_aspect_ratio():
    out, meta = letterbox(np.zeros((4000, 3, 3), np.uint8), size=64)

    assert out.shape == (64, 64, 3)
    assert meta["scale"] > 0


def test_predictor_uses_checkpoint_img_size(tmp_path):
    model = CandyEye("configs/yolo11.yaml", nc=2, img_size=64)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {"model": {"nc": 2, "img_size": 64},
                   "data": {"nc": 2, "names": ["a", "b"]}},
    }, ckpt)

    predictor = CandyEyePredictor(ckpt)
    assert predictor.imgsz == 64
    assert predictor.model.img_size == 64

    overridden = CandyEyePredictor(ckpt, imgsz=32)
    assert overridden.imgsz == 32
    assert overridden.model.img_size == 32
