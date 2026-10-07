"""Tests for per-class validation metrics."""
import pytest
import torch
from torch import nn

from eval.detection_metrics import evaluate_detector


class FixedPredictions(nn.Module):
    def forward(self, images):
        # One correct fruit box followed by one high-confidence false positive.
        rows = torch.tensor([
            [16.0, 16.0, 16.0, 16.0, .90],
            [30.0, 30.0, 2.0, 2.0, .80],
        ], dtype=images.dtype, device=images.device)
        return rows.t().unsqueeze(0).expand(images.shape[0], -1, -1)


def test_detector_reports_per_class_and_macro_metrics():
    batch = {
        "images": torch.zeros(1, 3, 32, 32),
        "targets": torch.tensor([[0, 0, .5, .5, .5, .5]]),
        "img_ids": ["image-a"],
    }

    metrics = evaluate_detector(
        FixedPredictions(), [batch], num_classes=1, class_names=["fruit"]
    )

    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["per_class"]["fruit"]["f1"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["fruit"]["ground_truth"] == 1
