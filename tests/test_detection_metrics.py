"""Tests for per-class validation metrics."""
import pytest
import torch
from torch import nn

from candyeye.eval.detection_metrics import evaluate_detector


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


class DifficultPredictions(nn.Module):
    def forward(self, images):
        # Detection on the difficult object (.9), a TP on the easy object
        # (.8), and a false positive (.7).
        rows = torch.tensor([
            [16.0, 16.0, 8.0, 8.0, .90],
            [8.0, 8.0, 8.0, 8.0, .80],
            [28.0, 28.0, 4.0, 4.0, .70],
        ], dtype=images.dtype, device=images.device)
        return rows.t().unsqueeze(0).expand(images.shape[0], -1, -1)


def test_difficult_objects_are_ignored_in_map_precision_and_confusion():
    batch = {
        "images": torch.zeros(1, 3, 32, 32),
        # easy box xyxy(4,4,12,12); difficult box xyxy(12,12,20,20)
        "targets": torch.tensor([[0, 0, .25, .25, .25, .25],
                                 [0, 0, .50, .50, .25, .25]]),
        "img_ids": ["image-a"],
        "difficult": [torch.tensor([False, True])],
    }

    metrics = evaluate_detector(
        DifficultPredictions(), [batch], num_classes=1, class_names=["fruit"],
        include_confusion_matrix=True,
    )

    # .9 on the difficult object is ignored, .8 is a TP, .7 is an FP.
    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(.5)
    assert metrics["recall"] == pytest.approx(1.0)
    # Difficult GT does not count toward ground truth (or as a miss).
    assert metrics["per_class"]["fruit"]["ground_truth"] == 1
    assert metrics["per_class"]["fruit"]["predictions"] == 3
    # Confusion: easy GT matched on the diagonal; one background FP row.
    # The detection on the difficult object must appear nowhere.
    matrix = metrics["confusion_matrix"]
    assert matrix.tolist() == [[1, 0], [1, 0]]
