"""Acceptance tests for TAL assignment and CandyEye's detection objective."""
import torch

from core import CandyEye
from training.assigner import TaskAlignedAssigner
from training.loss import DetectionLoss


def test_assigner_no_ground_truth_returns_empty_foreground():
    assigner = TaskAlignedAssigner(topk=2)
    result = assigner(
        torch.rand(1, 5, 3), torch.rand(1, 5, 4), torch.rand(5, 2),
        torch.zeros((1, 0), dtype=torch.long), torch.zeros((1, 0, 4)),
        torch.zeros((1, 0), dtype=torch.bool),
    )
    assert result["fg_mask"].shape == (1, 5)
    assert not result["fg_mask"].any()
    assert not result["scores"].any()


def test_assigner_selects_topk_inside_ground_truth():
    assigner = TaskAlignedAssigner(topk=2)
    anchors = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [8., 8.]])
    gt_boxes = torch.tensor([[[0., 0., 4., 4.]]])
    pred_boxes = torch.tensor([[[0., 0., 4., 4.]] * 4])
    scores = torch.tensor([[[.9], [.8], [.7], [.99]]])
    result = assigner(scores, pred_boxes, anchors,
                      torch.tensor([[0]]), gt_boxes, torch.tensor([[True]]))
    assert result["fg_mask"].sum().item() == 2
    assert torch.equal(result["labels"][result["fg_mask"]], torch.tensor([0, 0]))
    assert torch.all(result["scores"][result["fg_mask"]] > 0)


def test_detection_loss_backpropagates_for_empty_and_nonempty_targets():
    model = CandyEye("configs/yolo11.yaml", img_size=64)
    criterion = DetectionLoss(model, topk=3)
    model.train()
    features = model(torch.rand(1, 3, 64, 64))

    empty = criterion(features, torch.zeros((0, 6)))
    assert torch.isfinite(empty["loss"])
    assert empty["foreground"].item() == 0
    empty["loss"].backward()
    model.zero_grad(set_to_none=True)

    features = model(torch.rand(1, 3, 64, 64))
    targets = torch.tensor([[0, 3, .5, .5, .4, .4]])
    losses = criterion(features, targets)
    assert torch.isfinite(losses["loss"])
    assert losses["foreground"].item() > 0
    losses["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
