"""YOLO detection objective: class BCE, CIoU box loss, and DFL."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from core.modules.detect import make_anchors
from training.assigner import TaskAlignedAssigner


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor,
              eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU for corresponding xyxy boxes."""
    b1_wh = (box1[:, 2:] - box1[:, :2]).clamp(min=0)
    b2_wh = (box2[:, 2:] - box2[:, :2]).clamp(min=0)
    inter_lt = torch.maximum(box1[:, :2], box2[:, :2])
    inter_rb = torch.minimum(box1[:, 2:], box2[:, 2:])
    inter = (inter_rb - inter_lt).clamp(min=0).prod(1)
    union = b1_wh.prod(1) + b2_wh.prod(1) - inter + eps
    iou = inter / union

    enclosing_lt = torch.minimum(box1[:, :2], box2[:, :2])
    enclosing_rb = torch.maximum(box1[:, 2:], box2[:, 2:])
    c2 = (enclosing_rb - enclosing_lt).pow(2).sum(1) + eps
    center1 = (box1[:, :2] + box1[:, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2
    rho2 = (center1 - center2).pow(2).sum(1)
    v = (4 / torch.pi**2) * (
        torch.atan(b2_wh[:, 0] / (b2_wh[:, 1] + eps)) -
        torch.atan(b1_wh[:, 0] / (b1_wh[:, 1] + eps))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - rho2 / c2 - alpha * v


class DetectionLoss(nn.Module):
    """Compute losses from raw Detect maps and packed normalized targets.

    ``targets`` is ``(N,6)``: batch index, class id, normalized cx, cy, w, h.
    ``model`` must expose ``stride`` and a final Detect module.
    """
    def __init__(self, model: nn.Module, topk: int = 10,
                 box_gain: float = 7.5, cls_gain: float = .5,
                 dfl_gain: float = 1.5):
        super().__init__()
        detect = model.model[-1]
        self.nc = detect.nc
        self.reg_max = detect.reg_max
        self.stride = detect.stride.detach().clone()
        self.assigner = TaskAlignedAssigner(topk=topk, alpha=.5, beta=6.0)
        self.box_gain, self.cls_gain, self.dfl_gain = box_gain, cls_gain, dfl_gain

    def forward(self, features: list[torch.Tensor], targets: torch.Tensor):
        if len(features) != len(self.stride):
            raise ValueError("expected one feature map per model stride")
        batch_size = features[0].shape[0]
        flat = [f.permute(0, 2, 3, 1).reshape(batch_size, -1, f.shape[1])
                for f in features]
        pred = torch.cat(flat, dim=1)
        box_logits, cls_logits = pred.split((4 * self.reg_max, self.nc), dim=2)
        dist = box_logits.reshape(batch_size, -1, 4, self.reg_max)
        stride_values = self.stride.to(device=pred.device, dtype=pred.dtype)
        anchors_grid, stride_col = make_anchors(features, stride_values)
        anchor_points = anchors_grid * stride_col
        stride_per_anchor = stride_col.squeeze(1)
        pred_dist = dist.softmax(-1)
        bins = torch.arange(self.reg_max, device=pred.device, dtype=pred.dtype)
        pred_ltrb = (pred_dist * bins).sum(-1)
        pred_boxes_grid = torch.cat((anchors_grid[None] - pred_ltrb[..., :2],
                                     anchors_grid[None] + pred_ltrb[..., 2:]), dim=-1)
        pred_boxes = pred_boxes_grid * stride_per_anchor[None, :, None]

        gt_labels, gt_boxes, mask_gt = self._pad_targets(targets, batch_size,
                                                          pred.device, pred.dtype,
                                                          features[0].shape[-1] * stride_values[0])
        assigned = self.assigner(cls_logits.detach().sigmoid(), pred_boxes.detach(),
                                 anchor_points, gt_labels, gt_boxes, mask_gt)
        target_scores = assigned["scores"]
        score_sum = target_scores.sum().clamp(min=1.0)
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits, target_scores, reduction="sum") / score_sum

        fg = assigned["fg_mask"]
        if fg.any():
            weights = target_scores.sum(-1)[fg]
            pred_fg = pred_boxes[fg]
            target_fg = assigned["boxes"][fg]
            ciou = bbox_ciou(pred_fg, target_fg)
            box_loss = ((1 - ciou) * weights).sum() / score_sum

            stride_fg = stride_per_anchor[None, :].expand(batch_size, -1)[fg]
            gt_grid = target_fg / stride_fg[:, None]
            anchors_fg = anchors_grid[None, :, :].expand(batch_size, -1, -1)[fg]
            target_ltrb = torch.cat((anchors_fg - gt_grid[:, :2],
                                     gt_grid[:, 2:] - anchors_fg), dim=1)
            target_ltrb = target_ltrb.clamp(min=0, max=self.reg_max - 1 - 1e-2)
            dfl_logits = dist[fg].reshape(-1, self.reg_max)
            dfl_target = target_ltrb.reshape(-1)
            left = dfl_target.floor().long()
            right = left + 1
            right_weight = dfl_target - left
            left_weight = 1 - right_weight
            per_side = (F.cross_entropy(dfl_logits, left, reduction="none") * left_weight +
                        F.cross_entropy(dfl_logits, right, reduction="none") * right_weight)
            per_anchor = per_side.reshape(-1, 4).mean(1)
            dfl_loss = (per_anchor * weights).sum() / score_sum
        else:
            # Preserve graph connectivity so empty-GT batches can backpropagate.
            box_loss = box_logits.sum() * 0
            dfl_loss = box_logits.sum() * 0

        total = self.box_gain * box_loss + self.cls_gain * cls_loss + self.dfl_gain * dfl_loss
        return {"loss": total, "box": box_loss.detach(),
                "cls": cls_loss.detach(), "dfl": dfl_loss.detach(),
                "foreground": fg.sum().detach()}

    def _pad_targets(self, targets, batch_size, device, dtype, img_size):
        if targets.numel() == 0:
            return (torch.zeros((batch_size, 0), device=device, dtype=torch.long),
                    torch.zeros((batch_size, 0, 4), device=device, dtype=dtype),
                    torch.zeros((batch_size, 0), device=device, dtype=torch.bool))
        counts = [int((targets[:, 0] == i).sum()) for i in range(batch_size)]
        max_gt = max(counts, default=0)
        labels = torch.zeros((batch_size, max_gt), device=device, dtype=torch.long)
        boxes = torch.zeros((batch_size, max_gt, 4), device=device, dtype=dtype)
        mask = torch.zeros((batch_size, max_gt), device=device, dtype=torch.bool)
        for i in range(batch_size):
            row = targets[targets[:, 0] == i]
            n = len(row)
            if n == 0:
                continue
            labels[i, :n] = row[:, 1].long()
            cxcy = row[:, 2:4] * img_size
            wh = row[:, 4:6] * img_size
            boxes[i, :n, :2] = cxcy - wh / 2
            boxes[i, :n, 2:] = cxcy + wh / 2
            mask[i, :n] = True
        return labels, boxes, mask
