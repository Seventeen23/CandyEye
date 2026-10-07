"""Task Aligned Assigner for anchor-point object detection."""
from __future__ import annotations

import torch
from torch import nn


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor,
                 eps: float = 1e-9) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes shaped ``(N,4)`` and ``(M,4)``."""
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(1)
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


class TaskAlignedAssigner(nn.Module):
    """Select positives using class confidence and overlap (top-k per GT)."""
    def __init__(self, topk: int = 10, alpha: float = .5, beta: float = 6.0,
                 eps: float = 1e-9):
        super().__init__()
        self.topk, self.alpha, self.beta, self.eps = topk, alpha, beta, eps

    @torch.no_grad()
    def forward(self, pred_scores: torch.Tensor, pred_boxes: torch.Tensor,
                anchor_points: torch.Tensor, gt_labels: torch.Tensor,
                gt_boxes: torch.Tensor, mask_gt: torch.Tensor) -> dict[str, torch.Tensor]:
        """Assign ``(B,A,C)`` scores/``(B,A,4)`` xyxy predictions to padded GTs.

        Returns target labels, boxes, soft class scores, foreground mask, and
        the selected GT index for each anchor. GT labels have shape ``(B,M)``;
        boxes are pixel-space xyxy and ``mask_gt`` marks valid padded entries.
        """
        bs, na, nc = pred_scores.shape
        max_gt = gt_boxes.shape[1]
        device = pred_scores.device
        target_labels = torch.zeros((bs, na), dtype=torch.long, device=device)
        target_boxes = torch.zeros((bs, na, 4), dtype=pred_boxes.dtype, device=device)
        target_scores = torch.zeros((bs, na, nc), dtype=pred_scores.dtype, device=device)
        fg_mask = torch.zeros((bs, na), dtype=torch.bool, device=device)
        target_gt_idx = torch.zeros((bs, na), dtype=torch.long, device=device)
        if max_gt == 0 or not bool(mask_gt.any()):
            return {"labels": target_labels, "boxes": target_boxes,
                    "scores": target_scores, "fg_mask": fg_mask,
                    "target_gt_idx": target_gt_idx}

        for b in range(bs):
            valid = mask_gt[b].bool()
            if not bool(valid.any()):
                continue
            boxes = gt_boxes[b, valid]
            labels = gt_labels[b, valid].long().clamp(0, nc - 1)
            ious = pairwise_iou(boxes, pred_boxes[b]).clamp(min=0)
            in_gt = ((anchor_points[None, :, 0] > boxes[:, None, 0]) &
                     (anchor_points[None, :, 0] < boxes[:, None, 2]) &
                     (anchor_points[None, :, 1] > boxes[:, None, 1]) &
                     (anchor_points[None, :, 1] < boxes[:, None, 3]))
            cls = pred_scores[b, :, labels].transpose(0, 1).clamp(min=0)
            metric = cls.pow(self.alpha) * ious.pow(self.beta)
            metric = metric.masked_fill(~in_gt, 0)
            k = min(self.topk, na)
            values, indices = metric.topk(k, dim=1)
            topk_mask = torch.zeros_like(metric, dtype=torch.bool)
            topk_mask.scatter_(1, indices, values > 0)
            positives = topk_mask & in_gt

            # An anchor selected by overlapping GTs belongs to the one with
            # highest IoU, matching the usual TAL conflict resolution.
            multi = positives.sum(0) > 1
            if multi.any():
                best = ious.argmax(0)
                positives[:, multi] = False
                positives[best[multi], multi] = True

            assigned = positives.any(0)
            if not assigned.any():
                continue
            gt_idx = positives.float().argmax(0)
            local_boxes = boxes[gt_idx]
            local_labels = labels[gt_idx]
            target_boxes[b, assigned] = local_boxes[assigned]
            target_labels[b, assigned] = local_labels[assigned]
            target_gt_idx[b, assigned] = gt_idx[assigned]
            fg_mask[b] = assigned

            # Normalize alignment per GT, retaining IoU as the target quality.
            max_metric = metric.masked_fill(~positives, 0).amax(1, keepdim=True)
            max_iou = ious.masked_fill(~positives, 0).amax(1, keepdim=True)
            quality = (metric * max_iou / (max_metric + self.eps)).amax(0)
            # Scratch initialization can produce almost-zero IoUs. Keeping a
            # minimum positive target prevents box/class gradients from
            # vanishing before the detector learns its first useful boxes.
            quality = quality.clamp(min=0.25)
            target_scores[b, assigned, local_labels[assigned]] = quality[assigned]

        return {"labels": target_labels, "boxes": target_boxes,
                "scores": target_scores, "fg_mask": fg_mask,
                "target_gt_idx": target_gt_idx}
