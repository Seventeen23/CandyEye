"""Validation metrics for CandyEye object detection."""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import batched_nms

from eval.map import box_iou_xyxy, evaluate_map50


def evaluate_detector(model, loader, *, num_classes: int, class_names=None,
                      conf_threshold: float = 0.25, ap_conf_threshold: float = 0.001,
                      iou_threshold: float = 0.5, max_detections: int = 300,
                      criterion=None, include_confusion_matrix: bool = False) -> dict:
    """Evaluate a detector and report per-class/macro/micro metrics.

    Precision, recall, and F1 use ``conf_threshold``. AP uses detections above
    ``ap_conf_threshold`` and all-point interpolation at ``iou_threshold``.
    Boxes are matched one-to-one within each image and class.
    """
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("class_names length must equal num_classes")

    ground_truth = {}
    detections = []
    background_id = num_classes
    confusion_matrix = (np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
                        if include_confusion_matrix else None)
    val_loss_sums = {"loss": 0.0, "box": 0.0, "cls": 0.0,
                     "dfl": 0.0, "foreground": 0.0}
    val_batches = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch["images"]
                targets = batch["targets"]
                if criterion is not None:
                    # Keep BN in eval mode while still asking Detect for raw maps.
                    raw_features = model(images, decode=False)
                    losses = criterion(raw_features, targets)
                    for key in val_loss_sums:
                        val_loss_sums[key] += float(losses[key].detach())
                    val_batches += 1
                predictions = model(images)
                if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3:
                    raise ValueError("model eval forward must return (B, 4 + nc, anchors)")
                predictions = predictions.transpose(1, 2)
                size = images.shape[-1]
                for batch_index, image_id in enumerate(batch["img_ids"]):
                    target = targets[targets[:, 0] == batch_index]
                    if len(target):
                        cx, cy, width, height = (target[:, 2:6] * size).unbind(1)
                        boxes = torch.stack((cx - width / 2, cy - height / 2,
                                             cx + width / 2, cy + height / 2), dim=1)
                        gt_boxes = boxes.cpu().numpy()
                        gt_labels = target[:, 1].long().cpu().numpy()
                    else:
                        gt_boxes = np.zeros((0, 4), dtype=np.float32)
                        gt_labels = np.zeros((0,), dtype=np.int64)
                    ground_truth[image_id] = {
                        "boxes": gt_boxes, "labels": gt_labels,
                        "difficult": np.zeros(len(gt_labels), dtype=np.bool_),
                    }

                    pred = predictions[batch_index]
                    boxes = pred[:, :4]
                    class_scores, class_ids = pred[:, 4:].max(dim=1)
                    keep = class_scores >= ap_conf_threshold
                    boxes, class_scores, class_ids = boxes[keep], class_scores[keep], class_ids[keep]
                    if boxes.numel() == 0:
                        if confusion_matrix is not None:
                            for class_id in gt_labels:
                                confusion_matrix[int(class_id), background_id] += 1
                        continue
                    cx, cy, width, height = boxes.unbind(1)
                    xyxy = torch.stack((cx - width / 2, cy - height / 2,
                                        cx + width / 2, cy + height / 2), dim=1)
                    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, size)
                    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, size)
                    keep_indices = batched_nms(
                        xyxy, class_scores, class_ids, iou_threshold
                    )[:max_detections]
                    image_detections = []
                    for index in keep_indices.tolist():
                        detection = {
                            "image_id": image_id,
                            "class_id": int(class_ids[index]),
                            "score": float(class_scores[index]),
                            "box": xyxy[index].cpu().numpy(),
                        }
                        detections.append(detection)
                        if confusion_matrix is not None and detection["score"] >= conf_threshold:
                            image_detections.append(detection)

                    # Match detections to ground truth by IoU, regardless of
                    # class, so a wrong class is shown as an off-diagonal cell.
                    if confusion_matrix is not None:
                        pairs = []
                        for det_index, detection in enumerate(image_detections):
                            ious = box_iou_xyxy(detection["box"], gt_boxes)
                            pairs.extend((float(iou), det_index, gt_index)
                                         for gt_index, iou in enumerate(ious)
                                         if iou >= iou_threshold)
                        pairs.sort(reverse=True)
                        matched_detections, matched_targets = set(), set()
                        for _, det_index, gt_index in pairs:
                            if det_index in matched_detections or gt_index in matched_targets:
                                continue
                            detection = image_detections[det_index]
                            confusion_matrix[int(gt_labels[gt_index]), detection["class_id"]] += 1
                            matched_detections.add(det_index)
                            matched_targets.add(gt_index)
                        for gt_index, class_id in enumerate(gt_labels):
                            if gt_index not in matched_targets:
                                confusion_matrix[int(class_id), background_id] += 1
                        for det_index, detection in enumerate(image_detections):
                            if det_index not in matched_detections:
                                confusion_matrix[background_id, detection["class_id"]] += 1
    finally:
        model.train(was_training)

    map50, class_ap = evaluate_map50(
        ground_truth, detections, num_classes, iou_threshold=iou_threshold
    )
    per_class = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    valid_metrics = []
    for class_id, class_name in enumerate(class_names):
        gt_by_image = {
            image_id: np.asarray(entry["boxes"])[
                np.asarray(entry["labels"], dtype=np.int64) == class_id
            ]
            for image_id, entry in ground_truth.items()
        }
        gt_count = sum(len(boxes) for boxes in gt_by_image.values())
        ranked = sorted(
            (det for det in detections
             if det["class_id"] == class_id and det["score"] >= conf_threshold),
            key=lambda det: -det["score"],
        )
        matched = {image_id: np.zeros(len(boxes), dtype=np.bool_)
                   for image_id, boxes in gt_by_image.items()}
        tp = fp = 0
        for det in ranked:
            boxes = gt_by_image.get(det["image_id"], np.zeros((0, 4)))
            if not len(boxes):
                fp += 1
                continue
            overlaps = box_iou_xyxy(np.asarray(det["box"]), boxes)
            best = int(overlaps.argmax())
            if overlaps[best] >= iou_threshold and not matched[det["image_id"]][best]:
                matched[det["image_id"]][best] = True
                tp += 1
            else:
                fp += 1
        fn = gt_count - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt_count if gt_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision if gt_count else None,
            "recall": recall if gt_count else None,
            "f1": f1 if gt_count else None,
            "ap50": class_ap[class_id],
            "ground_truth": gt_count,
            "predictions": len(ranked),
        }
        if gt_count:
            valid_metrics.append((precision, recall, f1))
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    micro_precision = (totals["tp"] / (totals["tp"] + totals["fp"])
                       if totals["tp"] + totals["fp"] else 0.0)
    micro_recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall else 0.0)
    macro = np.mean(valid_metrics, axis=0) if valid_metrics else (0.0, 0.0, 0.0)
    return {
        "map50": map50,
        "precision": float(macro[0]), "recall": float(macro[1]),
        "f1": float(macro[2]),
        "micro_precision": micro_precision, "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "confusion_matrix_labels": ([*class_names, "background"]
                                    if confusion_matrix is not None else None),
        "confidence_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "val_loss": ({key: value / val_batches for key, value in val_loss_sums.items()}
                     if val_batches else None),
    }
