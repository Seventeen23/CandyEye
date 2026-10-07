"""PASCAL VOC detection metrics (all-point interpolated AP)."""
from __future__ import annotations

import numpy as np


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and a set of xyxy boxes."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float64)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    inter = np.prod(np.maximum(rb - lt, 0), axis=1)
    area_box = np.prod(np.maximum(box[2:] - box[:2], 0))
    area_boxes = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return inter / np.maximum(area_box + area_boxes - inter, 1e-12)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC all-point AP: integrate the precision envelope at recall changes."""
    mrec = np.concatenate(([0.0], recall.astype(np.float64), [1.0]))
    mpre = np.concatenate(([0.0], precision.astype(np.float64), [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    changes = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def evaluate_map50(ground_truth: dict, detections: list[dict],
                   num_classes: int, iou_threshold: float = 0.5):
    """Calculate VOC mAP@IoU for image-keyed GT and scored detections.

    Ground truth entries are ``{"boxes": (N,4), "labels": (N,),
    "difficult": (N,)}``. Detection entries are ``image_id``, ``class_id``,
    ``score``, and xyxy ``box``. A detection matched to a difficult GT is
    ignored; it is neither a true positive nor a false positive.

    Returns ``(mean_ap, per_class_ap)``. Classes without non-difficult GT are
    reported as ``None`` and excluded from the mean, as in VOC evaluation.
    """
    class_ap = {}
    valid_aps = []
    for class_id in range(num_classes):
        class_gts = {}
        npos = 0
        for image_id, entry in ground_truth.items():
            labels = np.asarray(entry["labels"], dtype=np.int64)
            boxes = np.asarray(entry["boxes"], dtype=np.float64).reshape(-1, 4)
            difficult = np.asarray(entry.get("difficult", np.zeros(len(labels))),
                                   dtype=np.bool_)
            selected = labels == class_id
            gt_boxes = boxes[selected]
            gt_difficult = difficult[selected]
            class_gts[image_id] = {"boxes": gt_boxes, "difficult": gt_difficult,
                                   "matched": np.zeros(len(gt_boxes), dtype=np.bool_)}
            npos += int((~gt_difficult).sum())

        if npos == 0:
            class_ap[class_id] = None
            continue
        ranked = sorted(
            (d for d in detections if int(d["class_id"]) == class_id),
            key=lambda d: -float(d["score"]),
        )
        tp, fp, ignored = [], [], []
        for det in ranked:
            record = class_gts.get(det["image_id"])
            if record is None or len(record["boxes"]) == 0:
                tp.append(0.0); fp.append(1.0); ignored.append(False)
                continue
            ious = box_iou_xyxy(np.asarray(det["box"], dtype=np.float64), record["boxes"])
            best = int(ious.argmax())
            if ious[best] < iou_threshold:
                tp.append(0.0); fp.append(1.0); ignored.append(False)
            elif record["difficult"][best]:
                tp.append(0.0); fp.append(0.0); ignored.append(True)
            elif not record["matched"][best]:
                record["matched"][best] = True
                tp.append(1.0); fp.append(0.0); ignored.append(False)
            else:
                tp.append(0.0); fp.append(1.0); ignored.append(False)

        keep = ~np.asarray(ignored, dtype=np.bool_)
        tp = np.cumsum(np.asarray(tp, dtype=np.float64)[keep])
        fp = np.cumsum(np.asarray(fp, dtype=np.float64)[keep])
        recall = tp / npos
        precision = tp / np.maximum(tp + fp, 1e-12)
        ap = average_precision(recall, precision)
        class_ap[class_id] = ap
        valid_aps.append(ap)

    mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0
    return mean_ap, class_ap
