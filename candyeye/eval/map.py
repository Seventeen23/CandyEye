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
    # Out-of-range class ids would be silently dropped by the per-class loop:
    # detections vanish (no false positive) and GT vanishes (no missed object),
    # inflating mAP. Fail loudly instead — this always means an nc mismatch
    # between the model, the dataset, and the evaluator.
    bad_ids = set()
    for entry in ground_truth.values():
        bad_ids.update(
            int(label) for label in np.asarray(entry["labels"], dtype=np.int64)
            if not 0 <= int(label) < num_classes
        )
    bad_ids.update(
        int(det["class_id"]) for det in detections
        if not 0 <= int(det["class_id"]) < num_classes
    )
    if bad_ids:
        raise ValueError(
            f"class ids {sorted(bad_ids)} outside 0..{num_classes - 1}: "
            "model/dataset num_classes mismatch"
        )
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


# COCO-style IoU sweep: 0.50, 0.55, ..., 0.95.
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def _default_area_ranges(image_size: float) -> dict:
    """COCO small/medium/large area cutoffs, rescaled to ``image_size``.

    COCO defines them in absolute pixels on ~640px inputs (32^2 and 96^2);
    scaling by ``image_size / 640`` keeps the buckets meaningful on our
    128px training budget.
    """
    scale = float(image_size) / 640.0
    small, medium = (32 * scale) ** 2, (96 * scale) ** 2
    return {"small": (0.0, small), "medium": (small, medium),
            "large": (medium, float("inf"))}


def _filter_area(ground_truth: dict, detections: list[dict],
                 low: float, high: float):
    """Keep only GT boxes and detections whose pixel area falls in [low, high)."""
    filtered_gt = {}
    for image_id, entry in ground_truth.items():
        boxes = np.asarray(entry["boxes"], dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(entry["labels"], dtype=np.int64)
        difficult = np.asarray(entry.get("difficult", np.zeros(len(boxes))),
                               dtype=np.bool_)
        areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
        keep = (areas >= low) & (areas < high)
        filtered_gt[image_id] = {"boxes": boxes[keep], "labels": labels[keep],
                                 "difficult": difficult[keep]}
    filtered_detections = []
    for det in detections:
        box = np.asarray(det["box"], dtype=np.float64)
        area = float(np.prod(np.maximum(box[2:] - box[:2], 0)))
        if low <= area < high:
            filtered_detections.append(det)
    return filtered_gt, filtered_detections


def _has_positives(ground_truth: dict) -> bool:
    """True when any non-difficult ground-truth box survives the filter."""
    for entry in ground_truth.values():
        labels = np.asarray(entry["labels"], dtype=np.int64)
        difficult = np.asarray(entry.get("difficult", np.zeros(len(labels))),
                               dtype=np.bool_)
        if len(labels) and np.any(~difficult):
            return True
    return False


def evaluate_map(ground_truth: dict, detections: list[dict], num_classes: int,
                 iou_thresholds=IOU_THRESHOLDS, image_size: float = 640,
                 size_buckets: bool = False, area_ranges: dict | None = None) -> dict:
    """COCO-style mAP@0.5:0.95 plus optional small/medium/large breakdown.

    Averaging per-class AP over the IoU sweep is the COCO ``mAP``; ``map50`` and
    ``map75`` are the values at those single thresholds. Size buckets re-run the
    matching on area-filtered GT and detections only.
    """
    per_threshold, per_class_by_threshold = {}, {}
    for threshold in iou_thresholds:
        key = round(float(threshold), 2)
        mean_ap, class_ap = evaluate_map50(ground_truth, detections, num_classes,
                                           iou_threshold=float(threshold))
        per_threshold[key] = mean_ap
        per_class_by_threshold[key] = class_ap

    per_class = {}
    for class_id in range(num_classes):
        values = [per_class_by_threshold[key][class_id]
                  for key in per_class_by_threshold
                  if per_class_by_threshold[key][class_id] is not None]
        per_class[class_id] = float(np.mean(values)) if values else None

    result = {
        "map": float(np.mean(list(per_threshold.values()))) if per_threshold else 0.0,
        "map50": per_threshold.get(0.5),
        "map75": per_threshold.get(0.75),
        "per_threshold": per_threshold,
        "per_class": per_class,
        "per_class_ap50": per_class_by_threshold.get(0.5, {}),
        "iou_thresholds": sorted(per_threshold),
        "size": None,
    }
    if size_buckets:
        ranges = area_ranges or _default_area_ranges(image_size)
        size = {}
        for name, (low, high) in ranges.items():
            bucket_gt, bucket_det = _filter_area(ground_truth, detections, low, high)
            if not _has_positives(bucket_gt):
                size[name] = None
                continue
            values = [
                evaluate_map50(bucket_gt, bucket_det, num_classes,
                               iou_threshold=float(threshold))[0]
                for threshold in iou_thresholds
            ]
            size[name] = float(np.mean(values)) if values else None
        result["size"] = size
    return result
