"""Minimal detection demo: letterbox -> forward -> decode -> NMS -> draw.

Run at nc=20 (the VOC training model) or nc=80 (full official load):
  PYTHONPATH=. venv/bin/python inference/predict.py runs/bus.jpg --nc 80
  PYTHONPATH=. venv/bin/python inference/predict.py runs/bus.jpg --nc 20

nc=80 gives *real* COCO detections (every official weight loads 1:1).
nc=20 skips/randomises the class branch (the `cv3` classification convs, see
core/convert_yolo11.py), so classes are placeholders until Phase 4 fine-tuning
— the image then proves the letterbox->forward->DFL-decode->NMS->draw
pipeline, not accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from core.convert_yolo11 import load_official_state_dict, load_weights
from core import CandyEye

# ---------------------------------------------------------------------------
# Class names.  nc=20 -> PASCAL VOC order; nc=80 -> COCO order (as in the
# official ultralytics names file).  Used only for drawing labels.
# ---------------------------------------------------------------------------
VOC_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def letterbox(im_bgr: np.ndarray, size: int = 128, color=114):
    """Pad (grey 114) an image to a square `size` while keeping aspect ratio.

    Uses the standard aspect-ratio-preserving letterbox: smallest scale factor to fit, then
    centred padding.  Returns the new image and the (scale, pad-left/top)
    metadata needed to map detections back to the ORIGINAL image coordinates.
    """
    h, w = im_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    out = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top: top + nh, left: left + nw] = cv2.resize(
        im_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return out, {"scale": scale, "pad": (left, top)}


def predict(
    model: CandyEye,
    im_bgr: np.ndarray,
    size: int = 128,
    conf: float = 0.35,
    iou_thr: float = 0.45,
    max_det: int = 100,
):
    """Boxes+labels for one BGR image.  Returns list of (cls_id, score, xyxy)."""
    padded, meta = letterbox(im_bgr, size)
    # BGR -> RGB -> CHW -> float 0..1 (same channel order as VOC training)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).unsqueeze(0)

    model.eval()  # BatchNorm -> running stats; Detect -> decoded _inference
    with torch.no_grad():
        preds = model(img)                # (1, 4 + nc, total_anchors)

    out = preds.squeeze(0).t()            # (A, 4 + nc)
    boxes = out[:, :4]                    # cxcywh in letterboxed pixels
    scores, cls_ids = out[:, 4:].max(dim=1)

    keep = scores >= conf
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    if boxes.numel() == 0:
        return []

    cx, cy, w, h = boxes.unbind(1)
    xyxy = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 1)
    # Suppress boxes within a class while retaining overlapping objects from
    # different classes, as expected by multi-class detection evaluation.
    sel = batched_nms(xyxy, scores, cls_ids, iou_thr)[: max_det]

    scale, (dx, dy) = meta["scale"], meta["pad"]
    mapped = (xyxy[sel] - torch.tensor([dx, dy, dx, dy])) / scale  # original px
    dets = [
        (int(cls_ids[i].item()), float(scores[i].item()), mapped[i].tolist())
        for i in range(len(sel))
    ]
    return dets


def draw(im_bgr: np.ndarray, dets, names, palette=None) -> np.ndarray:
    """Draw boxes + "class score" labels onto a copy of the image."""
    out = im_bgr.copy()
    width = max(1, round(im_bgr.shape[1] / 400))
    for i, (cls_id, score, (x1, y1, x2, y2)) in enumerate(dets):
        color = palette[cls_id % len(palette)] if palette else (0, 255, 0)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, width)
        label = f"{names[cls_id]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (int(x1), int(y1 - th - 4)), (int(x1) + tw, int(y1)),
                      color, -1)
        cv2.putText(out, label, (int(x1), int(y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--nc", type=int, default=20, choices=(20, 80))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cfg", default="configs/yolo11.yaml")
    parser.add_argument("--weights", default="weights/yolo11n.pth")
    parser.add_argument("--noprint", action="store_true")
    args = parser.parse_args()

    model = CandyEye(args.cfg, nc=args.nc, img_size=args.size)
    load_weights(model, load_official_state_dict(args.weights), verbose=True)

    im = cv2.imread(str(args.image))
    if im is None:
        raise SystemExit(f"cannot read image: {args.image}")
    names = VOC_NAMES if args.nc == 20 else COCO_NAMES
    dets = predict(model, im, size=args.size, conf=args.conf, iou_thr=args.iou)

    out_path = args.out or (args.image.parent / "predict.jpg")
    cv2.imwrite(str(out_path), draw(im, dets, names))
    print(f"saved: {out_path}  ({len(dets)} detections)")
    if not args.noprint:
        for cls_id, score, xyxy in dets[:10]:
            print(f"  {names[cls_id]:14s} {score:.3f} {[round(v,1) for v in xyxy]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
