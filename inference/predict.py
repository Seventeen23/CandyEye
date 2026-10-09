"""Run CandyEye on one image or a video and save the annotated result."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
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
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
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
    scale, (dx, dy) = meta["scale"], meta["pad"]
    mapped = (xyxy - torch.tensor([dx, dy, dx, dy])) / scale  # original px
    mapped[:, [0, 2]] = mapped[:, [0, 2]].clamp(0, im_bgr.shape[1])
    mapped[:, [1, 3]] = mapped[:, [1, 3]].clamp(0, im_bgr.shape[0])
    valid = (mapped[:, 2] > mapped[:, 0]) & (mapped[:, 3] > mapped[:, 1])
    mapped, scores, cls_ids = mapped[valid], scores[valid], cls_ids[valid]
    if not mapped.numel():
        return []
    sel = batched_nms(mapped, scores, cls_ids, iou_thr)[:max_det]
    dets = [
        (int(cls_ids[i].item()), float(scores[i].item()), mapped[i].tolist())
        for i in sel.tolist()
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


class CandyEyePredictor:
    """Load a CandyEye checkpoint and predict on image or video files.

    Example::

        predictor = CandyEyePredictor("runs/train/exp/best.pt", data="dataset.yaml")
        result = predictor.predict("image.jpg")
    """

    VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    def __init__(self, weights: str | Path, *, data: str | Path | None = None,
                 cfg: str | Path = "configs/yolo11.yaml", imgsz: int | None = None,
                 nc: int | None = None, output_dir: str | Path = "runs/predict"):
        self.weights = Path(weights)
        self.output_dir = Path(output_dir)
        if self.weights.suffix.lower() == ".pt":
            payload = torch.load(self.weights, map_location="cpu", weights_only=False)
            state = payload.get("model", payload) if isinstance(payload, dict) else payload
            saved_config = payload.get("config", {}) if isinstance(payload, dict) else {}
        else:
            state, saved_config = None, {}
        # Predictions must letterbox at the size the checkpoint was trained
        # with unless the caller explicitly overrides it.
        self.imgsz = int(imgsz if imgsz is not None
                         else saved_config.get("model", {}).get("img_size", 128))

        data_config = {}
        if data is not None:
            with Path(data).open(encoding="utf-8") as stream:
                data_config = yaml.safe_load(stream) or {}
        data_section = data_config.get("data", data_config)
        saved_data = saved_config.get("data", {})
        saved_model = saved_config.get("model", {})
        names = data_section.get("names") or saved_data.get("names")
        if isinstance(names, dict):
            names = [names[key] for key in sorted(names, key=lambda item: int(item))]
        self.nc = int(nc or data_section.get("nc") or saved_model.get("nc")
                      or (len(names) if names else 20))
        if names is None:
            names = VOC_NAMES if self.nc == 20 else (
                COCO_NAMES if self.nc == 80 else [str(i) for i in range(self.nc)])
        if len(names) != self.nc:
            raise ValueError(f"class names has {len(names)} entries but model nc={self.nc}")
        data_nc = data_section.get("nc")
        if data_nc is not None and int(data_nc) != self.nc:
            raise ValueError(f"dataset YAML has nc={data_nc}, but model has nc={self.nc}")
        self.names = names

        self.model = CandyEye(cfg, nc=self.nc, img_size=self.imgsz)
        if state is not None:
            self.model.load_state_dict(state)
        else:
            load_weights(self.model, load_official_state_dict(str(self.weights)))
        self.model.eval()

    def predict_image(self, source: str | Path | np.ndarray, *, output=None,
                      conf: float = 0.25, iou: float = 0.45,
                      max_det: int = 100) -> dict:
        """Predict one BGR image; save an annotated image and return detections."""
        if isinstance(source, np.ndarray):
            image = source
            stem = "image"
        else:
            source = Path(source)
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"could not read image: {source}")
            stem = source.stem
        detections = predict(self.model, image, self.imgsz, conf, iou, max_det)
        output_path = Path(output) if output else self.output_dir / f"{stem}_pred.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), draw(image, detections, self.names)):
            raise OSError(f"could not save annotated image: {output_path}")
        return {"output": output_path, "detections": detections}

    def predict_video(self, source: str | Path, *, output=None,
                      conf: float = 0.25, iou: float = 0.45,
                      max_det: int = 100) -> dict:
        """Predict every frame of a video and save an annotated MP4."""
        source = Path(source)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open video: {source}")
        output_path = Path(output) if output else self.output_dir / f"{source.stem}_pred.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise OSError(f"could not create output video: {output_path}")
        frame_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = predict(self.model, frame, self.imgsz, conf, iou, max_det)
                writer.write(draw(frame, detections, self.names))
                frame_count += 1
        finally:
            capture.release()
            writer.release()
        return {"output": output_path, "frames": frame_count}

    def predict(self, source: str | Path | np.ndarray, **kwargs) -> dict:
        """Predict an image array, image file, or video file and save the result."""
        if isinstance(source, np.ndarray):
            return self.predict_image(source, **kwargs)
        source = Path(source)
        if source.suffix.lower() in self.VIDEO_SUFFIXES:
            return self.predict_video(source, **kwargs)
        return self.predict_image(source, **kwargs)
