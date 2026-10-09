"""Dataset reader for image folders with normalized YOLO ``.txt`` labels."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from data.transforms import letterbox


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_paths(source: str | Path | list[str | Path]) -> list[Path]:
    """Expand an image directory, image-list file, or explicit path list."""
    if isinstance(source, (list, tuple)):
        paths = [Path(p) for p in source]
    else:
        path = Path(source)
        if path.is_dir():
            paths = sorted(p for p in path.rglob("*")
                           if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() == ".txt":
            paths = []
            for line in path.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if value:
                    item = Path(value)
                    paths.append(item if item.is_absolute() else path.parent / item)
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            paths = [path]
        else:
            raise ValueError(f"not an image directory/list: {source}")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"image paths not found (first): {missing[0]}")
    if not paths:
        raise ValueError(f"no supported images found in {source}")
    return paths


def label_path(image_path: Path) -> Path:
    """Map an image path to its YOLO label file across common layouts.

    Tries ``.../images/foo.jpg`` → ``.../labels/foo.txt``, a sibling ``labels``
    directory next to the parent, and a ``labels`` directory inside the parent.
    Returns the first candidate that exists, otherwise the primary mapping so
    callers can treat the image as background.
    """
    parts = list(image_path.parts)
    candidates = []
    image_dirs = [i for i, part in enumerate(parts[:-1]) if part.lower() == "images"]
    if image_dirs:
        replaced = list(parts)
        replaced[image_dirs[-1]] = "labels"
        candidates.append(Path(*replaced).with_suffix(".txt"))
    candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
    candidates.append(image_path.parent / "labels" / f"{image_path.stem}.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


class YoloTxtDataset(torch.utils.data.Dataset):
    """Read images plus YOLO class/cx/cy/w/h text labels.

    Missing label files are treated as background images. Labels are returned
    in the same pixel-space xyxy sample contract as ``VOCDataset``.
    """
    def __init__(self, images, img_size: int = 128,
                 transform: Callable | None = None,
                 mosaic_probability: float = 0.0,
                 num_classes: int | None = None):
        self.images = image_paths(images)
        self.img_size = img_size
        self.transform = transform
        self.mosaic_probability = mosaic_probability
        self.num_classes = num_classes
        # A wrong images→labels mapping would silently turn the whole dataset
        # into background images; fail loudly instead.
        found = sum(1 for path in self.images if label_path(path).is_file())
        if self.images and found == 0:
            raise ValueError(
                f"no label files found for {len(self.images)} images "
                f"(e.g. expected {label_path(self.images[0])}); "
                "check the dataset's images/labels layout"
            )
        if found < len(self.images):
            print(f"warning: {len(self.images) - found}/{len(self.images)} "
                  "images have no label file (treated as background)",
                  flush=True)

    def __len__(self):
        return len(self.images)

    def _load_sample(self, idx: int) -> dict:
        path = self.images[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise OSError(f"could not read image: {path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        boxes, labels = [], []
        annotation = label_path(path)
        if annotation.is_file():
            for line_no, line in enumerate(annotation.read_text(encoding="utf-8").splitlines(), 1):
                values = line.split()
                if not values:
                    continue
                if len(values) != 5:
                    raise ValueError(f"{annotation}:{line_no}: expected class cx cy w h")
                cls, cx, cy, bw, bh = map(float, values)
                if cls < 0 or not cls.is_integer():
                    raise ValueError(f"{annotation}:{line_no}: class id must be a nonnegative integer")
                if self.num_classes is not None and cls >= self.num_classes:
                    raise ValueError(
                        f"{annotation}:{line_no}: class id {int(cls)} outside "
                        f"configured range 0..{self.num_classes - 1}"
                    )
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                x1, x2 = np.clip([x1, x2], 0, w)
                y1, y2 = np.clip([y1, y2], 0, h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls))
        return {"image": image,
                "boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
                "labels": np.asarray(labels, np.int64),
                "difficult": np.zeros(len(labels), dtype=np.bool_),
                "img_id": str(path)}

    def __getitem__(self, idx: int) -> dict:
        if self.mosaic_probability > 0 and np.random.random() < self.mosaic_probability:
            indices = [idx] + np.random.randint(0, len(self), size=3).tolist()
            samples = [self._load_sample(i) for i in indices]
            image, boxes, labels, difficult = self._mosaic(samples)
            sample = {"image": image, "boxes": boxes, "labels": labels,
                      "difficult": difficult, "img_id": samples[0]["img_id"]}
        else:
            sample = self._load_sample(idx)
            sample["image"], sample["boxes"] = letterbox(
                sample["image"], sample["boxes"], self.img_size)
        if self.transform is not None:
            sample["image"], sample["boxes"], sample["labels"], sample["difficult"] = \
                self.transform(sample["image"], sample["boxes"],
                               sample["labels"], sample["difficult"])
        return sample

    def _mosaic(self, samples: list[dict]):
        half = self.img_size // 2
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        boxes_all, labels_all = [], []
        for i, sample in enumerate(samples):
            image, boxes = letterbox(sample["image"], sample["boxes"], half)
            dx, dy = (i % 2) * half, (i // 2) * half
            canvas[dy:dy + half, dx:dx + half] = image
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] += dx
                boxes[:, [1, 3]] += dy
                boxes_all.append(boxes)
                labels_all.append(sample["labels"])
        boxes = np.concatenate(boxes_all).astype(np.float32) if boxes_all else np.zeros((0, 4), np.float32)
        labels = np.concatenate(labels_all) if labels_all else np.zeros((0,), np.int64)
        return canvas, boxes, labels, np.zeros(len(labels), dtype=np.bool_)
