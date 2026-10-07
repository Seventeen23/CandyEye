"""Image and box transforms used by the VOC training pipeline."""
from __future__ import annotations

import random

import cv2
import numpy as np


def letterbox(image: np.ndarray, boxes: np.ndarray, size: int = 128,
              color: int = 114):
    """Resize RGB image to a square and pad; return transformed xyxy boxes."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(
        image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    boxes = boxes.astype(np.float32, copy=True)
    if len(boxes):
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (nw / w) + left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (nh / h) + top
    return out, boxes


class HSVJitter:
    """Apply random hue, saturation, and value gains to an RGB image."""
    def __init__(self, h: float = .015, s: float = .7, v: float = .4):
        self.h, self.s, self.v = h, s, v

    def __call__(self, image, boxes, labels, difficult):
        gains = np.random.uniform(-1, 1, 3) * [self.h, self.s, self.v] + 1
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] * gains[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return image, boxes, labels, difficult


class RandomHorizontalFlip:
    def __init__(self, probability: float = .5):
        self.probability = probability

    def __call__(self, image, boxes, labels, difficult):
        if random.random() < self.probability:
            image = np.ascontiguousarray(image[:, ::-1])
            boxes = boxes.copy()
            if len(boxes):
                width = image.shape[1]
                x1 = boxes[:, 0].copy()
                boxes[:, 0] = width - boxes[:, 2]
                boxes[:, 2] = width - x1
        return image, boxes, labels, difficult


class Compose:
    def __init__(self, transforms):
        self.transforms = tuple(transforms)

    def __call__(self, image, boxes, labels, difficult):
        for transform in self.transforms:
            image, boxes, labels, difficult = transform(
                image, boxes, labels, difficult)
        return image, boxes, labels, difficult
