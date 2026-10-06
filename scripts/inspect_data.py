"""Visualize 4 random VOC samples with their ground-truth boxes."""

import random
from pathlib import Path

import cv2
import numpy as np

from data.voc import VOCDataset, VOC_CLASSES

ROOT = "data/VOCdevkit"
OUT = Path("runs/inspect_data.jpg")
TILE_W, TILE_H = 400, 300


def main() -> None:
    train = VOCDataset(ROOT, "trainval")
    test = VOCDataset(ROOT, "test")
    print(f"trainval: {len(train)}  test: {len(test)}")  # expect 5011 / 4952

    # Seeded so the same 4 images show up every run
    indices = random.Random(0).sample(range(len(train)), 4)

    # One distinct BGR color per class (OpenCV hue range is 0-179)
    colors = [
        tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[i * 180 // len(VOC_CLASSES), 255, 255]]]),
            cv2.COLOR_HSV2BGR)[0, 0])
        for i in range(len(VOC_CLASSES))
    ]

    tiles = []
    for idx in indices:
        s = train[idx]
        # Dataset returns RGB; cv2 draws in BGR -> convert first
        img = cv2.cvtColor(s["image"], cv2.COLOR_RGB2BGR)

        for box, cls in zip(s["boxes"], s["labels"]):
            x1, y1, x2, y2 = map(int, box)
            color = colors[int(cls)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, VOC_CLASSES[int(cls)], (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # All tiles must share a size to be stitched -> resize AFTER drawing
        # (so boxes scale together with the image)
        tiles.append(cv2.resize(img, (TILE_W, TILE_H)))

    grid = np.vstack((
        np.hstack((tiles[0], tiles[1])),
        np.hstack((tiles[2], tiles[3])),
    ))

    OUT.parent.mkdir(parents=True, exist_ok=True)  # imwrite fails silently if missing!
    ok = cv2.imwrite(str(OUT), grid)
    print(f"saved {OUT} ({OUT.stat().st_size} bytes)" if ok else f"FAILED: {OUT}")

    # Spot-check stats for the first sample
    s = train[indices[0]]
    names = [VOC_CLASSES[int(c)] for c in s["labels"]]
    print(f"\nsample {s['img_id']}: shape={s['image'].shape}, "
          f"{len(names)} boxes, difficult={s['difficult'].tolist()}")
    print("labels:", names)


if __name__ == "__main__":
    main()