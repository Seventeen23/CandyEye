"""Render a training batch and its ground-truth boxes to runs/batch.jpg."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from candyeye.data.transforms import Compose, HSVJitter, RandomHorizontalFlip
from candyeye.data.voc import VOCDataset, VOC_CLASSES, collate_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/VOCdevkit")
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("runs/batch.jpg"))
    args = parser.parse_args()

    aug = Compose([HSVJitter(), RandomHorizontalFlip()]) if args.split == "trainval" else None
    ds = VOCDataset(args.root, args.split, transform=aug, img_size=args.size,
                    mosaic_probability=args.mosaic if args.split == "trainval" else 0)
    batch = next(iter(DataLoader(ds, batch_size=args.batch_size,
                                 num_workers=args.workers, collate_fn=collate_fn)))
    images, targets = batch["images"], batch["targets"]
    tiles = []
    for i, tensor in enumerate(images):
        rgb = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for row in targets[targets[:, 0] == i]:
            cls = int(row[1])
            cx, cy, w, h = (row[2:] * args.size).tolist()
            x1, y1 = round(cx - w / 2), round(cy - h / 2)
            x2, y2 = round(cx + w / 2), round(cy + h / 2)
            color = tuple(int(v) for v in cv2.cvtColor(
                np.uint8([[[cls * 180 // len(VOC_CLASSES), 255, 255]]]),
                cv2.COLOR_HSV2BGR)[0, 0])
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 1)
            cv2.putText(bgr, VOC_CLASSES[cls], (x1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1, cv2.LINE_AA)
        tiles.append(cv2.resize(bgr, (args.size * 3, args.size * 3),
                                interpolation=cv2.INTER_NEAREST))
    cols = min(2, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    blank = np.full_like(tiles[0], 114)
    tiles.extend([blank] * (rows * cols - len(tiles)))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), grid):
        raise OSError(f"failed to write {args.out}")
    print(f"batch images: {tuple(images.shape)} | targets: {tuple(targets.shape)}")
    print(f"saved {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
