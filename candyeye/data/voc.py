from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch

from candyeye.data.transforms import letterbox


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

CLASS_TO_ID = {name: i for i, name in enumerate(VOC_CLASSES)}


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
        img_size: int = 128,
        mosaic_probability: float = 0.0,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.mosaic_probability = mosaic_probability

        self.voc_dir = self.root / "VOC2007"

        image_sets_dir = self.voc_dir / "ImageSets" / "Main"
        split_file = image_sets_dir / f"{split}.txt"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        # Keep only the image IDs, ignoring any additional columns.
        with split_file.open("r", encoding="utf-8") as f:
            self.ids = [
                line.strip().split()[0]
                for line in f
                if line.strip()
            ]

        self.annotations_dir = self.voc_dir / "Annotations"
        self.images_dir = self.voc_dir / "JPEGImages"

    def __len__(self) -> int:
        return len(self.ids)

    def _load_sample(self, idx: int) -> dict:
        img_id = self.ids[idx]

        image_path = self.images_dir / f"{img_id}.jpg"
        annotation_path = self.annotations_dir / f"{img_id}.xml"

        # Load image
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        # Parse XML
        root = ET.parse(annotation_path).getroot()

        boxes = []
        labels = []
        difficult = []

        for obj in root.findall("object"):
            class_name = obj.findtext("name")

            if class_name not in CLASS_TO_ID:
                raise ValueError(
                    f"Unknown VOC class '{class_name}' "
                    f"in {annotation_path}"
                )

            # VOC XML stores difficult as 0/1.
            difficult_value = int(
                obj.findtext("difficult", default="0")
            )

            bndbox = obj.find("bndbox")

            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))

            # Clamp coordinates to image bounds
            xmin = np.clip(xmin, 0, width)
            ymin = np.clip(ymin, 0, height)
            xmax = np.clip(xmax, 0, width)
            ymax = np.clip(ymax, 0, height)

            # Drop degenerate boxes
            box_width = xmax - xmin
            box_height = ymax - ymin

            if box_width <= 0 or box_height <= 0:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_ID[class_name])
            difficult.append(bool(difficult_value))

        # Convert to required NumPy dtypes/shapes
        boxes = np.asarray(
            boxes,
            dtype=np.float32,
        ).reshape(-1, 4)

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        difficult = np.asarray(
            difficult,
            dtype=np.bool_,
        )

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "difficult": difficult,
            "img_id": img_id,
        }

    def __getitem__(self, idx: int) -> dict:
        if self.split == "trainval" and self.mosaic_probability > 0 and \
                np.random.random() < self.mosaic_probability:
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
        """Compose four letterboxed images into a fixed-size 2x2 mosaic."""
        half = self.img_size // 2
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        all_boxes, all_labels, all_difficult = [], [], []
        for i, sample in enumerate(samples):
            image, boxes = letterbox(sample["image"], sample["boxes"], half)
            xoff, yoff = (i % 2) * half, (i // 2) * half
            canvas[yoff:yoff + half, xoff:xoff + half] = image
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] += xoff
                boxes[:, [1, 3]] += yoff
                all_boxes.append(boxes)
                all_labels.append(sample["labels"])
                all_difficult.append(sample["difficult"])
        return (canvas,
                np.concatenate(all_boxes).astype(np.float32) if all_boxes else np.zeros((0, 4), np.float32),
                np.concatenate(all_labels) if all_labels else np.zeros((0,), np.int64),
                np.concatenate(all_difficult) if all_difficult else np.zeros((0,), np.bool_))


def collate_fn(batch: list[dict]) -> dict:
    """Stack images and pack targets as ``(batch_idx, cls, cx, cy, w, h)``."""
    images = torch.stack([
        torch.from_numpy(np.ascontiguousarray(s["image"]))
        .permute(2, 0, 1).float().div_(255.0) for s in batch
    ])
    targets = []
    for batch_idx, sample in enumerate(batch):
        boxes = torch.as_tensor(sample["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(sample["labels"], dtype=torch.float32).reshape(-1, 1)
        if len(boxes):
            x1, y1, x2, y2 = boxes.unbind(1)
            xywh = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2,
                                x2 - x1, y2 - y1), dim=1) / images.shape[-1]
            bi = torch.full((len(boxes), 1), batch_idx, dtype=torch.float32)
            targets.append(torch.cat((bi, labels, xywh), dim=1))
    targets = torch.cat(targets) if targets else torch.zeros((0, 6), dtype=torch.float32)
    difficult = [
        torch.as_tensor(
            s.get("difficult", np.zeros(len(s["labels"]), dtype=np.bool_)),
            dtype=torch.bool,
        ).reshape(-1)
        for s in batch
    ]
    return {"images": images, "targets": targets,
            "img_ids": [s["img_id"] for s in batch],
            "difficult": difficult}
