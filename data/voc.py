from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch


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
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

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

    def __getitem__(self, idx: int) -> dict:
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

        # Optional transform
        if self.transform is not None:
            image, boxes, labels, difficult = self.transform(
                image,
                boxes,
                labels,
                difficult,
            )

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "difficult": difficult,
            "img_id": img_id,
        }