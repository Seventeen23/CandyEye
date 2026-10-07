"""Tests for image transforms, VOC mosaic, and training batch packing."""
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch

from data.transforms import RandomHorizontalFlip, letterbox
from data.voc import VOCDataset, collate_fn


def test_letterbox_resizes_and_offsets_xyxy_boxes():
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    boxes = np.array([[1, .5, 3, 1.5]], dtype=np.float32)

    out, mapped = letterbox(image, boxes, size=8, color=114)

    assert out.shape == (8, 8, 3)
    assert np.all(out[:2] == 114)
    np.testing.assert_allclose(mapped, [[2, 3, 6, 5]])


def test_horizontal_flip_updates_box_coordinates(monkeypatch):
    monkeypatch.setattr("data.transforms.random.random", lambda: 0.0)
    image = np.arange(2 * 10 * 3, dtype=np.uint8).reshape(2, 10, 3)
    boxes = np.array([[1, 0, 5, 2]], dtype=np.float32)
    labels = np.array([3], dtype=np.int64)
    difficult = np.array([False])

    flipped, mapped, out_labels, out_difficult = RandomHorizontalFlip(0.5)(
        image, boxes, labels, difficult)

    np.testing.assert_array_equal(flipped, image[:, ::-1])
    np.testing.assert_array_equal(mapped, [[5, 0, 9, 2]])
    np.testing.assert_array_equal(out_labels, labels)
    np.testing.assert_array_equal(out_difficult, difficult)


def test_collate_stacks_images_and_normalizes_targets():
    batch = [
        {"image": np.zeros((8, 8, 3), np.uint8),
         "boxes": np.array([[2, 2, 6, 6]], np.float32),
         "labels": np.array([4]), "img_id": "a"},
        {"image": np.full((8, 8, 3), 255, np.uint8),
         "boxes": np.zeros((0, 4), np.float32),
         "labels": np.zeros((0,), np.int64), "img_id": "b"},
    ]

    packed = collate_fn(batch)

    assert packed["images"].shape == (2, 3, 8, 8)
    assert packed["images"].dtype == torch.float32
    assert packed["images"].min() == 0 and packed["images"].max() == 1
    assert packed["img_ids"] == ["a", "b"]
    torch.testing.assert_close(
        packed["targets"], torch.tensor([[0, 4, .5, .5, .5, .5]])
    )


def _write_voc_fixture(root):
    voc = root / "VOC2007"
    (voc / "JPEGImages").mkdir(parents=True)
    (voc / "Annotations").mkdir()
    (voc / "ImageSets" / "Main").mkdir(parents=True)
    ids = ["one", "two", "three", "four"]
    (voc / "ImageSets" / "Main" / "trainval.txt").write_text(
        "\n".join(ids) + "\n", encoding="utf-8")
    for img_id in ids:
        cv2.imwrite(str(voc / "JPEGImages" / f"{img_id}.jpg"),
                    np.zeros((8, 8, 3), dtype=np.uint8))
        annotation = ET.Element("annotation")
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "cat"
        ET.SubElement(obj, "difficult").text = "0"
        box = ET.SubElement(obj, "bndbox")
        for name, value in (("xmin", 1), ("ymin", 1),
                            ("xmax", 7), ("ymax", 7)):
            ET.SubElement(box, name).text = str(value)
        ET.ElementTree(annotation).write(voc / "Annotations" / f"{img_id}.xml")


def test_voc_dataset_mosaic_returns_fixed_size_sample(tmp_path, monkeypatch):
    _write_voc_fixture(tmp_path)
    # Ensure the mosaic branch is taken; sampled source IDs need not be fixed.
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    sample = VOCDataset(tmp_path, "trainval", img_size=16,
                        mosaic_probability=1.0)[0]

    assert sample["image"].shape == (16, 16, 3)
    assert sample["boxes"].shape == (4, 4)
    assert sample["labels"].shape == (4,)
    assert sample["difficult"].shape == (4,)
    assert np.all(sample["boxes"] >= 0)
    assert np.all(sample["boxes"] <= 16)
    packed = collate_fn([sample])
    assert packed["images"].shape == (1, 3, 16, 16)
    assert packed["targets"].shape == (4, 6)
