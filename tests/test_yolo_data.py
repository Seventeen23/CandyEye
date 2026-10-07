"""Roboflow-style YAML and normalized-label dataset tests."""
import cv2
import numpy as np
import pytest
from data.yolo import YoloTxtDataset
from data.voc import collate_fn
from training.trainer import _resolve_data


def _write_image_and_label(root, name="sample"):
    images = root / "train" / "images"
    labels = root / "train" / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images / f"{name}.jpg"), np.zeros((10, 20, 3), np.uint8))
    (labels / f"{name}.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def test_yolo_txt_labels_map_to_letterboxed_pixel_boxes(tmp_path):
    _write_image_and_label(tmp_path)
    ds = YoloTxtDataset(tmp_path / "train" / "images", img_size=32)

    sample = ds[0]
    batch = collate_fn([sample])

    assert sample["image"].shape == (32, 32, 3)
    np.testing.assert_allclose(sample["boxes"], [[8, 12, 24, 20]])
    assert sample["labels"].tolist() == [1]
    np.testing.assert_allclose(batch["targets"].numpy(), [[0, 1, .5, .5, .5, .25]])


def test_roboflow_yaml_resolves_paths_and_names(tmp_path):
    _write_image_and_label(tmp_path)
    (tmp_path / "valid" / "images").mkdir(parents=True)
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(
        "train: train/images\nvalid: valid/images\nnc: 2\nnames: [cat, dog]\n",
        encoding="utf-8",
    )

    config = _resolve_data(yaml_path)

    assert config["format"] == "yolo_txt"
    assert config["nc"] == 2
    assert config["names"] == ["cat", "dog"]
    assert config["train_images"] == str(tmp_path / "train" / "images")
    assert config["val_images"] == str(tmp_path / "valid" / "images")


def test_yolo_txt_rejects_class_outside_dataset_range(tmp_path):
    _write_image_and_label(tmp_path)
    ds = YoloTxtDataset(tmp_path / "train" / "images", num_classes=1)
    with pytest.raises(ValueError, match="outside configured range"):
        ds[0]
