"""Roboflow-style YAML and normalized-label dataset tests."""
import cv2
import numpy as np
import pytest
import torch
from data.yolo import YoloTxtDataset, label_path
from data.voc import collate_fn
from training.trainer import _resolve_data
from core import CandyEye
import training.trainer as trainer


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


def test_train_reads_yaml_and_resizes_class_head_automatically(tmp_path, monkeypatch):
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(
        "train: train/images\nvalid: valid/images\nnc: 2\nnames: [cat, dog]\n",
        encoding="utf-8",
    )
    model = CandyEye("configs/yolo11.yaml", img_size=64)
    assert model.nc == 20

    monkeypatch.setattr(trainer, "run_training", lambda _config, **kwargs: tmp_path / "run")
    results = trainer.train_model(
        model, data=yaml_path, epochs=1, imgsz=64, project=tmp_path, name="run"
    )

    assert model.nc == 2
    assert model.model[-1].nc == 2
    assert [feature.shape[1] for feature in model(torch.zeros(1, 3, 64, 64))] == [66] * 3
    assert results["best"] == tmp_path / "run" / "best.pt"


def test_label_path_falls_back_without_images_dir(tmp_path):
    train = tmp_path / "train"
    (train / "labels").mkdir(parents=True)
    image = train / "foo.jpg"
    image.write_bytes(b"")
    expected = train / "labels" / "foo.txt"
    expected.write_text("0 .5 .5 .1 .1\n", encoding="utf-8")

    # No "images" component: parent.parent/labels is wrong here.
    assert label_path(image) == expected


def test_label_path_prefers_images_to_labels_mapping(tmp_path):
    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image = images / "foo.jpg"
    image.write_bytes(b"")
    primary = labels / "foo.txt"
    primary.write_text("0 .5 .5 .1 .1\n", encoding="utf-8")
    # Decoy in the parent.parent layout must not win.
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "foo.txt").write_text("1 .5 .5 .1 .1\n", encoding="utf-8")

    assert label_path(image) == primary


def test_dataset_raises_when_no_labels_anywhere(tmp_path):
    images = tmp_path / "train" / "images"
    images.mkdir(parents=True)
    cv2.imwrite(str(images / "foo.jpg"), np.zeros((10, 10, 3), np.uint8))

    with pytest.raises(ValueError, match="no label files"):
        YoloTxtDataset(images, img_size=32)
