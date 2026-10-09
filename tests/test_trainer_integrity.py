"""Trainer integrity: log schema, output-dir safety, validation failures."""
import csv

import cv2
import numpy as np
import pytest

import training.trainer as trainer


def _tiny_config(tmp_path, *, names=("a", "b"), nc=2):
    train_images = tmp_path / "ds" / "train" / "images"
    train_labels = tmp_path / "ds" / "train" / "labels"
    val_images = tmp_path / "ds" / "val" / "images"
    val_labels = tmp_path / "ds" / "val" / "labels"
    for images, labels in ((train_images, train_labels), (val_images, val_labels)):
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
    for directory, count in ((train_images, 2), (val_images, 1)):
        for index in range(count):
            cv2.imwrite(str(directory / f"img{index}.jpg"),
                        np.zeros((32, 32, 3), np.uint8))
            label_file = (train_labels if directory is train_images
                          else val_labels) / f"img{index}.txt"
            label_file.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return {
        "name": "integrity",
        "model": {"type": "yolo", "cfg": "configs/yolo11.yaml",
                  "nc": nc, "img_size": 32},
        "train": {"epochs": 1, "output": str(tmp_path / "run"), "seed": 7,
                  "warmup_epochs": 1},
        "data": {"format": "yolo_txt",
                 "train_images": str(train_images),
                 "val_images": str(val_images),
                 "names": list(names), "nc": nc,
                 "batch_size": 2, "workers": 0},
    }


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.reader(stream))


def test_max_batches_zero_is_rejected(tmp_path):
    config = _tiny_config(tmp_path)

    with pytest.raises(ValueError, match="max_batches"):
        trainer.run_training(config, epochs=1, max_batches=0, threads=1)


def test_model_class_count_mismatch_raises_before_training(tmp_path):
    config = _tiny_config(tmp_path, names=("a", "b"), nc=3)

    with pytest.raises(ValueError, match="class names"):
        trainer.run_training(config, epochs=1, threads=1)


def test_resume_rotates_old_schema_metrics_log(tmp_path):
    config = _tiny_config(tmp_path)
    run_dir = trainer.run_training(config, epochs=1, threads=1)
    last = run_dir / "last.pt"
    assert last.exists()

    # Simulate a log written by the pre-fix 7-column schema.
    metrics = run_dir / "metrics.csv"
    metrics.write_text(
        "epoch,lr,loss,box,cls,dfl,foreground\n1,0.001,10,1,2,3,0.5\n",
        encoding="utf-8",
    )

    trainer.run_training(config, epochs=2, resume=str(last), threads=1)

    rotated = list(run_dir.glob("metrics_prev_*.csv"))
    assert len(rotated) == 1
    assert _rows(rotated[0])[0] == ["epoch", "lr", "loss", "box", "cls",
                                    "dfl", "foreground"]
    rows = _rows(metrics)
    assert rows[0][0:3] == ["epoch", "lr", "train_loss"]
    assert len(rows) == 2          # header + the resumed epoch
    assert rows[1][0] == "2"
    # The mixed-schema log must still be plottable.
    assert trainer._save_training_plots(metrics, run_dir) is not None


def test_plots_skip_gracefully_on_old_schema(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "epoch,lr,loss,box,cls,dfl,foreground\n1,0.001,10,1,2,3,0.5\n",
        encoding="utf-8",
    )

    assert trainer._save_training_plots(metrics, tmp_path) is None


def test_second_run_renames_directory_instead_of_truncating_log(tmp_path):
    config = _tiny_config(tmp_path)
    first = trainer.run_training(config, epochs=1, threads=1)
    original_rows = _rows(first / "metrics.csv")

    second = trainer.run_training(config, epochs=1, threads=1)

    assert second != first
    assert second.name == first.name + "2"
    assert _rows(first / "metrics.csv") == original_rows
    assert (second / "metrics.csv").exists()


def test_epoch_failure_still_writes_end_of_training_artifacts(tmp_path,
                                                              monkeypatch):
    config = _tiny_config(tmp_path)
    real = trainer.evaluate_detector
    calls = {"count": 0}

    def fail_second_validation(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer, "evaluate_detector", fail_second_validation)

    with pytest.raises(RuntimeError, match="simulated crash"):
        trainer.run_training(config, epochs=3, threads=1)

        run_dir = Path(config["train"]["output"])
        assert (run_dir / "best.pt").exists()
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "confusion_matrix.csv").exists()
