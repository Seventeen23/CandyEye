"""CPU trainer, CandyEye Python API, and experiment CLI."""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from core.backbone_mobilenet import MobileNetV3SmallDetector
from core.convert_yolo11 import load_official_state_dict, load_weights
from core import CandyEye
from data.transforms import Compose, HSVJitter, RandomHorizontalFlip
from data.yolo import YoloTxtDataset
from data.voc import VOC_CLASSES, VOCDataset, collate_fn
from eval.detection_metrics import evaluate_detector
from training.loss import DetectionLoss


def lr_factor(progress: float, epochs: int, warmup_epochs: float = 3.0,
              min_lr_ratio: float = .01, start_ratio: float = .1) -> float:
    """Linear warmup followed by cosine decay, evaluated in epoch units."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    progress = min(max(progress, 0.0), float(epochs))
    warmup = min(max(warmup_epochs, 0.0), float(epochs))
    if warmup == 0:
        phase = progress / epochs
        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * phase)) / 2
    if progress < warmup:
        return start_ratio + (1 - start_ratio) * progress / warmup
    if epochs <= warmup:
        return start_ratio + (1 - start_ratio) * progress / epochs
    phase = (progress - warmup) / (epochs - warmup)
    return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * phase)) / 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_experiment_model(config: dict, no_pretrained: bool = False):
    model_cfg = config["model"]
    init_cfg = config.get("initialization", {})
    model_type = model_cfg.get("type", "yolo")
    if model_type == "yolo":
        model = CandyEye(model_cfg.get("cfg", "configs/yolo11.yaml"),
                         nc=model_cfg.get("nc", 20),
                         img_size=model_cfg.get("img_size", 128))
        if init_cfg.get("type") == "yolo11n":
            path = Path(init_cfg.get("weights", "weights/yolo11n.pth"))
            if not path.exists():
                raise FileNotFoundError(f"pretrained YOLO11 source weights not found: {path}")
            report = load_weights(model, load_official_state_dict(str(path)))
            if report["unexpected"]:
                raise RuntimeError(f"unexpected pretrained keys: {report['unexpected'][:5]}")
            print(f"loaded {len(report['loaded'])} pretrained tensors; "
                  f"reinitialized {len(report['skipped'])} shape-mismatched tensors")
        return model
    if model_type == "mobilenetv3_small":
        pretrained = bool(init_cfg.get("pretrained", False)) and not no_pretrained
        return MobileNetV3SmallDetector(
            nc=model_cfg.get("nc", 20), img_size=model_cfg.get("img_size", 128),
            pretrained=pretrained)
    raise ValueError(f"unknown model type: {model_type}")


def run_training(config: dict, *, epochs: int | None = None,
                 max_batches: int | None = None, resume: str | None = None,
                 no_pretrained: bool = False, threads: int = 4,
                 model=None, patience: int | None = None) -> Path:
    train_cfg, data_cfg, model_cfg = config["train"], config["data"], config["model"]
    epochs = epochs or train_cfg["epochs"]
    if patience is None:
        patience = train_cfg.get("patience")
    seed_everything(int(train_cfg.get("seed", 23)))
    torch.set_num_threads(threads)
    device = torch.device("cpu")
    model = (model or build_experiment_model(
        config, no_pretrained=no_pretrained)).to(device)
    model_nc = int(getattr(model, "nc", model.model[-1].nc))

    augment = Compose(train_cfg.get(
        "api_transforms", [HSVJitter(), RandomHorizontalFlip()]))
    img_size = int(model_cfg.get("img_size", 128))
    mosaic_probability = float(data_cfg.get("mosaic_probability", 0.0))
    if data_cfg.get("format") == "yolo_txt":
        data_nc = int(data_cfg.get("nc", 0))
        if data_nc and data_nc != model_nc:
            raise ValueError(
                f"model has nc={model_nc}, but dataset YAML declares nc={data_nc}; "
                "construct CandyEye with matching nc"
            )
        dataset = YoloTxtDataset(
            data_cfg["train_images"], img_size=img_size, transform=augment,
            mosaic_probability=mosaic_probability,
            num_classes=data_cfg.get("nc") or None,
        )
        val_dataset = YoloTxtDataset(
            data_cfg["val_images"], img_size=img_size,
            num_classes=data_cfg.get("nc") or None,
        )
        class_names = data_cfg.get("names") or [str(i) for i in range(model_nc)]
    else:
        dataset = VOCDataset(
            data_cfg.get("root", "data/VOCdevkit"),
            data_cfg.get("train_split", "trainval"), transform=augment,
            img_size=img_size, mosaic_probability=mosaic_probability,
        )
        val_dataset = VOCDataset(
            data_cfg.get("root", "data/VOCdevkit"),
            data_cfg.get("val_split", "test"), img_size=img_size,
        )
        class_names = VOC_CLASSES
    loader = DataLoader(
        dataset, batch_size=int(data_cfg.get("batch_size", 16)), shuffle=True,
        num_workers=int(data_cfg.get("workers", 0)), collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(data_cfg.get("batch_size", 16)), shuffle=False,
        num_workers=int(data_cfg.get("workers", 0)), collate_fn=collate_fn,
        drop_last=False,
    )
    steps_per_epoch = min(len(loader), max_batches) if max_batches else len(loader)
    if steps_per_epoch <= 0:
        raise ValueError("training loader has no batches")

    criterion = DetectionLoss(model)
    base_lr = float(train_cfg.get("learning_rate", 2e-4))
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.SGD
    optimizer_kwargs = {"lr": base_lr,
                        "weight_decay": float(train_cfg.get("weight_decay", 5e-4))}
    if optimizer_name == "sgd":
        optimizer_kwargs["momentum"] = .9
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    warmup = float(train_cfg.get("warmup_epochs", 3))
    min_ratio = float(train_cfg.get("min_lr_ratio", .01))
    output = Path(train_cfg.get("output", f"runs/experiments/{config['name']}"))
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.csv"
    start_epoch = 0
    best_loss = float("inf")
    best_map50 = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        best_map50 = float(checkpoint.get("best_map50", best_map50))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = int(checkpoint.get("best_epoch", start_epoch))
        print(f"resumed {resume} at epoch {start_epoch}")

    write_header = not log_path.exists() or not resume
    with log_path.open("a" if resume else "w", newline="", encoding="utf-8") as csv_file:
        log = csv.writer(csv_file)
        if write_header:
            class_columns = [column for name in class_names
                             for column in (f"{name}/gt", f"{name}/precision",
                                            f"{name}/recall", f"{name}/f1",
                                            f"{name}/ap50")]
            log.writerow(["epoch", "lr", "train_loss", "train_box", "train_cls",
                          "train_dfl", "train_foreground", "val_loss", "val_box",
                          "val_cls", "val_dfl",
                          "val_foreground",
                          "precision_macro", "recall_macro", "f1_macro",
                          "precision_micro", "recall_micro", "f1_micro", "map50",
                          *class_columns])
        for epoch in range(start_epoch, epochs):
            model.train()
            sums = np.zeros(5, dtype=np.float64)
            for step, batch in enumerate(loader):
                if max_batches is not None and step >= max_batches:
                    break
                progress = epoch + (step + 1) / steps_per_epoch
                factor = lr_factor(progress, epochs, warmup, min_ratio)
                lr = base_lr * factor
                for group in optimizer.param_groups:
                    group["lr"] = lr
                images = batch["images"].to(device)
                targets = batch["targets"].to(device)
                optimizer.zero_grad(set_to_none=True)
                losses = criterion(model(images), targets)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, step {step}")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                sums[:4] += [losses[k].item() for k in ("loss", "box", "cls", "dfl")]
                sums[4] += losses["foreground"].item()

            means = sums / steps_per_epoch
            epoch_loss = float(means[0])
            best_loss = min(best_loss, epoch_loss)
            validation = evaluate_detector(
                model, val_loader, num_classes=int(model_nc), class_names=class_names,
                criterion=criterion,
            )
            val_loss = validation["val_loss"]
            map50 = validation["map50"]
            improved = (map50 > best_map50 + 1e-12 or
                        (abs(map50 - best_map50) <= 1e-12 and
                         val_loss["loss"] < best_val_loss))
            if improved:
                best_map50 = map50
                best_val_loss = val_loss["loss"]
                best_epoch = epoch + 1
            class_values = [
                field
                for value in validation["per_class"].values()
                for field in (
                    value["ground_truth"],
                    *(value[metric] if value[metric] is not None else "n/a"
                      for metric in ("precision", "recall", "f1", "ap50")),
                )
            ]
            log.writerow([
                epoch + 1, lr, *means[:4], means[4],
                val_loss["loss"], val_loss["box"], val_loss["cls"], val_loss["dfl"],
                val_loss["foreground"],
                validation["precision"], validation["recall"], validation["f1"],
                validation["micro_precision"], validation["micro_recall"],
                validation["micro_f1"], map50, *class_values,
            ])
            csv_file.flush()
            state = {"epoch": epoch + 1, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "best_loss": best_loss,
                     "best_map50": best_map50, "best_val_loss": best_val_loss,
                     "best_epoch": best_epoch,
                     "config": config}
            torch.save(state, output / "last.pt")
            if improved or not (output / "best.pt").exists():
                torch.save(state, output / "best.pt")
            print(
                f"epoch {epoch + 1}/{epochs} lr={lr:.2e} "
                f"train_loss={epoch_loss:.4f} train_box={means[1]:.4f} "
                f"train_cls={means[2]:.4f} train_dfl={means[3]:.4f} "
                f"val_loss={val_loss['loss']:.4f} val_box={val_loss['box']:.4f} "
                f"val_cls={val_loss['cls']:.4f} val_dfl={val_loss['dfl']:.4f} "
                f"val_fg={val_loss['foreground']:.1f} "
                f"P={validation['precision']:.3f} R={validation['recall']:.3f} "
                f"F1={validation['f1']:.3f} mAP50={map50:.3f}", flush=True
            )
            if patience is not None and patience > 0 and epoch + 1 - best_epoch >= patience:
                print(f"early stop: no validation mAP@0.5 improvement for {patience} epochs",
                      flush=True)
                break
    return output


def _resolve_data(data) -> dict:
    """Read the project VOC YAML or a Roboflow YOLO-format dataset YAML."""
    yaml_dir = Path.cwd()
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.is_dir():
            raw = {"root": str(path)}
        else:
            yaml_dir = path.resolve().parent
            with path.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
    elif isinstance(data, dict):
        raw = dict(data)
    else:
        raise TypeError("data must be a dataset YAML, VOC root path, or mapping")
    section = raw.get("data", raw)
    if "train" in section and ("val" in section or "valid" in section):
        data_root = Path(section.get("path", yaml_dir))
        if not data_root.is_absolute():
            data_root = (yaml_dir / data_root).resolve()

        def resolve_source(value):
            if isinstance(value, (list, tuple)):
                return [str((data_root / p).resolve()) if not Path(p).is_absolute()
                        else str(Path(p)) for p in value]
            value_path = Path(value)
            if value_path.is_absolute():
                return str(value_path)
            resolved = (data_root / value_path).resolve()
            if not resolved.exists():
                # Some exported YAMLs retain ../ split paths after data.yaml
                # has been placed alongside train/valid/test in its dataset.
                local_split = Path(*(part for part in value_path.parts
                                      if part not in ("..", ".")))
                local_resolved = (yaml_dir / local_split).resolve()
                if local_resolved.exists():
                    resolved = local_resolved
            return str(resolved)

        names = section.get("names", raw.get("names"))
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=lambda x: int(x))]
        nc = int(section.get("nc", raw.get("nc", len(names) if names else 0)))
        if names and nc != len(names):
            raise ValueError(f"dataset YAML declares nc={nc} but has {len(names)} names")
        val_source = section.get("val", section.get("valid"))
        return {"format": "yolo_txt", "train_images": resolve_source(section["train"]),
                "val_images": resolve_source(val_source),
                "test_images": resolve_source(section["test"]) if section.get("test") else None,
                "nc": nc, "names": names, "root": str(data_root),
                "mosaic_probability": 0.0}

    root = section.get("root", section.get("path", "data/VOCdevkit"))
    train_split = section.get("train_split", "trainval")
    val_split = section.get("val_split", "test")
    # Accept conventional VOC ImageSets split file declarations as well.
    if "train" in section and isinstance(section["train"], str):
        train_value = Path(section["train"])
        if train_value.suffix == ".txt":
            train_split = train_value.stem
    if "val" in section and isinstance(section["val"], str):
        val_value = Path(section["val"])
        if val_value.suffix == ".txt":
            val_split = val_value.stem
    return {"root": str(root), "train_split": train_split,
            "val_split": val_split, "mosaic_probability": 0.0}


def train_model(model, *, data, epochs: int = 100, imgsz: int = 128,
                batch: int = 16, patience: int = 50, workers: int = 0,
                device: str = "cpu", project: str | Path = "runs/train",
                name: str = "exp", resume: bool | str | Path = False,
                optimizer: str = "AdamW", lr0: float = 2e-4,
                weight_decay: float = 5e-4, warmup_epochs: float = 3,
                mosaic: float = .5, hsv: bool = True, fliplr: float = .5,
                pretrained: bool | str | Path = False, seed: int = 23,
                exist_ok: bool = False, max_batches: int | None = None,
                threads: int = 4) -> dict:
    """Train an existing CandyEye detector from Python.

    Each epoch reports train/validation losses and aggregate validation metrics;
    per-class metrics are stored in the CSV. `patience` monitors validation
    mAP@0.5. Training is CPU-only in this release.
    """
    if epochs <= 0 or imgsz <= 0 or imgsz % 32:
        raise ValueError("epochs must be positive and imgsz divisible by 32")
    if batch <= 0 or workers < 0:
        raise ValueError("batch must be positive and workers nonnegative")
    if device not in ("cpu", "auto"):
        raise ValueError("this trainer currently supports CPU only")
    optimizer_name = optimizer.lower()
    if optimizer_name not in ("adamw", "sgd"):
        raise ValueError("optimizer must be 'AdamW' or 'SGD'")

    data_cfg = _resolve_data(data)
    data_nc = int(data_cfg.get("nc", 0) or 0)
    if data_nc:
        model_nc = int(getattr(model, "nc", model.model[-1].nc))
        if data_nc != model_nc:
            if not hasattr(model, "set_classes"):
                raise ValueError(
                    f"model has nc={model_nc}, but dataset YAML declares nc={data_nc}"
                )
            model.set_classes(data_nc)
    data_cfg.update({"batch_size": batch, "workers": workers,
                     "mosaic_probability": mosaic})
    cfg_name = str(getattr(model, "yaml", "configs/yolo11.yaml"))
    nc = int(getattr(model, "nc", getattr(model.model[-1], "nc", 20)))
    config = {
        "name": name,
        "model": {"type": "yolo", "cfg": cfg_name, "nc": nc,
                  "img_size": imgsz},
        "initialization": {"type": "scratch"},
        "data": data_cfg,
        "train": {"epochs": epochs, "learning_rate": lr0,
                  "weight_decay": weight_decay, "warmup_epochs": warmup_epochs,
                  "min_lr_ratio": .01, "seed": seed},
    }
    output = Path(project) / name
    if isinstance(resume, (str, Path)):
        resume_path = Path(resume)
    elif resume:
        resume_path = output / "last.pt"
    else:
        resume_path = None
    if resume_path is None and output.exists() and not exist_ok:
        base, suffix = name, 2
        while (Path(project) / f"{base}{suffix}").exists():
            suffix += 1
        name = f"{base}{suffix}"
        output = Path(project) / name
        config["name"] = name
    if pretrained:
        weights_path = (Path(pretrained) if not isinstance(pretrained, bool)
                        else Path("weights/yolo11n.pth"))
        if not weights_path.exists():
            raise FileNotFoundError(f"pretrained weights not found: {weights_path}")
        report = load_weights(model, load_official_state_dict(str(weights_path)))
        print(f"loaded {len(report['loaded'])} matching pretrained tensors; "
              f"skipped {len(report['skipped'])}")
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

    # Update the loss-independent augmentation choices for this API invocation.
    transforms = []
    if hsv:
        transforms.append(HSVJitter())
    if fliplr > 0:
        transforms.append(RandomHorizontalFlip(probability=fliplr))
    # run_training uses the defaults for its CLI; attach per-run options here.
    config["train"]["api_transforms"] = transforms
    config["train"]["optimizer"] = optimizer_name
    config["train"]["output"] = str(output)
    config["train"]["patience"] = patience
    result_dir = run_training(config, epochs=epochs, max_batches=max_batches,
                              resume=str(resume_path) if resume_path else None,
                              threads=threads, model=model, patience=patience)
    return {"save_dir": result_dir, "best": result_dir / "best.pt",
            "last": result_dir / "last.pt", "results": result_dir / "metrics.csv"}


def train(model: str | Path | dict = "configs/yolo11.yaml", *, data,
          nc: int = 20, **kwargs) -> dict:
    """Create a CandyEye model from architecture YAML and train it."""
    candyeye = CandyEye(model, nc=nc, img_size=kwargs.get("imgsz", 128))
    return train_model(candyeye, data=data, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--epochs", type=int, help="override configured epoch count")
    parser.add_argument("--max-batches", type=int, help="limit batches per epoch (smoke runs)")
    parser.add_argument("--resume", type=str, help="resume from last/best checkpoint")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="skip optional ImageNet initialization (offline smoke runs)")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output = run_training(config, epochs=args.epochs, max_batches=args.max_batches,
                          resume=args.resume, no_pretrained=args.no_pretrained,
                          threads=args.threads)
    print(f"training complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
