"""CPU trainer and CLI for the VOC experiment matrix."""
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
from core.yolo import YOLO
from data.transforms import Compose, HSVJitter, RandomHorizontalFlip
from data.voc import VOCDataset, collate_fn
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
        model = YOLO(model_cfg.get("cfg", "configs/yolo11.yaml"),
                     nc=model_cfg.get("nc", 20),
                     img_size=model_cfg.get("img_size", 128))
        if init_cfg.get("type") == "yolo11n":
            path = Path(init_cfg.get("weights", "weights/yolo11n.pth"))
            if not path.exists():
                raise FileNotFoundError(f"pretrained YOLO weights not found: {path}")
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
                 no_pretrained: bool = False, threads: int = 4) -> Path:
    train_cfg, data_cfg, model_cfg = config["train"], config["data"], config["model"]
    epochs = epochs or train_cfg["epochs"]
    seed_everything(int(train_cfg.get("seed", 23)))
    torch.set_num_threads(threads)
    device = torch.device("cpu")

    augment = Compose([HSVJitter(), RandomHorizontalFlip()])
    dataset = VOCDataset(
        data_cfg.get("root", "data/VOCdevkit"),
        data_cfg.get("train_split", "trainval"), transform=augment,
        img_size=model_cfg.get("img_size", 128),
        mosaic_probability=float(data_cfg.get("mosaic_probability", 0.0)),
    )
    loader = DataLoader(
        dataset, batch_size=int(data_cfg.get("batch_size", 16)), shuffle=True,
        num_workers=int(data_cfg.get("workers", 0)), collate_fn=collate_fn,
        drop_last=False,
    )
    steps_per_epoch = min(len(loader), max_batches) if max_batches else len(loader)
    if steps_per_epoch <= 0:
        raise ValueError("training loader has no batches")

    model = build_experiment_model(config, no_pretrained=no_pretrained).to(device)
    criterion = DetectionLoss(model)
    base_lr = float(train_cfg.get("learning_rate", 1e-3))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
    )
    warmup = float(train_cfg.get("warmup_epochs", 3))
    min_ratio = float(train_cfg.get("min_lr_ratio", .01))
    output = Path(train_cfg.get("output", f"runs/experiments/{config['name']}"))
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.csv"
    start_epoch = 0
    best_loss = float("inf")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        print(f"resumed {resume} at epoch {start_epoch}")

    write_header = not log_path.exists() or not resume
    with log_path.open("a" if resume else "w", newline="", encoding="utf-8") as csv_file:
        log = csv.writer(csv_file)
        if write_header:
            log.writerow(["epoch", "lr", "loss", "box", "cls", "dfl", "foreground"])
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
            log.writerow([epoch + 1, lr, *means[:4], means[4]])
            csv_file.flush()
            state = {"epoch": epoch + 1, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "best_loss": best_loss,
                     "config": config}
            torch.save(state, output / "last.pt")
            if epoch_loss <= best_loss:
                torch.save(state, output / "best.pt")
            print(f"epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f} "
                  f"box={means[1]:.4f} cls={means[2]:.4f} dfl={means[3]:.4f} "
                  f"lr={lr:.2e}", flush=True)
    return output


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
