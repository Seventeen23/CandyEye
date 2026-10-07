"""Evaluate experiment checkpoints on VOC and write a per-checkpoint AP table.

Example:
  PYTHONPATH=. venv/bin/python scripts/evaluate.py \
    --run configs/experiments/tiny_scratch.yaml runs/experiments/tiny_scratch/best.pt
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import cv2
import torch
import yaml

from data.voc import VOCDataset, VOC_CLASSES
from eval.map import evaluate_map50
from inference.predict import predict
from training.trainer import build_experiment_model


def evaluate_checkpoint(config_path: str | Path, checkpoint_path: str | Path,
                        split: str = "test", limit: int | None = None,
                        conf: float = .001, nms_iou: float = .65,
                        threads: int = 4):
    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    # The detector weights come from the checkpoint; avoid reloading/downloading
    # its original initialization while rebuilding the architecture.
    config["initialization"] = {"type": "scratch", "pretrained": False}
    model = build_experiment_model(config, no_pretrained=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.set_num_threads(threads)

    size = int(config["model"].get("img_size", 128))
    data_cfg = config["data"]
    dataset = VOCDataset(data_cfg.get("root", "data/VOCdevkit"), split,
                         img_size=size, mosaic_probability=0)
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    ground_truth, detections = {}, []
    for index in range(count):
        # Dataset images are RGB; predict accepts OpenCV-style BGR input.
        sample = dataset._load_sample(index)
        ground_truth[sample["img_id"]] = {
            "boxes": sample["boxes"], "labels": sample["labels"],
            "difficult": sample["difficult"],
        }
        image_bgr = cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR)
        found = predict(model, image_bgr, size=size, conf=conf,
                        iou_thr=nms_iou, max_det=300)
        detections.extend({"image_id": sample["img_id"], "class_id": cls,
                           "score": score, "box": box}
                          for cls, score, box in found)
        if (index + 1) % 500 == 0 or index + 1 == count:
            print(f"{Path(checkpoint_path).name}: evaluated {index + 1}/{count}",
                  flush=True)

    mean_ap, class_ap = evaluate_map50(ground_truth, detections,
                                       num_classes=len(VOC_CLASSES),
                                       iou_threshold=.5)
    return {"config": str(config_path), "checkpoint": str(checkpoint_path),
            "images": count, "map50": mean_ap, "class_ap": class_ap}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, required=True,
                        metavar=("CONFIG", "CHECKPOINT"),
                        help="repeat once per config/checkpoint pair")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="evaluate only first N images")
    parser.add_argument("--conf", type=float, default=.001)
    parser.add_argument("--nms-iou", type=float, default=.65)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/eval/map.csv"))
    args = parser.parse_args()

    results = [evaluate_checkpoint(cfg, ckpt, split=args.split, limit=args.limit,
                                   conf=args.conf, nms_iou=args.nms_iou,
                                   threads=args.threads)
               for cfg, ckpt in args.run]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "checkpoint", "images", "mAP50"] + [
        f"AP_{name}" for name in VOC_CLASSES]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {"config": result["config"], "checkpoint": result["checkpoint"],
                   "images": result["images"], "mAP50": result["map50"]}
            row.update({f"AP_{VOC_CLASSES[c]}": ap
                        for c, ap in result["class_ap"].items()})
            writer.writerow(row)
            print(f"{Path(result['checkpoint']).name}: mAP@0.5="
                  f"{result['map50']:.4f} on {result['images']} images")
    print(f"saved AP table: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
