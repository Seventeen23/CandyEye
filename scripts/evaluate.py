"""Evaluate experiment checkpoints and write a per-checkpoint AP table.

Works with both PASCAL VOC and YOLO-txt (``format: yolo_txt``) experiment
configs. Reports COCO-style mAP@0.5:0.95 (averaged over the IoU sweep) as the
headline number, alongside mAP@0.5.

Example:
  PYTHONPATH=. venv/bin/python scripts/evaluate.py \
    --run configs/experiments/isda_exchange_dynamic.yaml \
          runs/experiments/isda_exchange_dynamic/best.pt
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
from data.yolo import YoloTxtDataset
from eval.map import evaluate_map
from inference.predict import predict
from training.trainer import build_experiment_model

_SPLIT_KEYS = {"train": "train_images", "val": "val_images",
               "valid": "val_images", "test": "test_images"}


def _build_dataset(config: dict, split: str, size: int):
    """Return ``(dataset, class_names, num_classes)`` for either data format."""
    data_cfg = config["data"]
    if data_cfg.get("format") == "yolo_txt":
        key = _SPLIT_KEYS.get(split, split)
        images = data_cfg.get(key)
        if not images:
            raise ValueError(
                f"data config has no {key!r}; available: "
                f"{[k for k in data_cfg if k.endswith('_images')]}"
            )
        dataset = YoloTxtDataset(images, img_size=size,
                                 num_classes=data_cfg.get("nc") or None)
        class_names = data_cfg.get("names")
        if not class_names:
            class_names = [str(i) for i in range(int(data_cfg.get("nc", 0)))]
        return dataset, class_names, len(class_names)
    dataset = VOCDataset(data_cfg.get("root", "data/VOCdevkit"), split,
                         img_size=size, mosaic_probability=0)
    return dataset, list(VOC_CLASSES), len(VOC_CLASSES)


def evaluate_checkpoint(config_path: str | Path, checkpoint_path: str | Path,
                        split: str = "test", limit: int | None = None,
                        conf: float = .001, nms_iou: float = .65,
                        threads: int = 4, size_buckets: bool = False):
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
    dataset, class_names, num_classes = _build_dataset(config, split, size)
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    ground_truth, detections = {}, []
    for index in range(count):
        # Dataset images are RGB; predict accepts OpenCV-style BGR input.
        sample = dataset._load_sample(index)
        ground_truth[sample["img_id"]] = {
            "boxes": sample["boxes"], "labels": sample["labels"],
            "difficult": sample.get("difficult"),
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

    metrics = evaluate_map(ground_truth, detections, num_classes=num_classes,
                           image_size=size, size_buckets=size_buckets)
    return {"config": str(config_path), "checkpoint": str(checkpoint_path),
            "images": count, "class_names": class_names, "map50": metrics["map50"],
            "map50_95": metrics["map"], "size_map": metrics["size"],
            "per_class_ap50": metrics["per_class_ap50"]}


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
    parser.add_argument("--size-buckets", action="store_true",
                        help="also report small/medium/large mAP")
    parser.add_argument("--out", type=Path, default=Path("runs/eval/map.csv"))
    args = parser.parse_args()

    results = [evaluate_checkpoint(cfg, ckpt, split=args.split, limit=args.limit,
                                   conf=args.conf, nms_iou=args.nms_iou,
                                   threads=args.threads,
                                   size_buckets=args.size_buckets)
               for cfg, ckpt in args.run]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "checkpoint", "images", "mAP50", "mAP50-95"]
    if args.size_buckets:
        fields += ["mAP_small", "mAP_medium", "mAP_large"]
    fields += [f"AP50_{name}" for name in results[0]["class_names"]]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {"config": result["config"], "checkpoint": result["checkpoint"],
                   "images": result["images"], "mAP50": result["map50"],
                   "mAP50-95": result["map50_95"]}
            if args.size_buckets and result["size_map"]:
                row.update({f"mAP_{name}": result["size_map"].get(name)
                            for name in ("small", "medium", "large")})
            row.update({f"AP50_{name}": ap for name, ap
                        in zip(result["class_names"], result["per_class_ap50"].values())})
            writer.writerow(row)
            print(f"{Path(result['checkpoint']).name}: mAP@0.5="
                  f"{result['map50']:.4f} mAP@0.5:0.95={result['map50_95']:.4f} "
                  f"on {result['images']} images")
    print(f"saved AP table: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
