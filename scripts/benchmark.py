"""Benchmark model size (params), compute (FLOPs), and CPU latency.

Builds each experiment config (optionally loading a checkpoint) and measures a
single forward pass at the configured image size. Uses PyTorch's built-in
``FlopCounterMode`` so no extra dependency is required.

Examples:
  PYTHONPATH=. venv/bin/python scripts/benchmark.py \
    --run configs/experiments/isda_exchange_dynamic.yaml \
          runs/experiments/isda_exchange_dynamic/best.pt
  PYTHONPATH=. venv/bin/python scripts/benchmark.py \
    --run configs/experiments/isda_baseline.yaml \
    --run configs/experiments/isda_exchange_none.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from candyeye.training.trainer import build_experiment_model


def build_model(config_path: str | Path, checkpoint: str | Path | None = None):
    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    config["initialization"] = {"type": "scratch", "pretrained": False}
    model = build_experiment_model(config, no_pretrained=True)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    model.eval()
    return model, int(config["model"].get("img_size", 128))


def measure(model, img_size: int, runs: int = 50, warmup: int = 10,
            threads: int = 4) -> dict:
    if img_size % 32:
        raise ValueError(f"img_size must be divisible by 32, got {img_size}")
    torch.set_num_threads(threads)
    example = torch.randn(1, 3, img_size, img_size)

    with torch.inference_mode():
        for _ in range(warmup):
            model(example)
        started = time.perf_counter()
        for _ in range(runs):
            model(example)
        latency_ms = (time.perf_counter() - started) / runs * 1000.0

    counter = FlopCounterMode(display=False)
    with counter, torch.inference_mode():
        model(example)
    flops = int(counter.get_total_flops())
    params = sum(p.numel() for p in model.parameters())
    return {"params": params, "flops": flops, "latency_ms": latency_ms,
            "img_size": img_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs="+", required=True,
                        metavar=("CONFIG", "CHECKPOINT"),
                        help="CONFIG optionally followed by a checkpoint")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/benchmark.csv"))
    args = parser.parse_args()

    rows = []
    for entry in args.run:
        config_path = entry[0]
        checkpoint = entry[1] if len(entry) > 1 else None
        model, img_size = build_model(config_path, checkpoint)
        metrics = measure(model, img_size, runs=args.runs, warmup=args.warmup,
                          threads=args.threads)
        name = Path(config_path).stem
        row = {"name": name, "config": str(config_path),
               "checkpoint": str(checkpoint) if checkpoint else "",
               "params": metrics["params"], "flops": metrics["flops"],
               "latency_ms": round(metrics["latency_ms"], 3),
               "img_size": metrics["img_size"]}
        rows.append(row)
        print(f"{name:32s} params={row['params']:,}  "
              f"GFLOPs={row['flops'] / 1e9:.3f}  "
              f"latency={row['latency_ms']:.2f} ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "config", "checkpoint", "params", "flops",
                           "latency_ms", "img_size"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved benchmark table: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
