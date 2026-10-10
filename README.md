# CandyEye

[![PyPI](https://img.shields.io/pypi/v/CandyEye.svg)](https://pypi.org/project/CandyEye/)
[![CI](https://github.com/Seventeen23/CandyEye/actions/workflows/ci.yml/badge.svg)](https://github.com/Seventeen23/CandyEye/actions/workflows/ci.yml)

Lightweight, CPU-first object detector written from scratch — YOLOv11-style
architecture, trained on PASCAL VOC, exported to ONNX for fast CPU inference.

**Status:** Phase 0 complete → Phase 1 (model) in progress → [STATUS.md](STATUS.md) ·
Full plan → [ROADMAP.md](ROADMAP.md)

## Why "CandyEye"?

The name comes from sugar chemistry: **fructose** is a simple sugar — the
smallest, sweetest, quickest source of energy in nature. The detector aims
to be the same for vision: *small, sweet, fast*.

Three design ideas, one borrowed from each reference model:

| Idea | From | What it means here |
|---|---|---|
| **Detection** | YOLO | One forward glance — boxes come straight from conv features in a single stage, no region proposals |
| **Efficient & lightweight** | MobileNet | ~2.6M params, CPU-only training, 128×128 input — fast fuel, runs on the laptop it was built on |
| **Parallel** | YOLACT++ | Shared conv features feed multiple branches simultaneously: the FPN neck runs P3/P4/P5 in parallel, and each Detect-head scale splits into **box and classification branches computed in parallel** — no mask branch, bounding boxes only |

## Goals

| Metric | Target |
|---|---|
| Input size | 128×128 (divisible by 32) |
| Parameters | ~2.6M (YOLO11n-class, scale `n`: depth 0.50 / width 0.25) |
| FLOPs | ~0.3 G @ 128px (6.5 G @ 640px, scaled) |
| Training | CPU-only (laptop, 8 cores) |
| Inference | ONNX Runtime, CPU; INT8 quantization optional |
| Dataset | PASCAL VOC 2007 (20 classes): trainval 5011 / test 4952 |
| Latency | < 10 ms/image @ 128px (target, Phase 8; fp32 eager measured **49.3 ms** on 8 CPU threads — target unmet until ONNX Runtime / INT8) |
| Accuracy | mAP@0.5 baseline TBD after first training run |

## Experiments

Three configs share the same neck/head, differing only in how the backbone is
initialized:

| Config | Init | Purpose |
|---|---|---|
| `tiny_scratch` | none | from-scratch baseline |
| `yolo11n_finetune` | YOLO11n COCO weights, detect head reinit | likely best accuracy |
| `mobilenetv3_small` | ImageNet-pretrained backbone, fresh neck/head | pretrained-backbone experiment |

### Adaptive cross-scale exchange neck

`candyeye/core/modules/exchange.py` adds an optional `ScaleExchange` neck that sits
between the existing FPN and the Detect head. Instead of only the fixed
top-down/bottom-up path, adjacent pyramid levels (P3↔P4, P4↔P5) pass a cheap
depthwise message whose admission is controlled by a gate:

| Gate | Behaviour |
|---|---|
| `none` | fixed 0.5 mix — ablation control |
| `static` | `sigmoid(learnable per-channel weight + bias)` — BiFPN-like |
| `dynamic` | gate computed at runtime from the two feature maps — content-conditioned |

The `dynamic` gate is content-conditioned: unlike BiFPN's fixed learned scalar
weights, its weights are computed at runtime from the features. Gates initialize
near-closed (`bias -4.0`), so an exchange model starts close to the plain
baseline. The neck is selectable through the architecture YAML
(`configs/yolo11_exchange.yaml`, layer 23) and as a `neck: exchange` option on
the MobileNet model, with experiment configs under `configs/experiments/`
(`isda_baseline`, `isda_exchange_{none,static,dynamic}`, and the
`isda_mobilenet_*` variants).

**Result (seed 23, 30 epochs, Isda 9-class, `valid` for selection / `test`
reported once).** The exchange neck did **not** improve detection over the plain
neck — all three arms land within ±0.012 mAP@0.5 of the baseline, in both
directions and with no consistent winner:

| Neck | Test mAP@0.5 | Test mAP@0.5:0.95 | Params | CPU lat. |
|---|---:|---:|---:|---:|
| baseline (fixed FPN/PAN) | 0.8923 | 0.6815 | 2.59 M | 49.3 ms |
| `none` | 0.9044 | 0.6727 | 2.69 M | 60.0 ms |
| `static` | 0.9030 | 0.6831 | 2.69 M | 66.3 ms |
| `dynamic` | 0.8952 | 0.6797 | 2.83 M | 59.2 ms |

The only consistent effect is cost (+10-17 ms, up to +34% CPU latency at 128px).
The small-object hypothesis is untestable here (size-bucket eval reports
`small = 0.0`, `medium = n/a`; every labelled fish is `large`). Gated
cross-scale fusion is an established idea (BiFPN, ASFF, Gated Fully Fusion,
DyFPN, RetinaGate); this neck is an engineering variant, not a new mechanism,
and remains opt-in. Full details: [ROADMAP.md](ROADMAP.md#adaptive-cross-scale-exchange--ablation-run-closed).

## Install

```bash
pip install CandyEye        # from PyPI (once published)
pip install -e .            # or, from a checkout
```

For a CPU-only PyTorch wheel (recommended on laptops):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

The wheel bundles the default `yolo11`/`yolo11_exchange` architecture configs, so
`CandyEye()` works with no checkout and no download. The clean `yolo11n` weights
are **not** shipped in the wheel or sdist (they derive from Ultralytics'
AGPL-3.0 checkpoints): a fresh install downloads the official checkpoint on the
first `train(..., pretrained=True)` and caches a clean state_dict under
`~/.cache/candyeye/`. In a source checkout, `candyeye/assets/yolo11n.pth` is used
directly for offline dev/tests. Install the optional converter with
`pip install CandyEye[weights]`, or point `$CANDYEYE_WEIGHTS` at an existing
`.pth` to stay offline.

## Training

Training is available through CandyEye's Python API. The model config defines
the architecture (bundled names such as `"yolo11"` or `"yolo11_exchange"`, a
YAML path, or a parsed dict); the data YAML points to your dataset and split
names.

```python
from candyeye import CandyEye

model = CandyEye()  # bundled yolo11 architecture
results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=128,
    batch=16,
    patience=20,
    workers=0,
    pretrained=True,  # download + cache clean yolo11n weights on first use
    project="runs/train",
    name="voc_yolo11",
)
print(results["best"])
```

This CPU-first trainer saves `best.pt`, `last.pt`, and `metrics.csv` in the
run directory. Each epoch reports train and validation total, box,
classification, and DFL losses, plus aggregate precision, recall, F1,
mAP@0.5, and mAP@0.5:0.95. Per-class precision, recall, F1, AP@0.5, and
COCO AP are stored in `metrics.csv` rather than printed every epoch, together
with a small/medium/large mAP breakdown. Precision/recall/F1 use confidence 0.25 and IoU 0.5; AP uses
confidence 0.001 and the IoU sweep 0.50:0.05:0.95. `best.pt` is selected by validation mAP@0.5, and
`patience` stops after that metric fails to improve. Resume with `resume=True`
or pass a checkpoint path. `imgsz` must be divisible by 32. Detection
“accuracy” is not a standard object-detection metric, so use precision, recall,
F1, and AP/mAP to assess the model.

> **Note on `val_split: test`:** by default the trainer validates (and selects
> `best.pt`) on the VOC **test** split, so model selection sees the test set and
> reported test numbers are mildly optimistic. This default is kept for
> convenience; for an unbiased benchmark set `val_split` to a held-out split
> (e.g. `train`) in your data config.

The console prints a compact summary for each epoch. At the end of training,
the run directory also contains `results.png` (total/component losses,
validation metrics, and
learning-rate curves), `confusion_matrix.png` (for the best checkpoint at
confidence 0.25 and IoU 0.5), and `confusion_matrix.csv`. The matrix includes a
background row and column to show false positives and missed objects. The CSV
history remains available as `metrics.csv`.

### Run a trained model on an image or video

Use the Python predictor with an image path or video path. It loads class names
from the dataset YAML and saves annotated output under `runs/predict/` by
default:

```python
from candyeye import CandyEyePredictor

predictor = CandyEyePredictor(
    weights="runs/train/fruit_smoke_test/best.pt",
    data="data/test_data/fruits.v5i.yolov11/data.yaml",
    imgsz=128,
)

# Run one image
image_result = predictor.predict(
    "data/test_data/fruits.v5i.yolov11/test/images/15_jpg.rf.bdcbfdcabfa19ea0ca4b42a986bcb604.jpg",
    conf=0.25,
)
print(image_result["output"])
print(image_result["detections"])

# Or run a video instead
video_result = predictor.predict("/path/to/clip.mp4", conf=0.25)
print(video_result["output"], video_result["frames"])
```

Pass `output="path/to/result.jpg"` (or an `.mp4` path for video) to choose a
specific output file.

For a function-style entry point:

```python
from candyeye import train

results = train(data="dataset.yaml", epochs=100, imgsz=128, batch=16,
                patience=20, pretrained=True)
```

### Evaluation and benchmarking

```bash
# COCO-style mAP@0.5:0.95 (+ optional size buckets) for saved checkpoints
python scripts/evaluate.py \
  --run configs/experiments/isda_exchange_dynamic.yaml \
        runs/experiments/isda_exchange_dynamic/best.pt --size-buckets

# params, GFLOPs, and CPU latency at the configured image size
python scripts/benchmark.py \
  --run configs/experiments/isda_baseline.yaml \
  --run configs/experiments/isda_exchange_dynamic.yaml
```

`scripts/evaluate.py` accepts both VOC and `format: yolo_txt` configs and
writes a per-checkpoint AP table; `scripts/benchmark.py` uses PyTorch's
built-in FLOP counter (no extra dependency).

## Data formats

CandyEye accepts the common Roboflow image-folder and normalized `.txt` label
export. Its `data.yaml` can look like this (paths may be relative to the YAML):

```yaml
path: /datasets/my-export
train: train/images
valid: valid/images
test: test/images
nc: 2
names: [cat, dog]
```

Each image needs a same-stem label file under the matching `labels/` folder;
each nonempty row is `class_id center_x center_y width height`, normalized to
the image dimensions. Pass the YAML path directly; CandyEye reads `nc` (or
counts `names`) and configures its class head automatically:

```python
from candyeye import CandyEye

data_yaml = "/datasets/my-export/data.yaml"
model = CandyEye()  # or CandyEye("yolo11_exchange") for the exchange neck
results = model.train(data=data_yaml, epochs=100, imgsz=128,
                      batch=16, patience=20)
```

See [ROADMAP.md](ROADMAP.md) for current evaluation and export work.

## Project structure

```
CandyEye/
├── candyeye/                   # pip-installable package (import candyeye)
│   ├── __init__.py             # public API: CandyEye, train, CandyEyePredictor
│   ├── paths.py                # packaged config + weight resolution
│   ├── configs/                # bundled architecture YAMLs (package data)
│   │   ├── yolo11.yaml
│   │   └── yolo11_exchange.yaml
│   ├── assets/yolo11n.pth      # clean weights for dev/tests (not shipped in the wheel)
│   ├── core/                   # model builder + nn.Module blocks
│   │   ├── functions/          # stateless helpers (autopad, make_divisible)
│   │   ├── modules/            # Conv, Bottleneck, C3k2, SPPF, C2PSA, DFL, Detect, ScaleExchange
│   │   ├── candyeye.py         # YAML builder + forward graph
│   │   └── convert_yolo11.py   # yolo11n.pt / ONNX → our state_dict
│   ├── data/                   # VOC + YOLO-txt datasets, transforms
│   ├── training/               # assigner, loss, trainer
│   ├── eval/                   # mAP (mAP@0.5, mAP@0.5:0.95, size buckets)
│   └── inference/              # predict.py (CandyEyePredictor)
├── configs/experiments/        # ablation configs (Isda 9-class)
├── data/                       # datasets + VOCdevkit (gitignored)
├── export/                     # ONNX export, benchmark, INT8 (Phase 8)
├── scripts/                    # inspect_data, evaluate, benchmark, bootstrap_weights
├── runs/                       # outputs (gitignored)
└── tests/                      # shape, assigner, loss, mAP, exchange, ONNX parity
```

## How it's built

- **Written from scratch** — no Ultralytics runtime dependency; the official
  `yolo11n.pt` is used only as a weight source (see license note below).
- **Phase-driven** — each phase has a spec, an acceptance checklist, and a
  code review before moving on. Current state always lives in
  [STATUS.md](STATUS.md).
- **Architecture basis:** YOLOv11 (anchor-free head, DFL, Task-Aligned
  assignment, C3k2/SPPF blocks), MobileNetV3 as the efficiency reference
  (Experiment B), and a YOLACT-style *parallel* structure — neck scales and
  the head's box/class branches compute over shared conv features
  simultaneously. YOLACT's prototype-mask half is out of scope: bounding
  boxes only (revisit only if segmentation is ever added).

## References

- [YOLO11](https://docs.ultralytics.com/models/yolo11/) — Ultralytics
- [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689) — Bolya et al. (parallel neck/head inspiration; mask branch out of scope)
- [Task-Aligned One-stage Object Detection (TOOD)](https://arxiv.org/abs/2108.07755) — assignment strategy
- [Generalized Focal Loss](https://arxiv.org/abs/2006.04388) — Distribution Focal Loss
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

## License

[GPL-3.0](LICENSE) — Copyright (c) 2026 Seventeen23.

**Pretrained weights note:** YOLO11 weights are licensed AGPL-3.0 by
Ultralytics. They are used here for personal experimentation/fine-tuning only;
the original checkpoint is `.gitignore`d and never redistributed. A clean
derived state_dict (`candyeye/assets/yolo11n.pth`) is kept in the repo for
offline dev/tests/CI but is excluded from the published wheel and sdist. Models
trained from those weights may inherit AGPL obligations — check
[Ultralytics' license](https://ultralytics.com/license) before distributing
trained weights.
