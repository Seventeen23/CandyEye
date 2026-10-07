# CandyEye

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
| Latency | < 10 ms/image @ 128px (TBD in Phase 8) |
| Accuracy | mAP@0.5 baseline TBD after first training run |

## Experiments (planned)

Three configs share the same neck/head, differing only in how the backbone is
initialized:

| Config | Init | Purpose |
|---|---|---|
| `tiny_scratch` | none | from-scratch baseline |
| `yolo11n_finetune` | YOLO11n COCO weights, detect head reinit | likely best accuracy |
| `mobilenetv3_small` | ImageNet-pretrained backbone, fresh neck/head | pretrained-backbone experiment |

## Quickstart

```bash
# Environment (Python 3.14, CPU-only PyTorch)
python3 -m venv venv
source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install onnx onnxruntime opencv-python-headless pyyaml tqdm pillow pytest
pip install -e .

# PASCAL VOC 2007 (~870 MB download, ~2 GB extracted)
python scripts/download_voc.py
```

## Training

Training is available through CandyEye's Python API. The model YAML defines
the architecture; the data YAML points to the VOC dataset and split names.

```python
from core import CandyEye

model = CandyEye("configs/yolo11.yaml")
results = model.train(
    data="configs/default.yaml",
    epochs=100,
    imgsz=128,
    batch=16,
    patience=20,
    workers=0,
    project="runs/train",
    name="voc_yolo11",
)
print(results["best"])
```

This CPU-first trainer saves `best.pt`, `last.pt`, and `metrics.csv` in the
run directory. Each epoch reports train and validation total, box,
classification, and DFL losses, plus aggregate precision, recall, F1, and
mAP@0.5. Per-class precision, recall, F1, and AP@0.5 are stored in
`metrics.csv` rather than printed every epoch. Precision/recall/F1 use confidence 0.25 and IoU 0.5; AP uses
confidence 0.001 and IoU 0.5. `best.pt` is selected by validation mAP@0.5, and
`patience` stops after that metric fails to improve. Resume with `resume=True`
or pass a checkpoint path. `imgsz` must be divisible by 32. Detection
“accuracy” is not a standard object-detection metric, so use precision, recall,
F1, and AP/mAP to assess the model.

### Run a trained model on an image or video

Use the Python predictor with an image path or video path. It loads class names
from the dataset YAML and saves annotated output under `runs/predict/` by
default:

```python
from inference.predict import CandyEyePredictor

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
from core import train

results = train(model="configs/yolo11.yaml", data="configs/default.yaml",
                epochs=100, imgsz=128, batch=16, patience=20)
```

### Roboflow YOLO detection export

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
from core import CandyEye

data_yaml = "/datasets/my-export/data.yaml"
model = CandyEye("configs/yolo11.yaml")
results = model.train(data=data_yaml, epochs=100, imgsz=128,
                      batch=16, patience=20)
```

See [ROADMAP.md](ROADMAP.md) for current evaluation and export work.

> **Note:** package installation currently doesn't expose the modules outside
> the repo root — run scripts from the repo root (see
> [STATUS.md](STATUS.md#known-issues) for details).

## Project structure

```
CandyEye/
├── configs/
│   ├── default.yaml            # img_size, splits, class count
│   └── yolo11.yaml             # model architecture, nc=20      (Phase 1)
├── data/
│   ├── voc.py                  # VOCDataset (XML parsing, clamping, collate_fn)
│   └── VOCdevkit/              # VOC2007 (gitignored; scripts/download_voc.py)
├── core/
│   ├── functions/              # stateless helpers (autopad, make_divisible) ✓
│   ├── modules/                # nn.Module blocks (Conv ✓ · Bottleneck, C3k2,
│   │                            #   SPPF, C2PSA, DFL, Detect        Phase 1)
│   ├── candyeye.py             # YAML builder + forward graph    (Phase 1)
│   └── convert_yolo11.py       # yolo11n.pt / ONNX → our state_dict (Phase 2)
├── training/                   # assigner, loss, trainer          (Phases 4–5)
├── eval/                       # mAP@0.5                          (Phase 6)
├── inference/                  # predict.py already (Phase 2 preview of 7)
├── export/                     # ONNX export, benchmark, INT8     (Phase 8)
├── scripts/                    # download_voc, inspect_data, train/eval/predict CLIs
├── runs/                       # outputs (gitignored)
└── tests/                      # shape, assigner, loss, mAP, ONNX parity
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
weights files are `.gitignore`d and are not redistributed with this repository.
Models trained from those weights may inherit AGPL obligations — check
[Ultralytics' license](https://ultralytics.com/license) before distributing
trained weights.
