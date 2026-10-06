# CandyEye

Lightweight, CPU-first object detector written from scratch — YOLOv11-style
architecture, trained on PASCAL VOC, exported to ONNX for fast CPU inference.

**Status:** Phase 0 (environment & data) in progress → [STATUS.md](STATUS.md) ·
Full plan → [ROADMAP.md](ROADMAP.md)

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

Training/eval/predict commands are not available yet — see
[ROADMAP.md](ROADMAP.md) for what lands in which phase.

> **Note:** package installation currently doesn't expose the modules outside
> the repo root — run scripts from the repo root (see
> [STATUS.md](STATUS.md#known-issues) for details).

## Project structure

```
CandyEye/
├── configs/default.yaml        # img_size, splits, class count
├── data/voc.py                 # VOCDataset (XML parsing, box clamping)
├── scripts/
│   ├── download_voc.py         # VOC2007 download + verify + cleanup
│   └── inspect_data.py         # visualize GT boxes        (planned)
├── core/                       # blocks, backbone, neck, head  (Phase 1)
│   ├── blocks.py  backbone.py  backbone_mobilenet.py
│   ├── neck.py    head.py      model.py
│   └── convert_yolo11.py       # yolo11n.pt → our state_dict  (Phase 2)
├── training/                   # assigner, loss, trainer      (Phases 4–5)
├── eval/                       # mAP@0.5                      (Phase 6)
├── inference/                  # predict, visualize           (Phase 7)
├── export/                     # ONNX export, benchmark, INT8 (Phase 8)
├── scripts/                    # train / eval / predict / benchmark CLIs
└── tests/                      # shape, assigner, loss, mAP, ONNX parity
```

## How it's built

- **Written from scratch** — no Ultralytics runtime dependency; the official
  `yolo11n.pt` is used only as a weight source (see license note below).
- **Phase-driven** — each phase has a spec, an acceptance checklist, and a
  code review before moving on. Current state always lives in
  [STATUS.md](STATUS.md).
- **Architecture basis:** YOLOv11 (anchor-free head, DFL, Task-Aligned
  assignment, C3k2/SPPF blocks). YOLACT was studied but its prototype-mask
  approach + DCNv2/CUDA dependency was dropped — detection only.

## References

- [YOLO11](https://docs.ultralytics.com/models/yolo11/) — Ultralytics
- [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689) — Bolya et al. (inspiration, not used)
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
