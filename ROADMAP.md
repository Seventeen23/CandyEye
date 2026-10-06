# Roadmap

Phases in order. Nothing skips ahead — each phase ends with its acceptance
checks passing and a code review. See [STATUS.md](STATUS.md) for live progress.

Status legend: `todo` · `in progress` · `done` · `blocked`

---

## Phase 0 — Environment & data

**Goal:** reproducible env, VOC2007 on disk, dataset class that loads samples.

- [x] venv + CPU-only PyTorch + deps installed
- [x] `pyproject.toml`, `configs/default.yaml`, `.gitignore`
- [x] `scripts/download_voc.py` (download / extract / verify / cleanup)
- [x] `data/voc.py` — `VOCDataset` (XML parse, clamp, drop degenerate)
- [ ] Run download, verify `trainval=5011` / `test=4952`
- [ ] `scripts/inspect_data.py` — draw GT boxes, eyeball correctness
- [ ] DataLoader smoke test (`num_workers=2`)
- [ ] Acceptance checklist in [STATUS.md](STATUS.md#acceptance-checklist)

## Phase 1 — Model

**Goal:** forward pass at 128×128 → 3 scale outputs, YOLO11-compatible layout.

- [ ] `core/blocks.py` — Conv(CBS), Bottleneck, C3k2, SPPF, C2PSA
- [ ] `core/backbone.py` — layers 0–10 (indexed ModuleList)
- [ ] `core/neck.py` — PAN-FPN, layers 11–22
- [ ] `core/head.py` — decoupled Detect head, DFL (reg_max=16)
- [ ] `core/model.py` — assembly + forward graph
- [ ] Shape tests: `1×3×128×128` → P3/P4/P5 correct channels, param count ~2.6M

## Phase 2 — Pretrained demo (Experiment A)

**Goal:** official `yolo11n.pt` loads and detects on sample images, from our code.

- [ ] Download `yolo11n.pt` → plain state_dict converter (`core/convert_yolo11.py`)
- [ ] Converter check: **0 missing / 0 unexpected keys**
- [ ] Minimal inference: letterbox → forward → DFL projection → NMS → draw
- [ ] Visual check: COCO detections on sample images @ 128×128
- [ ] Unit test: forward parity vs converter output

## Phase 3 — Data pipeline

**Goal:** training-ready batches.

- [ ] Letterbox + resize transforms (raw xyxy → normalized cxcywh after)
- [ ] Mosaic, HSV jitter, horizontal flip
- [ ] Collate (image stack + `(img_idx, cls, cx, cy, w, h)` targets)
- [ ] Batch visualization with GT boxes

## Phase 4 — Assigner & loss

**Goal:** loss that can overfit a tiny set.

- [ ] Task-Aligned assigner (top-k=10, α=0.5, β=6.0)
- [ ] `DetectionLoss` = BCE(cls) + CIoU(box) + CrossEntropy(DFL)
- [ ] Assigner unit tests (no-GT edge case, top-k counts)
- [ ] Overfit 32 images: loss ↓ > 70%, predictions converge

## Phase 5 — Trainer & experiment matrix

**Goal:** all three configs train to completion on CPU.

- [ ] `training/trainer.py` — AdamW, 3-ep warmup + cosine, ckpt/resume, logs
- [ ] `core/backbone_mobilenet.py` — MobileNetV3-Small adapter (Experiment B)
- [ ] Configs: `tiny_scratch`, `yolo11n_finetune`, `mobilenetv3_small`
- [ ] Smoke run: 2–3 epochs, sane loss curves
- [ ] Full runs:

| Config | Init | Status |
|---|---|---|
| `tiny_scratch` | none | todo |
| `yolo11n_finetune` | YOLO11n COCO, head reinit | todo |
| `mobilenetv3_small` | ImageNet backbone | todo |

## Phase 6 — Evaluation

**Goal:** comparable accuracy numbers.

- [ ] VOC mAP@0.5 (all-point interpolation)
- [ ] Toy-example unit test (hand-computed)
- [ ] Difficult-flag handling
- [ ] mAP table per checkpoint

## Phase 7 — Inference polish

**Goal:** usable predictor.

- [ ] High-level `predict()` (image / folder)
- [ ] Visualization, per-class colors
- [ ] (optional) video/webcam

## Phase 8 — ONNX export & benchmark

**Goal:** real CPU numbers.

- [ ] `torch.onnx.export` for all three configs
- [ ] Parity test: PyTorch vs ONNX Runtime (max abs diff)
- [ ] Latency benchmark @ 128px (median, warmup) — target < 10 ms
- [ ] Model size table (FP32 / INT8)
- [ ] (optional) INT8 dynamic quantization + mAP delta

---

## Post-1.0 ideas (not committed)

- OpenVINO export (Intel CPU)
- VOC 07+12 trainset (16.5k images)
- Input-size sweep (96 / 160 / 192 / 320)
- INT8 static quantization with calibration set
- Instance segmentation heads (YOLACT-style prototype masks), revisit if needed
