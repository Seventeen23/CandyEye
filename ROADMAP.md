# Roadmap

Phases in order. Nothing skips ahead — each phase ends with its acceptance
checks passing and a code review. See [STATUS.md](STATUS.md) for live progress.

Status legend: `todo` · `in progress` · `done` · `blocked`

**Design DNA — why the name:** *CandyEye* = fructose-simple sugar (small,
sweet, quick energy) + one glance. Three pillars, one per reference model:
**detection** (YOLO: single-stage, one forward pass) · **efficient &
lightweight** (MobileNet: ~2.6M params, CPU-only, 128×128) · **parallel**
(YOLACT++: FPN neck scales + the Detect head's box/class branches compute
simultaneously over shared conv features — no mask branch, boxes only).

---

## Phase 0 — Environment & data

**Goal:** reproducible env, VOC2007 on disk, dataset class that loads samples.

- [x] venv + CPU-only PyTorch + deps installed
- [x] `pyproject.toml`, `configs/default.yaml`, `.gitignore`
- [x] `scripts/download_voc.py` (download / extract / verify / cleanup)
- [x] `data/voc.py` — `VOCDataset` (XML parse, clamp, drop degenerate)
- [x] Run download, verify `trainval=5011` / `test=4952`
- [x] `scripts/inspect_data.py` — draw GT boxes, eyeball correctness
- [x] DataLoader smoke test (`num_workers=2`)
- [x] Acceptance checklist in [STATUS.md](STATUS.md#acceptance-checklist)

## Phase 1 — Model

**Goal:** forward pass at 128×128 → 3 scale outputs, YOLO11-compatible layout.

- [x] `core/functions/layer_utils.py` — autopad, make_divisible
- [x] `core/modules/conv.py` — Conv (verified: params/keys match official layer 0)
- [ ] `core/modules/blocks.py` — Bottleneck, C3k, C3k2, SPPF, Attention, PSABlock, C2PSA
- [ ] `core/modules/detect.py` — DFL + Detect (parallel box/class branches)
- [ ] `configs/yolo11.yaml` — architecture transcription, nc=20, scale `n`
- [ ] `core/yolo.py` — YAML parser/builder + forward graph
- [ ] Shape tests: `1×3×128×128` → P3/P4/P5 correct channels, param count 2,592,740

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
