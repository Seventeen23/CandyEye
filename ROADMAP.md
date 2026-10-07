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
- [x] `core/modules/blocks.py` — Bottleneck, C3k, C3k2, SPPF, Attention, PSABlock, C2PSA
- [x] `core/modules/detect.py` — DFL + Detect (parallel box/class branches)
- [x] `configs/yolo11.yaml` — architecture transcription, nc=20, scale `n`
- [x] `core/candyeye.py` — YAML parser/builder + forward graph
- [x] Shape tests: `1×3×128×128` → P3/P4/P5 correct channels, param count 2,593,740

## Phase 2 — Pretrained demo (Experiment A)

**Goal:** official `yolo11n.pt` loads and detects on sample images, from our code.

- [x] Download `yolo11n.pt` → plain state_dict converter (`core/convert_yolo11.py`)
- [x] Converter check: **0 missing / 0 unexpected keys** (nc=80 full; nc=20 = exactly the class branch)
- [x] Minimal inference: letterbox → forward → DFL projection → NMS → draw (`inference/predict.py`)
- [x] Visual check: COCO detections on `bus.jpg` / `zidane.jpg` @ 640×640
- [x] Unit test: numerical parity vs official ONNX (fused fp32, `tests/test_convert.py`)
- [x] Parity fixes surfaced by the test: BatchNorm `eps=1e-3` (not torch default 1e-5) and SPPF `cv1` uses SiLU — both now bit-exact vs the official runtime

## Phase 3 — Data pipeline

**Goal:** training-ready batches.

- [x] Letterbox + resize transforms (raw xyxy → normalized cxcywh after)
- [x] Mosaic, HSV jitter, horizontal flip
- [x] Roboflow YOLO detection export reader (image folders + normalized labels)
- [x] Collate (image stack + `(img_idx, cls, cx, cy, w, h)` targets)
- [x] Batch visualization with GT boxes

## Phase 4 — Assigner & loss

**Goal:** loss that can overfit a tiny set.

- [x] Task-Aligned assigner (top-k=10, α=0.5, β=6.0)
- [x] `DetectionLoss` = BCE(cls) + CIoU(box) + CrossEntropy(DFL)
- [x] Assigner unit tests (no-GT edge case, top-k counts)
- [x] Overfit 32 images: loss ↓ > 70%, predictions converge

## Phase 5 — Trainer & experiment matrix

**Goal:** all three configs train to completion on CPU.

- [x] `training/trainer.py` — AdamW, 3-ep warmup + cosine, ckpt/resume, logs
- [x] Public training API: `CandyEye.train(data=..., epochs=..., imgsz=..., ...)`
- [x] `core/backbone_mobilenet.py` — MobileNetV3-Small adapter (Experiment B)
- [x] Configs: `tiny_scratch`, `yolo11n_finetune`, `mobilenetv3_small`
- [x] Smoke run: 2 epochs, finite losses, checkpoints and CSV logs
- [ ] Full runs:

| Config | Init | Status |
|---|---|---|
| `tiny_scratch` | none | smoke pass; 100-epoch run pending |
| `yolo11n_finetune` | YOLO11n COCO, head reinit | smoke pass; 100-epoch run pending |
| `mobilenetv3_small` | ImageNet backbone | random-init smoke pass; ImageNet weights + 100-epoch run pending |

## Phase 6 — Evaluation

**Goal:** comparable accuracy numbers.

- [x] VOC mAP@0.5 (all-point interpolation)
- [x] Toy-example unit test (hand-computed)
- [x] Difficult-flag handling
- [x] mAP table per checkpoint (CSV evaluator)
- [ ] Full VOC test-split results for trained checkpoints

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

### Proposed research direction: molecular-inspired adaptive scale exchange

**Status:** hypothesis only; not implemented and not a novelty claim.

The fructose/glucose analogy can motivate a design principle: local interactions
between feature scales should combine into a useful global representation. Treat
P3/P4/P5 as interacting feature groups. Let neighboring scales exchange
lightweight depthwise/pointwise feature messages, with learned gates controlling
how much information is passed. Keep the existing YOLO-style backbone option,
decoupled detection head, loss, and training workflow so the new neck remains an
optional experiment rather than a setup-breaking fork.

The molecular idea is a metaphor for adaptive local interaction, not a physical
simulation. Cross-scale fusion and gating are established research directions;
review related work before making any originality claim.

**Testable hypothesis:** adaptive cross-scale exchange improves validation
detection quality, especially for small objects, at an acceptable increase in
CPU latency and model size compared with the current fixed-fusion neck.

**Suggested experiment:** expose the neck as a config choice while keeping the
current neck as the default. Compare (1) the current neck, (2) the proposed
exchange without gates, and (3) the gated exchange. Keep data splits, image size,
training schedule, and seeds consistent; use a validation split from training
data for model selection and reserve the test split for final results. Report
mAP@0.5 and mAP@0.5:0.95, with small-object results where available, parameter
count, compute, CPU inference latency, and time to train. Repeat promising
comparisons across multiple seeds before drawing conclusions.
