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

All three pillars are delivered by **one configurable detector family**: a
YOLO11-CSP backbone or a MobileNetV3-Small backbone + light FPN, both feeding
the shared anchor-free parallel `Detect` head (see Phase 9).

---

## Phase 0 — Environment & data

**Goal:** reproducible env, VOC2007 on disk, dataset class that loads samples.

- [x] venv + CPU-only PyTorch + deps installed
- [x] `pyproject.toml`, `configs/default.yaml`, `.gitignore`
- [x] Pip-installable `candyeye/` package: public API (`CandyEye`, `train`,
  `CandyEyePredictor`), bundled architecture configs as package data, clean
  `yolo11n` weights resolved on demand (checkout asset → cache → download),
  CPU-first training defaults, no repo-root paths required
- [x] Release tooling — `CHANGELOG.md`, `MANIFEST.in`, and
  `.github/workflows/publish.yml` (TestPyPI + PyPI via Trusted Publishing)
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
- [x] COCO mAP@0.5:0.95 + small/medium/large size buckets
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
- [x] PyTorch params / GFLOPs / CPU-latency table (`scripts/benchmark.py`)
- [x] Latency benchmark @ 128px (100-run mean after warmup, 8 CPU threads, fp32 eager) — measured **baseline 49.3 ms** (exchange arms 59-66 ms); the < 10 ms target is unmet at fp32 and deferred to ONNX Runtime / INT8 (see below)
- [ ] Model size table (FP32 / INT8)
- [ ] ONNX Runtime / INT8 latency follow-up — fp32 eager is ~5× over target; expect the bulk of the < 10 ms goal to come from `torch.onnx.export` + optional INT8 quantization
- [ ] (optional) INT8 dynamic quantization + mAP delta

## Phase 9 — Unified detector family

**Goal:** one configurable detector family that combines all three design
pillars — MobileNet backbone, YOLO detection, YOLACT++-style parallel branches —
with **boxes only**. Keep the existing **light FPN** as the default MobileNet
neck (CPU-first, ~2.6M params); the full YOLO11 neck (SPPF + C2PSA + C3k2) is a
deferred, opt-in accuracy experiment (see Post-1.0).

**Non-goal:** instance segmentation / prototype masks.

- [ ] `core/factory.py` — `build_detector(cfg, nc, img_size, model_type)` shared by trainer and predictor
- [ ] `MobileNetV3SmallDetector` — add `set_classes()` and a `train(data=...)` dispatch mirroring `CandyEye.train`; keep the light FPN (no SPPF/C2PSA)
- [ ] `training/trainer.py` — build via the factory and record the real `model.type` in the saved config
- [ ] `inference/predict.py` — rebuild the correct family from `saved_config["model"]["type"]`
- [ ] Config for the combined model (reuse `mobilenetv3_small` or add a named config)
- [ ] Tests — factory selection, train-smoke → save → predictor reload round-trip, `set_classes`
- [ ] Docs — update `ARCHITECTURE.md` / `README.md` to describe the unified family

**Novelty note:** this is an engineering integration of established components
(efficient backbone + anchor-free YOLO head + parallel branches), not a new
architecture claim.

---

## Post-1.0 ideas (not committed)

- OpenVINO export (Intel CPU)
- VOC 07+12 trainset (16.5k images)
- Input-size sweep (96 / 160 / 192 / 320)
- INT8 static quantization with calibration set
- Full YOLO11 neck (SPPF + C2PSA + C3k2) on the MobileNet backbone as an opt-in accuracy experiment (default stays the light FPN)
- Instance segmentation heads (YOLACT-style prototype masks), revisit if needed

### Adaptive cross-scale exchange — ablation run (closed)

**Status:** neck implemented *and* the controlled comparison executed
(2026-10-10). Verdict: **no detection-quality improvement over the plain neck**;
the exchange only adds CPU latency and parameters. It stays an optional,
off-by-default experiment — not a claimed contribution.

**What was run.** Four arms share the YOLO11n backbone, dataloader,
augmentation, image size (128), schedule (30 epochs, AdamW lr 1e-4, 3-epoch
warmup + cosine), seed 23, and use the `valid` split for model selection; the
`test` split is reported once. Arms differ only in the neck
(`configs/experiments/isda_*.yaml`, `core/modules/exchange.py`).

| Arm | Neck | Val mAP@0.5 (best ep) | Val mAP@0.5:0.95 | Test mAP@0.5 | Test mAP@0.5:0.95 | Params | GFLOPs@128 | CPU lat. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | fixed FPN/PAN | 0.8994 | 0.6818 | 0.8923 | 0.6815 | 2.59 M | 0.253 | 49.3 ms |
| exchange-none | ungated (fixed 0.5 mix) | 0.8623 | 0.6674 | 0.9044 | 0.6727 | 2.69 M | 0.263 | 60.0 ms |
| exchange-static | learnable per-channel gate | 0.9013 | 0.6853 | 0.9030 | 0.6831 | 2.69 M | 0.263 | 66.3 ms |
| exchange-dynamic | content-conditioned gate | 0.8967 | 0.6926 | 0.8952 | 0.6797 | 2.83 M | 0.263 | 59.2 ms |

Latency is a 100-run mean after warmup on 8 CPU threads (`scripts/benchmark.py`,
`runs/benchmark.csv`); test numbers from `scripts/evaluate.py`
(`runs/eval/isda_exchange_ablation.csv`). The four arms trained concurrently, so
the per-arm wall time (~3.0-3.2 h) is contention-inflated and not a clean
single-run time.

**Reading the result.** The three exchange arms land within ±0.012 mAP@0.5 of
the baseline, in *both* directions and with no consistent winner: `static` edges
ahead on validation and on test mAP@0.5:0.95, `none` leads test mAP@0.5 but
trails validation, and `dynamic` trails the baseline on both. None of the gaps
exceeds the spread expected from a single seed, so **no improvement is
demonstrated**. The only consistent effect is cost: +97k-236k params, +0.01
GFLOPs, and +10-17 ms (up to +34%) CPU latency at 128px. The small-object half of
the hypothesis is untestable on this dataset — the size-bucket evaluation reports
`small = 0.0` and `medium = n/a`; every labelled fish is `large`.

**Honest framing / related work.** Content-conditioned, gated cross-scale fusion
is well established. BiFPN (EfficientDet) learns scalar fusion weights; ASFF
learns spatial per-level weight maps; Gated Fully Fusion applies pixelwise gates
across levels; DyFPN and the Fine-Grained Dynamic Head use input-dependent gates
to combine FPN scales; RetinaGate (2025) is a gated FPN for the same multi-scale
problem. `ScaleExchange` is an engineering variant — a cheap depthwise-
bottlenecked, near-closed-initialised post-neck drop-in — **not a new
mechanism**, and this ablation does not support an accuracy claim for it.

**Next steps if pursued.** (1) Repeat across ≥3 seeds; the deltas are at noise
level. (2) Test on a dataset with genuine small objects (COCO, or VOC at 640px)
to exercise the small-object hypothesis. (3) Longer/leaner gate schedules — the
near-closed init may starve the exchange of gradient signal early.
