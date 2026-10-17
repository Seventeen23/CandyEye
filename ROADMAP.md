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
YOLO11-CSP backbone or a MobileNetV3-Small backbone, both feeding the shared
anchor-free parallel `Detect` head. The MobileNetV3-Small backbone is now
expressed in the same YAML layer table as the CSP one (profile `d`, the default;
`candyeye/configs/mobilenetv3_small.yaml`), with the stock YOLO11n neck copied
verbatim — see Phase 9.

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
| `mobilenetv3_small` | ImageNet backbone | ImageNet load wired (208 tensors, 0 skip, bit-exact parity) + API smoke pass; 100-epoch run pending |

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

### Delivered: YAML-native MobileNetV3-Small backbone (profile `d`)

The mobile backbone no longer needs the standalone adapter — it is a
first-class YAML architecture (`candyeye/configs/mobilenetv3_small.yaml`):

- [x] `core/modules/mobilenet.py` — `MobileStem`, `MBConv` (inverted residual +
  SE), key/bn layout bit-identical to torchvision
- [x] Builder support — `MobileStem`/`MBConv` in `_MODULE_MAP` with an
  explicit-channel set (`_EXPLICIT_CHANNEL_MODULES`) so torchvision channel
  counts are used verbatim (no YOLO width multiplier)
- [x] `core/convert_mobilenet.py` — torchvision ImageNet state_dict remap
  (208 tensors, 0 skipped/unexpected) + backbone forward-parity check
- [x] Trainer wiring — `PROFILES["d"]` (default), `train()` boosts the profile
  cfg and enables ImageNet init; `initialization.type=imagenet` for the CLI path
- [x] Head kept verbatim from `yolo11.yaml`; layer 4 duplicates torchvision
  `features.3` (a stride-1 /8 repeat) so the head's P3/P4/P5 references land on
  /8, /16, /32
- [x] Tests + docs

**CPU latency caveat (re-measured 2026-10):** contrary to the original
single-run figure, MobileNetV3-Small is **not** slower than profile `c` on this
host — at 128px both land near ~25 ms (d ≈ 24.8 vs c ≈ 24.6), and at 320px
d ≈ 61 vs c ≈ 66. The earlier "61 ms at 128 / 2.5× slower" number was a
loaded-host artifact (61 ms is actually d's *320* latency). Depthwise/SE blocks
remain memory-bound, so depthwise-friendly runtimes (ONNX Runtime / OpenVINO) or
fusion/quantisation stay the follow-up; the legacy monolithic
`core/backbone_mobilenet.py` (light FPN) stays for A/B.

### Delivered: MobileNetV3-Large + P2 pyramid profiles (`e`/`f`/`g`)

Adds small-object capability and more backbone capacity to the mobile family:

- [x] `mobilenetv3_small_p2.yaml` (profile `e`) — Small + SPPF/C2PSA + P2,
  strides `[4,8,16,32]`, 1.58M params
- [x] `mobilenetv3_large_p2.yaml` (profile `f`) — full-depth (16-block)
  MobileNetV3-Large + SPPF/C2PSA + P2, strides `[4,8,16,32]`, 4.14M params
- [x] `mobilenetv3_large.yaml` (profile `g`) — Large + SPPF/C2PSA, 3 heads
  `[8,16,32]`, no P2 (large-object data), 4.10M params
- [x] `core/convert_mobilenet.py` generalised to two variants (`small`/`large`)
  via a `mobilenet_variant` YAML key; Large remaps the full 16-block
  `features` prefix (302 tensors, 0 skipped/unexpected, bit-exact parity)
- [x] `MBConv` SE squeeze switched to torchvision's `_make_divisible` (the
  0.9-correction differs from the YOLO nearest-rounding for hidden=72)
- [x] `PROFILES` `e`/`f`/`g`; tests + `MODEL_SPECS.md` rows

---

## Post-1.0 ideas (not committed)

- OpenVINO export (Intel CPU)
- VOC 07+12 trainset (16.5k images)
- Input-size sweep (96 / 160 / 192 / 320)
- INT8 static quantization with calibration set
- Full YOLO11 neck (SPPF + C2PSA + C3k2) on the MobileNet backbone as an opt-in accuracy experiment (default stays the light FPN)
- Instance segmentation heads (YOLACT-style prototype masks), revisit if needed

### Adaptive cross-scale exchange — follow-up

**Why the first ablation was neutral:** the current gate is a *global,
per-channel* scalar broadcast over the whole map (`sigmoid(w·[gap(F_i),gap(F_j)]
+b)`), and the Isda eval has no `small` objects (`small = 0.0`; every labelled
fish is `large`). A global scalar cannot route exchange *where* it helps, so
the only measured effect was CPU cost. The redesign below makes admission
**spatial** (and, later, **budgeted**), then re-tests on a dataset that can
actually show a per-size effect.

#### Spatially-selective gate (proposed module)

For each adjacent pair `(F_i, F_j)` at stride `s`, build a cheap H×W×1
content-conditioned gate instead of H×W×C:

```
g = sigmoid( Conv1x1( ReLU( Conv1x1([F_i_down, F_j_up]) ) ) + b )   # H×W×1
out_i = F_i + g * dw_exchange(F_j_up)                                # broadcast over C
out_j = F_j + (1 - g) * dw_exchange(F_i_down)
```

- `dw_exchange` is a depthwise conv (no cross-channel cost); the gate uses two
  1×1 convs to a single channel → `~C + 2` params per pair, negligible FLOPs.
- Broadcast `g` over channels (not per-channel), so it selects *regions*, not
  feature channels — the missing ingredient for small objects.
- Keep it ONNX-friendly: fixed H/W per input size, no dynamic shapes, no
  data-dependent branching.

#### Gate initialization & warm-start

- [ ] Sweep gate bias `b ∈ {-3.0, -2.5, -2.0}` vs current `-4.0` (σ≈0.018).
- [ ] Optionally warm-start: freeze gates for the first N epochs (exchange off
      ≈ baseline), then unfreeze — avoids starving the exchange path early.
- [ ] No extra FLOPs; only changes the `sigmoid` operating point.

#### Budgeted sparse exchange (research track, only if spatial gating helps)

- [ ] Tiny router over adjacent pairs (depending on profile: `P2↔P3`,
      `P3↔P4`, `P4↔P5`, or `none`). Top-1 / sparse selection via
      straight-through or Gumbel-softmax.
- [ ] Add a **measured CPU-ms budget penalty** to the objective (not FLOPs) so
      the model pays for exchange only where it buys accuracy.
- [ ] Prefer static-per-input-scale routing over per-batch (ONNX-friendly).

#### Evaluation requirements

Any exchange variant claiming improvement must:

1. Test on a **small-object dataset** (COCO subset or VOC@320) using **profile
   a** (P2/P3/P4/P5) so `small`/`medium` buckets are non-empty.
2. Report **per-size AP** (small/medium/large), not just aggregate mAP.
3. Run **≥3 seed-sequential runs** (avoid CPU contention) and report variance.
4. Compare against baseline *and* the current dynamic neck on an
   **accuracy / CPU-ms Pareto** (end-to-end CPU ms including decode + NMS),
   not FLOPs alone.

*(Matches the RetinaGate observation: gated-fusion gains are size-dependent;
the all-large Isda set explains the neutral first result.)*
