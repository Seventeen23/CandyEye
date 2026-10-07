# Status — Current Phase

**Phase 5: Trainer & experiment matrix — IN PROGRESS**
Last updated: 2026-10-07

---

## Where we are

```
Phase 0 ██████████ complete
Phase 1 ██████████ complete   (model, 5/5 tests, reviewed)
Phase 2 ██████████ complete   (official weights load + ONNX parity + live demo)
Phase 3 ██████████ complete  (data pipeline)
Phase 4 ██████████ complete  (assigner & loss)
Phase 5 ██████░░░░ in progress (trainer implemented; full runs pending)
```

Detailed per-phase specs + run commands: local `SPECS.md` (git-ignored).

## Environment (verified on this machine)

| Item | Value |
|---|---|
| OS / Python | CachyOS · Python 3.14.7 |
| torch / torchvision | 2.14.1+cpu / 0.29.1+cpu |
| onnx / onnxruntime | 1.23.2 / 1.30.0 |
| opencv / numpy / pillow | opencv-python-headless 5.0.0.93 · numpy 2.5.2 · pillow 12.3.0 |
| extras | pyyaml 6.0.3, tqdm 4.70.1, pytest 9.1.1 |
| Hardware | 8 CPU cores, 13 GB RAM (CPU-only training) |
| Disk | ~17 GB free (needs ~3 GB total) |
| venv | `CandyEye/venv/` |
| Remote | https://github.com/Seventeen23/CandyEye |

## Phase 0 checklist

- [x] venv + CPU PyTorch + all deps installed
- [x] `pyproject.toml` + editable install attempted
- [x] `configs/default.yaml` (img_size 128, nc 20, VOC07 splits)
- [x] `scripts/download_voc.py` written (270 lines: download/extract/verify/cleanup)
- [x] `data/voc.py` — `VOCDataset` written (XML parse, clamp, degenerate drop, difficult flag)
- [x] **Run** `scripts/download_voc.py` — dataset on disk, verified
- [x] `scripts/inspect_data.py` written **and run** — `runs/inspect_data.jpg` eyeball pass
- [x] DataLoader smoke test — `batch=4, num_workers=2` iterates (named `collate_fn`, forkserver-safe)
- [x] Acceptance checklist below complete

### Acceptance checklist

| # | Check | Result |
|---|---|---|
| 1 | `import torch` → `2.14.1+cpu` | **pass** (verified) |
| 2 | Download completes; `trainval=5011`, `test=4952` | **pass** (verified) |
| 3 | `len(VOCDataset('data/VOCdevkit','trainval')) == 5011` | **pass** (verified: 5011 / 4952) |
| 4 | `d[0]` shapes/dtypes/bounds correct | **pass** (RGB uint8, boxes f32, labels i64 in 0–19, 100-sample bounds = 0 bad) |
| 5 | `inspect_data.py` → `runs/inspect_data.jpg` eyeball pass | **pass** (232 KB, user eyeball 2026-10-07) |
| 6 | DataLoader `batch_size=4, num_workers=2` iterates | **pass** (shapes `(375,500,3)`-style, boxes `[5,1,4,1]`) |
| 7 | `df -h` ≥ 13 GB free | pass (17 GB) |

## Known issues

1. ~~**`.gitignore` no longer ignores the dataset.**~~ **FIXED** — `# data/`
   replaced with `data/downloads/` + `data/VOCdevkit/`; `data/voc.py` stays
   tracked. Verified: dataset dirs ignored, `data/voc.py` not ignored.

2. ~~**Dataset path mismatch (double `VOCdevkit`).**~~ **FIXED** —
   `data/voc.py:31` now `self.root / "VOC2007"`. Verified: resolves to
   `data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt` (matches download
   output).

3. ~~**Editable install exposes nothing.**~~ **RESOLVED (won't fix)** —
   decision 2026-10-07: flat layout stays (`data/`, `core/`, … top-level,
   no `__init__.py`), run from repo root with `PYTHONPATH=.`, `pyproject.toml`
   left as-is. Scripts are always launched from the repo root anyway.

4. ~~**`CandyEye.egg-info/` is tracked in git.**~~ **FIXED** — untracked via
   `git rm -r --cached`, ignored via `*.egg-info/` (files remain on disk).

5. ~~**`scripts/inspect_data.py` empty.**~~ **FIXED + RUN** — written,
   executed, eyeball pass.

6. *Minor:* `Tqdm_Download` class in `download_voc.py` is dead code (a second
   progress-bar impl, never used — `download_file` builds its own tqdm). Safe
   to delete whenever.

7. *Note:* Python 3.14 forkserver DataLoader needs picklable worker args —
   `collate_fn` must be module-level (`data/voc.py`), lambdas from `__main__`
   raise `PicklingError`. Same rule applies to Phase 5's trainer.

8. *Note:* empty `data/__init.py` / `scripts/__init.py` files exist (misnamed,
   0 bytes) — unnecessary under flat layout; user to delete or rename.

## Phase 1 checklist

- [x] `core/functions/layer_utils.py` — autopad, make_divisible
- [x] `core/modules/conv.py` — Conv (+DWConv) (params/keys match official layer 0)
- [x] `core/modules/blocks.py` — Bottleneck/C3k/C3k2/SPPF/Attention/PSABlock/C2PSA (param-exact, verified)
- [x] `core/modules/detect.py` — DFL + Detect (parallel box/class branches, param-exact)
- [x] `configs/yolo11.yaml` — architecture transcription, nc=20, scale `n`
- [x] `core/yolo.py` — YAML parser/builder + forward graph (stride fill, imgsz guard)
- [x] `tests/test_model.py` — **5/5 pass** (`pytest tests/test_model.py -q`)

### Phase 1 acceptance

| Check | Result |
|---|---|
| build + forward `(1,3,128,128)` → `[(1,84,16,16),(1,84,8,8),(1,84,4,4)]` | **pass** |
| `sum(p.numel()) == 2_593_740` (trainable 2,593,724; DFL 16 frozen) | **pass** (backbone+neck 2,159,168; Detect 434,572) |
| strides `[8, 16, 32]`; eval decode `(1,24,336)` | **pass** |
| backward: all grads present / finite / nonzero | **pass** |
| key layout `model.0.conv.weight` … `model.23.dfl.conv.weight` | **pass** |
| conv-weight keys vs official ONNX (model.2/10/23) | **pass** — only `*.bn.*` differ, and that's ONNX BN-fusion (official `.pt` keeps them) |
| `imgsz` guard rejects 127 | **pass** |

Bugs caught during build-out (all in `core/yolo.py`, now fixed):
1. `module.n = n` clobbered `SPPF.self.n` (pool count 3→1) and `C2PSA.n` →
   wrapped as `.rep`.
2. `ch[i]` was layer-`i-1` output → absolute refs (`ch[16]`=192, `[13]`=256)
   inflated Detect → replaced with `ch = {i: c2}` + `cur` for `-1`.
3. YAML `[None, 2, "nearest"]` parses `None` as the *string* `"None"` (YAML
   null is `null`/`~`) → `ast.literal_eval` on string args (mirrors official).
4. `_detect_stride` ran `nn.Sequential`'s forward, which can't feed Concat a
   list → stride now computed via our own manual loop (`self(...)`).

## Next actions

1. **Review → user commit** (working tree: `core/yolo.py` fixes, `core/convert_yolo11.py`,
   `inference/predict.py`, `tests/test_convert.py`, `scripts/bootstrap_weights.py`,
   `core/modules/conv.py` + `blocks.py` fixes) → Phase 3
2. Complete the three configured Phase 5 full training runs; MobileNet's
   ImageNet checkpoint is not cached on this machine.

## Phase 2 checklist

- [x] `scripts/bootstrap_weights.py` — dev-only ultralytics use; official `.pt` → clean `weights/yolo11n.pth` (499 tensors, fp16-source) + `runs/bus.jpg` / `runs/zidane.jpg`
- [x] `core/convert_yolo11.py` — shape-checked loader `load_weights` (loaded/skipped/unexpected) + `EXPECTED_NC20_SKIPPED` (51 unfused) + `load_fused_from_onnx` + `EXPECTED_FUSED_NC20_SKIPPED` (24)
- [x] `inference/predict.py` — letterbox → forward → DFL decode → torchvision NMS → cv2 draw
- [x] `YOLO.fuse()` — BN folded into convs (state_dict then matches ONNX initializers 1:1)
- [x] `tests/test_convert.py` — **10/10 pass** (`pytest tests/ -q`)

### Phase 2 acceptance

| Check | Result |
|---|---|
| nc=80 full load: 0 missing / 0 unexpected, params 2,624,080 | **pass** |
| nc=20 sparse load: exactly the class-branch skip set, params 2,593,740 | **pass** (51 unfused / 24 fused) |
| fused layout == official ONNX key set (no `.bn.*`) | **pass** |
| num parity vs official ONNX @640 (fp32 initializer load): boxes < 1e-2, cls < 1e-4 | **pass** (box ~3e-3, cls ~4e-7) |
| arch parity vs ultralytics runtime, same src weights: every module bit-exact (0.000000) | **pass** |
| demo: `bus.jpg` nc=80 → bus 0.92; `zidane.jpg` nc=80 → tie/car; `bus.jpg` nc=20 → pipeline proof | **pass** (real COCO detections) |

Bugs caught via parity testing (in core, now fixed):
1. **BatchNorm `eps`** — official ultralytics uses `eps=1e-3` (+`momentum=0.03`); our Conv used the torch default `1e-5`. Every map drifted a little (first symptom: layer-0 maxdiff ~10).
2. **SPPF `cv1` activation** — ours used `act=False`; official `SPPF.cv1` is a plain `Conv` with SiLU. Divergence appeared exactly at module 9.

Details: the `.pt`/`.pth` ship fp16 weights, so "checkpoint-load" parity vs the fp32 ONNX is ~1e-3 noise; the exact parity test therefore loads the ONNX's own fused fp32 initializers into a `fuse()`d model.

**Param target corrected 2026-10-07:** nc=20 total is **2,593,740**
(layers 0–22 = 2,159,168 + Detect = 434,572); earlier 2,592,740 was off by 1k.
Detect(nc=80)=464,912 matches the official model exactly.

## Phase 3 checklist

- [x] `data/transforms.py` — square letterbox with xyxy box remap, HSV jitter,
  horizontal flip, and composable transforms
- [x] `VOCDataset` — fixed-size samples and optional four-image mosaic
- [x] `collate_fn` — normalized BCHW float images and YOLO `(batch_idx, cls,
  cx, cy, w, h)` targets
- [x] `scripts/visualize_batch.py` — training batch and GT overlay saved to
  `runs/batch.jpg`

### Phase 3 acceptance

| Check | Result |
|---|---|
| Real VOC batch shape | **pass** `(4, 3, 128, 128)` |
| Mosaic + augmentation target packing | **pass** `(44, 6)` targets |
| Batch visualization | **pass** (`runs/batch.jpg`, 242015 bytes; box alignment inspected) |
| DataLoader workers=2 in sandbox | unavailable (sandbox denied forkserver bind); workers=0 pass |

## Phase 4 checklist

- [x] `training/assigner.py` — Task-Aligned assignment (top-k 10, α=.5, β=6),
  inside-GT filtering, conflict resolution, IoU-aligned soft class targets
- [x] `training/loss.py` — classification BCE, CIoU box loss, interpolated DFL
- [x] `tests/test_loss.py` — no-GT, top-k positive count, empty/nonempty loss and
  backward checks (**3 passed**)
- [x] 32-image VOC overfit: AdamW lr=.001, batch=4, 10 epochs at 64px;
  mean loss 1038.80 → 148.22 (**85.7% reduction**, target >70%)

## Phase 5 checklist

- [x] `training/trainer.py` — CPU AdamW training, 3-epoch warmup + cosine,
  CSV metrics, last/best checkpoints, and resume
- [x] `core/backbone_mobilenet.py` — MobileNetV3-Small feature adapter, FPN,
  and Detect head; optional torchvision ImageNet initialization
- [x] Experiment configs for scratch, YOLO11n fine-tune, and MobileNetV3-Small
- [x] Two-epoch smoke run for all three configs; all losses finite and
  checkpoints/logs written
- [x] Resume smoke: scratch run resumed at epoch 2 and finished epoch 3
- [x] Full suite: **19 tests passed**
- [ ] Full 100-epoch runs (CPU jobs estimated to take hours; MobileNet also
  requires its ImageNet weights download)

---

Update this file at every checkpoint: tick boxes, refresh "Last updated",
move the phase bar when a phase closes.
