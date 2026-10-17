# Changelog

All notable changes to this project will be documented in this file.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **MobileNetV3-Small YAML backbone (profile `d`, now the default)** — a
  non-YOLO feature extractor expressed in the same YAML layer table as the CSP
  backbones. New `candyeye/configs/mobilenetv3_small.yaml` (mobile backbone +
  verbatim `yolo11.yaml` head), module types `MobileStem` / `MBConv` in
  `core/modules/mobilenet.py`, and an explicit-channel set
  (`_EXPLICIT_CHANNEL_MODULES`) so torchvision channel counts are used
  verbatim (no YOLO width multiplier). `core/convert_mobilenet.py` remaps the
  torchvision ImageNet state_dict (208 tensors, 0 skipped/unexpected) and
  exposes a bit-exact forward-parity check. `train()` targets profile `d` by
  default and initialises the backbone from ImageNet (skip with
  `pretrained=False`); `initialization.type=imagenet` covers the CLI path.
  Layer 4 duplicates torchvision `features.3` (a stride-1 /8 repeat) so the
  head's P3/P4/P5 references land on /8, /16, /32.
- **Profiles** — `profile="a" | "b" | "c" | "d"` selects a bundled
  architecture/strides bundle and its augmentation defaults; `d`
  (`mobilenetv3_small`) is the default. New configs
  `candyeye/configs/yolo11_p2.yaml` (a), `yolo11_p2_small.yaml` (b), and
  `mobilenetv3_small.yaml` (d); `PROFILES` / `resolve_profile()` in
  `training/trainer.py`. P2 fusion is appended after the neck so pretrained
  indices 0–22 stay unchanged (only Detect is remapped).
- **EMA** — `EMA` (decay `0.999`, warm-up `(1+u)/(10+u)`, integer buffers
  copied not averaged). Wired through training: updated after every optimizer
  step, used for validation and `best.pt` selection, and saved as `ema` (+
  `ema_enabled`, `updates`) alongside the live `model`. `ema=True` by default;
  resume restores both.
- **Scale jitter** — `RandomAffine` (scale + translate, no rotation) in
  `data/transforms.py`; `scale_jitter=(lo, hi)` in the training API, defaulting
  to `(0.5, 1.5)` for profile `a`.
- **Library-first API** — `train(..., profile=, ema=, scale_jitter=)` and
  `CandyEye.train(...)` forward all three; `CandyEyePredictor` rebuilds the
  architecture from the checkpoint's saved config and prefers the checkpoint's
  EMA weights. `persistent_workers` / `prefetch_factor` set automatically when
  `workers>0`.

### Fixed

- **`CandyEyePredictor` architecture mismatch** — the predictor now reads
  `config.model.cfg` from the checkpoint when `cfg` is not passed, so
  P2/exchange checkpoints no longer fail `load_state_dict` against the default
  `yolo11` (3-head) graph.
- **`ScaleExchange` N-level parsing** — the constructor now distinguishes the
  keyword form (`channels..., gate=, iters=`) from the positional YAML form
  (`channels..., gate, iters`); a bare trailing int is treated as a channel, not
  `iters`, so a 4th level (e.g. `64, 128, 256, 32`) no longer misparses.
  `parse_model` builds the module with explicit `gate=` / `iters=` keywords.
- **`nn.Upsample` channel resolution** — `c2` now resolves against the source
  level (`ch[f]`) when `f` is a non-negative absolute reference instead of
  always using the current channel count.

### Changed

- **Default training profile is now `d`** (`mobilenetv3_small`). `train(...)`
  with no `model`/`profile` builds the MobileNetV3-Small backbone and
  initialises it from ImageNet; pass `profile="a"` (or another) to restore the
  previous YOLO11-P2 default.
- **Exchange neck result recorded as negative.** The controlled four-arm
  ablation (baseline / none / static / dynamic, Isda 9-class, seed 23, 30
  epochs) showed no detection-quality gain — all arms within ±0.012 mAP@0.5 of
  baseline, no consistent winner across mAP@0.5 and mAP@0.5:0.95 — at +~4%
  FLOPs, +97k-236k params, and +10-17 ms CPU latency. README / ROADMAP / STATUS
  / ARCHITECTURE updated to report this honestly; the neck stays opt-in and is
  not claimed as a contribution.
- **Latency target corrected in README** — fp32 eager PyTorch measures ~49 ms
  at 128×128 on 8 CPU threads; the < 10 ms goal is deferred to ONNX Runtime /
  INT8 work in Phase 8.

### Added

- **MobileNetV3-Large + P2 pyramid profiles (`e`/`f`/`g`)** — new bundled
  configs extending the mobile family: `mobilenetv3_small_p2.yaml` (`e`),
  `mobilenetv3_large_p2.yaml` (`f`), and `mobilenetv3_large.yaml` (`g`, no P2).
  Each adds SPPF + C2PSA to the mobile backbone; `e`/`f` append a stride-4 P2
  fusion level after the neck for small objects. Configs carry a
  `mobilenet_variant: small|large` key; `PROFILES` and `resolve_profile()` select
  `e`/`f`/`g`. Params: `e` 1.58M, `f` 4.14M, `g` 4.10M.
- **`convert_mobilenet` two-variant support** — `imagenet_mobilenetv3_large_state`
  / `imagenet_mobilenetv3_small_state`, a variant-aware feature map, and
  `load_imagenet_backbone(model, state, variant=...)`; the Large backbone remaps
  302 tensors (0 skipped/unexpected) with bit-exact forward parity.
- `research/scale_exchange_paper_draft.md` — result tables, per-class AP,
  ablation reading, Discussion, and Conclusion filled from the measured run,
  with single-seed / provenance caveats retained.

### Fixed

- **`MBConv` SE squeeze channel count** — the squeeze/excite `make_divisible`
  now matches torchvision's `_make_divisible` (0.9 round-down correction) instead
  of the YOLO nearest-rounding, which under-sized the SE hidden layer for
  `hidden=72` (18 → 24, not 16).

## [0.1.0] — 2026-10-09

First public release: a lightweight, CPU-first object detector.

### Added

- **Installable package** — code lives in `candyeye/` with the public API
  `from candyeye import CandyEye, train, CandyEyePredictor`. CPU-first training
  defaults (128×128 input, AdamW, warmup + cosine schedule, all logical cores).
- **Bundled architecture configs** (`candyeye/configs/yolo11.yaml`,
  `yolo11_exchange.yaml`) resolved via `candyeye.paths` with no repo-root paths.
- **Adaptive cross-scale exchange neck** (`ScaleExchange`) with `none` / `static`
  / `dynamic` gates, selectable from YAML and as `neck: exchange` on the
  MobileNet model.
- **Evaluation** — mAP@0.5, COCO-style mAP@0.5:0.95, and small/medium/large size
  buckets; `scripts/evaluate.py` and `scripts/benchmark.py` (params, GFLOPs,
  CPU latency).
- **Download-on-demand weights** — `candyeye.paths.download_default_weights`
  fetches and converts the official `yolo11n` checkpoint into the per-user
  cache; the distribution itself ships no weights.
- **CI** — `.github/workflows/ci.yml` (Python 3.10/3.12/3.14, CPU-only torch)
  and an opt-in `.github/workflows/onnx-parity.yml`.
- **Publishing** — `.github/workflows/publish.yml` (TestPyPI + PyPI, Trusted
  Publishing on `v*` tags).

### Changed

- Imports moved from top-level `core` / `training` / `data` / `eval` /
  `inference` modules to the `candyeye.*` package.
- `pyproject.toml`: `requires-python >= 3.10`, SPDX `GPL-3.0-or-later`, version
  single-sourced from `candyeye.__version__`.
- Experiment configs (`configs/experiments/*`) dropped the hard-coded
  `weights/yolo11n.pth` path; weights now resolve through
  `candyeye.paths.default_weights_path()`.
- Docs (README, ARCHITECTURE, STATUS, ROADMAP) refreshed for the packaged
  layout, on-demand weights, and the MobileNetV3-Small variant.

### Notes

- Pretrained weights derive from Ultralytics' YOLO11 checkpoints (AGPL-3.0) and
  are downloaded on first use rather than redistributed.

## [Unreleased] — Phase 0

### Added

- `pyproject.toml` — package metadata (CandyEye 0.0.1, Python ≥ 3.14)
- `configs/default.yaml` — img_size 128, 20 classes, VOC07 trainval/test splits
- `scripts/download_voc.py` — VOC 2007 download, safe extract, line-count
  verification (5011/4952), tar cleanup
- `data/voc.py` — `VOCDataset`: XML annotation parsing, class-id mapping
  (0–19), coordinate clamping, degenerate-box drop, `difficult` flag
- `LICENSE` — GPL-3.0
- `README.md`, `ROADMAP.md`, `STATUS.md`, this changelog
