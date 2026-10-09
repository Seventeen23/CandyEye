# Changelog

All notable changes to this project will be documented in this file.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
