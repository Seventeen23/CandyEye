# Changelog

All notable changes to this project will be documented in this file.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
