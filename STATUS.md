# Status — Current Phase

**Phase 0: Environment & data — COMPLETE → Phase 1: Model (in progress)**  
Last updated: 2026-10-07 (through `da54fd9`; Phase 1 started: `core/` layout)

---

## Where we are

```
Phase 0 ██████████ complete
Phase 1 ██████░░░░ in progress — Conv/blocks/Detect ✓ (param-exact); builder + tests next
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

## Next actions

1. `configs/yolo11.yaml` (layer table from SPECS) + `core/yolo.py` builder
   (ModuleList assembly, `-1`/list `from`, channel scaling, stride fill)
2. `tests/test_model.py` — shapes + `params == 2,593,740` + grad flow + key layout
3. Review → commit → Phase 2 (weight loading from official `yolo11n.pt`)

**Param target corrected 2026-10-07:** nc=20 total is **2,593,740**
(layers 0–22 = 2,159,168 + Detect = 434,572); earlier 2,592,740 was off by 1k.
Detect(nc=80)=464,912 matches the official model exactly.

---

Update this file at every checkpoint: tick boxes, refresh "Last updated",
move the phase bar when a phase closes.
