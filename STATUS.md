# Status — Current Phase

**Phase 0: Environment & data — in progress**  
Last updated: 2026-10-06 (after `84f7fed`; known issues 1/2/4 fixed — uncommitted)

---

## Where we are

```
Phase 0 ██░░░░░░░░  env done, dataset not yet downloaded
```

Phases 1–8 not started — see [ROADMAP.md](ROADMAP.md).

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
- [ ] **Run** `scripts/download_voc.py` — dataset not on disk yet
- [ ] `scripts/inspect_data.py` — file exists but is **empty (0 bytes)**
- [ ] Acceptance checklist below

### Acceptance checklist

| # | Check | Result |
|---|---|---|
| 1 | `import torch` → `2.14.1+cpu` | **pass** (verified) |
| 2 | Download completes; `trainval=5011`, `test=4952` | not run |
| 3 | `len(VOCDataset('data/VOCdevkit','trainval')) == 5011` | blocked (no data); import works from repo root |
| 4 | `d[0]` shapes/dtypes/bounds correct | blocked |
| 5 | `inspect_data.py` → `runs/inspect_data.jpg` eyeball pass | not written |
| 6 | DataLoader `batch_size=4, num_workers=2` iterates | blocked |
| 7 | `df -h` ≥ 13 GB free | pass (17 GB) |

## Known issues

1. ~~**`.gitignore` no longer ignores the dataset.**~~ **FIXED** — `# data/`
   replaced with `data/downloads/` + `data/VOCdevkit/`; `data/voc.py` stays
   tracked. Verified: dataset dirs ignored, `data/voc.py` not ignored.

2. ~~**Dataset path mismatch (double `VOCdevkit`).**~~ **FIXED** —
   `data/voc.py:31` now `self.root / "VOC2007"`. Verified: resolves to
   `data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt` (matches download
   output).

3. **Editable install exposes nothing.** (open — decide before Phase 1)  
   `pyproject.toml` has `include = ["CandyEye*"]` but no package named
   `CandyEye` exists (modules are top-level `data/`, `core/`, … with **no
   `__init__.py`**). Result: `top_level.txt` is empty, and
   `from data.voc import …` only works when cwd is the repo root — it fails
   from anywhere else, and `python scripts/foo.py` fails outright
   (`scripts/` becomes `sys.path[0]`, not the repo root).  
   **Workaround until fixed:** run scripts from repo root with `PYTHONPATH=.`
   (verified working).  
   **Proper fix (decide before Phase 1):** restructure into a real package,
   e.g. `candyeye/data/voc.py` + `__init__.py` files + matching
   `pyproject.toml` include.

4. ~~**`CandyEye.egg-info/` is tracked in git.**~~ **FIXED** — untracked via
   `git rm -r --cached`, ignored via `*.egg-info/` (files remain on disk).

5. **`scripts/inspect_data.py` is 0 bytes** — written in this phase.

6. *Minor:* `Tqdm_Download` class in `download_voc.py` is dead code (a second
   progress-bar impl, never used — `download_file` builds its own tqdm). Safe
   to delete whenever.

## Next actions

1. Commit the issue 1/2/4 fixes
2. `python scripts/download_voc.py` from repo root (~870 MB, verify counts)
3. Write `scripts/inspect_data.py`, eyeball `runs/inspect_data.jpg`
4. DataLoader smoke test
5. Close acceptance checklist, commit, review → **Phase 1 (model)**
6. *(before Phase 1)* decide on the packaging restructure (issue 3)

---

Update this file at every checkpoint: tick boxes, refresh "Last updated",
move the phase bar when a phase closes.
