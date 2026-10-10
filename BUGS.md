# Bugs found and fixed (Phases 3–6)

This document summarizes all bugs identified during the read-only audit, with concrete repro evidence and the minimal fixes applied. The fixes change no existing test contracts.

## HIGH (H1–H4)

### H1 — Trainer mAP ignored `difficult` objects
**Locations:** `data/voc.py:193–211` (collate_fn), `eval/detection_metrics.py:11–76, 140–176` (evaluation).

**Problem:** `collate_fn` dropped the `difficult` flag (only returned `images`, `targets`, `img_ids`), and `evaluate_detector` hardcoded `"difficult": np.zeros(...)` for every GT. This made difficult objects count as non-difficult targets: a detection on a difficult GT was treated as a false positive in P/R and could also be paired incorrectly in the confusion matrix. `eval/map.py` already implemented correct difficult semantics (ignore dets matched to difficult GT, don't count them as FP and don't count difficult GT as a miss), but its inputs came from `evaluate_detector` with an all-zero mask.

**Evidence:** `collate_fn([sample with difficult=[True]])` produced keys `{'images','img_ids','targets'}`; `evaluate_detector` wrote all `difficult = 0`. `scripts/evaluate.py` reads labels directly (so it handled `difficult` correctly) while trainer evaluation didn't — this explained disagreements between trainer-reported mAP and script-reported mAP for the same checkpoint (swing observed up to ~±0.19).

**Fix:** `collate_fn` now returns `"difficult": [...]` (list of boolean tensors per image, 1 per target row in sample order, falling back to all `False`). `evaluate_detector` reads `batch.get("difficult")[batch_index]` when building `ground_truth`; for P/R it builds easy/difficult views (detections whose best GT is difficult are **ignored** — neither TP nor FP), and for the confusion matrix it excludes difficult GTs from FP-to-background accounting and ignores detections that only matched a difficult GT (same VOC behavior). New tests: `tests/test_data_pipeline.py::test_collate_carries_difficult_flags_per_image`, `tests/test_detection_metrics.py::test_difficult_objects_are_ignored_in_map_precision_and_confusion`.

### H2 — Assigner conflict resolution ignored containment (could assign to a non-claiming/non-containing GT)
**Location:** `training/assigner.py:71–75`.

**Problem:** when multiple GTs "claimed" an anchor, conflict resolution did `best = ious.argmax(0)` over **all** pairwise IoUs (unmasked). An anchor that was already `positives[:, a]` only because it was inside/claimed by GT0 could be reassigned to the GT with the largest IoU against the *predicted box* — including a GT that never claimed it and may not contain the anchor point. The resulting `target_boxes` can put the anchor **outside** the assigned target box, causing `target_ltrb` (lat/lon/rd/bt) to be negative → clamped to 0 and producing wrong box supervision. The 2025-10-08 `quality.clamp(min=0.25)` floor then also propagated a soft score computed from the "wrong" GT.

**Evidence:** random stress (300 trials, 16 anchors each) with the old code could attach positives whose target box did **not** contain their anchor point; after the fix this becomes 0/1152 in the sampled case. Adversarial repro showed an anchor claimed only by a tiny GT was correctly assigned to that tiny GT with quality ≤ 0.2501, whereas the unmasked-max allowed the unclaimed large GT's metric/normalization to dominate.

**Fix:** restrict the `argmax` to GTs that *actually claimed* the anchor (`positives` mask): `best = ious.masked_fill(~positives, -1).argmax(0)`. This guarantees the assigned GT claimed the anchor (and since `positives ⊆ in_gt`, the anchor point is inside that GT's box by construction). Test: `tests/test_assigner.py::test_conflict_resolution_keeps_anchor_inside_assigned_target`.

### H3 — Soft-target quality numerator not masked to the assigned (claiming) GT
**Location:** `training/assigner.py:89–91`.

**Problem:** the per-anchor quality was computed as `quality = (metric * max_iou / (max_metric + eps)).amax(0)` after normalizing by `max_metric = metric.masked_fill(~positives, 0).amax(1, keepdim=True)` (GT-wise max over *claiming* ones) but the **numerator** still used `metric` (unmasked by the final assignment) in the product with `max_iou` in a way that could let an alignment computed against a *different* GT win the final `amax(0)` across the unmasked metric rows. In effect, a soft score could be derived from an unclaimed GT even though the anchor was assigned to another GT — directly corrupting `target_scores` for that positive.

**Evidence:** adversarial two-GT case (small GT claims anchor0; large GT contains it but ranks anchor0 outside its top-k): after the fix the assigned GT is the small one and `target_scores` for that (anchor,label) are ≤ 0.2501 (the clamped alignment of the assigned GT). The old unmasked logic allowed the large GT's high metric to leak through.

**Fix:** mask `metric` to the final positives before normalizing: `metric_pos = metric.masked_fill(~positives, 0); max_metric = metric_pos.amax(1, keepdim=True); quality = (metric_pos * max_iou / (max_metric + eps)).amax(0)` (with `quality = quality.clamp(min=0.25)` unchanged). Test: `tests/test_assigner.py::test_target_quality_comes_from_the_assigned_gt`.

### H4 — Out-of-range class ids silently dropped (mAP inflation)
**Location:** `eval/map.py:29–93`.

**Problem:** `evaluate_map50` iterated only over `range(num_classes)` and filtered detections/GT by class, so a detection with `class_id >= num_classes` or a GT label outside `[0,num_classes)` was never considered — it **disappeared** entirely. That means such a detection produced no FP (and the corresponding GT produced no miss) — an impossible silent correctness bug that can artificially inflate mAP. The two code paths (AP and the subsequent per-class P/R in `evaluate_detector`) behaved inconsistently in failure semantics.

**Evidence:** calling `evaluate_map50` with `num_classes=2` and a detection `class_id=5` returned without error and effectively ignored it (no FP recorded in that class loop), hiding a dataset/model `nc` mismatch.

**Fix:** scan all GT labels and all detections up front; if any class id is outside `[0,num_classes)`, raise `ValueError(f"class ids {sorted(bad_ids)} outside 0..{num_classes - 1}: model/dataset num_classes mismatch")`. Tests: `tests/test_map.py::test_out_of_range_detection_class_raises`, `tests/test_map.py::test_out_of_range_gt_label_raises`.

## MEDIUM (M2–M9 + M1/M10/M11 context)

### M2 — `CandyEyePredictor` ignored checkpoint `img_size`
**Location:** `inference/predict.py:133–174, 188`.

**Problem:** `CandyEyePredictor.__init__` defaulted `imgsz = 128` and passed it through even when loading a `.pt` checkpoint that stored `config["model"]["img_size"]` (e.g. `64`). Letterboxing and model construction then used the wrong input size, changing the effective resolution and anchor grid alignment relative to training.

**Evidence:** a checkpoint trained at `imgsz=64` with `config.model.img_size=64` was loaded as `imgsz=128` unless explicitly overridden.

**Fix:** `imgsz` is now `int(imgsz if imgsz is not None else saved_config.get("model", {}).get("img_size", 128))`; the constructed `CandyEye` uses `self.imgsz`. Test: `tests/test_predict.py::test_predictor_uses_checkpoint_img_size`.

### M3 — `letterbox` could create zero-width/height resized tiles
**Location:** `inference/predict.py:48–57`.

**Problem:** `nh, nw = int(round(h*scale)), int(round(w*scale))` can round to 0 for images with extreme aspect ratios (e.g. 4000×3). `cv2.resize` will then receive invalid sizes or the paste region can be a no-op in a surprising way; the function previously didn't guard against this.

**Evidence:** constructed 4000×3 image → `scale = min(64/4000, 64/3)~0.016`; `nw = round(3*0.016)=0` → invalid resize.

**Fix:** clamp to at least 1: `nh, nw = max(1, int(round(h*scale))), max(1, int(round(w*scale)))`. Test: `tests/test_predict.py::test_letterbox_survives_extreme_aspect_ratio`.

### M4 — YOLO label path mapping was brittle; missing labels silently became background
**Location:** `data/yolo.py:45–54, 62–110`.

**Problem:** `label_path` only tried `images→labels` replacement or a single `parent.parent/labels/...` fallback. Layouts like `train/foo.jpg` + `train/labels/foo.txt` (no `images/` dir) were mapped to the wrong location and returned a non-existent path — `YoloTxtDataset` treated that as "background image" (no boxes) with **no error**. Also, if the entire dataset was mis-mapped, it silently became all-background.

**Evidence:** in a flat `train/` layout with a `train/labels/` sibling, the primary logic preferred `train.parent.parent/labels/foo.txt` which didn't exist, so no labels were loaded and the dataset appeared empty-of-annotations.

**Fix:** `label_path` tries candidates in order: `.../images/...`→`.../labels/...` (last `images/` dir), `image_path.parent.parent / "labels" / ...`, `image_path.parent / "labels" / ...`; returns the first that exists, else the primary mapping. In `__init__`, preflight: if `images` non-empty and **zero** label files exist for them, raise `ValueError("no label files found … check the dataset's images/labels layout")`; if some are missing, print a single warning with counts (preserves existing behavior of treating missing files as background). Tests: `tests/test_yolo_data.py::test_label_path_falls_back_without_images_dir`, `tests/test_label_path_prefers_images_to_labels_mapping`, `tests/test_dataset_raises_when_no_labels_anywhere`.

### M5/M11 — CSV schema drift corrupted resumes and crashed plots
**Locations:** `training/trainer.py:36–80 (_save_training_plots)`, `training/trainer.py:263–308 (log header/write)`.

**Problem:** old runs used 7-column metrics (`epoch,lr,loss,box,cls,dfl,foreground`); new code writes 60+-column CSVs with `train_loss/val_loss/.../map50` and per-class columns. On resume, the code did `write_header = not log_path.exists() or not resume` and always appended — resuming into an old-schema CSV produced a mixed header+rows file. Later `_save_training_plots` unconditionally accessed keys like `row["train_loss"]`, `row["val_loss"]`, `row["map50"]` and raised `KeyError` (which aborted the end-of-training artifacts block in some cases). Also, if a previous run died between opening `w` and writing the header, `metrics.csv` could be 0 bytes (resume would then append without a header in that edge case).

**Evidence:** `fruit_finetune2/metrics.csv` was header-only (crashed mid-epoch 1); resuming from a checkpoint into a directory that still held an old-schema log was observed to be problematic in the audit. Plots crashed on old schemas.

**Fix:** Build the header row once; on resume, read the existing header (if any). If it differs from the expected header → **rotate** the existing log to `metrics_prev_<YYYYMMDD-HHMMSS>.csv` (with a printed notice) and write a fresh header. In `_save_training_plots`, check `required = {"epoch","lr","train_loss","val_loss",...,"map50"}` is a subset of `rows[0]` keys; if not, print `"Plots skipped: metrics schema lacks ... columns"` and return `None` (graceful degradation, no KeyError). Tests: `tests/test_trainer_integrity.py::test_resume_rotates_old_schema_metrics_log`, `tests/test_plots_skip_gracefully_on_old_schema`.

### M6 — `max_batches=0` → `UnboundLocalError: 'lr'` and `max_batches` validation
**Location:** `training/trainer.py:226–284`.

**Problem:** `steps_per_epoch = min(len(loader), max_batches) if max_batches else len(loader)` treats `max_batches=0` as falsy → uses `len(loader)` (no guard), but the step loop checks `if max_batches is not None and step >= max_batches: break`. If `max_batches==0`, the loop never executes any step (breaks immediately or runs 0 steps), so `sums` remain all zeros, `means = 0.0`, validation runs, but **`lr` is never assigned** (it's only set inside the step loop: `lr = base_lr * factor` at `:284`). The CSV row later writes `log.writerow([... , lr, ...])` → `UnboundLocalError: 'lr'`. Also `max_batches < 1` should be rejected explicitly.

**Evidence:** matches the signature of `runs/train/fruit_finetune2` (header-only CSV, no checkpoints) — ran past validation into log writing with `lr` unbound.

**Fix:** upfront validation `if max_batches is not None and max_batches < 1: raise ValueError(f"max_batches must be >= 1, got {max_batches}")`; compute `steps_per_epoch = len(loader) if max_batches is None else min(len(loader), max_batches)`; initialize `lr = base_lr` before the epoch loop. Test: `tests/test_trainer_integrity.py::test_max_batches_zero_is_rejected`.

### M7 — No up-front `nc` ↔ class-names check on VOC path (crashed after epoch 1)
**Location:** `training/trainer.py:215–216` (was after building datasets).

**Problem:** The YOLO-txt branch already rejected `data_nc != model_nc` at `:188–194`, but the VOC branch never checked that `model_nc == len(class_names)` (VOC_CLASSES length). If the model was constructed with `nc=11` but dataset is VOC (20 classes) or names count mismatched, `evaluate_detector` raises `ValueError("class_names length must equal num_classes")` **after** completing a full epoch (and after possibly writing the epoch row/checkpointing in the old structure) — losing that epoch's work and producing a confusing error message far from the root cause.

**Evidence:** audit noted crash after epoch 1 with mismatch between model nc and dataset class count.

**Fix:** After choosing `class_names` (for both branches), do `if len(class_names) != model_nc: raise ValueError(f"model has nc={model_nc}, but the dataset provides {len(class_names)} class names; construct the model with matching nc (or fix the dataset names)")`. Test: `tests/test_trainer_integrity.py::test_model_class_count_mismatch_raises_before_training`.

### M8 — CLI re-runs silently clobbered `metrics.csv`
**Location:** `training/trainer.py:505–511 (train_model rename)`, `training/trainer.py:551–567 (main CLI)`, `training/trainer.py:261 (run_training mkdir/log)`.

**Problem:** `main()` calls `run_training` directly with no `exist_ok` and no directory rename. If the target output already contains `metrics.csv` (a previous run), `run_training` opened `metrics.csv` in `"w"` mode (when `resume is None`) — truncating the old log. `train_model` had a rename loop (to `name2`, `name3`, …) but the **CLI entrypoint did not**. So `python -m training.trainer configs/experiments/tiny_scratch.yaml` could silently destroy the previous experiment's `metrics.csv` (and overwrite `best.pt`/`last.pt` on the next improvement).

**Evidence:** `fruit_finetune4` and others showed overwritten/ambiguous directories in the observed runs; the rename logic existed only in `train_model`, not in `run_training`/`main`.

**Fix:** Move the safety check into `run_training` (single source of truth): if `resume is None and not exist_ok and (output / "metrics.csv").exists()`, pick the next available sibling name (`name2`, `name3`, …), print the rename notice, update `config["name"]` and `config["train"]["output"]`, then proceed. Add `--exist-ok` to `main()` (also plumbed through `train_model`→`run_training`); on resume the check is skipped. Test: `tests/test_trainer_integrity.py::test_second_run_renames_directory_instead_of_truncating_log` (second run uses a fresh renamed dir, original log preserved).

### M9 — End-of-training artifacts skipped on exceptions/interrupts; return paths sometimes lied
**Location:** `training/trainer.py:358–377, 537–541`.

**Problem:** The "best checkpoint → final evaluation → confusion matrix → plots" block ran **after** the epoch loop but **outside** any `try/finally`. Any exception inside the loop (e.g. `FloatingPointError` at `291–292`, I/O error, or `KeyboardInterrupt`) skipped that entire block — no `results.png`, `confusion_matrix.csv`/`.png`, and the user got no "artifacts" printed. Also `train_model` returned hardcoded paths like `result_dir / "results.png"` even if they didn't exist (returned `None` only in the plotting function itself, not filtered at the API boundary).

**Evidence:** `fruit_finetune4` (5/30 epochs, then nothing) had no `results.png` or `confusion_matrix.*`; across all observed runs the artifacts block rarely/never completed in interrupted cases. `train_model`'s return dict advertised `plots`/`confusion_matrix` as existing paths unconditionally.

**Fix:** Wrap the epoch loop + CSV writing in `try/finally`. Track `completed` and `interrupted = (KeyboardInterrupt caught)`. In `finally`: if interrupted → attempt to save plots from the existing log (best-effort) and print a clear message; else if `best.pt` exists → reload best, run a final evaluation with `include_confusion_matrix=True` (guarded per step, catching and reporting exceptions without masking the original one), save confusion matrix and plots, print a summary (including epoch of best). If no checkpoint was produced (ended before epoch 1 finished) → print that. `train_model` now returns only paths that actually exist (`existing(path) or None`). Tests: `tests/test_trainer_integrity.py::test_epoch_failure_still_writes_end_of_training_artifacts` (best.pt, metrics.csv, confusion_matrix.csv exist even when the second validation raises).

### M10 — README promised plots but matplotlib missing (editable install)
**Locations:** `README.md:95–101`, environment.

**Evidence:** `egg-info`/`dist-info` from an editable install dated 2026-10-06 21:44 (predating `d0d3b2c` which added `matplotlib>=3.8` to dependencies), so the active venv lacked `matplotlib` and `results.png`/`confusion_matrix.png` were never produced in observed runs (code already degraded to CSV-only with warnings).

**Fix:** `pip install -e .` in the venv (installed `matplotlib 3.11.2`). README also softened wording (minor fix) and added a note about `val_split: test` bias (M1).

### M1 — `val_split: test` means model selection sees test set (documented)
**Location:** `configs/default.yaml:4`, `README.md:88–94`.

**Decision (user):** keep `val_split: test` as default and document the bias rather than changing the default behavior.

**Fix:** Added inline comment in `configs/default.yaml`: `val_split: test # selects best.pt; use a held-out split for unbiased benchmarks`. Added a README callout explaining `best.pt` selection uses the validation split (test by default) and advising users to set `val_split` to a held-out split for an unbiased benchmark.

### Core safety/robustness tweaks (supporting M2/M3)
- `core/candyeye.py:189–196` (`_detect_stride`): run probe in `eval()` mode and `decode=False` to avoid BatchNorm issues with 1×1 feature maps (e.g. `imgsz=32`); restore original training mode.
- `_save_training_plots` schema guard (M11) as above.

## Verification

- All existing tests still pass: **45 passed** at the time of this audit; the
  suite has since grown to **60 passed** (57 excluding the opt-in ONNX-parity
  tests).
- New unit tests cover the correctness bugs: difficult plumbing end-to-end, assigner conflict resolution and quality masking, `evaluate_map50` out-of-range rejection, YOLO label-path fallbacks and preflight, predictor `imgsz` resolution and letterbox edge case, trainer log-rotation/schema tolerance, `max_batches==0` rejection, nc↔names mismatch, output-dir rename, and end-of-run artifact preservation on exceptions.
- In-memory repros: H2 stress → 0/1152 positives with anchor outside assigned target (was >1%); H3 adversarial → assigned to claiming GT with `quality <= 0.2501`; H4 raises on out-of-range ids; H1 `collate_fn` carries `difficult` and evaluator ignores dets on difficult GT.

## Deferred (LOW/NIT, out of scope)
Per user scope (HIGH + MEDIUM), the following were not changed: loss batch-size invariance/tiny-GT edge cases, mosaic gate `split == "trainval"`, resume RNG/dir-identity, `"type": "yolo"` hardcode in checkpoint `config`, double forward per val batch (perf), `epochs=0` treated as unset, `_resolve_data(dir)` not looking for `data.yaml`, SPECS/README minor tree staleness. These remain documented in the earlier audit but are not introduced by these fixes.

## Files changed (summary)
- `data/voc.py` — collate carries `difficult`
- `data/yolo.py` — robust `label_path` + preflight
- `eval/detection_metrics.py` — use real `difficult`, VOC-consistent P/R + confusion
- `eval/map.py` — out-of-range validation
- `training/assigner.py` — mask conflict resolution + quality to claiming GTs
- `training/trainer.py` — header/schema rotation, `max_batches>=1`, nc check, `lr` init, `try/finally` artifacts, `exist_ok` + CLI rename, honest return paths, plot schema guard
- `inference/predict.py` — ckpt `imgsz` + `letterbox` `max(1)`
- `core/candyeye.py` — stride probe in eval/decode=False
- `README.md`, `configs/default.yaml` — docs/notes
- `tests/*` — new tests + one small extension to existing tests

---

## Addendum (2026-10-10) — ScaleExchange N-level refactor

### H5 — `ScaleExchange` positional parsing ate a trailing channel as `iters`
**Locations:** `candyeye/core/modules/exchange.py` (`__init__`),
`candyeye/core/candyeye.py` (`parse_model`).

**Problem:** the in-flight refactor from a fixed 3-level signature
(`c1, c2, c3, gate="none", iters=1`) to arbitrary level counts made the
constructor guess that *any* trailing int was `iters`. That is ambiguous when a
gate is given by keyword or omitted: a 4-level channel list such as
`ScaleExchange(64, 128, 256, 32)` was parsed as `channels=(64,128,256)` with
`iters=32` — silently dropping the finest (P2-class) level. `parse_model`
also spliced the gate/iters into a positional arg list, which further confused
the heuristic. Symptom in tests: wrong channel counts / `iters` for positional
and keyword call sites (test failures, no silent corruption in memory because
the channel mismatch surfaced on the first forward).

**Fix:** the constructor accepts both keyword form (`channels..., gate=, iters=`)
and the YAML-positional form (`channels..., gate, iters`); `iters` is read from
the positional tail **only** when it immediately follows a positional gate
string, so a bare trailing int is always a channel. `parse_model` now constructs
`ScaleExchange(*channels, gate=gate, iters=iters)` with explicit keywords.
Full suite: 60 passed.

### Extra hunk (pre-existing, working tree)
`nn.Upsample` in `parse_model` now resolves `c2 = ch[f]` when `f` is a
non-negative absolute reference (was always `cur`). Kept; no test contract
changed. **Not** from this session's changes — was already uncommitted in the
working tree.
