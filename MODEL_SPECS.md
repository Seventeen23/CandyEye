# MODEL_SPECS.md

Real specifications for CandyEye (fp32, CPU benchmarked on this machine: AMD Ryzen 5 PRO 3500U, AVX2, torch 2.14.1+cpu).

## Profiles

- **d (default)**: `mobilenetv3_small` — P3/P4/P5 (strides 8/16/32). Non-YOLO `MobileNetV3-Small` backbone + verbatim YOLO11n neck, ImageNet-initialised (torchvision). Lightest by parameters.
- **e**: `mobilenetv3_small_p2` — P2/P3/P4/P5 (strides 4/8/16/32). MobileNetV3-Small + SPPF/C2PSA + P2 fusion; adds small-object capability to `d`.
- **f**: `mobilenetv3_large_p2` — P2/P3/P4/P5 (strides 4/8/16/32). Full-depth `MobileNetV3-Large` + SPPF/C2PSA + P2; most capacity (~2.9× `d`).
- **g**: `mobilenetv3_large` — P3/P4/P5 (strides 8/16/32). `MobileNetV3-Large` + SPPF/C2PSA, no P2 (for large-object data; no extra P2 latency).
- **a**: `yolo11_p2` — P2/P3/P4/P5 (strides 4/8/16/32). Published general model. Designed for 320; works at 128/640 (div by 32).
- **b**: `yolo11_p2_small` — P2/P3/P4 (strides 4/8/16). Small-object only; drops large head.
- **c**: `yolo11` — P3/P4/P5 (strides 8/16/32). Fastest; stock.

## Measured (forward, decode=False, torch.no_grad, CPU 8 threads)

### Parameters (trainable)
| Config | nc=9 | nc=20 |
|---|---:|---:|
| base (c) | 2,591,595 | 2,593,740* |
| p2_small (b) | 2,435,195 | 2,437,300 |
| p2 (a) | 2,636,740 | 2,638,885 |
| mobilenetv3_small (d) | 1,434,547 | 1,436,692 |
| mobilenetv3_small_p2 (e) | 1,583,404 | 1,584,856 |
| mobilenetv3_large (g) | 4,099,227 | 4,101,372 |
| mobilenetv3_large_p2 (f) | 4,143,092 | 4,144,544 |

*legacy baseline.

### FLOPs (GFLOPs @1x, decode=False)
| Config | 128 | 320 | 640 |
|---|---:|---:|---:|
| base (c) | 1.35 | 8.44 | 34.04 |
| p2_small (b) | — | 13.31 | — |
| p2 (a) | — | 13.56 | — |
| mobilenetv3_small (d) | 0.15 | 0.94 | — |
| mobilenetv3_small_p2 (e) | 0.29 | 1.83 | — |
| mobilenetv3_large (g) | 0.27 | 1.65 | — |
| mobilenetv3_large_p2 (f) | 0.40 | 2.51 | — |

### CPU latency (ms/img, decode=False)
| Config | 128 | 320 | 640 |
|---|---:|---:|---:|
| base (c) | 24.6 | 65.9 | — |
| p2_small (b) | — | 101.1 | — |
| p2 (a) | 33.2** | 98.3 | 290.3 |
| mobilenetv3_small (d)† | 24.8 | 61.1 | — |
| mobilenetv3_small_p2 (e)† | 37.4 | 106.9 | — |
| mobilenetv3_large (g)† | 49.2 | 98.1 | — |
| mobilenetv3_large_p2 (f)† | 39.7 | 125.8 | — |

**approximate at 128.
†best-of-3 (60 runs, 4 threads) in a separate session. Latency on this machine is highly load-dependent; the c/a/b rows are from an earlier session, so compare only within a column and loosely. (An earlier profile-d figure of 61.1 ms at 128 was taken while the host was loaded — that value is actually d's *320* latency.)

Note: decode/NMS omitted (timed raw maps). At inference time decode dominates for 4-head models — expect higher end-to-end ms.

Latency note: contrary to an earlier note, MobileNetV3-Small (`d`) is **not** slower than stock YOLO11n (`c`) on this host — at 128 both land near ~25 ms and at 320 near ~60–66 ms. MobileNetV3-Large (`f`/`g`) costs roughly 1.5–2× `c` at equal input. Depthwise/SE blocks are still memory-bound, so a fuse/quantise or depthwise-friendly runtime remains the follow-up (see ROADMAP Phase 9).

## Training features

- **EMA (default on)**: decay `0.999`, warm-up `(1+u)/(10+u)` per update;
  shadow copy used for validation, `best.pt` selection, and inference. No
  wall-clock cost (frozen averaged copy updated once per optimizer step).
- **Scale jitter**: profile `a` defaults to `RandomAffine((0.5, 1.5), translate=0.1)`
  (scale + translate, no rotation).
- **Profiles**: `profile="d"` (default, `mobilenetv3_small`, ImageNet-init),
  `"e"` (`mobilenetv3_small_p2`), `"f"` (`mobilenetv3_large_p2`),
  `"g"` (`mobilenetv3_large`), `"a"` (`yolo11_p2`), `"b"` (`yolo11_p2_small`),
  `"c"` (`yolo11`) — pass to `train(...)` or `CandyEye.train(...)`.
- **ImageNet init**: the MobileNetV3 profiles (`d`/`e`/`f`/`g`) load the
  torchvision backbone (Small: 208 tensors; Large: 302 tensors; 0
  skipped/unexpected, bit-exact forward parity) by default; disable with
  `pretrained=False`. Heads (and SPPF/C2PSA) stay random-init.
- Checkpoints save live `model` + `ema` weights (+ `ema_enabled`, `updates`);
  `CandyEyePredictor` prefers `ema` when present.
