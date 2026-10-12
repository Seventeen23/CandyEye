# MODEL_SPECS.md

Real specifications for CandyEye (fp32, CPU benchmarked on this machine: AMD Ryzen 5 PRO 3500U, AVX2, torch 2.14.1+cpu).

## Profiles

- **d (default)**: `mobilenetv3_small` — P3/P4/P5 (strides 8/16/32). Non-YOLO `MobileNetV3-Small` backbone + verbatim YOLO11n neck, ImageNet-initialised (torchvision). Lightest by parameters, but depthwise/SE-heavy on CPU (see latency note).
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

*legacy baseline.

### FLOPs (GFLOPs @1x, decode=False)
| Config | 128 | 320 | 640 |
|---|---:|---:|---:|
| base (c) | 1.35 | 8.44 | 34.04 |
| p2_small (b) | — | 13.31 | — |
| p2 (a) | — | 13.56 | — |
| mobilenetv3_small (d) | 0.15 | 0.94 | — |

### CPU latency (ms/img, 100 runs after 20 warmups, decode=False)
| Config | 128 | 320 | 640 |
|---|---:|---:|---:|
| base (c) | 24.6 | 65.9 | — |
| p2_small (b) | — | 101.1 | — |
| p2 (a) | 33.2** | 98.3 | 290.3 |
| mobilenetv3_small (d)† | 61.1 | 153.6 | — |

**approximate at 128.
†measured on 4 threads (the 8-thread host was under load during the run).

Note: decode/NMS omitted (timed raw maps). At inference time decode dominates for 4-head models — expect higher end-to-end ms.

Latency note (d): MobileNetV3-Small has ~9× fewer FLOPs than profile c at 128 but ~2.5× the latency — depthwise 3×3/5×5 convolutions and squeeze-excite blocks are memory-bound and poorly amortised on this CPU. The mobile backbone is a **parameter/energy** win, not (yet) a CPU wall-clock win; fuse/quantise or a depthwise-friendly runtime is the follow-up (see ROADMAP Phase 9).

## Training features

- **EMA (default on)**: decay `0.999`, warm-up `(1+u)/(10+u)` per update;
  shadow copy used for validation, `best.pt` selection, and inference. No
  wall-clock cost (frozen averaged copy updated once per optimizer step).
- **Scale jitter**: profile `a` defaults to `RandomAffine((0.5, 1.5), translate=0.1)`
  (scale + translate, no rotation).
- **Profiles**: `profile="d"` (default, `mobilenetv3_small`, ImageNet-init),
  `"a"` (`yolo11_p2`), `"b"` (`yolo11_p2_small`), `"c"` (`yolo11`) — pass to
  `train(...)` or `CandyEye.train(...)`.
- **ImageNet init**: profile `d` loads the torchvision MobileNetV3-Small
  backbone (208 tensors, 0 skipped/unexpected; bit-exact forward parity) by
  default; disable with `pretrained=False`. Head stays random-init.
- Checkpoints save live `model` + `ema` weights (+ `ema_enabled`, `updates`);
  `CandyEyePredictor` prefers `ema` when present.
