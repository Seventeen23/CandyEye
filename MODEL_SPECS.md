# MODEL_SPECS.md

Real specifications for CandyEye (fp32, CPU benchmarked on this machine: AMD Ryzen 5 PRO 3500U, AVX2, torch 2.14.1+cpu).

## Profiles

- **a**: `yolo11_p2` — P2/P3/P4/P5 (strides 4/8/16/32). Default profile (published general model). Designed for 320; works at 128/640 (div by 32).
- **b**: `yolo11_p2_small` — P2/P3/P4 (strides 4/8/16). Small-object only; drops large head.
- **c**: `yolo11` — P3/P4/P5 (strides 8/16/32). Fastest; stock.

## Measured (forward, decode=False, torch.no_grad, CPU 8 threads)

### Parameters (trainable)
| Config | nc=9 | nc=20 |
|---|---:|---:|
| base (c) | 2,591,595 | 2,593,740* |
| p2_small (b) | 2,435,195 | 2,437,300 |
| p2 (a) | 2,636,740 | 2,638,885 |

*legacy baseline.

### FLOPs (GFLOPs @1x, decode=False)
| Config | 128 | 320 | 640 |
|---|---:|---:|---:|
| base (c) | 1.35 | 8.44 | 34.04 |
| p2_small (b) | — | 13.31 | — |
| p2 (a) | — | 13.56 | — |

### CPU latency (ms/img, 100 runs after 20 warmups, decode=False)
| Config | 128 | 320 | 640 |
|---|---:|---:|---:|
| base (c) | 24.6 | 65.9 | — |
| p2_small (b) | — | 101.1 | — |
| p2 (a) | 33.2** | 98.3 | 290.3 |

**approximate at 128.

Note: decode/NMS omitted (timed raw maps). At inference time decode dominates for 4-head models — expect higher end-to-end ms.
