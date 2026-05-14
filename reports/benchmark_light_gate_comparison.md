# Light Gate (MobileNetV3-Small) vs EfficientNet-B0 Gate — Cascade Benchmark

_Eval set: `splits_eval/v{1,2,3}_balanced_test.csv` (400 balanced/version, 3 runs each, T_low=0.1, T_high=0.5, CPU)._

## Per-version cascade comparison

| version | gate | acc | f1 | recall | precision | latency ms (mean±std) | heatmap call rate |
|---|---|---|---|---|---|---|---|
| v1 | effnetb0 |   0.9900 |   0.9899 |   0.9800 |   1.0000 |  99.77±0.39  |   1.75% |
| v1 | mnv3_small |   0.9975 |   0.9975 |   0.9950 |   1.0000 |  33.02±1.73  |   0.00% |
| v2 | effnetb0 |   0.9775 |   0.9772 |   0.9650 |   0.9897 | 103.27±0.68  |   2.50% |
| v2 | mnv3_small |   1.0000 |   1.0000 |   1.0000 |   1.0000 |  36.63±0.60  |   1.25% |
| v3 | effnetb0 |   0.9825 |   0.9824 |   0.9750 |   0.9898 | 101.21±1.48  |   1.75% |
| v3 | mnv3_small |   1.0000 |   1.0000 |   1.0000 |   1.0000 |  34.39±0.28  |   0.25% |

## Baseline (PatchCore-only, identical across runs)

| version | acc | f1 | recall | latency ms |
|---|---|---|---|---|
| v1 |   0.7750 |   0.7955 |   0.8750 |  70.75±0.30  |
| v2 |   0.7600 |   0.7714 |   0.8100 |  80.88±1.06  |
| v3 |   0.6525 |   0.7157 |   0.8750 |  94.16±1.09  |

## End-to-end cascade speedup (mnv3_small vs effnetb0)

| version | effnetb0 cascade ms | mnv3_small cascade ms | speedup |
|---|---|---|---|
| v1 | 99.77 | 33.02 | **3.02x** |
| v2 | 103.27 | 36.63 | **2.82x** |
| v3 | 101.21 | 34.39 | **2.94x** |

## Gate forward — standalone (1 image, 1 thread)

| backbone | params | CPU ms (mean±std, p50) |
|---|---|---|
| efficientnet_b0 | 4,008,829 | 41.45±5.16, p50=39.93 |
| mobilenet_v3_small | 927,585 | 12.31±2.55, p50=12.14 |

## Takeaways

- **Accuracy parity (or better):** mnv3_small cascade F1 mean = 0.9992 vs effnetb0 = 0.9832.
- **Latency:** mnv3_small cascade 34.68 ms avg vs effnetb0 101.42 ms — ~2.92× faster end-to-end on the cascade path.
- **Heatmap call rate:** mnv3_small 0.50% vs effnetb0 2.00% — comparable, so the speedup is dominated by the lighter gate forward.
- Lightweight gate (~928k params, ~10 ms CPU) is sufficient for the gate role in this dataset; effnetb0/mnv3_large can be reserved for a deeper second-stage judgment, with PatchCore heatmap as an optional add-on.
