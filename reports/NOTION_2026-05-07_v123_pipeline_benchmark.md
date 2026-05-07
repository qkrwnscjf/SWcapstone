# v1·v2·v3 데이터 버전별 Baseline vs Gate-Cascade 파이프라인 성능 비교

> **담당 영역**: 새로운 데이터 버전 3개(v1/v2/v3)에 대해, Baseline (PatchCore-only)과 Gate-Cascade (EfficientNet Gate → PatchCore) 파이프라인의 정확도·추론시간을 평균/표준편차로 평가.
> **Date**: 2026-05-07

## 1. 평가 대상

| 항목 | 내용 |
|---|---|
| 데이터 버전 | v1, v2, v3 (각각 별도 split) |
| 평가 셋 | `splits_eval/v{1,2,3}_balanced_test.csv` — 각 400장 (anomaly:normal = 200:200) |
| Baseline | PatchCore (ResNet18 backbone, coreset_ratio=0.10, k=9) — 모든 입력에 heatmap 추론 |
| Gate-Cascade | EfficientNet-B0 게이트 → 임계 외부면 즉시 결정, 중간대(`T_low ≤ p ≤ T_high`)만 PatchCore 호출 |
| 임계값 | T_low = 0.1, T_high = 0.5 (게이트 학습 시 권장값) |
| 반복 | 각 버전·파이프라인 3 run, latency는 run간 평균 |
| 디바이스 | CPU |
| 스크립트 | `scripts/benchmark_pipeline.py` |
| 결과 원본 | `reports/benchmark_v1v2v3_balanced_new.{json,md,log}` |

## 2. 정확도 (Accuracy / F1)

| ver | pipeline | accuracy | precision | recall | F1 | TP / FP / TN / FN |
|---|---|---:|---:|---:|---:|---|
| v1 | baseline | 0.7750 | 0.7292 | 0.8750 | 0.7955 | 175 / 65 / 135 / 25 |
| v1 | **gate_cascade** | **0.9900** | **1.0000** | 0.9800 | **0.9899** | 196 / 0 / 200 / 4 |
| v2 | baseline | 0.7600 | 0.7364 | 0.8100 | 0.7714 | 162 / 58 / 142 / 38 |
| v2 | **gate_cascade** | **0.9775** | 0.9897 | 0.9650 | **0.9772** | 193 / 2 / 198 / 7 |
| v3 | baseline | 0.6525 | 0.6055 | 0.8750 | 0.7157 | 175 / 114 / 86 / 25 |
| v3 | **gate_cascade** | **0.9825** | 0.9898 | 0.9750 | **0.9824** | 195 / 2 / 198 / 5 |

> **표준편차 주석**: 모델이 결정론적(eval 모드, no dropout, 고정 입력 순서)이므로 3 run의 정확도 지표 std는 모두 0.0000. 따라서 표에서 ±std 컬럼은 생략. 변동성은 추론시간(latency)에만 의미 있게 발생.

### F1 개선폭 (gate_cascade − baseline)

| ver | ΔF1 | Δaccuracy |
|---|---:|---:|
| v1 | **+0.1944** (+19.4%p) | +0.2150 |
| v2 | **+0.2058** (+20.6%p) | +0.2175 |
| v3 | **+0.2667** (+26.7%p) | +0.3300 |

세 버전 모두에서 cascade가 baseline을 압도. **버전 난이도가 가장 높은(분포 변화가 큰) v3에서 개선폭이 가장 큼** — 게이트가 "확실히 정상/이상"인 표본을 깔끔히 분리하고 PatchCore가 처리해야 할 애매한 표본 자체를 줄여서 false positive를 크게 억제.

## 3. 추론시간 (per-image latency, ms)

| ver | pipeline | mean | per-image std | median | p95 | total (400장) |
|---|---|---:|---:|---:|---:|---:|
| v1 | baseline | 70.75 | 16.64 | 66.12 | 109.32 | 28,299 ms |
| v1 | gate_cascade | 99.77 | 22.22 | 93.67 | 161.91 | 39,908 ms |
| v2 | baseline | 80.88 | 15.69 | 77.81 | 91.91 | 32,352 ms |
| v2 | gate_cascade | 103.27 | 28.79 | 94.43 | 174.38 | 41,309 ms |
| v3 | baseline | 94.16 | 21.94 | 88.97 | 130.01 | 37,664 ms |
| v3 | gate_cascade | 101.21 | 26.41 | 93.92 | 171.37 | 40,484 ms |

> per-image std = 한 run 내 400장 latency의 표준편차를 3 run 평균. mean/median/p95 모두 3 run 평균.

### Heatmap 호출률 (cascade의 PatchCore 우회 효과)

| ver | heatmap call rate |
|---|---:|
| v1 | **1.8%** (≈ 7/400) |
| v2 | **2.5%** (≈ 10/400) |
| v3 | **1.8%** (≈ 7/400) |

세 버전 모두 **PatchCore가 처리하는 입력은 전체의 2% 내외**. 나머지 ~98%는 게이트가 직접 short-circuit으로 결정.

### 추론시간 해석

- **현재 측정값으로는 cascade가 baseline보다 평균 +7~29 ms 느림.** 원인은 단일 CPU·배치=1 환경에서 (a) gate forward 자체가 ~70-90 ms 들고, (b) PatchCore 호출도 ~2%지만 무거워서 p95에 영향, (c) MPS/배치/오버헤드 최적화 없음.
- 정확도 개선폭(+20~27%p)을 고려하면 latency 비용은 작은 수준. p50(median) 차이는 v3에서 88.97→93.92ms로 ~5ms 정도.
- GPU 또는 배치화 시 gate forward 비용이 거의 무시할 수준이 되므로, **실서비스에서는 PatchCore 호출 2%만 의미 있는 latency 비용**으로 줄어들 것.

## 4. 결론

| 지표 | 결론 |
|---|---|
| 정확도 | gate_cascade가 세 버전 모두에서 F1 0.97+ 달성, baseline 대비 +20~27%p |
| 정밀도 | precision 0.99+ 로 false positive 0~2건 수준 (baseline은 v3에서 FP 114건) |
| 추론시간 (CPU 단일) | cascade가 평균 +7~29 ms 느리지만, 정확도 이득 대비 무시 가능 |
| 작업 분담 효율 | PatchCore 호출이 2%로 떨어져, GPU·배치화 시 cascade가 latency도 더 빠를 가능성 큼 |
| 권장 | **세 버전 모두 cascade를 default로 채택. v3처럼 분포가 어려운 경우일수록 cascade 효과 큼** |

## 5. 재현 방법

```bash
# 1. 학습
bash scripts/train_all_v123.sh        # v1/v2/v3 모두 PatchCore + Gate 학습

# 2. 벤치마크
python scripts/benchmark_pipeline.py \
  --versions v1 v2 v3 \
  --test-csv splits_eval/v1_balanced_test.csv splits_eval/v2_balanced_test.csv splits_eval/v3_balanced_test.csv \
  --gate-model models/v1_effnetb0_gate.pt models/v2_effnetb0_gate.pt models/v3_effnetb0_gate.pt \
  --gate-calib models/v1_effnetb0_calibrator.pkl models/v2_effnetb0_calibrator.pkl models/v3_effnetb0_calibrator.pkl \
  --patchcore-model models/v1_patchcore_r18_patchcore.pt models/v2_patchcore_r18_patchcore.pt models/v3_patchcore_r18_patchcore.pt \
  --t-low 0.1 --t-high 0.5 \
  --device cpu \
  --output reports/benchmark_v1v2v3_balanced_new.json
```

## 6. 부가 사항 (참고)

수정 이전 첫 실행에서는 cascade가 모든 버전에서 F1≈0 이 나왔음. 원인은 `backend/src/gate_model.py` 의 체크포인트 키 리매핑 버그(분류 head 가중치가 `strict=False`로 누락 로드됨). 수정 후 위 결과 확보. 자세한 내용은 [`fix/gate-load-key-remap`](https://github.com/SWcapstone/SWcapstone/tree/fix/gate-load-key-remap) 브랜치 참조.
