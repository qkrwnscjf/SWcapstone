# Balanced vs Original Benchmark Comparison

원래 평가(`benchmark_v1v2v3.json`, n=300)는 `--limit 300`이 CSV 앞에서부터 잘라
v3는 anomaly-only(300/0), v2는 277/23, v1은 222/78의 불균형 셋이 되어 있었음.
이번 재평가(`benchmark_v1v2v3_balanced.json`, n=400)는 각 버전 anomaly:normal=1:1 (200+200, seed=42).

## 1. Baseline (PatchCore 단독) — 정확도

| ver | original acc (n_norm) | balanced acc | Δ |
|---|---:|---:|---:|
| v1 | 0.8067 (78 norm)  | 0.7925 | −0.0142 |
| v2 | 0.9300 (23 norm)  | 0.7225 | **−0.2075** |
| v3 | 1.0000 (0 norm)   | 0.7250 | **−0.2750** |

해석:
- **v1은 거의 변하지 않음** → 원래도 비교적 정직한 평가였음.
- **v2/v3의 90~100%대 정확도는 평가 셋의 anomaly 편중에서 오는 거짓 상승**. balanced에서 보면 세 버전 baseline은 ~72~79% 비슷한 수준.
- v3 baseline이 v2보다 살짝 높지만 의미 있는 차이 아님 (모두 ±0.03 안쪽).

## 2. Gate Cascade — 정확도와 라우팅

| ver | original acc | balanced acc | original heatmap_call_rate | balanced heatmap_call_rate |
|---|---:|---:|---:|---:|
| v1 | 0.8067 | 0.7925 | 1.000  | 1.000  |
| v2 | **0.0700** | 0.7550 | 0.007 | 0.255 |
| v3 | **0.9933** | 0.7225 | 0.013 | 0.933 |

원래 결과의 v2 “0.07 (완전 망가짐)”과 v3 “0.9933 (거의 무손실)”은 **둘 다 anomaly-only/heavy 셋에서 게이트 confidence 분포가 한쪽으로 쏠려 만든 환상**.
balanced에서 보면:
- **v1 게이트는 여전히 무용지물** (전부 PatchCore로 넘김) — 정확도는 baseline과 동일하고 latency만 +7ms.
- **v2 게이트는 normal로 과도하게 라우팅** — recall 0.51 (FN=98), precision 1.0. acc=0.755로 baseline(0.7225)보다 **오히려 좋음**, 단 anomaly 절반 놓침.
- **v3 게이트는 거의 모든 샘플을 uncertain으로 보냄** (heatmap_call_rate 0.933) — 사실상 baseline과 같이 동작. 정확도도 baseline에 근접.

## 3. 추론 시간 (per-image, ms)

| ver | pipeline | original mean | balanced mean | balanced std (run간) | balanced run내 std 평균 |
|---|---|---:|---:|---:|---:|
| v1 | baseline     | 110.62 | 95.19  | 0.79 | 19.05 |
| v1 | gate_cascade | 125.25 | 102.67 | 1.05 | 18.99 |
| v2 | baseline     | 165.20 | 168.70 | 6.40 | 26.42 |
| v2 | gate_cascade | **9.38** | **51.75** | 4.16 | 74.26 |
| v3 | baseline     | 239.06 | 241.79 | 3.07 | 33.48 |
| v3 | gate_cascade | **10.74** | **233.43** | 1.97 | 68.70 |

해석:
- v3 gate_cascade의 **“22배 속도 개선”은 fake**였음. balanced에서 latency 233ms ≈ baseline 242ms — 게이트가 거의 모든 이미지를 PatchCore로 넘기므로 속도 이득 사실상 없음.
- v2 gate_cascade는 51.75ms로 여전히 baseline 대비 약 3.3배 빠름 (heatmap_call_rate 25.5%) — 단, recall 0.51로 “빠르지만 anomaly를 절반 놓침”.
- v1은 latency 거의 같음 (게이트가 100% 우회 → 게이트 추가 비용만큼 느림).

## 4. 데이터 버전 효과 (balanced 기준)

각 버전 baseline에서:
- recall: v1=0.84, v2=0.84, v3=0.795
- precision: v1=0.767, v2=0.680, v3=0.697
- F1: v1=0.802, v2=0.752, v3=0.743

**v3가 “가장 좋다”는 결론은 잘못됨.** balanced에서는 v1 baseline이 약간 더 좋고, v2/v3는 비슷한 수준. 증강이 늘어났지만 (additional set의 anomaly가 v1<v2<v3로 다양해지면서) anomaly 변동성이 커져 baseline PatchCore가 더 어려워한 것으로 해석 가능.

## 5. 결론

1. **원래 보고서의 v2 게이트 0.07 / v3 게이트 0.9933 / v3 baseline 100%는 모두 평가 셋 불균형이 만든 인공물.** 실제 모델 성능 비교에 부적합.
2. **balanced 평가에서 가장 흥미로운 결과**: v2 gate_cascade — precision 1.0, latency 3.3× 단축. recall은 절반이지만 “정확도가 중요하지만 anomaly를 놓치는 것보다 false positive 줄이는 게 우선”인 시나리오에서는 의미 있음.
3. **v1/v3 게이트는 현재 사실상 무용지물.** v1은 gate가 자신감 없어서 전부 통과, v3는 gate가 너무 자신감 있어서 거의 전부 uncertain으로 분류. 둘 다 t_low/t_high 또는 게이트 재학습 필요.
4. **run 간 std=0** 유지 — 결정론적 평가. 실제 통계적 변동성을 측정하려면 매 run마다 sample 셔플 필요 (현재 `--shuffle`은 한 번만 적용).

## 6. 권장 다음 단계

1. **임계값 스윕**: `scripts/threshold_sweep.py`로 v1, v3 게이트의 t_low/t_high 재탐색 (목표: heatmap_call_rate를 합리적인 0.2~0.5 영역으로).
2. **v2/v3 게이트 재학습 또는 캘리브레이션 재조정**: balanced set에서 calibrated probability 분포 확인.
3. **운영 시나리오별 평가 매트릭스**: 단순 accuracy 외에 “precision-우선 (false positive 최소화)” 시나리오에서 latency-accuracy trade-off 곡선 그리기.
