# Gate Cascade 재현성 회복 — `gate_model.load` 키 매핑 버그 수정

> **요약**: v1/v2/v3 학습 직후 cascade 평가에서 F1≈0.0이 나오던 문제를 추적해, gate 체크포인트 헤드 가중치가 로드 시 사일런트로 누락되는 버그를 수정. 결과적으로 cascade F1이 baseline 대비 +20~27%p 회복.
> **브랜치**: [`fix/gate-load-key-remap`](https://github.com/SWcapstone/SWcapstone/tree/fix/gate-load-key-remap) · 커밋 `b3dc317`
> **Date**: 2026-05-07

## 1. 증상

`scripts/train_all_v123.sh` 학습 종료 후 `splits_eval/v{1,2,3}_balanced_test.csv`(각 400장, anomaly:normal=200:200)로 `scripts/benchmark_pipeline.py` 실행:

| ver | pipeline | F1 | TP/FP/TN/FN | heatmap call rate |
|---|---|---:|---|---:|
| v1 | baseline | 0.7955 | 175/65/135/25 | — |
| v1 | gate_cascade | **0.000** | 0/0/200/200 | 0.0% |
| v2 | gate_cascade | **0.000** | 0/0/200/200 | 0.0% |
| v3 | gate_cascade | **0.020** | 2/0/200/198 | 53.5% |

학습 로그상 v3 EfficientNet은 **Test AUROC 0.9953 / F1 0.9616** — 학습은 정상이었음.

## 2. 원인

`backend/src/gate_model.py:779-803` 의 키 리매핑이 잘못돼 있었음.

### 모델 구조 vs 체크포인트 키

```
Sequential(
  (0): EfficientNet(features=..., classifier=Identity())   # 백본 head는 Identity로 교체됨
  (1): Sequential(Dropout, Linear(1280, 1))                 # 실제 분류 head
)
```

| 체크포인트 키 | 모델 위치 | 기존 remap | 결과 |
|---|---|---|---|
| `features.*` | `0.features.*` | `→ 0.features.*` | ✅ |
| `classifier.1.{weight,bias}` | `1.1.{weight,bias}` | `→ 0.classifier.1.*` (Identity, 받는 곳 없음) | ❌ silently dropped |

`load_state_dict(strict=False)` 가 누락된 head 가중치를 경고 없이 무시 → **inference 단계의 head는 항상 random init**. 학습 루프는 같은 인-메모리 모델을 그대로 평가해서 좋은 점수가 나왔지만, 디스크 round-trip 후엔 무용지물.

### 왜 이전 round1 벤치마크는 "작동"한 것처럼 보였나

`reports/benchmark_v1v2v3_balanced.md`(t_low=0.3, t_high=0.7) 에서는 cascade와 baseline이 거의 동일했음. 깨진 head가 calibrated score를 0.5 근처 상수로 뱉었고, `[0.3, 0.7]` 사이라 **모든 샘플이 PatchCore로 라우팅** → cascade ≡ baseline. 게이트는 그때도 일을 안 하고 있었음.

## 3. 수정

**파일**: `backend/src/gate_model.py:779-803`

```python
for k, v in state_dict.items():
    if k.startswith("features"):
        new_key = f"0.{k}"
    elif k.startswith("classifier."):
        new_key = f"1.{k.split('.', 1)[1]}"   # classifier.1.* → 1.1.*
    else:
        new_key = k
    new_state_dict[new_key] = v

result = instance.model.load_state_dict(new_state_dict, strict=False)
if result.missing_keys or result.unexpected_keys:
    logger.warning("Gate load key mismatch: missing=%s unexpected=%s",
                   result.missing_keys, result.unexpected_keys)
```

mismatch 발생 시 사일런트 실패 방지 위한 경고 로그 추가.

## 4. 수정 후 결과

`reports/benchmark_v1v2v3_balanced_new.md` (t_low=0.1, t_high=0.5):

| ver | pipeline | F1 | TP/FP/TN/FN | heatmap rate | latency/img |
|---|---|---:|---|---:|---:|
| v1 | baseline | 0.7955 | 175/65/135/25 | — | 70.8 ms |
| v1 | **gate_cascade** | **0.9899** | 196/0/200/4 | 1.8% | 99.8 ms |
| v2 | baseline | 0.7714 | 162/58/142/38 | — | 80.9 ms |
| v2 | **gate_cascade** | **0.9772** | 193/2/198/7 | 2.5% | 103.3 ms |
| v3 | baseline | 0.7157 | 175/114/86/25 | — | 94.2 ms |
| v3 | **gate_cascade** | **0.9824** | 195/2/198/5 | 1.8% | 101.2 ms |

- cascade F1: baseline 대비 **+19.4%p (v1) / +20.6%p (v2) / +26.7%p (v3)**
- Gate가 ~98% 샘플을 직접 결정, PatchCore는 1.8~2.5%만 호출
- false positive 1~2건 수준으로 억제

## 5. 검증한 디버깅 단서

수정 직후 raw logit 샘플 (v1 게이트, balanced eval):

| label | logit 범위 | sigmoid | calibrated |
|---|---|---|---|
| Normal | -5.5 ~ -4.7 | 0.004~0.009 | 0.0000 |
| Anomaly | +6.2 ~ +12.9 | 0.998~1.000 | 0.9936 |

`1.1.weight.norm()` 값이 체크포인트의 `classifier.1.weight.norm()`과 정확히 일치(`0.8999`). 헤드가 실제로 들어왔음을 확인.

## 6. 후속 / Open Items

- [ ] 같은 `load` 경로를 쓰는 `backend/src/serve.py` 도 동일 fix 적용 필요(현재 자체 IsotonicCalibrator stub만 있고 GateModel 사용 여부는 추가 확인 필요).
- [ ] `train_gate.py` save 형식을 wrapper 통째로 저장하도록 바꾸면 remap 자체가 불필요. 다만 기존 round1/2/3 ckpt와 호환 깨짐 → load 쪽 보강이 더 안전.
- [ ] 로드 후 한 batch dummy forward로 logit 분리도(min<-3 / max>+3 등) 자동 검증하는 sanity check 도입 고려.
- [ ] `dev-ui` / `dev-serve` 브랜치도 동일 버그 가능성 — 머지 전 확인.

## 7. 관련 파일

- 수정: `backend/src/gate_model.py`
- 신규: `scripts/{benchmark_pipeline,train_gate,train_patchcore,make_splits_v123,build_*_test_csv,inspect_dataset}.py`, `scripts/train_all_v123.sh`
- 데이터: `splits/v{1,2,3}_*.csv`, `splits_eval/v{1,2,3}_*.csv`
- 모델 메타: `models/v{1,2,3}_{effnetb0,patchcore_r18}_{config.json,calibrator.pkl}` (`.pt` 가중치는 `.gitignore` 처리)
- 리포트: `reports/benchmark_v1v2v3_balanced_new.{json,md,log}`, `reports/assets/v{1,2,3}_*.png`
