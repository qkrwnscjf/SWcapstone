# PatchCore — MVTec AD `grid` category

MVTec AD `grid` 카테고리용 PatchCore 이상탐지 모델. 최종 학습 산출물(체크포인트 + 메모리뱅크)과 베이스라인 지표를 포함.

> 이 브랜치(`model/patchcore-grid`)는 **코드와 분리된 orphan 브랜치**입니다. 학습 파이프라인/서빙 코드는 `main` 브랜치를 참조하세요.

---

## 최종 성능

| 모델 | Split | AUROC | AUPRC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| **Wide-ResNet50-2** | val  | **0.957** | 0.963 | 0.923 | 0.960 | 0.889 |
| **Wide-ResNet50-2** | test | **0.938** | 0.942 | 0.889 | **1.000** | 0.800 |
| ResNet-18           | val  | 0.690 | 0.624 | 0.625 | 0.714 | 0.556 |
| ResNet-18           | test | 0.811 | 0.754 | 0.754 | 0.667 | 0.867 |

- 데이터: MVTec AD grid (정상 264 + 결함 57), seed=42, 70:15:15 층화 분할 (train 185 / val 67 / test 69)
- 결함 카테고리 5종: bent / broken / glue / metal_contamination / thread

---

## 추론 속도 (이미지당, warmup 후 median)

| 디바이스 | wrn50 | r18 |
|---|---|---|
| Apple M-series (MPS) — single | **68 ms** | 22 ms |
| Apple M-series (MPS) — batch-8 | 48 ms/img | 15 ms/img |
| CPU — single | 669 ms | 155 ms |

배포 시 **GPU(또는 Apple Silicon) 필수** — CPU 환경에선 wrn50 기준 100ms 예산 초과.

---

## 구성

```
models/
├── grid_patchcore_wrn50.pt    # 최종 (권장)   116MB · LFS
└── grid_patchcore_r18.pt      # 경량 버전      29MB · LFS

reports/
└── grid_baseline_summary.csv  # AUROC/F1/threshold 기록

patchcore_grid_baseline.ipynb  # 학습 + 평가 노트북
```

### 체크포인트 내용

```python
{
    "backbone_name":  "wide_resnet50_2" | "resnet18",
    "coreset_ratio":  0.10,
    "num_neighbors":  1,
    "coreset_method": "greedy",
    "memory_bank":    torch.Tensor  # (N_patches, C)
}
```
- wrn50: memory_bank shape `(18944, 1536)`
- r18:   memory_bank shape `(18944, 384)`

---

## 하이퍼파라미터 (최종)

| 항목 | 값 |
|---|---|
| backbone | `wide_resnet50_2` (ImageNet 사전학습) |
| feature layers | `layer2` + `layer3` |
| locally-aware pooling | 3×3 avg pool (stride 1, pad 1) |
| input size | 256 |
| coreset sampling | **greedy (furthest-point sampling)**, ratio 0.10 |
| num_neighbors (k) | 1 |

---

## 다운로드 & 사용

### Clone (LFS 포함)

```bash
git lfs install
git clone --branch model/patchcore-grid --single-branch \
    https://github.com/qkrwnscjf/SWcapstone.git
```

`git lfs install` 이 선행되지 않으면 `.pt` 파일이 LFS 포인터 텍스트로만 받아집니다.

### 추론 (최소 예시)

```python
import torch
from torchvision.models import wide_resnet50_2
from torchvision import transforms
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("models/grid_patchcore_wrn50.pt",
                  map_location=device, weights_only=False)
memory_bank = ckpt["memory_bank"].to(device)

# Feature extractor: layer2 + layer3 에 hook + 3x3 avg pool, bilinear upsample,
# channel-wise concat 후 (B, H*W, C) 로 reshape — 노트북 FeatureExtractor 참조.
# 스코어: cdist(features, memory_bank).min(-1).max(patch_dim) 를 per-image score로 사용.
```

전체 구현은 `patchcore_grid_baseline.ipynb` 참고.

---

## 결정 임계값

`reports/grid_baseline_summary.csv` 의 `threshold` 열 기준:

| 모델 | split | threshold |
|---|---|---|
| wrn50 | val  | 19.237 |
| wrn50 | test | 19.599 |
| r18   | val  | 2.276 |
| r18   | test | 2.149 |

운영 배포 시엔 `val` 기준 임계값을 고정하고, 이후 drift 모니터링으로 주기 재보정 권장.
