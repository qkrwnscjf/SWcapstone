# SteelVision MLOps v2.0: AI-Integrated Manufacturing Quality Control System
**Technical Specification & API Handover Document**
**Date:** 2026-05-08
**Version:** v2.0 (Final Release)
**Author:** Gemini CLI (Backend Engineering Team)

---

## 1. 서론 (Introduction)
본 문서는 SteelVision 프로젝트의 MLOps 백엔드 서버(v2.0)에 구현된 모든 기능적, 기술적 사양을 상술한다. 본 시스템은 철강 표면 결함 탐지를 위한 **Cascade Inference Engine**과 데이터 중심의 지속적 개선을 위한 **MLOps Pipeline**이 통합된 구조를 가진다. 프론트엔드 개발자는 본 명세서를 바탕으로 대시보드 및 제어 UI를 구성할 수 있다.

---

## 2. 시스템 아키텍처 (System Architecture)

### 2.1. 하이브리드 추론 엔진 (Cascade Inference)
- **Gate Model (EfficientNet-B0):** 고속 바이너리 분류기로서, 유입되는 모든 이미지의 정상/이상 여부를 1차 판정한다.
- **Heatmap Model (PatchCore-R18):** Gate에서 이상으로 판정된 데이터에 대해 정밀 픽셀 단위 분석을 수행하여 결함 부위를 시각화한다.
- **Ensemble Logic:** 두 모델의 결과값을 결합하여 최종 신뢰도를 산출하며, 불필요한 연산을 줄이기 위한 조건부 활성화(Triggering) 메커니즘을 포함한다.

### 2.2. 하드웨어 추상화 계층 (Hardware Abstraction)
- `device_utils.py`를 통해 실행 환경을 자동 감지한다.
    - **Apple Silicon:** MPS(Metal Performance Shaders) 가속 사용.
    - **NVIDIA GPU:** CUDA 가속 사용.
    - **Standard:** 가속기 부재 시 CPU 연산 수행.

### 2.3. 저장소 및 상태 관리 (Storage & State)
- **Centralized S3 Registry:** MinIO를 활용하여 모델 아티팩트(`.pt`), 학습 데이터셋, 사용자 피드백 이미지를 통합 관리한다.
- **State Persistence:** `state.json`을 통해 시스템의 전역 상태(배포 모델 정보, 학습 이력, 로그 등)를 영속화하며 실시간 동기화한다.

---

## 3. API 상세 명세 (API Specifications)

### 3.1. 추론 및 진단 (Inference & Diagnosis)
- **Endpoint:** `POST /predict`
- **Description:** 업로드된 이미지에 대해 실시간 결함 탐지를 수행한다.
- **Payload:** `Multipart/form-data` (file: Image)
- **Response Key Elements:**
    - `gate_score`: Gate 모델의 정상/이상 확률 (0~1).
    - `decision`: "normal" 또는 "anomaly".
    - `heatmap_score`: PatchCore 기반 정밀 이상 점수.
    - `heatmap_overlay`: 결함 부위가 강조된 이미지 (Base64 PNG).
    - `ensemble_active`: Heatmap 모델 가동 여부.

### 3.2. MLOps 관리 대시보드 (Management Dashboard)
- **Endpoint:** `GET /mlops/dashboard`
- **Description:** 시스템의 전체 상태 정보를 반환한다. 프론트엔드의 메인 대시보드 렌더링에 사용된다.
- **Response Data:**
    - `available_model_files`: S3 및 로컬에서 사용 가능한 모든 `.pt` 파일 목록 (드롭다운 메뉴용).
    - `model_versions`: 모델별 메트릭(F1, Acc), 상태(Production/Candidate), 계보(Lineage).
    - `training_runs`: 과거 학습 이력 및 결과.
    - `feedback_items`: 수집된 현장 피드백 데이터셋.
    - `runtime_config`: 현재 배포 중인 모델 ID 및 파일명, 앙상블 활성화 여부.
    - `logs`: 시스템 주요 이벤트 로그.

### 3.3. 비동기 재학습 파이프라인 (Retraining Pipeline)
- **Endpoint:** `POST /mlops/train`
- **Description:** 새로운 모델 학습을 비동기적으로 시작한다.
- **Request Body:** `architecture`, `epochs`, `batch_size`, `learning_rate` 등.
- **Endpoint:** `GET /mlops/training/status`
- **Description:** 현재 진행 중인 학습의 상태를 폴링한다.
- **Response Data:**
    - `is_running`: 학습 진행 여부.
    - `progress`: 진행률 (0~100%).
    - `message`: 현재 단계 (예: "EPOCH 5/10", "PUSHING TO S3").
    - `metrics`: 실시간 Loss 및 Accuracy.

### 3.4. 무중단 모델 배포 (Hot-swap Deployment)
- **Endpoint:** `POST /mlops/deploy`
- **Description:** 특정 모델을 운영 환경에 즉시 적용한다.
- **Mechanism:** 서버 재시작 없이 메모리 상에서 모델 객체를 교체(Hot-swap)하며, 파일이 로컬에 없을 경우 S3에서 자동 다운로드한다.
- **Payload:** `model_id`, `gate_file`, `heatmap_file`, `ensemble_enabled`.

### 3.5. 데이터 피드백 루프 (Data Feedback Loop)
- **Endpoint:** `POST /mlops/feedback`
- **Description:** 추론 결과에 대한 사용자의 라벨링 데이터를 저장한다.
- **Sync:** 데이터는 로컬 저장소에 저장됨과 동시에 S3의 `feedback` 버킷으로 자동 업로드되어 향후 재학습 데이터셋으로 활용된다.

---

## 4. 프론트엔드 구현 가이드 (Frontend Implementation Guide)

1.  **실시간 동기화:** 모델 학습 시작 시 `GET /mlops/training/status`를 1~2초 간격으로 폴링하여 프로그레스 바를 업데이트하십시오.
2.  **유동적 드롭다운:** `GET /mlops/dashboard`에서 `available_model_files`를 받아와 모델 선택 드롭다운을 구성하십시오. 학습이 완료되면 이 API를 재호출하여 리스트를 갱신해야 합니다.
3.  **이미지 처리:** `/predict`에서 반환되는 `heatmap_overlay`는 `data:image/png;base64,...` 형식을 사용하여 즉시 이미지 태그의 `src`에 할당할 수 있습니다.
4.  **배포 승인:** 새로운 모델 학습 완료 후, 사용자가 '배포' 버튼을 누르면 `/mlops/deploy` API를 통해 즉시 운영 모델을 교체할 수 있습니다.

---

## 5. 결론 (Conclusion)
본 서버는 유연한 동적 부팅(Bootstrap Strategy)과 강력한 하드웨어 가속, 그리고 완전한 데이터 피드백 루프를 지원한다. 프론트엔드 담당자는 상기 API 명세를 활용하여 AI 운영 효율을 극대화하는 관리 도구를 완성할 수 있다.

---
**Document Ends.**
