# SteelVision MLOps Pipeline: Architectural Flow & Visual Guide
**Version:** v2.0
**Target:** AI Visualization Tool (Prompt Engineering) & System Architects

---

## 1. 사용된 기술 스택 (Technology Stack)
- **Framework:** FastAPI (High-performance Async Backend)
- **ML Engine:** PyTorch (Inference & Training)
- **Storage:** MinIO / S3 (Centralized Artifact & Data Registry)
- **Containerization:** Docker & Docker Compose (Microservices)
- **Database/State:** JSON-based State Persistence (State.json)
- **Hardware Acceleration:** MPS (Apple Silicon), CUDA (NVIDIA), CPU

---

## 2. MLOps 파이프라인 작동 흐름 (Operational Flow)

### Step 1: Data Ingestion & Feedback Loop (데이터 유입)
- 현장(Edge)에서 발생한 추론 결과에 대해 작업자가 피드백을 주면, 이미지는 즉시 **MinIO Feedback Bucket**으로 업로드됩니다.
- 새로운 학습 데이터셋은 **S3 Datasets Bucket**에 적재되어 버전 관리됩니다.

### Step 2: Asynchronous Retraining (비동기 학습)
- 사용자가 UI에서 학습을 트리거하면, **TrainEngine**이 비동기로 가동됩니다.
- 학습 데이터는 S3에서 스트리밍되어 PyTorch 엔진으로 유입되며, 매 에폭마다 지표(Loss, Acc)가 **State Manager**를 통해 대시보드로 전달됩니다.

### Step 3: Artifact Registry & Auto-Push (모델 저장)
- 학습이 완료된 모델(`.pt`)은 **MinIO Models Bucket**에 고유 타임스탬프와 함께 자동 푸시됩니다.
- 동시에 `state.json`에 새로운 모델의 메타데이터(ID, 성능 지표, 경로)가 등록됩니다.

### Step 4: Hot-swap Deployment (운영 배포)
- 사용자가 특정 모델을 선택하여 '배포'하면, 서버는 운영 중인 모델 객체를 메모리에서 즉시 교체합니다.
- 새로운 모델은 실시간으로 `/predict` API에 반영되어 현장에 투입됩니다.

---

## 3. 시각화 AI를 위한 프롬프트 가이드 (AI Visualization Prompts)

그림 생성 AI(예: Midjourney, DALL-E 3)에 입력하여 시스템 구조도를 그리기 위한 구체적인 묘사입니다.

### [Option A: Isometric System Diagram]
> "A futuristic, isometric 3D architectural diagram of an AI MLOps pipeline named 'SteelVision'. The flow starts from a 'Steel Factory Edge Device' sending images to a 'FastAPI Backend'. Centralized 'MinIO S3 Storage' icon is in the middle, connected with lines representing data flow: 'Feedback Data', 'Training Datasets', and 'AI Models (.pt)'. One side shows a 'PyTorch Training Engine' with glowing neural network nodes, and the other side shows a 'React Dashboard' with real-time charts. High-tech, clean blue and dark grey theme, professional UI/UX design style."

### [Option B: Circular Life-cycle Flow]
> "A circular flow chart representing a continuous MLOps lifecycle. 1. Data Collection (icons of steel plates), 2. S3 Storage (cloud/database icon), 3. GPU Training (lightning and chip icons), 4. Model Registry (shelf icon with .pt files), 5. Deployment (rocket icon). In the center, a large 'FastAPI' logo acts as the heart of the system. Professional, flat vector illustration, infographics style, high resolution."

---

## 4. 파이프라인 핵심 메커니즘 (Internal Logic for AI)
- **Event-Driven:** 모든 학습 진행 상황은 Event Callbacks를 통해 프론트엔드와 실시간 동기화됨.
- **Decoupled:** 모델 저장소와 실행 엔진이 분리되어 있어, 어떤 환경에서도 최신 모델을 다운로드하여 즉시 가동 가능(Bootstrap Strategy).
- **Universal:** `device_utils`가 하드웨어를 자동 감지하여 인프라에 구애받지 않는 파이프라인 구축.

---
**Guide Ends.**
