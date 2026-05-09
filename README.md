# SteelVision MLOps Console

철강 표면 이미지의 이상 여부를 판정하고, 현장 피드백을 다시 학습 데이터와 배포 흐름으로 연결하는 MLOps 콘솔입니다.

## 실행 준비

- Docker Desktop
- Git
- Windows PowerShell 또는 터미널

## Docker로 실행

프로젝트 루트로 이동합니다.

```powershell
cd C:\OSYSTUDY\260423\SWcapstone
```

이미지를 빌드합니다.

```powershell
docker compose build
```

서비스를 실행합니다.

```powershell
docker compose up --no-build
```

접속 주소는 다음과 같습니다.

- Frontend: http://localhost:5173
- Backend API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001
- MinIO 계정: `minioadmin` / `minioadmin`

서비스를 중지하려면 실행 중인 터미널에서 `Ctrl + C`를 누른 뒤 아래 명령을 실행합니다.

```powershell
docker compose down
```

프론트엔드 코드만 빠르게 다시 확인할 때는 다음 명령을 사용할 수 있습니다.

```powershell
docker compose up -d frontend
```

## 로컬 개발 실행

Docker 대신 로컬에서 나누어 실행할 수도 있습니다.

Backend:

```powershell
cd C:\OSYSTUDY\260423\SWcapstone\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd C:\OSYSTUDY\260423\SWcapstone\frontend
npm install
npm run dev -- --host 0.0.0.0
```

로컬 프론트엔드가 다른 백엔드를 보게 하려면 `.env` 또는 실행 환경에 `VITE_API_BASE=http://localhost:8000`을 지정합니다.

## 화면 구성

상단 화면은 `현장`과 `전산` 두 가지로 구성됩니다. 기존 요약 화면의 역할은 `전산 > 운영` 화면으로 통합되었습니다.

## 화면별 사용법

### 현장 화면

- 상단 라인 선택에서 `LINE-A`, `LINE-B`, `LINE-C`를 고릅니다.
- 이미지 업로드 후 `이상감지 실행`을 누르면 `/predict` API로 Gate/Heatmap 추론을 수행합니다.
- `RAW`, `HEATMAP`, `OVERLAY` 버튼으로 원본과 결과 이미지를 바꿔 봅니다.
- 오탐, 미탐, 입력 품질 저하, 판정 보류 버튼은 `/mlops/feedback`으로 현장 피드백을 저장합니다.

### 관리자 운영 화면

- Production 모델, 데이터셋, 학습 상태, 배포 리스크를 요약해서 확인합니다.
- 라인별 운영 상태, 수율, 평균 지연 시간, 카메라 상태를 확인합니다.
- Runtime Artifact와 다음 배포 의사결정 항목으로 현재 운영 기준을 확인합니다.
- 최근 피드백 목록에서 현장 입력 샘플과 코멘트를 확인합니다.

### 학습 및 배포 화면

- Dataset, Base Model, Model Name, Epoch을 지정한 뒤 학습을 시작합니다.
- 레시피 목록에서 기존 JSON 레시피를 선택하거나 값을 수정해 커스텀 레시피로 저장합니다.
- 필수 항목은 Dataset, Base Model, Model Name, Epoch, 레시피 선택, Model Candidate입니다.
- 선택 항목은 레시피 수정 저장, Architecture Registry, 학습 상태 확인, Canary Line, Ensemble입니다.
- Architecture Registry는 새로운 모델 구조를 추가할 때만 사용합니다. 기존 Gate/Heatmap 구조로 재학습할 때는 비워둬도 됩니다.
- Architecture Registry에서 Gate/Heatmap 명세 파일을 업로드하면 `/mlops/architectures/upload`에 등록됩니다.
- 학습 상태 카드에서 진행률, 현재 단계, 최근 로그를 확인합니다.
- 배포 영역에서 Model Candidate를 고르면 해당 후보의 Gate/Heatmap 모델 파일이 자동으로 연결됩니다.
- `Canary 시작`은 선택 모델을 특정 라인 검증 후보로 표시하고, 선택 모델 파일을 런타임에 반영합니다.
- `배포 승인`은 선택 모델과 자동 연결된 `.pt` 파일을 production으로 승격합니다.
- `롤백`은 이전 production 후보로 되돌립니다.

### 버전 화면

- 데이터셋 버전, 샘플 수, 피드백 수, 원본 데이터셋 정보를 확인합니다.
- 피드백 묶음을 기존 데이터셋에 추가하거나 새 데이터셋으로 materialize할 수 있습니다.
- 이미지 파일을 직접 업로드해 데이터셋에 반영할 수 있습니다.
- 학습/배포된 모델 목록과 상태, F1, 레시피, 데이터셋 연결 정보를 확인합니다.

### 로그 화면

- 시스템 이벤트 로그를 시간순으로 확인합니다.
- Backend가 내려주는 모델 인터페이스 명세와 저장소 구성을 확인합니다.

## 주요 API 연결

- `POST /predict`: 이미지 이상 감지
- `GET /mlops/dashboard`: 대시보드 상태 조회
- `POST /mlops/feedback`: 현장 피드백 저장
- `POST /mlops/datasets/upload`: 데이터셋 파일 업로드
- `POST /mlops/train`: 백그라운드 학습 시작
- `GET /mlops/training/status`: 학습 상태 조회
- `DELETE /mlops/training/runs/{run_id}`: 학습 이력 삭제
- `POST /mlops/deploy`: production 모델 및 Gate/Heatmap 런타임 파일 교체
- `POST /mlops/architectures/upload`: 모델 아키텍처 명세 등록

## 데이터와 산출물 위치

- 모델 파일: `models/`, `backend/models/`
- MLOps 상태 파일: `storage/mlops/state.json`
- 업로드 이미지: `storage/mlops/assets/uploads/`
- 피드백 이미지: `storage/mlops/assets/feedback/`
- 아키텍처 명세 파일: `storage/mlops/assets/architectures/`
- 리포트 이미지: `reports/assets/`
