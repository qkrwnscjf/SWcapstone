#!/usr/bin/env python3
"""
FastAPI serving endpoint implementing the Gate-Heatmap cascade logic.

Cascade decision flow:
    1. Gate model outputs p_gate(anomaly) in [0, 1].
    2. If p_gate >= T_high  -> call Heatmap model  (likely anomaly).
    3. If p_gate <= T_low   -> normal exit          (confident normal).
    4. Else (uncertain zone) -> call Heatmap model.
    5. Override: always call Heatmap if any override flag is active
       (quality_low, drift_suspected, critical_line).

Endpoints:
    POST /predict   - image upload -> JSON prediction
    GET  /health    - health check
    GET  /metrics   - runtime statistics

Environment variables / config:
    CASCADE_T_LOW           (default 0.3)
    CASCADE_T_HIGH          (default 0.7)
    OVERRIDE_QUALITY_LOW    (default false)
    OVERRIDE_DRIFT_SUSPECTED(default false)
    OVERRIDE_CRITICAL_LINE  (default false)
    GATE_MODEL_PATH         (default models/gate_model.pt)
    HEATMAP_MODEL_PATH      (default models/heatmap_model.pt)
    SERVE_HOST              (default 0.0.0.0)
    SERVE_PORT              (default 8000)
"""

from __future__ import annotations

import io
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms

# ---------------------------------------------------------------------------
# Project imports -- these modules are expected to expose:
#   gate_model.load_gate_model(path, device) -> nn.Module
#   gate_model.gate_predict(model, tensor)   -> float  (p_anomaly)
#   heatmap_model.load_heatmap_model(path, device) -> object
#   heatmap_model.heatmap_predict(model, tensor)   -> float  (anomaly score)
# ---------------------------------------------------------------------------
#from src.gate_model import load_gate_model, gate_predict
#from src.heatmap_model import load_heatmap_model, heatmap_predict

from src.gate_model import GateModel

def gate_predict(model, image_tensor):
    return model.predict(image_tensor)

from src.heatmap_model import PatchCoreModel

def heatmap_predict(model, image_tensor):
    """PatchCoreModel 클래스와 기존의 heatmap_predict 호출을 연결"""
    # PatchCore 모델의 추론 메서드가 predict인지 확인 후 맞춰주세요.
    # 보통 score만 반환하도록 구현되어 있을 것입니다.
    return model.predict(image_tensor)

#추가
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cascade_serve")

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Cascade configuration (read once at import / startup)
# ---------------------------------------------------------------------------
#T_LOW: float = _env_float("CASCADE_T_LOW", 0.3)
#T_HIGH: float = _env_float("CASCADE_T_HIGH", 0.7)
T_LOW: float = _env_float("CASCADE_T_LOW", 0.1)
T_HIGH: float = _env_float("CASCADE_T_HIGH", 0.5)

OVERRIDE_QUALITY_LOW: bool = _env_bool("OVERRIDE_QUALITY_LOW", False)
OVERRIDE_DRIFT_SUSPECTED: bool = _env_bool("OVERRIDE_DRIFT_SUSPECTED", False)
OVERRIDE_CRITICAL_LINE: bool = _env_bool("OVERRIDE_CRITICAL_LINE", False)

GATE_MODEL_PATH: str = _env_str(
    "GATE_MODEL_PATH", str(PROJECT_ROOT / "models" / "gate_model.pt")
)
HEATMAP_MODEL_PATH: str = _env_str(
    "HEATMAP_MODEL_PATH", str(PROJECT_ROOT / "models" / "heatmap_model.pt")
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# ---------------------------------------------------------------------------
# Image preprocessing (must match training pipeline)
# ---------------------------------------------------------------------------
INPUT_SIZE = 224
_preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Read raw bytes -> PIL -> preprocessed tensor [1, C, H, W]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor.to(DEVICE)


# ---------------------------------------------------------------------------
# Runtime metrics accumulator
# ---------------------------------------------------------------------------
class _Metrics:
    """Thread-safe (GIL-level) runtime statistics."""

    def __init__(self) -> None:
        self.total_predictions: int = 0
        self.heatmap_calls: int = 0
        self.gate_latency_sum_ms: float = 0.0
        self.heatmap_latency_sum_ms: float = 0.0
        self.total_latency_sum_ms: float = 0.0

    # -- mutators ----------------------------------------------------------
    def record(
        self,
        called_heatmap: bool,
        gate_latency_ms: float,
        heatmap_latency_ms: float,
        total_latency_ms: float,
    ) -> None:
        self.total_predictions += 1
        if called_heatmap:
            self.heatmap_calls += 1
        self.gate_latency_sum_ms += gate_latency_ms
        self.heatmap_latency_sum_ms += heatmap_latency_ms
        self.total_latency_sum_ms += total_latency_ms

    # -- queries -----------------------------------------------------------
    @property
    def heatmap_call_rate(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.heatmap_calls / self.total_predictions

    @property
    def avg_gate_latency_ms(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.gate_latency_sum_ms / self.total_predictions

    @property
    def avg_heatmap_latency_ms(self) -> float:
        if self.heatmap_calls == 0:
            return 0.0
        return self.heatmap_latency_sum_ms / self.heatmap_calls

    @property
    def avg_total_latency_ms(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_latency_sum_ms / self.total_predictions

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "heatmap_calls": self.heatmap_calls,
            "heatmap_call_rate": round(self.heatmap_call_rate, 4),
            "avg_gate_latency_ms": round(self.avg_gate_latency_ms, 2),
            "avg_heatmap_latency_ms": round(self.avg_heatmap_latency_ms, 2),
            "avg_total_latency_ms": round(self.avg_total_latency_ms, 2),
        }


metrics = _Metrics()

# ---------------------------------------------------------------------------
# Cascade decision logic
# ---------------------------------------------------------------------------

def _should_call_heatmap(p_gate: float) -> bool:
    """Return True if the heatmap stage should be invoked."""
    # Override flags take precedence
    if OVERRIDE_QUALITY_LOW or OVERRIDE_DRIFT_SUSPECTED or OVERRIDE_CRITICAL_LINE:
        return True
    # Confident normal -> skip heatmap
    if p_gate <= T_LOW:
        return False
    # Confident anomaly or uncertain zone -> call heatmap
    return True  # p_gate > T_LOW (covers both uncertain and >= T_HIGH)


def _cascade_decision(p_gate: float, heatmap_score: Optional[float]) -> str:
    """
    Produce a human-readable decision string.

    Returns one of:
        "normal"  - gate-only exit, predicted normal
        "anomaly" - heatmap confirmed anomaly
        "normal (heatmap)" - heatmap overrode gate, predicted normal
    """
    if heatmap_score is None:
        # Gate-only exit
        return "normal"

    # When heatmap was called, use the heatmap score as the final arbiter.
    # The heatmap anomaly threshold is conventionally 0.5 for a calibrated
    # score; adjust as needed for your PatchCore calibration.
    HEATMAP_THRESHOLD = 0.5
    if heatmap_score >= HEATMAP_THRESHOLD:
        return "anomaly"
    return "normal (heatmap)"


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cascade Anomaly Detection Service",
    description="Gate -> Heatmap cascade serving endpoint",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],      # GET, POST 등 모든 메소드 허용
    allow_headers=["*"],      # 모든 헤더 허용
)

# -- model singletons (loaded at startup) ----------------------------------
_gate_model: Any = None
_heatmap_model: Any = None


@app.on_event("startup")
async def _load_models() -> None:
    global _gate_model, _heatmap_model

    logger.info("Loading gate model from %s", GATE_MODEL_PATH)
    _gate_model = GateModel.load(GATE_MODEL_PATH, DEVICE)
    #_gate_model = load_gate_model(GATE_MODEL_PATH, DEVICE)
    logger.info("Gate model loaded on %s", DEVICE)

    logger.info("Loading heatmap model from %s", HEATMAP_MODEL_PATH)
    _heatmap_model = PatchCoreModel.load(HEATMAP_MODEL_PATH, DEVICE)
    #_heatmap_model = load_heatmap_model(HEATMAP_MODEL_PATH, DEVICE)
    logger.info("Heatmap model loaded on %s", DEVICE)

    logger.info(
        "Cascade config  T_low=%.3f  T_high=%.3f  overrides=[quality_low=%s, "
        "drift_suspected=%s, critical_line=%s]",
        T_LOW,
        T_HIGH,
        OVERRIDE_QUALITY_LOW,
        OVERRIDE_DRIFT_SUSPECTED,
        OVERRIDE_CRITICAL_LINE,
    )


# --------------------------------------------------------------------------
# POST /predict
# --------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image upload, run the cascade, return JSON result.

    Response schema:
    {
        "gate_score": float,         // p_gate(anomaly)
        "decision": str,             // "normal" | "anomaly" | "normal (heatmap)"
        "heatmap_called": bool,
        "heatmap_score": float|null, // only when heatmap was invoked
        "override_active": bool,
        "latency": {
            "gate_latency_ms": float,
            "heatmap_latency_ms": float,
            "total_latency_ms": float
        }
    }
    """
    t_start = time.perf_counter()

    # --- read & preprocess ------------------------------------------------
    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read/preprocess image: {exc}",
        )

    # --- gate inference ---------------------------------------------------
    t_gate_start = time.perf_counter()
    try:
        #p_gate: float = gate_predict(_gate_model, tensor)
        p_gate: float = _gate_model.predict(tensor)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gate model inference failed: {exc}",
        )
    t_gate_end = time.perf_counter()
    gate_latency_ms = (t_gate_end - t_gate_start) * 1000.0

    # --- cascade decision -------------------------------------------------
    override_active = (
        OVERRIDE_QUALITY_LOW
        or OVERRIDE_DRIFT_SUSPECTED
        or OVERRIDE_CRITICAL_LINE
    )
    call_heatmap = _should_call_heatmap(p_gate)

    heatmap_score: Optional[float] = None
    heatmap_latency_ms: float = 0.0
    """
    if call_heatmap:
        t_hm_start = time.perf_counter()
        try:
            heatmap_score = heatmap_predict(_heatmap_model, tensor)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Heatmap model inference failed: {exc}",
            )
        t_hm_end = time.perf_counter()
        heatmap_latency_ms = (t_hm_end - t_hm_start) * 1000.0

    decision = _cascade_decision(p_gate, heatmap_score)

    t_end = time.perf_counter()
    total_latency_ms = (t_end - t_start) * 1000.0
    """
    # --- cascade decision (히트맵 결과 백엔드 전송 로직) (2/22 수정)
    heatmap_data_for_ui: Optional[list] = None 

    if call_heatmap:
        t_hm_start = time.perf_counter()
        try:
            hm_result = heatmap_predict(_heatmap_model, tensor)
            
            # 터미널 로그에서 확인한 dict 구조 분석 및 추출
            if isinstance(hm_result, dict):
                # 1. 점수 추출
                heatmap_score = hm_result.get("score") or hm_result.get("anomaly_score") or 0.0
                
                # 2. 4채널(RGBA) 배열 찾기 
                for key, value in hm_result.items():
                    
                    if hasattr(value, "shape") and len(value.shape) == 3 and value.shape[2] == 4:
                        if torch.is_tensor(value):
                            heatmap_data_for_ui = value.detach().cpu().numpy().tolist()
                        elif isinstance(value, np.ndarray):
                            heatmap_data_for_ui = value.tolist()
                        else:
                            heatmap_data_for_ui = list(value)
                        logger.info(f"Found RGBA heatmap data in key: '{key}'")
                        break
                
                
                if heatmap_data_for_ui is None:
                    raw_heatmap = hm_result.get("heatmap") or hm_result.get("anomaly_map")
                    if raw_heatmap is not None:
                        heatmap_data_for_ui = raw_heatmap.tolist() if hasattr(raw_heatmap, "tolist") else raw_heatmap
            else:
                heatmap_score = hm_result
                
        except Exception as exc:
            logger.error(f"Heatmap inference failed: {exc}")
            
        t_hm_end = time.perf_counter()
        heatmap_latency_ms = (t_hm_end - t_hm_start) * 1000.0
    """
    heatmap_array_list: Optional[list] = None # 히트맵 배열 저장용 변수 추가 (2/22)

    if call_heatmap:
        t_hm_start = time.perf_counter()
        try:
            # heatmap_predict가 반환하는 값 전체
            hm_result = heatmap_predict(_heatmap_model, tensor)
            
            logger.info(f"DEBUG: hm_result type={type(hm_result)}, value={hm_result}")
            
            # 만약 hm_result가 딕셔너리라면 score만 추출하고, 아니면 그대로 사용
            if isinstance(hm_result, dict):
                #heatmap_score = hm_result.get("score", 0.0)
                # 나중에 React에 전달하기 위해 히트맵 배열도 따로 저장
                heatmap_score = hm_result.get("score") or hm_result.get("anomaly_score") or 0.0
                heatmap_array = hm_result.get("heatmap", None)
            else:
                heatmap_score = hm_result
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Heatmap model inference failed: {exc}",
            )
        t_hm_end = time.perf_counter()
        heatmap_latency_ms = (t_hm_end - t_hm_start) * 1000.0
        """

    # --- record metrics ---------------------------------------------------
    
    decision = _cascade_decision(p_gate, heatmap_score)

    t_end = time.perf_counter()
    total_latency_ms = (t_end - t_start) * 1000.0  # 초를 밀리초(ms)로 변환

    metrics.record(
        called_heatmap=call_heatmap,
        gate_latency_ms=gate_latency_ms,
        heatmap_latency_ms=heatmap_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    # --- log --------------------------------------------------------------
    logger.info(
        "predict  gate=%.4f  heatmap=%s  decision=%s  "
        "gate_ms=%.1f  hm_ms=%.1f  total_ms=%.1f",
        p_gate,
        f"{heatmap_score:.4f}" if heatmap_score is not None else "N/A",
        decision,
        gate_latency_ms,
        heatmap_latency_ms,
        total_latency_ms,
    )
    """
    return JSONResponse(
        content={
            "gate_score": round(float(p_gate), 6),
            "decision": decision,
            "heatmap_called": call_heatmap,
            "heatmap_score": (
                round(float(heatmap_score), 6) if heatmap_score is not None else None
            ),
            "override_active": override_active,
            "latency": {
                "gate_latency_ms": round(gate_latency_ms, 2),
                "heatmap_latency_ms": round(heatmap_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
            },
        }
    )
    """
    # --- return 부분 수정 --- (2/22)
    return JSONResponse(
        content={
            "gate_score": round(float(p_gate), 6),
            "decision": decision,
            "heatmap_called": call_heatmap,
            "heatmap_score": (
                round(float(heatmap_score), 6) if heatmap_score is not None else None
            ),
            "heatmap_data": heatmap_data_for_ui, # null이 아닌 리스트
            "mlflow_info": {
                "backbone": _gate_model.backbone_name,
                "threshold": _gate_model.threshold,
            },
            "latency": {
                "gate_latency_ms": round(gate_latency_ms, 2),
                "heatmap_latency_ms": round(heatmap_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
            },
        }
    )


# --------------------------------------------------------------------------
# GET /health
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Simple health check."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "gate_model_loaded": _gate_model is not None,
        "heatmap_model_loaded": _heatmap_model is not None,
        "cascade_config": {
            "T_low": T_LOW,
            "T_high": T_HIGH,
            "override_quality_low": OVERRIDE_QUALITY_LOW,
            "override_drift_suspected": OVERRIDE_DRIFT_SUSPECTED,
            "override_critical_line": OVERRIDE_CRITICAL_LINE,
        },
    }


# --------------------------------------------------------------------------
# GET /metrics
# --------------------------------------------------------------------------
@app.get("/metrics")
async def get_metrics():
    """Return current runtime statistics."""
    return metrics.snapshot()


# --------------------------------------------------------------------------
# Entrypoint (for `python -m src.serve`)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = _env_str("SERVE_HOST", "0.0.0.0")
    port = int(_env_float("SERVE_PORT", 8000))
    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(
        "src.serve:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
