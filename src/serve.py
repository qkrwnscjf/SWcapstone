#!/usr/bin/env python3
"""
FastAPI serving endpoint implementing the Gate-Heatmap cascade logic.

핵심 수정 사항
-------------
1. GateModel.load() 사용 안 함
   -> train_gate.py의 build_gate_model()과 동일한 구조로 직접 로드

2. gate checkpoint 안의 T_low / T_high 사용
   -> 서빙에서 하드코딩 threshold 제거

3. calibrator(.pkl) 자동 로드
   -> 학습 시 저장한 calibration 반영

4. heatmap 출력 계약 통일
   -> PatchCoreModel.predict() 결과(dict)를 그대로 받아 anomaly_score / overlay 사용

5. Gate score / threshold / call 여부 로그 강화
"""

from __future__ import annotations

import io
import os
import time
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models

from src.heatmap_model import PatchCoreModel

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class PlattCalibrator:
    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)

    def fit(self, probs, labels) -> None:
        self.lr.fit(probs.reshape(-1, 1), labels)

    def predict_proba(self, probs):
        return self.lr.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs, labels) -> None:
        self.ir.fit(probs, labels)

    def predict_proba(self, probs):
        return self.ir.predict(probs)
    
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cascade_serve")

# ---------------------------------------------------------------------------
# Paths / env helpers
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
# Override flags
# ---------------------------------------------------------------------------
OVERRIDE_QUALITY_LOW: bool = _env_bool("OVERRIDE_QUALITY_LOW", False)
OVERRIDE_DRIFT_SUSPECTED: bool = _env_bool("OVERRIDE_DRIFT_SUSPECTED", False)
OVERRIDE_CRITICAL_LINE: bool = _env_bool("OVERRIDE_CRITICAL_LINE", False)

GATE_MODEL_PATH: str = _env_str(
    "GATE_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "round1_effnetb0_gate.pt"),
)
HEATMAP_MODEL_PATH: str = _env_str(
    "HEATMAP_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "round1_patchcore_r18_patchcore.pt"),
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# ---------------------------------------------------------------------------
# Runtime thresholds
#   실제 값은 startup에서 gate checkpoint로부터 덮어씀
# ---------------------------------------------------------------------------
T_LOW: float = 0.3
T_HIGH: float = 0.7

# ---------------------------------------------------------------------------
# Image preprocessing
#   기본값 224. startup에서 gate checkpoint의 input_size를 반영해 갱신 가능
# ---------------------------------------------------------------------------
INPUT_SIZE = 224


def build_preprocess(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


_preprocess = build_preprocess(INPUT_SIZE)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Read raw bytes -> PIL -> preprocessed tensor [1, C, H, W]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0)
    return tensor.to(DEVICE)


# ---------------------------------------------------------------------------
# Gate model builder (train_gate.py와 동일 구조)
# ---------------------------------------------------------------------------
def build_gate_model(gate_name: str, num_classes: int = 1) -> nn.Module:
    if gate_name == "effnetb0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif gate_name == "mnv3_large":
        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )
    else:
        raise ValueError(f"Unsupported gate model: {gate_name}")

    return model

def load_gate_checkpoint(model_path: str, device: torch.device) -> Dict[str, Any]:
    """train_gate.py에서 저장한 checkpoint를 직접 로드."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    gate_name = ckpt.get("gate_name")
    if gate_name is None:
        raise ValueError(f"'gate_name' not found in checkpoint: {model_path}")

    model = build_gate_model(gate_name, num_classes=1)

    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        logger.info("Gate checkpoint loaded with strict=True")
    except Exception as e:
        logger.warning("strict=True load failed: %s", e)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        logger.warning("Gate load missing keys: %s", missing)
        logger.warning("Gate load unexpected keys: %s", unexpected)

    model.to(device)
    model.eval()

    bundle = {
        "model": model,
        "gate_name": gate_name,
        "backbone": ckpt.get("backbone", gate_name),
        "input_size": int(ckpt.get("input_size", 224)),
        "T_low": float(ckpt.get("T_low", 0.3)),
        "T_high": float(ckpt.get("T_high", 0.7)),
        "checkpoint_path": model_path,
    }
    return bundle

def load_gate_calibrator(cal_path: Optional[str]) -> Optional[Any]:
    if cal_path is None:
        logger.info("No calibrator file found for gate model.")
        return None

    try:
        with open(cal_path, "rb") as f:
            obj = pickle.load(f)

        calibrator = obj.get("calibrator", None)
        method = obj.get("method", "unknown")
        logger.info("Loaded gate calibrator from %s (method=%s)", cal_path, method)
        return calibrator

    except Exception as e:
        logger.warning(
            "Failed to load gate calibrator from %s. Proceeding without calibration. Error: %s",
            cal_path,
            e,
        )
        return None

def find_calibrator_path(gate_model_path: str) -> Optional[str]:
    """
    예:
      round1_effnetb0_gate.pt -> round1_effnetb0_calibrator.pkl
    """
    gate_path = Path(gate_model_path)
    stem = gate_path.stem

    if stem.endswith("_gate"):
        cal_name = stem[:-5] + "_calibrator.pkl"
    else:
        cal_name = stem + "_calibrator.pkl"

    cal_path = gate_path.with_name(cal_name)
    if cal_path.exists():
        return str(cal_path)
    return None


@torch.no_grad()
def gate_predict(
    bundle: Dict[str, Any],
    image_tensor: torch.Tensor,
    calibrator: Optional[Any] = None,
) -> float:
    """Return calibrated p_gate(anomaly) if calibrator exists."""
    model = bundle["model"]

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    model_device = next(model.parameters()).device
    image_tensor = image_tensor.to(model_device)

    logits = model(image_tensor).squeeze(-1)
    prob = torch.sigmoid(logits).item()

    if calibrator is not None:
        prob_arr = np.array([prob], dtype=np.float32)

        # custom PlattCalibrator / IsotonicCalibrator 둘 다 predict_proba 제공
        if hasattr(calibrator, "predict_proba"):
            out = calibrator.predict_proba(prob_arr)
            out = np.asarray(out)

            # sklearn logistic 계열이면 (N, 2)
            if out.ndim == 2:
                prob = float(out[0, 1])
            else:
                prob = float(out[0])

        # 혹시 직접 sklearn isotonic object가 들어온 경우
        elif hasattr(calibrator, "predict"):
            prob = float(calibrator.predict(prob_arr)[0])

        prob = float(np.clip(prob, 0.0, 1.0))

    return prob


# ---------------------------------------------------------------------------
# Heatmap inference contract
# ---------------------------------------------------------------------------
def heatmap_predict(model: PatchCoreModel, image_tensor: torch.Tensor):
    """
    PatchCoreModel.predict() 결과(dict)를 그대로 반환.
    기대 key:
      - anomaly_score
      - raw_score_heatmap
      - normalized_score_heatmap
      - heatmap_overlay
    """
    return model.predict(image_tensor)


# ---------------------------------------------------------------------------
# Runtime metrics accumulator
# ---------------------------------------------------------------------------
class _Metrics:
    def __init__(self) -> None:
        self.total_predictions: int = 0
        self.heatmap_calls: int = 0
        self.gate_latency_sum_ms: float = 0.0
        self.heatmap_latency_sum_ms: float = 0.0
        self.total_latency_sum_ms: float = 0.0

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
    """
    Override가 있으면 항상 heatmap.
    p_gate <= T_LOW 이면 confident normal -> skip.
    그 외에는 heatmap 호출.
    """
    if OVERRIDE_QUALITY_LOW or OVERRIDE_DRIFT_SUSPECTED or OVERRIDE_CRITICAL_LINE:
        return True

    if p_gate <= T_LOW:
        return False

    if p_gate >= T_HIGH:
        return True

    # uncertain zone
    return True


def _cascade_decision(p_gate: float, heatmap_score: Optional[float]) -> str:
    """
    Returns:
        "normal"
        "anomaly"
        "normal (heatmap)"
    """
    if heatmap_score is None:
        return "normal"

    HEATMAP_THRESHOLD = 0.5
    if heatmap_score >= HEATMAP_THRESHOLD:
        return "anomaly"
    return "normal (heatmap)"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cascade Anomaly Detection Service",
    description="Gate -> Heatmap cascade serving endpoint",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_gate_bundle: Optional[Dict[str, Any]] = None
_gate_calibrator: Optional[Any] = None
_heatmap_model: Optional[PatchCoreModel] = None


@app.on_event("startup")
async def _load_models() -> None:
    global _gate_bundle, _gate_calibrator, _heatmap_model
    global T_LOW, T_HIGH, INPUT_SIZE, _preprocess

    logger.info("Loading gate model from %s", GATE_MODEL_PATH)
    _gate_bundle = load_gate_checkpoint(GATE_MODEL_PATH, DEVICE)

    # checkpoint에 저장된 input size 반영
    INPUT_SIZE = int(_gate_bundle["input_size"])
    _preprocess = build_preprocess(INPUT_SIZE)

    cal_path = find_calibrator_path(GATE_MODEL_PATH)
    _gate_calibrator = load_gate_calibrator(cal_path)

    # checkpoint에서 추천 threshold 반영
    T_LOW = float(_gate_bundle["T_low"])
    T_HIGH = float(_gate_bundle["T_high"])

    logger.info(
        "Gate loaded: gate_name=%s, backbone=%s, input_size=%d, T_low=%.3f, T_high=%.3f",
        _gate_bundle["gate_name"],
        _gate_bundle["backbone"],
        INPUT_SIZE,
        T_LOW,
        T_HIGH,
    )

    logger.info("Loading heatmap model from %s", HEATMAP_MODEL_PATH)
    _heatmap_model = PatchCoreModel.load(str(HEATMAP_MODEL_PATH), device=str(DEVICE))
    logger.info("Heatmap model loaded on %s", DEVICE)

    logger.info(
        "Cascade config  T_low=%.3f  T_high=%.3f  overrides=[quality_low=%s, drift_suspected=%s, critical_line=%s]",
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
    Response schema:
    {
        "gate_score": float,
        "decision": str,
        "heatmap_called": bool,
        "heatmap_score": float|null,
        "heatmap_data": list|null,
        "override_active": bool,
        "thresholds": {
            "T_low": float,
            "T_high": float
        },
        "mlflow_info": {
            "gate_name": str,
            "backbone": str,
            "input_size": int
        },
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
        if _gate_bundle is None:
            raise RuntimeError("Gate model is not loaded")
        p_gate = gate_predict(_gate_bundle, tensor, _gate_calibrator)
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
    heatmap_data_for_ui: Optional[list] = None

    if call_heatmap:
        t_hm_start = time.perf_counter()
        try:
            if _heatmap_model is None:
                raise RuntimeError("Heatmap model is not loaded")

            hm_result = heatmap_predict(_heatmap_model, tensor)

            if isinstance(hm_result, dict):
                # 1) anomaly score
                heatmap_score = float(
                    hm_result.get("anomaly_score", hm_result.get("score", 0.0))
                )

                # 2) UI용 overlay 우선
                overlay = hm_result.get("heatmap_overlay", None)
                if overlay is not None:
                    if torch.is_tensor(overlay):
                        heatmap_data_for_ui = overlay.detach().cpu().numpy().tolist()
                    elif isinstance(overlay, np.ndarray):
                        heatmap_data_for_ui = overlay.tolist()
                    else:
                        try:
                            heatmap_data_for_ui = list(overlay)
                        except Exception:
                            heatmap_data_for_ui = None

                # 3) overlay 없으면 normalized heatmap fallback
                if heatmap_data_for_ui is None:
                    norm_map = hm_result.get("normalized_score_heatmap", None)
                    if norm_map is not None:
                        if torch.is_tensor(norm_map):
                            heatmap_data_for_ui = norm_map.detach().cpu().numpy().tolist()
                        elif isinstance(norm_map, np.ndarray):
                            heatmap_data_for_ui = norm_map.tolist()

                # 4) raw heatmap fallback
                if heatmap_data_for_ui is None:
                    raw_map = hm_result.get("raw_score_heatmap", None)
                    if raw_map is not None:
                        if torch.is_tensor(raw_map):
                            heatmap_data_for_ui = raw_map.detach().cpu().numpy().tolist()
                        elif isinstance(raw_map, np.ndarray):
                            heatmap_data_for_ui = raw_map.tolist()

            else:
                # 혹시 float만 반환하는 구현이라면
                heatmap_score = float(hm_result)

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Heatmap model inference failed: {exc}",
            )
        t_hm_end = time.perf_counter()
        heatmap_latency_ms = (t_hm_end - t_hm_start) * 1000.0

    # --- final decision ---------------------------------------------------
    decision = _cascade_decision(p_gate, heatmap_score)

    t_end = time.perf_counter()
    total_latency_ms = (t_end - t_start) * 1000.0

    # --- metrics ----------------------------------------------------------
    metrics.record(
        called_heatmap=call_heatmap,
        gate_latency_ms=gate_latency_ms,
        heatmap_latency_ms=heatmap_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    # --- log --------------------------------------------------------------
    logger.info(
        "predict  gate=%.6f  T_low=%.3f  T_high=%.3f  heatmap_called=%s  heatmap=%s  decision=%s  gate_ms=%.2f  hm_ms=%.2f  total_ms=%.2f",
        p_gate,
        T_LOW,
        T_HIGH,
        call_heatmap,
        f"{heatmap_score:.6f}" if heatmap_score is not None else "N/A",
        decision,
        gate_latency_ms,
        heatmap_latency_ms,
        total_latency_ms,
    )

    return JSONResponse(
        content={
            "gate_score": round(float(p_gate), 6),
            "decision": decision,
            "heatmap_called": call_heatmap,
            "heatmap_score": (
                round(float(heatmap_score), 6)
                if heatmap_score is not None else None
            ),
            "heatmap_data": heatmap_data_for_ui,
            "override_active": override_active,
            "thresholds": {
                "T_low": round(float(T_LOW), 6),
                "T_high": round(float(T_HIGH), 6),
            },
            "mlflow_info": {
                "gate_name": _gate_bundle["gate_name"] if _gate_bundle else None,
                "backbone": _gate_bundle["backbone"] if _gate_bundle else None,
                "input_size": _gate_bundle["input_size"] if _gate_bundle else None,
            },
            "latency": {
                "gate_latency_ms": round(gate_latency_ms, 2),
                "heatmap_latency_ms": round(heatmap_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
            },
        }
    )


# --------------------------------------------------------------------------
# POST /predict_gate_only
#   Gate score 분포 디버깅용
# --------------------------------------------------------------------------
@app.post("/predict_gate_only")
async def predict_gate_only(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read/preprocess image: {exc}",
        )

    try:
        if _gate_bundle is None:
            raise RuntimeError("Gate model is not loaded")
        p_gate = gate_predict(_gate_bundle, tensor, _gate_calibrator)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gate model inference failed: {exc}",
        )

    return {
        "gate_score": round(float(p_gate), 6),
        "T_low": round(float(T_LOW), 6),
        "T_high": round(float(T_HIGH), 6),
        "would_call_heatmap": _should_call_heatmap(p_gate),
    }


# --------------------------------------------------------------------------
# GET /health
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "gate_model_loaded": _gate_bundle is not None,
        "gate_calibrator_loaded": _gate_calibrator is not None,
        "heatmap_model_loaded": _heatmap_model is not None,
        "cascade_config": {
            "T_low": T_LOW,
            "T_high": T_HIGH,
            "override_quality_low": OVERRIDE_QUALITY_LOW,
            "override_drift_suspected": OVERRIDE_DRIFT_SUSPECTED,
            "override_critical_line": OVERRIDE_CRITICAL_LINE,
        },
        "gate_info": {
            "gate_name": _gate_bundle["gate_name"] if _gate_bundle else None,
            "backbone": _gate_bundle["backbone"] if _gate_bundle else None,
            "input_size": _gate_bundle["input_size"] if _gate_bundle else None,
            "checkpoint_path": _gate_bundle["checkpoint_path"] if _gate_bundle else None,
        },
    }


# --------------------------------------------------------------------------
# GET /metrics
# --------------------------------------------------------------------------
@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


# --------------------------------------------------------------------------
# Entrypoint
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