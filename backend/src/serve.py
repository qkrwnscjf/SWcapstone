#!/usr/bin/env python3
"""
SteelVision MLOps 통합 서버 - [적재 및 삭제 완전판]
------------------------------------------
1. Gate-Heatmap Cascade 추론 (Ensemble 지원)
2. vLLM 기반 지능형 결함 진단 (Diagnosis)
3. MLOps 모든 API (Feedback, Upload, Dashboard, Delete)
4. 상태 영속성 (state.json) 및 실시간 동기화
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import pickle
import shutil
import logging
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torchvision import transforms, models

from src.heatmap_model import PatchCoreModel
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# --- Classes for Calibration ---
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

sys.modules["__main__"].PlattCalibrator = PlattCalibrator
sys.modules["__main__"].IsotonicCalibrator = IsotonicCalibrator

# --- Global Config ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("steelvision")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLOPS_ROOT = PROJECT_ROOT / "storage" / "mlops"
MLOPS_ASSETS_ROOT = MLOPS_ROOT / "assets"
MLOPS_STATE_PATH = MLOPS_ROOT / "state.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_production_gate = None
_heatmap_model = None
_training_status = {"is_running": False, "progress": 0, "message": "IDLE", "epoch": 0}

# --- State Management Helpers ---
def _iso_now() -> str: return datetime.now().isoformat(timespec="seconds")
def _ensure_dir(path: Path) -> None: path.mkdir(parents=True, exist_ok=True)

def _save_state(state: Dict[str, Any]):
    with open(MLOPS_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def _load_state() -> Dict[str, Any]:
    if not MLOPS_STATE_PATH.exists():
        _ensure_dir(MLOPS_ROOT)
        _ensure_dir(MLOPS_ASSETS_ROOT / "feedback")
        _ensure_dir(MLOPS_ASSETS_ROOT / "uploads")
        initial_state = {
            "active_dataset_id": "DATA-V3-PRODUCTION",
            "dataset_versions": [{"id": "DATA-V3-PRODUCTION", "name": "Round 3 Pool", "status": "locked", "sample_count": 14850, "feedback_count": 0, "samples": [], "updated_at": _iso_now()}],
            "model_versions": [{"id": "MODEL-R3-FINAL", "status": "production", "metrics": {"f1": 0.94}, "lineage": "Round 3 Cascade"}],
            "training_runs": [], "feedback_items": [], "logs": [], "architectures": [
                {"id": "ARCH-GATE-EFF", "name": "EffNet-B0", "kind": "gate"},
                {"id": "ARCH-HM-PC", "name": "PatchCore-R18", "kind": "heatmap"}
            ]
        }
        _save_state(initial_state)
        return initial_state
    with open(MLOPS_STATE_PATH, "r", encoding="utf-8") as f: return json.load(f)

def _append_log(state: Dict[str, Any], level: str, msg: str):
    state["logs"].insert(0, {"id": uuid.uuid4().hex[:8].upper(), "time": _iso_now(), "level": level, "message": msg})

# --- AI Diagnosis (vLLM Mock) ---
async def vllm_diagnose(img_bytes: bytes, score: float) -> str:
    await asyncio.sleep(0.3)
    if score > 0.8: return "심각한 물리적 파손 및 딥 스크래치 (공정 라인 점검 필요)"
    if score > 0.5: return "표면 오염 또는 미세 스크래치 감지 (세척 후 재검사 권고)"
    return "정상 범주 내의 특이점 (모니터링 유지)"

# --- FastAPI Setup ---
app = FastAPI(title="SteelVision Final Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/mlops-assets", StaticFiles(directory=str(MLOPS_ASSETS_ROOT)), name="mlops-assets")

@app.on_event("startup")
async def startup():
    global _production_gate, _heatmap_model
    _load_state()
    p_path = PROJECT_ROOT / "models" / "round3_effnetb0_gate.pt"
    if p_path.exists():
        ckpt = torch.load(p_path, map_location=DEVICE, weights_only=False)
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.load_state_dict(ckpt["model_state_dict"])
        _production_gate = {"model": model.to(DEVICE).eval(), "input_size": 224, "T_low": 0.1, "T_high": 0.5}
    h_path = PROJECT_ROOT / "models" / "round3_patchcore_r18_patchcore.pt"
    if h_path.exists(): _heatmap_model = PatchCoreModel.load(str(h_path), device=str(DEVICE))
    logger.info("Systems Online.")

# --- Core Inference API ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _production_gate: raise HTTPException(500, "Model not loaded")
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): prob = torch.sigmoid(_production_gate["model"](tensor)).item()
    heatmap_res = _heatmap_model.predict(tensor) if prob > 0.1 and _heatmap_model else None
    score = heatmap_res["anomaly_score"] if heatmap_res else 0.0
    decision = "anomaly" if score > 0.5 else "normal"
    issue_type = await vllm_diagnose(img_bytes, score) if decision == "anomaly" else "정상"
    
    import base64
    def _to_b64(arr):
        if arr is None: return None
        if arr.dtype != np.uint8: arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "gate_score": prob, "decision": decision, "issueType": issue_type, "heatmap_score": score,
        "heatmap_overlay": _to_b64(heatmap_res.get("heatmap_overlay") if heatmap_res else None),
        "latency": {"total_latency_ms": 150} # Mock latency
    }

# --- MLOps Dashboard & Management ---
@app.get("/mlops/dashboard")
async def get_dashboard():
    return _load_state()

@app.post("/mlops/feedback")
async def create_feedback(
    file: UploadFile = File(...), feedback_type: str = Form(...), label: str = Form("unlabeled"),
    operator: str = Form(""), comment: str = Form(""), line: str = Form(""), predicted_label: str = Form("")
):
    state = _load_state()
    ext = Path(file.filename or "sample.png").suffix
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = MLOPS_ASSETS_ROOT / "feedback" / fname
    _ensure_dir(fpath.parent)
    with open(fpath, "wb") as f: shutil.copyfileobj(file.file, f)
    
    fb_item = {
        "id": f"FDBK-{uuid.uuid4().hex[:8].upper()}",
        "feedback_type": feedback_type, "label": label, "operator": operator,
        "comment": comment, "image_url": f"/mlops-assets/feedback/{fname}", "created_at": _iso_now()
    }
    state["feedback_items"].insert(0, fb_item)
    state["dataset_versions"][0]["feedback_count"] += 1
    _append_log(state, "info", f"Data Ingested: {feedback_type} ({fname})")
    _save_state(state)
    return {"message": "Success", "feedback_item": fb_item, "dataset_version": state["dataset_versions"][0]}

@app.delete("/mlops/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    state = _load_state()
    item_to_remove = next((item for item in state["feedback_items"] if item["id"] == feedback_id), None)
    if not item_to_remove: raise HTTPException(404, "Feedback not found")
    
    # 물리 파일 삭제
    img_rel_path = item_to_remove["image_url"].replace("/mlops-assets/", "")
    img_abs_path = MLOPS_ASSETS_ROOT / img_rel_path
    if img_abs_path.exists(): os.remove(img_abs_path)
    
    # 리스트에서 제거
    state["feedback_items"] = [item for item in state["feedback_items"] if item["id"] != feedback_id]
    state["dataset_versions"][0]["feedback_count"] = max(0, state["dataset_versions"][0]["feedback_count"] - 1)
    _append_log(state, "warning", f"Data Removed: {feedback_id}")
    _save_state(state)
    return {"message": "Deleted"}

@app.post("/mlops/datasets/upload")
async def upload_data(files: List[UploadFile] = File(...)):
    state = _load_state()
    added = 0
    for f in files:
        fname = f"{uuid.uuid4().hex}{Path(f.filename or '.png').suffix}"
        fpath = MLOPS_ASSETS_ROOT / "uploads" / fname
        _ensure_dir(fpath.parent)
        with open(fpath, "wb") as out: shutil.copyfileobj(f.file, out)
        added += 1
    state["dataset_versions"][0]["sample_count"] += added
    _append_log(state, "info", f"Bulk upload: {added} files")
    _save_state(state)
    return {"message": "Success", "added_count": added, "dataset_version": state["dataset_versions"][0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
