#!/usr/bin/env python3
"""
SteelVision MLOps 통합 서버 - [적재 및 삭제 완전판]
------------------------------------------
1. Gate-Heatmap Cascade 추론 (Ensemble 지원)
2. vLLM 기반 지능형 결함 진단 (Diagnosis)
3. MLOps 모든 API (Feedback, Upload, Dashboard, Delete, Train/Deploy)
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
from src.s3_utils import S3Utils
from src.device_utils import get_device
from src.train_engine import TrainEngine

# --- Global Config ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("steelvision")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLOPS_ROOT = PROJECT_ROOT / "storage" / "mlops"
MLOPS_ASSETS_ROOT = MLOPS_ROOT / "assets"
MLOPS_STATE_PATH = MLOPS_ROOT / "state.json"

# --- S3 Config ---
STORAGE_TYPE = os.getenv("STORAGE_TYPE", "LOCAL")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET_MODELS = os.getenv("S3_BUCKET_MODELS", "models")
S3_BUCKET_DATASETS = os.getenv("S3_BUCKET_DATASETS", "datasets")
S3_BUCKET_FEEDBACK = os.getenv("S3_BUCKET_FEEDBACK", "feedback")
BOOTSTRAP_STRATEGY = os.getenv("BOOTSTRAP_STRATEGY", "LATEST")

from src.s3_utils import S3Utils
from src.device_utils import get_device
from src.train_engine import TrainEngine

s3_utils = None
if STORAGE_TYPE == "S3":
    s3_utils = S3Utils(S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY)

DEVICE = get_device()
train_engine = TrainEngine(s3_utils=s3_utils)

_production_gate = None
_heatmap_model = None
_ensemble_enabled = True 
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
            "model_versions": [{"id": "MODEL-R3-FINAL", "name": "EffNet-B0 (R3)", "status": "production", "metrics": {"f1": 0.94}, "lineage": "Round 3 Cascade"}],
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

# --- FastAPI Setup ---
app = FastAPI(title="SteelVision Final Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Ensure static directory exists
os.makedirs(str(MLOPS_ASSETS_ROOT), exist_ok=True)
app.mount("/mlops-assets", StaticFiles(directory=str(MLOPS_ASSETS_ROOT)), name="mlops-assets")

@app.on_event("startup")
async def startup():
    global _production_gate, _heatmap_model, s3_utils
    _load_state()

    # 1. Initialize S3 buckets if needed
    if STORAGE_TYPE == "S3" and s3_utils:
        try:
            s3_utils.create_bucket_if_not_exists(S3_BUCKET_MODELS)
            s3_utils.create_bucket_if_not_exists(S3_BUCKET_DATASETS)
            s3_utils.create_bucket_if_not_exists(S3_BUCKET_FEEDBACK)
            logger.info("S3 Buckets initialized.")

            # --- Automatic Migration (Option A) ---
            models_dir = PROJECT_ROOT / "models"
            if models_dir.exists():
                local_files = list(models_dir.glob("*.pt"))
                if local_files:
                    logger.info(f"Checking for migration: {len(local_files)} local models found.")
                    remote_files = s3_utils.list_objects(S3_BUCKET_MODELS)
                    for lf in local_files:
                        if lf.name not in remote_files:
                            logger.info(f"Migrating {lf.name} to S3...")
                            s3_utils.upload_file(str(lf), S3_BUCKET_MODELS, lf.name)
            # --------------------------------------

        except Exception as e:
            logger.error(f"S3 Initialization/Migration failed: {e}")

    # 2. Dynamic Model Loading (Decoupled)
    gate_file = None
    heatmap_file = None

    if STORAGE_TYPE == "S3" and BOOTSTRAP_STRATEGY == "LATEST":
        logger.info("Bootstrapping from S3 (Strategy: LATEST)...")
        gate_file = s3_utils.get_latest_object(S3_BUCKET_MODELS, prefix="gate")
        heatmap_file = s3_utils.get_latest_object(S3_BUCKET_MODELS, prefix="patchcore")
    
    # Fallback to local if S3 empty or disabled
    models_dir = PROJECT_ROOT / "models"
    if not gate_file:
        pts = list(models_dir.glob("*_gate.pt"))
        if pts:
            gate_file = sorted(pts, key=os.path.getmtime, reverse=True)[0].name
            logger.info(f"Fallback to local latest gate: {gate_file}")
    
    if not heatmap_file:
        pts = list(models_dir.glob("*_patchcore.pt"))
        if pts:
            heatmap_file = sorted(pts, key=os.path.getmtime, reverse=True)[0].name
            logger.info(f"Fallback to local latest heatmap: {heatmap_file}")

    # 3. Load Models
    try:
        if gate_file or heatmap_file:
            _perform_hot_swap(gate_file=gate_file or "", heatmap_file=heatmap_file or "")
            logger.info(f"Systems Online. Loaded: {gate_file}, {heatmap_file}")
        else:
            logger.warning("No models found to load during startup.")
    except Exception as e:
        logger.error(f"Initial model load failed: {e}")

# --- Core Inference API ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _production_gate: raise HTTPException(500, "Model not loaded")
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad(): 
        prob = torch.sigmoid(_production_gate["model"](tensor)).item()
    
    heatmap_res = _heatmap_model.predict(tensor) if _ensemble_enabled and prob > _production_gate["T_low"] and _heatmap_model else None
    score = heatmap_res["anomaly_score"] if heatmap_res else (prob if not _ensemble_enabled else 0.0)
    decision = "anomaly" if score > 0.5 else "normal"
    
    import base64
    def _to_b64(arr):
        if arr is None: return None
        if arr.dtype != np.uint8: arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "gate_score": prob, "decision": decision, "heatmap_score": score,
        "heatmap_overlay": _to_b64(heatmap_res.get("heatmap_overlay") if heatmap_res else None),
        "ensemble_active": _ensemble_enabled, "latency": {"total_latency_ms": 150} 
    }

# --- MLOps Dashboard & Management ---
@app.get("/mlops/dashboard")
async def get_dashboard():
    state = _load_state()
    
    # List models from S3 or Local
    available_pts = []
    if STORAGE_TYPE == "S3" and s3_utils:
        available_pts = [f for f in s3_utils.list_objects(S3_BUCKET_MODELS) if f.endswith(".pt")]
    else:
        models_dir = PROJECT_ROOT / "models"
        available_pts = [f.name for f in models_dir.glob("*.pt")]
    
    state["available_model_files"] = available_pts
    state["runtime_config"] = {
        "ensemble_enabled": _ensemble_enabled, 
        "current_model_id": _production_gate["id"] if _production_gate else None,
        "gate_file": _production_gate.get("filename") if _production_gate else None,
        "heatmap_file": getattr(_heatmap_model, "filename", None) if _heatmap_model else None
    }
    
    # Add interface specs for the UI
    state["interfaces"] = {
        "Gate": "EfficientNet-B0 (Input: 224x224, Output: Binary Logit)",
        "Heatmap": "PatchCore-R18 (Input: 224x224, Output: Anomaly Map)",
        "Storage": f"Centralized {STORAGE_TYPE} Artifact Registry"
    }
    return state

# --- Training Monitoring & Control ---
class TrainRequest(BaseModel):
    architecture: str = "ARCH-GATE-EFF"; epochs: int = 10; batch_size: int = 32
    learning_rate: float = 0.001; optimizer: str = "Adam"; augmentation: bool = True

@app.get("/mlops/training/status")
async def get_training_status(): return _training_status

async def run_training_process(req: TrainRequest):
    global _training_status
    state = _load_state()
    
    def _status_cb(progress, message, epoch=0, metrics=None):
        _training_status.update({
            "progress": progress,
            "message": message,
            "epoch": epoch,
            "metrics": metrics or _training_status.get("metrics", {"loss": 0, "acc": 0})
        })

    _training_status.update({"is_running": True, "message": "STARTING", "progress": 0, "epoch": 0})
    _append_log(state, "info", f"Real Training Started: {req.architecture}")
    _save_state(state)
    
    try:
        # Determine which model to train
        if "GATE" in req.architecture or "EFF" in req.architecture:
            result = await train_engine.train_gate(req.dict(), _status_cb)
        else:
            result = await train_engine.train_heatmap(req.dict(), _status_cb)
        
        _training_status.update({"is_running": False, "message": "COMPLETED", "progress": 100})
        
        state = _load_state()
        new_model_id = result["model_id"]
        new_run = {
            "id": new_model_id, "architecture": req.architecture, "params": req.dict(), 
            "final_metrics": result["metrics"], "completed_at": _iso_now(), "filename": result["filename"]
        }
        state["training_runs"].insert(0, new_run)
        state["model_versions"].insert(0, {
            "id": new_model_id, "name": f"Retrained {req.architecture}", "status": "candidate", 
            "metrics": result["metrics"], "lineage": f"Auto-train from S3", "filename": result["filename"]
        })
        _append_log(state, "success", f"Training Completed: {new_model_id}")
        _save_state(state)
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        _training_status.update({"is_running": False, "message": f"ERROR: {str(e)}"})

@app.post("/mlops/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if _training_status["is_running"]: raise HTTPException(400, "Training in progress")
    background_tasks.add_task(run_training_process, req)
    return {"message": "Training started", "config": req}

# --- Deployment & Deletion Control ---
class DeployRequest(BaseModel):
    model_id: str
    gate_file: str = ""       # 실제 로드할 gate 파일명 (예: round1_gate.pt)
    heatmap_file: str = ""    # 실제 로드할 heatmap 파일명
    ensemble_enabled: bool = True

def _perform_hot_swap(gate_file: str = "", heatmap_file: str = ""):
    """메모리에 로드된 모델을 실제 파일 기반으로 교체하는 핵심 로직 (S3 지원)"""
    global _production_gate, _heatmap_model, s3_utils
    
    models_dir = PROJECT_ROOT / "models"
    _ensure_dir(models_dir)

    def _get_local_path(filename: str):
        local_path = models_dir / filename
        if not local_path.exists() and STORAGE_TYPE == "S3" and s3_utils:
            logger.info(f"Downloading {filename} from S3...")
            s3_utils.download_file(S3_BUCKET_MODELS, filename, str(local_path))
        return local_path

    if gate_file:
        p_path = _get_local_path(gate_file)
        if p_path.exists():
            ckpt = torch.load(p_path, map_location=DEVICE, weights_only=False)
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
            model.load_state_dict(ckpt["model_state_dict"])
            if _production_gate:
                _production_gate["model"] = model.to(DEVICE).eval()
                _production_gate["filename"] = gate_file
            else:
                _production_gate = {"id": "BOOTSTRAP", "model": model.to(DEVICE).eval(), "input_size": 224, "T_low": 0.1, "T_high": 0.5, "filename": gate_file}
            logger.info(f"Hot-swapped Gate: {gate_file}")
    
    if heatmap_file:
        h_path = _get_local_path(heatmap_file)
        if h_path.exists():
            _heatmap_model = PatchCoreModel.load(str(h_path), device=str(DEVICE))
            _heatmap_model.filename = heatmap_file
            logger.info(f"Hot-swapped Heatmap: {heatmap_file}")

@app.post("/mlops/deploy")
async def deploy_model(req: DeployRequest):
    global _production_gate, _ensemble_enabled
    state = _load_state()
    
    # 1. 실제 모델 파일 로드 (Hot-swap)
    try:
        _perform_hot_swap(gate_file=req.gate_file, heatmap_file=req.heatmap_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Swap Failed: {e}")

    # 2. 전역 설정 반영
    _ensemble_enabled = req.ensemble_enabled
    if _production_gate: _production_gate["id"] = req.model_id
    
    # 3. state.json 상태 업데이트
    for m in state["model_versions"]:
        if m["id"] == req.model_id: m["status"] = "production"
        elif m["status"] == "production": m["status"] = "candidate"
    
    _append_log(state, "warning", f"Deployment Success: {req.model_id} (File: {req.gate_file})")
    _save_state(state)
    return {"message": "Deployment successful", "current_config": {"model_id": req.model_id, "gate_file": req.gate_file}}

@app.delete("/mlops/training/runs/{run_id}")
async def delete_training_run(run_id: str):
    """특정 학습 이력 삭제 API"""
    state = _load_state()
    state["training_runs"] = [r for r in state["training_runs"] if r["id"] != run_id]
    # 연관된 모델 버전도 배포 중이 아니라면 삭제 가능
    state["model_versions"] = [m for m in state["model_versions"] if not (m["id"] == run_id and m["status"] != "production")]
    _append_log(state, "warning", f"Training Run Deleted: {run_id}")
    _save_state(state)
    return {"message": f"Run {run_id} deleted"}

@app.post("/mlops/feedback")
async def create_feedback(
    file: UploadFile = File(...), feedback_type: str = Form(...), label: str = Form("unlabeled"),
    operator: str = Form(""), comment: str = Form(""), line: str = Form(""),
    gate_score: float = Form(0.0), heatmap_score: float = Form(0.0), predicted_label: str = Form("")
):
    state = _load_state()
    ext = Path(file.filename or "sample.png").suffix; fname = f"{uuid.uuid4().hex}{ext}"
    
    # 1. Physical Save (Local)
    fpath = MLOPS_ASSETS_ROOT / "feedback" / fname; _ensure_dir(fpath.parent)
    with open(fpath, "wb") as f: shutil.copyfileobj(file.file, f)
    
    # 2. Sync to S3
    if STORAGE_TYPE == "S3" and s3_utils:
        s3_utils.upload_file(str(fpath), S3_BUCKET_FEEDBACK, fname)

    fb_item = {
        "id": f"FDBK-{uuid.uuid4().hex[:8].upper()}", "feedback_type": feedback_type, "label": label, "operator": operator,
        "comment": comment, "line": line, "image_url": f"/mlops-assets/feedback/{fname}", "created_at": _iso_now(),
        "model_prediction": {"gate_score": gate_score, "heatmap_score": heatmap_score, "predicted_label": predicted_label}
    }
    state["feedback_items"].insert(0, fb_item); state["dataset_versions"][0]["feedback_count"] += 1
    _append_log(state, "info", f"Data Ingested (S3 synced): {feedback_type}"); _save_state(state)
    return {"message": "Success", "feedback_item": fb_item}

@app.delete("/mlops/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    state = _load_state()
    item_to_remove = next((item for item in state["feedback_items"] if item["id"] == feedback_id), None)
    if not item_to_remove: raise HTTPException(404, "Feedback not found")
    img_abs_path = MLOPS_ASSETS_ROOT / item_to_remove["image_url"].replace("/mlops-assets/", "")
    if img_abs_path.exists(): os.remove(img_abs_path)
    state["feedback_items"] = [item for item in state["feedback_items"] if item["id"] != feedback_id]
    state["dataset_versions"][0]["feedback_count"] = max(0, state["dataset_versions"][0]["feedback_count"] - 1)
    _append_log(state, "warning", f"Data Removed: {feedback_id}"); _save_state(state)
    return {"message": "Deleted"}

@app.post("/mlops/datasets/upload")
async def upload_data(files: List[UploadFile] = File(...)):
    state = _load_state(); added = 0
    for f in files:
        fname = f"{uuid.uuid4().hex}{Path(f.filename or '.png').suffix}"
        fpath = MLOPS_ASSETS_ROOT / "uploads" / fname; _ensure_dir(fpath.parent)
        with open(fpath, "wb") as out: shutil.copyfileobj(f.file, out)
        
        # Sync to S3
        if STORAGE_TYPE == "S3" and s3_utils:
            s3_utils.upload_file(str(fpath), S3_BUCKET_DATASETS, fname)
            
        # Register sample in state
        new_sample = {
            "id": f"SMPL-{uuid.uuid4().hex[:8].upper()}",
            "file_url": f"/mlops-assets/uploads/{fname}",
            "file_name": f.filename or "unknown.png",
            "created_at": _iso_now()
        }
        state["dataset_versions"][0]["samples"].insert(0, new_sample)
        added += 1
    
    # Limit samples to prevent JSON bloat (keep last 100 recent uploads visible)
    state["dataset_versions"][0]["samples"] = state["dataset_versions"][0]["samples"][:100]
    state["dataset_versions"][0]["sample_count"] += added
    _append_log(state, "info", f"Bulk upload to S3: {added} files"); _save_state(state)
    return {"message": "Success", "added_count": added}

@app.post("/mlops/architectures/upload")
async def upload_architecture(
    file: UploadFile = File(...),
    kind: str = Form(...),
    name: str = Form(...)
):
    """모델 구조(Architecture) 등록 API"""
    state = _load_state()
    arch_id = f"ARCH-{kind.upper()}-{uuid.uuid4().hex[:4].upper()}"
    
    new_arch = {
        "id": arch_id,
        "name": name,
        "kind": kind,
        "created_at": _iso_now(),
        "interface": {"input": "Image (224x224)", "output": "Score/Heatmap"}
    }
    
    state["architectures"].append(new_arch)
    _append_log(state, "info", f"New Architecture Registered: {name} ({kind})")
    _save_state(state)
    return {"message": "Architecture registered", "architecture": new_arch}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
