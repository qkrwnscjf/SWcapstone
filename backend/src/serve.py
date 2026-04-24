#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import logging
import os
import pickle
import shutil
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torchvision import models, transforms

from src.heatmap_model import PatchCoreModel


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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("steelvision")
KST = ZoneInfo("Asia/Seoul")


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    candidates = [current.parent.parent]
    try:
        candidates.append(current.parents[2])
    except IndexError:
        pass

    for candidate in candidates:
        if (candidate / "models").exists():
            return candidate
    return candidates[0]


PROJECT_ROOT = _resolve_project_root()
MLOPS_ROOT = PROJECT_ROOT / "storage" / "mlops"
MLOPS_ASSETS_ROOT = MLOPS_ROOT / "assets"
MLOPS_STATE_PATH = MLOPS_ROOT / "state.json"
MLOPS_ASSETS_ROOT.mkdir(parents=True, exist_ok=True)


def _has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if _has_mps() else "cpu"))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _iso_now() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "artifact"


def _public_asset_url(path: Path) -> str:
    return f"/mlops-assets/{path.relative_to(MLOPS_ASSETS_ROOT).as_posix()}"


def _np_to_base64_png(arr: np.ndarray) -> str:
    import base64

    if arr.dtype != np.uint8:
        arr_max = float(np.max(arr)) if arr.size else 0.0
        if arr_max <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    if arr.ndim == 2:
        image = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 4:
        image = Image.fromarray(arr, mode="RGBA")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        image = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


GATE_MODEL_PATH = _env_str(
    "GATE_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "round3_effnetb0_gate.pt"),
)
HEATMAP_MODEL_PATH = _env_str(
    "HEATMAP_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "round3_patchcore_r18_patchcore.pt"),
)

T_LOW = 0.3
T_HIGH = 0.7
INPUT_SIZE = 224


def build_preprocess(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


_preprocess = build_preprocess(INPUT_SIZE)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image).unsqueeze(0)
    return tensor.to(DEVICE)


def _normalize_gate_name(gate_name: str) -> str:
    return gate_name.strip().lower().replace("-", "").replace("_", "")


def _build_effnet_gate_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


def _build_legacy_effnet_gate_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def build_gate_model(gate_name: str) -> nn.Module:
    normalized = _normalize_gate_name(gate_name)

    if normalized in {"effnetb0", "efficientnetb0"}:
        return _build_effnet_gate_model()

    if normalized in {"mnv3large", "mobilenetv3large"}:
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 1),
        )
        return model

    raise ValueError(f"Unsupported gate model: {gate_name}")


def load_gate_checkpoint(model_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_state = ckpt.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint does not contain model_state_dict: {model_path}")

    gate_name = str(ckpt.get("gate_name") or ckpt.get("backbone") or "effnetb0")
    builders: List[tuple[str, nn.Module]] = []
    try:
        builders.append((gate_name, build_gate_model(gate_name)))
    except ValueError:
        builders.append(("effnetb0", _build_effnet_gate_model()))

    if _normalize_gate_name(gate_name) in {"effnetb0", "efficientnetb0"}:
        builders.append(("effnetb0-legacy", _build_legacy_effnet_gate_model()))

    loaded_model: Optional[nn.Module] = None
    load_errors: List[str] = []
    for builder_name, candidate in builders:
        try:
            candidate.load_state_dict(model_state, strict=True)
            loaded_model = candidate
            gate_name = builder_name
            break
        except Exception as exc:  # pragma: no cover - compatibility path
            load_errors.append(f"{builder_name}: {exc}")

    if loaded_model is None:
        raise ValueError(" / ".join(load_errors))

    loaded_model.to(device)
    loaded_model.eval()

    input_size_value = ckpt.get("input_size", 224)
    if isinstance(input_size_value, (list, tuple)):
        input_size = int(input_size_value[0])
    else:
        input_size = int(input_size_value)

    bundle = {
        "model": loaded_model,
        "gate_name": gate_name,
        "backbone": ckpt.get("backbone", gate_name),
        "input_size": input_size,
        "T_low": float(ckpt.get("T_low", 0.3)),
        "T_high": float(ckpt.get("T_high", 0.7)),
        "checkpoint_path": model_path,
    }
    return bundle


def find_calibrator_path(gate_model_path: str) -> Optional[str]:
    gate_path = Path(gate_model_path)
    stem = gate_path.stem

    if stem.endswith("_gate"):
        calibrator_name = stem[:-5] + "_calibrator.pkl"
    else:
        calibrator_name = stem + "_calibrator.pkl"

    calibrator_path = gate_path.with_name(calibrator_name)
    return str(calibrator_path) if calibrator_path.exists() else None


def load_gate_calibrator(calibrator_path: Optional[str]) -> Optional[Any]:
    if not calibrator_path:
        return None

    try:
        with open(calibrator_path, "rb") as handle:
            obj = pickle.load(handle)
        calibrator = obj.get("calibrator", obj)
        logger.info("Loaded gate calibrator from %s", calibrator_path)
        return calibrator
    except Exception as exc:  # pragma: no cover - compatibility path
        logger.warning("Failed to load gate calibrator from %s: %s", calibrator_path, exc)
        return None


@torch.no_grad()
def gate_predict(bundle: Dict[str, Any], image_tensor: torch.Tensor, calibrator: Optional[Any] = None) -> float:
    model = bundle["model"]
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    logits = model(image_tensor.to(next(model.parameters()).device)).squeeze(-1)
    probability = float(torch.sigmoid(logits).item())

    if calibrator is not None:
        probs = np.array([probability], dtype=np.float32)
        if hasattr(calibrator, "predict_proba"):
            calibrated = np.asarray(calibrator.predict_proba(probs))
            probability = float(calibrated[0, 1] if calibrated.ndim == 2 else calibrated[0])
        elif hasattr(calibrator, "predict"):
            probability = float(np.asarray(calibrator.predict(probs))[0])

    return float(np.clip(probability, 0.0, 1.0))


def heatmap_predict(model: PatchCoreModel, image_tensor: torch.Tensor) -> Dict[str, Any]:
    return model.predict(image_tensor)


class _Metrics:
    def __init__(self) -> None:
        self.total_predictions = 0
        self.heatmap_calls = 0
        self.gate_latency_sum_ms = 0.0
        self.heatmap_latency_sum_ms = 0.0
        self.total_latency_sum_ms = 0.0

    def record(
        self,
        *,
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

    def snapshot(self) -> Dict[str, Any]:
        total = max(self.total_predictions, 1)
        heatmap_total = max(self.heatmap_calls, 1)
        return {
            "total_predictions": self.total_predictions,
            "heatmap_calls": self.heatmap_calls,
            "heatmap_call_rate": round(self.heatmap_calls / total, 4),
            "avg_gate_latency_ms": round(self.gate_latency_sum_ms / total, 2),
            "avg_heatmap_latency_ms": round(self.heatmap_latency_sum_ms / heatmap_total, 2)
            if self.heatmap_calls
            else 0.0,
            "avg_total_latency_ms": round(self.total_latency_sum_ms / total, 2),
        }


metrics = _Metrics()


def _default_state() -> Dict[str, Any]:
    now = _iso_now()
    return {
        "active_dataset_id": "DATA-V3-PRODUCTION",
        "dataset_versions": [
            {
                "id": "DATA-V3-PRODUCTION",
                "name": "Round 3 Pool",
                "status": "locked",
                "created_at": now,
                "updated_at": now,
                "source_dataset_id": None,
                "sample_count": 14850,
                "feedback_count": 0,
                "samples": [],
                "notes": "Current production baseline dataset.",
            }
        ],
        "architectures": [
            {
                "id": "ARCH-GATE-EFF",
                "name": "EfficientNet-B0 Gate",
                "kind": "gate",
                "source": "builtin",
                "created_at": now,
                "interface": {
                    "input": "Tensor[B,3,224,224]",
                    "output": "anomaly probability",
                },
                "file_url": None,
            },
            {
                "id": "ARCH-HM-PC",
                "name": "PatchCore ResNet18",
                "kind": "heatmap",
                "source": "builtin",
                "created_at": now,
                "interface": {
                    "input": "Tensor[B,3,224,224]",
                    "output": "anomaly_score + normalized heatmap + overlay",
                },
                "file_url": None,
            },
        ],
        "training_runs": [],
        "model_versions": [
            {
                "id": "MODEL-R3-FINAL",
                "name": "Round 3 Production Cascade",
                "status": "production",
                "dataset_version_id": "DATA-V3-PRODUCTION",
                "gate_architecture_id": "ARCH-GATE-EFF",
                "heatmap_architecture_id": "ARCH-HM-PC",
                "created_at": now,
                "updated_at": now,
                "metrics": {"f1": 0.94, "latency_ms": 53},
                "lineage": "EfficientNet gate -> PatchCore heatmap",
                "training_run_id": None,
                "target_line": None,
            }
        ],
        "feedback_items": [],
        "logs": [
            {
                "id": f"LOG-{uuid.uuid4().hex[:10].upper()}",
                "level": "info",
                "time": now,
                "message": "MLOps storage initialized.",
            }
        ],
        "deployment": {
            "production_model_id": "MODEL-R3-FINAL",
            "staging_model_id": None,
            "canary_model_id": None,
            "canary_line": None,
            "previous_production_model_id": None,
            "last_action": "initialized",
            "last_action_at": now,
        },
    }


def _ensure(target: Dict[str, Any], key: str, default: Any) -> bool:
    if key not in target:
        target[key] = default
        return True
    return False


def _sync_deployment(state: Dict[str, Any]) -> None:
    deployment = state.setdefault("deployment", {})
    deployment["production_model_id"] = next(
        (model["id"] for model in state["model_versions"] if model.get("status") == "production"),
        None,
    )
    deployment["staging_model_id"] = next(
        (model["id"] for model in state["model_versions"] if model.get("status") == "staging"),
        None,
    )
    deployment["canary_model_id"] = next(
        (model["id"] for model in state["model_versions"] if model.get("status") == "canary"),
        None,
    )
    if not deployment.get("canary_model_id"):
        deployment["canary_line"] = None


def _normalize_state(state: Dict[str, Any]) -> bool:
    changed = False
    defaults = _default_state()

    for key in ("active_dataset_id", "dataset_versions", "architectures", "training_runs", "model_versions", "feedback_items", "logs", "deployment"):
        changed |= _ensure(state, key, deepcopy(defaults[key]))

    if not state["dataset_versions"]:
        state["dataset_versions"] = deepcopy(defaults["dataset_versions"])
        changed = True
    if not state["architectures"]:
        state["architectures"] = deepcopy(defaults["architectures"])
        changed = True
    if not state["model_versions"]:
        state["model_versions"] = deepcopy(defaults["model_versions"])
        changed = True

    gate_arch_default = next(
        (arch["id"] for arch in state["architectures"] if arch.get("kind") == "gate"),
        defaults["architectures"][0]["id"],
    )
    heatmap_arch_default = next(
        (arch["id"] for arch in state["architectures"] if arch.get("kind") == "heatmap"),
        defaults["architectures"][1]["id"],
    )

    for dataset in state["dataset_versions"]:
        changed |= _ensure(dataset, "name", dataset.get("id", "Dataset"))
        changed |= _ensure(dataset, "status", "draft")
        changed |= _ensure(dataset, "created_at", dataset.get("updated_at", _iso_now()))
        changed |= _ensure(dataset, "updated_at", dataset["created_at"])
        changed |= _ensure(dataset, "source_dataset_id", None)
        changed |= _ensure(dataset, "sample_count", 0)
        changed |= _ensure(dataset, "feedback_count", 0)
        changed |= _ensure(dataset, "samples", [])
        changed |= _ensure(dataset, "notes", "")

    for arch in state["architectures"]:
        changed |= _ensure(arch, "name", arch.get("id", "Architecture"))
        changed |= _ensure(arch, "kind", "gate")
        changed |= _ensure(arch, "source", "builtin")
        changed |= _ensure(arch, "created_at", _iso_now())
        changed |= _ensure(
            arch,
            "interface",
            {
                "input": "Tensor[B,3,224,224]",
                "output": "model-specific output",
            },
        )
        changed |= _ensure(arch, "file_url", None)

    for run in state["training_runs"]:
        changed |= _ensure(run, "status", "configured")
        changed |= _ensure(run, "created_at", _iso_now())
        changed |= _ensure(run, "dataset_version_id", state["active_dataset_id"])
        changed |= _ensure(run, "gate_architecture_id", gate_arch_default)
        changed |= _ensure(run, "heatmap_architecture_id", heatmap_arch_default)
        changed |= _ensure(run, "train_strategy", "cascade")
        changed |= _ensure(run, "notes", "")
        changed |= _ensure(run, "lineage", "EfficientNet gate -> PatchCore heatmap")
        changed |= _ensure(run, "sample_count", 0)

    for model in state["model_versions"]:
        changed |= _ensure(model, "name", model.get("id", "Model"))
        changed |= _ensure(model, "status", "staging")
        changed |= _ensure(model, "dataset_version_id", None)
        changed |= _ensure(model, "gate_architecture_id", gate_arch_default)
        changed |= _ensure(model, "heatmap_architecture_id", heatmap_arch_default)
        changed |= _ensure(model, "created_at", _iso_now())
        changed |= _ensure(model, "updated_at", model["created_at"])
        changed |= _ensure(model, "metrics", {"f1": None, "latency_ms": None})
        changed |= _ensure(model, "lineage", "EfficientNet gate -> PatchCore heatmap")
        changed |= _ensure(model, "training_run_id", None)
        changed |= _ensure(model, "target_line", None)

    for item in state["feedback_items"]:
        changed |= _ensure(item, "sample_id", "")
        changed |= _ensure(item, "dataset_version_id", state["active_dataset_id"])
        changed |= _ensure(item, "feedback_type", "needs_review")
        changed |= _ensure(item, "label", "unlabeled")
        changed |= _ensure(item, "operator", "")
        changed |= _ensure(item, "comment", "")
        changed |= _ensure(item, "line", "")
        changed |= _ensure(item, "predicted_label", "")
        changed |= _ensure(item, "image_url", "")
        changed |= _ensure(item, "created_at", _iso_now())

    for log in state["logs"]:
        changed |= _ensure(log, "id", f"LOG-{uuid.uuid4().hex[:10].upper()}")
        changed |= _ensure(log, "level", "info")
        changed |= _ensure(log, "time", _iso_now())
        changed |= _ensure(log, "message", "")

    deployment = state["deployment"]
    changed |= _ensure(deployment, "production_model_id", None)
    changed |= _ensure(deployment, "staging_model_id", None)
    changed |= _ensure(deployment, "canary_model_id", None)
    changed |= _ensure(deployment, "canary_line", None)
    changed |= _ensure(deployment, "previous_production_model_id", None)
    changed |= _ensure(deployment, "last_action", "loaded")
    changed |= _ensure(deployment, "last_action_at", _iso_now())

    before = json.dumps(deployment, ensure_ascii=False, sort_keys=True)
    _sync_deployment(state)
    after = json.dumps(state["deployment"], ensure_ascii=False, sort_keys=True)
    if before != after:
        changed = True

    return changed


def _save_state(state: Dict[str, Any]) -> None:
    _ensure_dir(MLOPS_ROOT)
    with open(MLOPS_STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def _ensure_mlops_storage() -> None:
    for subdir in (
        MLOPS_ROOT,
        MLOPS_ASSETS_ROOT / "feedback",
        MLOPS_ASSETS_ROOT / "uploads",
        MLOPS_ASSETS_ROOT / "architectures",
    ):
        _ensure_dir(subdir)

    if not MLOPS_STATE_PATH.exists():
        _save_state(_default_state())


def _load_state() -> Dict[str, Any]:
    _ensure_mlops_storage()
    with open(MLOPS_STATE_PATH, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    if _normalize_state(state):
        _save_state(state)
    return state


def _append_log(state: Dict[str, Any], level: str, message: str) -> None:
    state["logs"].insert(
        0,
        {
            "id": f"LOG-{uuid.uuid4().hex[:10].upper()}",
            "level": level,
            "time": _iso_now(),
            "message": message,
        },
    )
    state["logs"] = state["logs"][:100]


def _get_dataset(state: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    dataset = next((item for item in state["dataset_versions"] if item["id"] == dataset_id), None)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset version not found: {dataset_id}")
    return dataset


def _get_model(state: Dict[str, Any], model_version_id: str) -> Dict[str, Any]:
    model = next((item for item in state["model_versions"] if item["id"] == model_version_id), None)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model version not found: {model_version_id}")
    return model


def _next_version_id(prefix: str, items: List[Dict[str, Any]]) -> str:
    today = datetime.now(KST).strftime("%Y.%m.%d")
    revision = sum(1 for item in items if str(item.get("id", "")).startswith(f"{prefix}-{today}")) + 1
    return f"{prefix}-{today}-r{revision}"


def _next_run_id(state: Dict[str, Any]) -> str:
    return f"RUN-{datetime.now(KST).strftime('%Y%m%d')}-{len(state['training_runs']) + 1:02d}"


def _copy_dataset_snapshot(state: Dict[str, Any], source_dataset_id: str) -> Dict[str, Any]:
    source = _get_dataset(state, source_dataset_id)
    snapshot = {
        "id": _next_version_id("DATA", state["dataset_versions"]),
        "name": f"Snapshot of {source['id']}",
        "status": "locked",
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
        "source_dataset_id": source["id"],
        "sample_count": source.get("sample_count", 0),
        "feedback_count": source.get("feedback_count", 0),
        "samples": list(source.get("samples", [])),
        "notes": f"Training snapshot created from {source['id']}",
    }
    state["dataset_versions"].insert(0, snapshot)
    return snapshot


def _register_sample(
    dataset: Dict[str, Any],
    *,
    file_url: str,
    file_name: str,
    label: str,
    source_type: str,
    feedback_type: Optional[str] = None,
    comment: str = "",
    line: str = "",
    operator: str = "",
    predicted_label: str = "",
) -> Dict[str, Any]:
    sample = {
        "id": f"SAMPLE-{uuid.uuid4().hex[:12].upper()}",
        "file_url": file_url,
        "file_name": file_name,
        "label": label,
        "source_type": source_type,
        "feedback_type": feedback_type,
        "comment": comment,
        "line": line,
        "operator": operator,
        "predicted_label": predicted_label,
        "created_at": _iso_now(),
    }
    dataset["samples"].insert(0, sample)
    dataset["sample_count"] = int(dataset.get("sample_count", 0)) + 1
    if feedback_type:
        dataset["feedback_count"] = int(dataset.get("feedback_count", 0)) + 1
    dataset["updated_at"] = _iso_now()
    return sample


def _should_call_heatmap(probability: float) -> bool:
    if probability <= T_LOW:
        return False
    return True


def _cascade_decision(probability: float, heatmap_score: Optional[float]) -> str:
    if heatmap_score is None:
        return "anomaly" if probability >= T_HIGH else "normal"
    return "anomaly" if heatmap_score >= 0.5 else "normal (heatmap)"


class TrainRunRequest(BaseModel):
    dataset_version_id: Optional[str] = None
    gate_architecture_id: str
    heatmap_architecture_id: str
    train_strategy: str = "cascade"
    notes: str = ""


class PromoteModelRequest(BaseModel):
    model_version_id: str
    target_status: Literal["staging", "production"] = "production"


class CanaryRequest(BaseModel):
    model_version_id: str
    line: str = "LINE-B"


class RollbackRequest(BaseModel):
    model_version_id: Optional[str] = None


app = FastAPI(
    title="SteelVision Final Server",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/mlops-assets", StaticFiles(directory=str(MLOPS_ASSETS_ROOT)), name="mlops-assets")

_gate_bundle: Optional[Dict[str, Any]] = None
_gate_calibrator: Optional[Any] = None
_heatmap_model: Optional[PatchCoreModel] = None


@app.on_event("startup")
async def startup() -> None:
    global _gate_bundle, _gate_calibrator, _heatmap_model, INPUT_SIZE, T_LOW, T_HIGH, _preprocess

    _load_state()

    try:
        _gate_bundle = load_gate_checkpoint(GATE_MODEL_PATH, DEVICE)
        INPUT_SIZE = int(_gate_bundle["input_size"])
        T_LOW = float(_gate_bundle["T_low"])
        T_HIGH = float(_gate_bundle["T_high"])
        _preprocess = build_preprocess(INPUT_SIZE)
        _gate_calibrator = load_gate_calibrator(find_calibrator_path(GATE_MODEL_PATH))
        logger.info(
            "Gate loaded: gate_name=%s, input_size=%d, T_low=%.3f, T_high=%.3f",
            _gate_bundle["gate_name"],
            INPUT_SIZE,
            T_LOW,
            T_HIGH,
        )
    except Exception as exc:  # pragma: no cover - startup path
        _gate_bundle = None
        logger.exception("Failed to load gate model: %s", exc)

    try:
        if Path(HEATMAP_MODEL_PATH).exists():
            _heatmap_model = PatchCoreModel.load(HEATMAP_MODEL_PATH, device=str(DEVICE))
            logger.info("Heatmap model loaded from %s", HEATMAP_MODEL_PATH)
        else:
            logger.warning("Heatmap model file not found: %s", HEATMAP_MODEL_PATH)
    except Exception as exc:  # pragma: no cover - startup path
        _heatmap_model = None
        logger.exception("Failed to load heatmap model: %s", exc)


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
        },
    }


@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _gate_bundle is None:
        raise HTTPException(status_code=500, detail="Gate model is not loaded.")

    started_at = time.perf_counter()

    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc

    gate_started_at = time.perf_counter()
    try:
        gate_score = gate_predict(_gate_bundle, tensor, _gate_calibrator)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gate model inference failed: {exc}") from exc
    gate_latency_ms = (time.perf_counter() - gate_started_at) * 1000.0

    heatmap_called = False
    heatmap_score: Optional[float] = None
    heatmap_latency_ms = 0.0
    overlay_b64: Optional[str] = None
    heatmap_b64: Optional[str] = None

    if _should_call_heatmap(gate_score) and _heatmap_model is not None:
        heatmap_called = True
        heatmap_started_at = time.perf_counter()
        try:
            hm_result = heatmap_predict(_heatmap_model, tensor)
            heatmap_score = float(hm_result.get("anomaly_score", hm_result.get("score", 0.0)))

            overlay = hm_result.get("heatmap_overlay")
            if overlay is not None:
                if torch.is_tensor(overlay):
                    overlay = overlay.detach().cpu().numpy()
                overlay_b64 = _np_to_base64_png(np.asarray(overlay))

            norm_map = hm_result.get("normalized_score_heatmap")
            if norm_map is not None:
                if torch.is_tensor(norm_map):
                    norm_map = norm_map.detach().cpu().numpy()
                heatmap_b64 = _np_to_base64_png(np.asarray(norm_map))
        except Exception as exc:
            logger.exception("Heatmap inference failed: %s", exc)
            heatmap_called = False
            heatmap_score = None
            overlay_b64 = None
            heatmap_b64 = None
        heatmap_latency_ms = (time.perf_counter() - heatmap_started_at) * 1000.0

    decision = _cascade_decision(gate_score, heatmap_score)
    total_latency_ms = (time.perf_counter() - started_at) * 1000.0

    metrics.record(
        called_heatmap=heatmap_called,
        gate_latency_ms=gate_latency_ms,
        heatmap_latency_ms=heatmap_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    logger.info(
        "predict gate=%.4f heatmap_called=%s heatmap=%s decision=%s total_ms=%.2f",
        gate_score,
        heatmap_called,
        f"{heatmap_score:.4f}" if heatmap_score is not None else "N/A",
        decision,
        total_latency_ms,
    )

    return {
        "gate_score": round(float(gate_score), 6),
        "decision": decision,
        "heatmap_called": heatmap_called,
        "heatmap_score": None if heatmap_score is None else round(float(heatmap_score), 6),
        "override_active": False,
        "latency": {
            "gate_latency_ms": round(gate_latency_ms, 3),
            "heatmap_latency_ms": round(heatmap_latency_ms, 3),
            "total_latency_ms": round(total_latency_ms, 3),
        },
        "heatmap_overlay": overlay_b64,
        "normalized_score_heatmap": heatmap_b64,
        "cascade": {
            "gate_architecture": _gate_bundle["gate_name"] if _gate_bundle else None,
            "heatmap_architecture": "patchcore",
            "t_low": round(float(T_LOW), 6),
            "t_high": round(float(T_HIGH), 6),
        },
    }


@app.get("/mlops/dashboard")
async def mlops_dashboard():
    state = _load_state()
    return {
        "active_dataset_id": state["active_dataset_id"],
        "dataset_versions": state["dataset_versions"][:10],
        "architectures": state["architectures"],
        "training_runs": state["training_runs"][:10],
        "model_versions": state["model_versions"][:10],
        "feedback_items": state["feedback_items"][:12],
        "logs": state["logs"][:20],
        "deployment": state["deployment"],
        "interfaces": {
            "gate_input": f"Tensor[B,3,{INPUT_SIZE},{INPUT_SIZE}]",
            "gate_output": "Anomaly probability",
            "heatmap_input": f"Tensor[B,3,{INPUT_SIZE},{INPUT_SIZE}]",
            "heatmap_output": "anomaly_score + normalized heatmap + overlay",
            "cascade": "EfficientNet gate -> PatchCore heatmap",
        },
    }


@app.post("/mlops/feedback")
async def create_feedback(
    file: UploadFile = File(...),
    feedback_type: Literal["false_positive", "false_negative", "confirmed_anomaly", "needs_review"] = Form(...),
    label: Literal["normal", "anomaly", "unlabeled"] = Form("unlabeled"),
    operator: str = Form(""),
    comment: str = Form(""),
    line: str = Form(""),
    predicted_label: str = Form(""),
):
    state = _load_state()
    dataset = _get_dataset(state, state["active_dataset_id"])

    extension = Path(file.filename or "sample.png").suffix or ".png"
    target_dir = MLOPS_ASSETS_ROOT / "feedback" / datetime.now(KST).strftime("%Y%m%d")
    _ensure_dir(target_dir)
    target_path = target_dir / f"{uuid.uuid4().hex}{extension}"
    with open(target_path, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    sample = _register_sample(
        dataset,
        file_url=_public_asset_url(target_path),
        file_name=file.filename or target_path.name,
        label=label,
        source_type="operator_feedback",
        feedback_type=feedback_type,
        comment=comment,
        line=line,
        operator=operator,
        predicted_label=predicted_label,
    )

    feedback_item = {
        "id": f"FDBK-{uuid.uuid4().hex[:12].upper()}",
        "sample_id": sample["id"],
        "dataset_version_id": dataset["id"],
        "feedback_type": feedback_type,
        "label": label,
        "operator": operator,
        "comment": comment,
        "line": line,
        "predicted_label": predicted_label,
        "image_url": sample["file_url"],
        "created_at": _iso_now(),
    }
    state["feedback_items"].insert(0, feedback_item)
    _append_log(state, "info", f"Operator feedback added to {dataset['id']} ({feedback_type}).")
    _save_state(state)
    return {
        "message": "Feedback added to active dataset.",
        "feedback_item": feedback_item,
        "dataset_version": dataset,
    }


@app.post("/mlops/datasets/upload")
async def upload_dataset_files(
    files: List[UploadFile] = File(...),
    label: Literal["normal", "anomaly", "unlabeled"] = Form("unlabeled"),
    source_type: Literal["bulk_upload", "field_capture", "reviewed_set"] = Form("bulk_upload"),
    line: str = Form(""),
    comment: str = Form(""),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    state = _load_state()
    dataset = _get_dataset(state, state["active_dataset_id"])
    ingest_dir = MLOPS_ASSETS_ROOT / "uploads" / datetime.now(KST).strftime("%Y%m%d") / uuid.uuid4().hex[:8]
    _ensure_dir(ingest_dir)

    added_samples: List[Dict[str, Any]] = []
    for upload in files:
        extension = Path(upload.filename or "sample.png").suffix or ".png"
        clean_name = Path(upload.filename or f"{uuid.uuid4().hex}{extension}").name
        target_path = ingest_dir / clean_name
        with open(target_path, "wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        sample = _register_sample(
            dataset,
            file_url=_public_asset_url(target_path),
            file_name=clean_name,
            label=label,
            source_type=source_type,
            comment=comment,
            line=line,
        )
        added_samples.append(sample)

    _append_log(state, "info", f"{len(added_samples)} files ingested into {dataset['id']}.")
    _save_state(state)
    return {
        "message": f"{len(added_samples)} files added to the active dataset.",
        "dataset_version": dataset,
        "added_samples": added_samples[:12],
        "added_count": len(added_samples),
    }


@app.post("/mlops/architectures/upload")
async def upload_architecture(
    file: UploadFile = File(...),
    kind: Literal["gate", "heatmap"] = Form(...),
    name: str = Form(...),
):
    state = _load_state()
    extension = Path(file.filename or "model.py").suffix or ".py"
    architecture_dir = MLOPS_ASSETS_ROOT / "architectures" / kind
    _ensure_dir(architecture_dir)
    target_path = architecture_dir / f"{_slugify(name)}-{uuid.uuid4().hex[:8]}{extension}"
    with open(target_path, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    architecture = {
        "id": f"ARCH-{kind.upper()}-{uuid.uuid4().hex[:8].upper()}",
        "name": name,
        "kind": kind,
        "source": "custom",
        "created_at": _iso_now(),
        "interface": {
            "input": f"Tensor[B,3,{INPUT_SIZE},{INPUT_SIZE}]",
            "output": "Gate: anomaly probability / Heatmap: anomaly_score + maps",
        },
        "file_url": _public_asset_url(target_path),
    }
    state["architectures"].insert(0, architecture)
    _append_log(state, "info", f"Custom {kind} architecture registered: {name}.")
    _save_state(state)
    return {"message": "Architecture registered.", "architecture": architecture}


@app.post("/mlops/train")
async def create_training_run(payload: TrainRunRequest):
    state = _load_state()
    dataset_id = payload.dataset_version_id or state["active_dataset_id"]
    snapshot = _copy_dataset_snapshot(state, dataset_id)
    run_id = _next_run_id(state)
    model_version_id = _next_version_id("MODEL", state["model_versions"])

    run = {
        "id": run_id,
        "status": "configured",
        "created_at": _iso_now(),
        "dataset_version_id": snapshot["id"],
        "gate_architecture_id": payload.gate_architecture_id,
        "heatmap_architecture_id": payload.heatmap_architecture_id,
        "train_strategy": payload.train_strategy,
        "notes": payload.notes,
        "lineage": "EfficientNet gate -> PatchCore heatmap",
        "sample_count": snapshot["sample_count"],
    }
    state["training_runs"].insert(0, run)

    model_version = {
        "id": model_version_id,
        "name": f"Cascade Candidate {model_version_id}",
        "status": "staging",
        "dataset_version_id": snapshot["id"],
        "gate_architecture_id": payload.gate_architecture_id,
        "heatmap_architecture_id": payload.heatmap_architecture_id,
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
        "metrics": {"f1": None, "latency_ms": None},
        "lineage": "EfficientNet gate -> PatchCore heatmap",
        "training_run_id": run_id,
        "target_line": None,
    }
    state["model_versions"].insert(0, model_version)
    state["deployment"]["last_action"] = "train_created"
    state["deployment"]["last_action_at"] = _iso_now()
    _sync_deployment(state)
    _append_log(state, "info", f"Training run {run_id} created from dataset snapshot {snapshot['id']}.")
    _save_state(state)

    return {
        "message": "Training run configured. Dataset snapshot and staging model version were created.",
        "training_run": run,
        "dataset_snapshot": snapshot,
        "model_version": model_version,
    }


@app.post("/mlops/deployments/canary")
async def start_canary(payload: CanaryRequest):
    state = _load_state()
    target = _get_model(state, payload.model_version_id)

    if target["status"] == "production":
        raise HTTPException(status_code=400, detail="Production model cannot be re-used as a canary candidate.")

    for model in state["model_versions"]:
        if model["id"] != target["id"] and model.get("status") == "canary":
            model["status"] = "staging"
            model["updated_at"] = _iso_now()

    target["status"] = "canary"
    target["target_line"] = payload.line
    target["updated_at"] = _iso_now()
    state["deployment"]["canary_line"] = payload.line
    state["deployment"]["last_action"] = "canary_started"
    state["deployment"]["last_action_at"] = _iso_now()
    _sync_deployment(state)
    _append_log(state, "info", f"Canary started: {target['id']} on {payload.line}.")
    _save_state(state)

    return {
        "message": f"Canary started on {payload.line}.",
        "model_version": target,
        "deployment": state["deployment"],
    }


@app.post("/mlops/models/promote")
async def promote_model(payload: PromoteModelRequest):
    state = _load_state()
    target = _get_model(state, payload.model_version_id)

    if payload.target_status == "production":
        current_production = next(
            (model for model in state["model_versions"] if model.get("status") == "production"),
            None,
        )
        if current_production and current_production["id"] != target["id"]:
            current_production["status"] = "archived"
            current_production["updated_at"] = _iso_now()
            state["deployment"]["previous_production_model_id"] = current_production["id"]

        for model in state["model_versions"]:
            if model["id"] != target["id"] and model.get("status") == "canary":
                model["status"] = "staging"
                model["target_line"] = None
                model["updated_at"] = _iso_now()

        target["status"] = "production"
        target["target_line"] = None
    else:
        target["status"] = "staging"

    target["updated_at"] = _iso_now()
    state["deployment"]["last_action"] = f"promoted_{payload.target_status}"
    state["deployment"]["last_action_at"] = _iso_now()
    _sync_deployment(state)
    _append_log(state, "info", f"Model {target['id']} promoted to {payload.target_status}.")
    _save_state(state)
    return {
        "message": "Model version updated.",
        "model_version": target,
        "deployment": state["deployment"],
    }


@app.post("/mlops/deployments/rollback")
async def rollback_deployment(payload: RollbackRequest):
    state = _load_state()
    deployment = state["deployment"]

    target_id = payload.model_version_id or deployment.get("previous_production_model_id")
    if not target_id:
        archived = next((model for model in state["model_versions"] if model.get("status") == "archived"), None)
        if archived is not None:
            target_id = archived["id"]

    if not target_id:
        raise HTTPException(status_code=400, detail="No rollback candidate is available.")

    target = _get_model(state, target_id)
    current_production = next(
        (model for model in state["model_versions"] if model.get("status") == "production"),
        None,
    )

    if current_production and current_production["id"] != target["id"]:
        current_production["status"] = "archived"
        current_production["updated_at"] = _iso_now()

    for model in state["model_versions"]:
        if model.get("status") == "canary":
            model["status"] = "staging"
            model["target_line"] = None
            model["updated_at"] = _iso_now()

    target["status"] = "production"
    target["target_line"] = None
    target["updated_at"] = _iso_now()
    deployment["last_action"] = "rolled_back"
    deployment["last_action_at"] = _iso_now()
    _sync_deployment(state)
    _append_log(state, "warning", f"Rollback completed. Production restored to {target['id']}.")
    _save_state(state)

    return {
        "message": f"Rollback completed to {target['id']}.",
        "model_version": target,
        "deployment": state["deployment"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serve:app",
        host=_env_str("SERVE_HOST", "0.0.0.0"),
        port=int(_env_float("SERVE_PORT", 8000)),
        reload=False,
        log_level="info",
    )
