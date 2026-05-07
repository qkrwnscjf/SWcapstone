"""
Gate Model Training Pipeline (v1/v2/v3 / round-tagged)
=======================================================
Trains a binary classifier (EfficientNet-B0 or MobileNetV3-Large) on the gate
training split identified by `--tag`, calibrates probabilities, runs a
threshold sweep, and saves model + calibrator + reproducibility config.

Recovered from git commit e361cee:src/train_gate.py and adapted:
  - MLflow optional (no-op if not installed)
  - --round int → --tag str (accepts v1/v2/v3 or round1/round2/round3)
  - splits/{tag}_train_gate_mix.csv etc. (no `split=...` filter; per-split files)
  - Saves models/{run_tag}_config.json alongside .pt for reproducibility
  - Removed mlruns/ writes (keeps only matplotlib reports)

Usage
-----
    python scripts/train_gate.py --tag v1 --gate effnetb0
    python scripts/train_gate.py --tag v2 --gate effnetb0 --device mps --batch_size 8 --epochs 3
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import pickle
import random
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import yaml
from PIL import Image
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

# -- MLflow optional --------------------------------------------------------
try:
    import mlflow as _mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

class _MlflowStub:
    class _CtxNoop:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def start_run(self, *args, **kwargs):
        return self._CtxNoop()
    def set_tracking_uri(self, *a, **k): pass
    def set_experiment(self, *a, **k): pass
    def log_param(self, *a, **k): pass
    def log_metric(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass

mlflow = _mlflow if _MLFLOW_AVAILABLE else _MlflowStub()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "assets"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GateDataset(Dataset):
    """Binary classification dataset (normal=0, anomaly=1).

    Loads a CSV that already represents a single split (per-split file pattern:
    `{tag}_train_gate_mix.csv`, `{tag}_val_gate.csv`, `{tag}_test_gate.csv`).
    No `split` filter applied — the CSV's contents define the split.
    """

    def __init__(self, csv_path: str | Path, transform: Optional[T.Compose] = None) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {"normal": 0, "anomaly": 1}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = self.label_map.get(row["label"], 0)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].map(self.label_map).values

    def class_counts(self) -> Tuple[int, int]:
        labels = self.labels
        return int((labels == 0).sum()), int((labels == 1).sum())


def get_train_transforms(input_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(input_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_gate_model(gate_name: str, num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    if gate_name == "effnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif gate_name == "mnv3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
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


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels_f = labels.float().to(device)
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels_f)
        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        correct += (preds == labels.to(device).long()).sum().item()
        total += imgs.size(0)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
    return (running_loss / max(total, 1), correct / max(total, 1),
            np.array(all_probs), np.array(all_labels))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class PlattCalibrator:
    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    def fit(self, probs, labels):
        self.lr.fit(probs.reshape(-1, 1), labels)
    def predict_proba(self, probs):
        return self.lr.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Same shape as backend/src/serve.py expects: .ir is the IsotonicRegression
    and .predict(arr) returns a 1-d numpy array. (Used by benchmark loader.)"""

    def __init__(self) -> None:
        self.ir = IsotonicRegression(out_of_bounds="clip")
        self.method = "isotonic"

    def fit(self, probs, labels):
        self.ir.fit(probs, labels)

    def predict_proba(self, probs):
        return self.ir.predict(probs)

    # Compatibility with serve.IsotonicCalibrator and benchmark_pipeline stub
    def predict(self, x):
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return np.asarray(self.ir.predict(arr)).reshape(-1)


def build_calibrator(method: str):
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method == "none":
        return None
    raise ValueError(f"Unknown calibration method: {method}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, probs):
    if len(y_true) == 0:
        return {}, 0.5
    auroc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds_pr[min(best_idx, len(thresholds_pr) - 1)])
    y_pred = (probs >= optimal_threshold).astype(int)
    return ({
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "optimal_threshold": optimal_threshold,
    }, optimal_threshold)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_roc(y_true, probs, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
    ax.legend(loc="lower right"); fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_pr(y_true, probs, save_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR")
    ax.legend(loc="lower left"); fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_cm(y_true, y_pred, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix"); fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "anomaly"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "anomaly"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="train"); ax1.plot(epochs, val_losses, label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, train_accs, label="train"); ax2.plot(epochs, val_accs, label="val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy"); ax2.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def threshold_sweep(y_true, probs, t_low_range, t_high_range,
                    optimize_for: str = "recall",
                    max_heatmap_call_rate: float = 0.50) -> Tuple[Dict, pd.DataFrame]:
    results = []
    for t_low, t_high in itertools.product(t_low_range, t_high_range):
        if t_low >= t_high:
            continue
        pred_normal = probs <= t_low
        pred_anomaly = probs >= t_high
        uncertain = ~pred_normal & ~pred_anomaly
        heatmap_calls = int(pred_anomaly.sum()) + int(uncertain.sum())
        heatmap_call_rate = heatmap_calls / max(len(probs), 1)
        y_pred_cascade = np.zeros_like(y_true)
        y_pred_cascade[pred_anomaly] = 1
        y_pred_cascade[uncertain] = 1
        results.append({
            "T_low": t_low, "T_high": t_high,
            "recall": recall_score(y_true, y_pred_cascade, zero_division=0),
            "precision": precision_score(y_true, y_pred_cascade, zero_division=0),
            "f1": f1_score(y_true, y_pred_cascade, zero_division=0),
            "heatmap_call_rate": heatmap_call_rate,
        })
    df = pd.DataFrame(results)
    feasible = df[df["heatmap_call_rate"] <= max_heatmap_call_rate]
    if len(feasible) == 0:
        feasible = df
    key = "recall" if optimize_for == "recall" else "f1"
    best_row = feasible.loc[feasible[key].idxmax()]
    return best_row.to_dict(), df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate model training pipeline")
    p.add_argument("--tag", type=str, required=True,
                   help="Version tag (e.g. v1, v2, v3, round1, round2, round3). "
                        "CSVs read from splits/{tag}_train_gate_mix.csv etc.")
    p.add_argument("--gate", type=str, required=True,
                   choices=["effnetb0", "mnv3_large"], help="Gate model variant")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--optimizer", type=str, choices=["AdamW", "Adam", "SGD"])
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--scheduler", type=str, choices=["cosine", "step", "none"])
    p.add_argument("--run_tag", type=str, help="Override run tag (default: {tag}_{gate})")
    p.add_argument("--base_model_path", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    gate_cfg = cfg["gate"]["models"][args.gate]
    input_size = gate_cfg["input_size"]
    if isinstance(input_size, list):
        input_size = input_size[0]

    batch_size = args.batch_size if args.batch_size is not None else gate_cfg["batch_size"]
    epochs = args.epochs if args.epochs is not None else gate_cfg["epochs"]
    lr = args.lr if args.lr is not None else gate_cfg["lr"]
    optimizer_name = args.optimizer if args.optimizer is not None else gate_cfg.get("optimizer", "AdamW")
    weight_decay = args.weight_decay if args.weight_decay is not None else gate_cfg["weight_decay"]
    scheduler_type = args.scheduler if args.scheduler is not None else gate_cfg.get("scheduler", "cosine")
    pos_weight_auto = gate_cfg.get("pos_weight_auto", True)
    early_stopping_patience = gate_cfg.get("early_stopping_patience", 7)

    cal_cfg = cfg.get("calibration", {})
    cal_methods = cal_cfg.get("methods", ["platt", "isotonic"])

    sweep_cfg = cfg.get("threshold_sweep", {})
    t_low_range = sweep_cfg.get("T_low_range", [0.1, 0.2, 0.3, 0.4, 0.5])
    t_high_range = sweep_cfg.get("T_high_range", [0.5, 0.6, 0.7, 0.8, 0.9])
    optimize_for = sweep_cfg.get("optimize_for", "recall")
    max_heatmap_call_rate = sweep_cfg.get("max_heatmap_call_rate", 0.50)

    device = args.device or cfg.get("training", {}).get("device", "cpu")
    num_workers = cfg.get("training", {}).get("num_workers", 0)

    tag = args.tag
    run_tag = args.run_tag or f"{tag}_{args.gate}"

    print("=" * 60)
    print("Gate Model Training Pipeline")
    print(f"  tag         : {tag}")
    print(f"  model       : {args.gate}")
    print(f"  device      : {device}")
    print(f"  batch_size  : {batch_size}")
    print(f"  epochs      : {epochs}")
    print(f"  lr          : {lr}")
    print(f"  optimizer   : {optimizer_name}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  scheduler   : {scheduler_type}")
    print(f"  run_tag     : {run_tag}")
    print(f"  seed        : {seed}")
    print(f"  mlflow      : {'enabled' if _MLFLOW_AVAILABLE else 'stub (not installed)'}")
    print("=" * 60)

    splits_dir = Path(cfg.get("data", {}).get("splits_dir", str(SPLITS_DIR)))
    if not splits_dir.is_absolute():
        splits_dir = PROJECT_ROOT / splits_dir
    if not splits_dir.exists() or not (splits_dir / f"{tag}_train_gate_mix.csv").exists():
        local = PROJECT_ROOT / "splits"
        if (local / f"{tag}_train_gate_mix.csv").exists():
            print(f"  splits_dir from config ({splits_dir}) missing; falling back to {local}")
            splits_dir = local
    train_csv = splits_dir / f"{tag}_train_gate_mix.csv"
    val_csv = splits_dir / f"{tag}_val_gate.csv"
    test_csv = splits_dir / f"{tag}_test_gate.csv"

    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    train_transform = get_train_transforms(input_size)
    eval_transform = get_eval_transforms(input_size)

    train_ds = GateDataset(train_csv, transform=train_transform)
    val_ds = GateDataset(val_csv, transform=eval_transform)
    test_ds = GateDataset(test_csv, transform=eval_transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_neg, n_pos = train_ds.class_counts()
    print(f"\nTrain set: {len(train_ds)} images (normal={n_neg}, anomaly={n_pos})")
    print(f"Val   set: {len(val_ds)} images")
    print(f"Test  set: {len(test_ds)} images")

    if pos_weight_auto and n_pos > 0:
        pw = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        print(f"Auto pos_weight: {pw.item():.4f}")
    else:
        pw = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    model = build_gate_model(args.gate, num_classes=1, pretrained=not bool(args.base_model_path))
    if args.base_model_path:
        base_path = Path(args.base_model_path)
        if base_path.exists():
            checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded base checkpoint: {base_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print(f"[WARN] Base checkpoint not found, starting from pretrained: {base_path}")
    model = model.to(device)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    if _MLFLOW_AVAILABLE:
        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
        if not tracking_uri.startswith("file://") and not tracking_uri.startswith("http"):
            tracking_uri = f"file://{PROJECT_ROOT / tracking_uri}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(f"surface_defect_detection_gate_{tag}")

    with mlflow.start_run(run_name=run_tag):
        mlflow.log_param("tag", tag); mlflow.log_param("gate_model", args.gate)
        mlflow.log_param("device", device); mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs); mlflow.log_param("lr", lr)
        mlflow.log_param("optimizer", optimizer_name); mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("scheduler", scheduler_type); mlflow.log_param("seed", seed)
        mlflow.log_param("train_normal", n_neg); mlflow.log_param("train_anomaly", n_pos)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        print("\n--- Training ---")
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
            v_loss, v_acc, _, _ = evaluate(model, val_dl, criterion, device)
            train_losses.append(t_loss); val_losses.append(v_loss)
            train_accs.append(t_acc); val_accs.append(v_acc)
            if scheduler is not None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("train/loss", t_loss, step=epoch)
            mlflow.log_metric("val/loss", v_loss, step=epoch)
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={t_loss:.4f} train_acc={t_acc:.4f}  "
                  f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}  lr={current_lr:.6f}")
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                             REPORTS_DIR / f"{run_tag}_training_curves.png")

        print("\n--- Calibration on val set ---")
        _, _, val_probs, val_labels = evaluate(model, val_dl, criterion, device)
        best_calibrator = None
        best_cal_name = "none"
        best_cal_auroc = 0.0
        if len(val_labels) > 0 and len(np.unique(val_labels)) >= 2:
            best_cal_auroc = roc_auc_score(val_labels, val_probs)
            print(f"  Uncalibrated val AUROC: {best_cal_auroc:.4f}")
            for cal_method in cal_methods:
                try:
                    calibrator = build_calibrator(cal_method)
                    if calibrator is None:
                        continue
                    calibrator.fit(val_probs, val_labels)
                    cal_probs = calibrator.predict_proba(val_probs)
                    cal_auroc = roc_auc_score(val_labels, cal_probs)
                    print(f"  {cal_method} val AUROC: {cal_auroc:.4f}")
                    if cal_auroc > best_cal_auroc:
                        best_cal_auroc = cal_auroc
                        best_calibrator = calibrator
                        best_cal_name = cal_method
                except Exception as e:
                    print(f"  [WARN] Calibration '{cal_method}' failed: {e}")
        else:
            print("  [WARN] Val set empty or single-class; skipping calibration.")
        print(f"  Selected calibration: {best_cal_name}")

        print("\n--- Evaluation on test set ---")
        _, _, test_probs_raw, test_labels = evaluate(model, test_dl, criterion, device)
        test_probs = best_calibrator.predict_proba(test_probs_raw) if best_calibrator else test_probs_raw

        test_metrics = {}
        opt_thr = 0.5
        if len(test_labels) > 0 and len(np.unique(test_labels)) >= 2:
            test_metrics, opt_thr = compute_metrics(test_labels, test_probs)
            for k, v in test_metrics.items():
                print(f"  test/{k}: {v:.4f}")
            y_pred_test = (test_probs >= opt_thr).astype(int)
            plot_roc(test_labels, test_probs, REPORTS_DIR / f"{run_tag}_test_roc.png")
            plot_pr(test_labels, test_probs, REPORTS_DIR / f"{run_tag}_test_pr.png")
            plot_cm(test_labels, y_pred_test, REPORTS_DIR / f"{run_tag}_test_cm.png")
        else:
            print("  [WARN] Only one class in test set; skipping metrics.")

        print("\n--- Threshold Sweep ---")
        if len(test_labels) > 0 and len(np.unique(test_labels)) >= 2:
            best_thresholds, sweep_df = threshold_sweep(
                y_true=test_labels, probs=test_probs,
                t_low_range=t_low_range, t_high_range=t_high_range,
                optimize_for=optimize_for, max_heatmap_call_rate=max_heatmap_call_rate,
            )
            print(f"  Recommended T_low : {best_thresholds['T_low']:.3f}")
            print(f"  Recommended T_high: {best_thresholds['T_high']:.3f}")
            print(f"  Recall            : {best_thresholds['recall']:.4f}")
            print(f"  Precision         : {best_thresholds['precision']:.4f}")
            print(f"  F1                : {best_thresholds['f1']:.4f}")
            print(f"  Heatmap call rate : {best_thresholds['heatmap_call_rate']:.4f}")
        else:
            print("  [WARN] Skipping threshold sweep (single class in test).")
            best_thresholds = {"T_low": 0.3, "T_high": 0.7}

        # ---- Save model + calibrator ----
        model_path = MODELS_DIR / f"{run_tag}_gate.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "gate_name": args.gate,
            "backbone": gate_cfg.get("backbone", args.gate),
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "T_low": best_thresholds["T_low"],
            "T_high": best_thresholds["T_high"],
        }, model_path)
        print(f"\nModel saved: {model_path}")

        cal_path = None
        if best_calibrator is not None:
            cal_path = MODELS_DIR / f"{run_tag}_calibrator.pkl"
            with open(cal_path, "wb") as f:
                pickle.dump({"method": best_cal_name, "calibrator": best_calibrator}, f)
            print(f"Calibrator saved: {cal_path}")

        # ---- Reproducibility config ----
        config_path = MODELS_DIR / f"{run_tag}_config.json"
        config_payload = {
            "tag": tag,
            "run_tag": run_tag,
            "model_kind": "gate",
            "gate": args.gate,
            "device": device,
            "seed": seed,
            "batch_size": batch_size,
            "epochs": epochs,
            "actual_epochs": len(train_losses),
            "lr": lr,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
            "scheduler": scheduler_type,
            "early_stopping_patience": early_stopping_patience,
            "input_size": input_size,
            "splits_dir": str(splits_dir),
            "train_csv_path": str(train_csv),
            "val_csv_path": str(val_csv),
            "test_csv_path": str(test_csv),
            "train_normal": n_neg,
            "train_anomaly": n_pos,
            "pos_weight": float(pw.item()) if pw is not None else None,
            "calibration": best_cal_name,
            "calibration_path": str(cal_path) if cal_path else None,
            "best_val_loss": float(best_val_loss),
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "T_low": float(best_thresholds["T_low"]),
            "T_high": float(best_thresholds["T_high"]),
            "git_sha": get_git_sha(),
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "mlflow_used": _MLFLOW_AVAILABLE,
        }
        config_path.write_text(json.dumps(config_payload, indent=2))
        print(f"Config saved: {config_path}")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if test_metrics:
            print(f"  Test AUROC    : {test_metrics['auroc']:.4f}")
            print(f"  Test AUPRC    : {test_metrics['auprc']:.4f}")
            print(f"  Test F1       : {test_metrics['f1']:.4f}")
            print(f"  Test Recall   : {test_metrics['recall']:.4f}")
            print(f"  Test Precision: {test_metrics['precision']:.4f}")
            print(f"  Calibration   : {best_cal_name}")
        print(f"  Recommended T_low : {best_thresholds['T_low']:.3f}")
        print(f"  Recommended T_high: {best_thresholds['T_high']:.3f}")
        print(f"\n  Model saved : {model_path}")
        print("=" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()
