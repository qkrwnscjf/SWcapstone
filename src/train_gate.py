"""
Gate Model Training Pipeline
=============================
Trains a binary classifier (EfficientNet-B0 or MobileNetV3-Large) on
the gate split (normal + anomaly), calibrates probabilities, runs a
threshold sweep, and logs everything to MLflow.

Usage
-----
    python src/train_gate.py --round 1 --gate effnetb0
    python src/train_gate.py --round 2 --gate mnv3_large --device cuda
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import pickle
import random
import sys
import warnings
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
from sklearn.calibration import CalibratedClassifierCV
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

try:
    import mlflow
except ImportError:
    raise ImportError("mlflow is required. Install via: pip install mlflow")

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


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GateDataset(Dataset):
    """Binary classification dataset (normal=0, anomaly=1)."""

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[T.Compose] = None,
    ) -> None:
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
        n_neg = int((labels == 0).sum())
        n_pos = int((labels == 1).sum())
        return n_neg, n_pos


def get_train_transforms(input_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(input_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_gate_model(gate_name: str, num_classes: int = 1) -> nn.Module:
    """Build a pretrained backbone with a custom classifier head.

    Output is single logit for BCEWithLogitsLoss.
    """
    if gate_name == "effnetb0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Train one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate on a dataset. Returns (avg_loss, acc, probs, labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs: List[float] = []
    all_labels: List[int] = []

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

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.array(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """Platt scaling: fit logistic regression on logits/probs."""

    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        self.lr.fit(probs.reshape(-1, 1), labels)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibration."""

    def __init__(self) -> None:
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        self.ir.fit(probs, labels)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return self.ir.predict(probs)


def build_calibrator(method: str):
    """Factory for calibrator objects."""
    if method == "platt":
        return PlattCalibrator()
    elif method == "isotonic":
        return IsotonicCalibrator()
    elif method == "none":
        return None
    else:
        raise ValueError(f"Unknown calibration method: {method}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray, probs: np.ndarray
) -> Tuple[dict, float]:
    """Compute standard binary classification metrics.

    Returns (metrics_dict, optimal_threshold).
    """
    auroc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)

    # Optimal threshold via F1
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds_pr[min(best_idx, len(thresholds_pr) - 1)])

    y_pred = (probs >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    prec_val = precision_score(y_true, y_pred, zero_division=0)

    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "recall": recall_val,
        "precision": prec_val,
        "optimal_threshold": optimal_threshold,
    }
    return metrics, optimal_threshold


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_roc(y_true: np.ndarray, probs: np.ndarray, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr(y_true: np.ndarray, probs: np.ndarray, save_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    classes = ["normal", "anomaly"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="train")
    ax1.plot(epochs, val_losses, label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="train")
    ax2.plot(epochs, val_accs, label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def threshold_sweep(
    y_true: np.ndarray,
    probs: np.ndarray,
    t_low_range: List[float],
    t_high_range: List[float],
    optimize_for: str = "recall",
    max_heatmap_call_rate: float = 0.50,
) -> Tuple[Dict, Path]:
    """Sweep T_low / T_high and find optimal cascade thresholds.

    Cascade logic:
      p <= T_low  -> predict normal (no heatmap call)
      p >= T_high -> predict anomaly / call heatmap
      T_low < p < T_high -> uncertain, call heatmap

    Returns (best_result_dict, plot_path).
    """
    results: List[Dict] = []

    for t_low, t_high in itertools.product(t_low_range, t_high_range):
        if t_low >= t_high:
            continue

        # Cascade predictions
        pred_normal = probs <= t_low
        pred_anomaly = probs >= t_high
        uncertain = ~pred_normal & ~pred_anomaly

        # Heatmap call rate = fraction that is uncertain or high
        heatmap_calls = int(pred_anomaly.sum()) + int(uncertain.sum())
        heatmap_call_rate = heatmap_calls / max(len(probs), 1)

        # For recall calculation, we assume heatmap catches all true anomalies
        # that are sent to it. Missed anomalies are those predicted normal.
        y_pred_cascade = np.zeros_like(y_true)
        y_pred_cascade[pred_anomaly] = 1
        y_pred_cascade[uncertain] = 1  # assume heatmap catches them

        rec = recall_score(y_true, y_pred_cascade, zero_division=0)
        prec = precision_score(y_true, y_pred_cascade, zero_division=0)
        f1 = f1_score(y_true, y_pred_cascade, zero_division=0)

        results.append(
            {
                "T_low": t_low,
                "T_high": t_high,
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "heatmap_call_rate": heatmap_call_rate,
            }
        )

    df = pd.DataFrame(results)

    # Filter by max call rate
    feasible = df[df["heatmap_call_rate"] <= max_heatmap_call_rate]
    if len(feasible) == 0:
        feasible = df  # fall back to all if no feasible solutions

    if optimize_for == "recall":
        best_row = feasible.loc[feasible["recall"].idxmax()]
    elif optimize_for == "f1":
        best_row = feasible.loc[feasible["f1"].idxmax()]
    else:
        best_row = feasible.loc[feasible["f1"].idxmax()]

    best_result = best_row.to_dict()
    return best_result, df


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    best: Dict,
    save_path: Path,
) -> None:
    """Plot threshold sweep results as a heatmap-style grid."""
    t_lows = sorted(sweep_df["T_low"].unique())
    t_highs = sorted(sweep_df["T_high"].unique())

    # Build a pivot for recall
    pivot_recall = sweep_df.pivot_table(
        index="T_low", columns="T_high", values="recall", aggfunc="first"
    )
    pivot_call = sweep_df.pivot_table(
        index="T_low", columns="T_high", values="heatmap_call_rate", aggfunc="first"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Recall heatmap
    im1 = ax1.imshow(
        pivot_recall.values,
        aspect="auto",
        cmap="YlOrRd",
        origin="lower",
    )
    ax1.set_xticks(range(len(pivot_recall.columns)))
    ax1.set_xticklabels([f"{v:.1f}" for v in pivot_recall.columns], fontsize=7)
    ax1.set_yticks(range(len(pivot_recall.index)))
    ax1.set_yticklabels([f"{v:.1f}" for v in pivot_recall.index], fontsize=7)
    ax1.set_xlabel("T_high")
    ax1.set_ylabel("T_low")
    ax1.set_title("Anomaly Recall")
    fig.colorbar(im1, ax=ax1)

    # Call-rate heatmap
    im2 = ax2.imshow(
        pivot_call.values,
        aspect="auto",
        cmap="YlGnBu",
        origin="lower",
    )
    ax2.set_xticks(range(len(pivot_call.columns)))
    ax2.set_xticklabels([f"{v:.1f}" for v in pivot_call.columns], fontsize=7)
    ax2.set_yticks(range(len(pivot_call.index)))
    ax2.set_yticklabels([f"{v:.1f}" for v in pivot_call.index], fontsize=7)
    ax2.set_xlabel("T_high")
    ax2.set_ylabel("T_low")
    ax2.set_title("Heatmap Call Rate")
    fig.colorbar(im2, ax=ax2)

    fig.suptitle(
        f"Threshold Sweep  |  Best: T_low={best['T_low']:.2f}, "
        f"T_high={best['T_high']:.2f}, Recall={best['recall']:.3f}, "
        f"Call Rate={best['heatmap_call_rate']:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate model training pipeline"
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Training round (1, 2, or 3)",
    )
    parser.add_argument(
        "--gate",
        type=str,
        required=True,
        choices=["effnetb0", "mnv3_large"],
        help="Gate model variant",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: mps | cpu | cuda (overrides config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    # ---- Load config ----
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Resolve gate model config
    gate_cfg = cfg["gate"]["models"][args.gate]
    input_size = gate_cfg["input_size"]
    if isinstance(input_size, list):
        input_size = input_size[0]
    batch_size = gate_cfg["batch_size"]
    epochs = gate_cfg["epochs"]
    lr = gate_cfg["lr"]
    weight_decay = gate_cfg["weight_decay"]
    scheduler_type = gate_cfg.get("scheduler", "cosine")
    pos_weight_auto = gate_cfg.get("pos_weight_auto", True)
    early_stopping_patience = gate_cfg.get("early_stopping_patience", 7)

    # Calibration config
    cal_cfg = cfg.get("calibration", {})
    cal_methods = cal_cfg.get("methods", ["platt", "isotonic"])

    # Threshold sweep config
    sweep_cfg = cfg.get("threshold_sweep", {})
    t_low_range = sweep_cfg.get("T_low_range", [0.1, 0.2, 0.3, 0.4, 0.5])
    t_high_range = sweep_cfg.get("T_high_range", [0.5, 0.6, 0.7, 0.8, 0.9])
    optimize_for = sweep_cfg.get("optimize_for", "recall")
    max_heatmap_call_rate = sweep_cfg.get("max_heatmap_call_rate", 0.50)

    # Device
    device = args.device
    if device is None:
        device = cfg.get("training", {}).get("device", "cpu")
    num_workers = cfg.get("training", {}).get("num_workers", 0)

    rnd = args.round
    run_tag = f"round{rnd}_{args.gate}"

    print("=" * 60)
    print(f"Gate Model Training Pipeline")
    print(f"  round       : {rnd}")
    print(f"  model       : {args.gate}")
    print(f"  device      : {device}")
    print(f"  batch_size  : {batch_size}")
    print(f"  epochs      : {epochs}")
    print(f"  lr          : {lr}")
    print(f"  scheduler   : {scheduler_type}")
    print(f"  seed        : {seed}")
    print("=" * 60)

    # ---- CSV paths ----
    #splits_dir = Path(cfg.get("data", {}).get("splits_dir", str(SPLITS_DIR)))
    # ---- CSV paths 수정 ----
    # 현송님의 개인 경로가 들어있을 수 있는 cfg.get 부분을 주석 처리
    # 강제로 프로젝트 루트 기준 splits 폴더 설정
    splits_dir = SPLITS_DIR # 강제 지정
    train_csv = splits_dir / f"round{rnd}_train_gate_mix.csv"
    val_csv = splits_dir / f"round{rnd}_val_gate.csv"
    test_csv = splits_dir / f"round{rnd}_test_gate.csv"

    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    # ---- Transforms ----
    train_transform = get_train_transforms(input_size)
    eval_transform = get_eval_transforms(input_size)

    # ---- Datasets & Loaders ----
    train_ds = GateDataset(train_csv, transform=train_transform)
    val_ds = GateDataset(val_csv, transform=eval_transform)
    test_ds = GateDataset(test_csv, transform=eval_transform)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    n_neg, n_pos = train_ds.class_counts()
    print(f"\nTrain set: {len(train_ds)} images (normal={n_neg}, anomaly={n_pos})")
    print(f"Val   set: {len(val_ds)} images")
    print(f"Test  set: {len(test_ds)} images")

    # ---- Loss with optional pos_weight ----
    if pos_weight_auto and n_pos > 0:
        pw = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        print(f"Auto pos_weight: {pw.item():.4f}")
    else:
        pw = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ---- Build model ----
    model = build_gate_model(args.gate, num_classes=1)
    model = model.to(device)

    # ---- Optimizer & Scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    # ---- MLflow ----
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    if not tracking_uri.startswith("file://") and not tracking_uri.startswith("http"):
        abs_path = (PROJECT_ROOT / tracking_uri).resolve()
        tracking_uri = abs_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = mlflow_cfg.get("experiment_name", "surface_defect_detection")
    experiment_name = f"{experiment_name}_gate_round{rnd}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_tag):
        # Log params
        mlflow.log_param("round", rnd)
        mlflow.log_param("gate_model", args.gate)
        mlflow.log_param("backbone", gate_cfg["backbone"])
        mlflow.log_param("device", device)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("scheduler", scheduler_type)
        mlflow.log_param("pos_weight_auto", pos_weight_auto)
        mlflow.log_param("pos_weight", pw.item() if pw is not None else "none")
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train_normal", n_neg)
        mlflow.log_param("train_anomaly", n_pos)
        mlflow.log_param("early_stopping_patience", early_stopping_patience)

        # ---- Training loop ----
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        print("\n--- Training ---")
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_one_epoch(
                model, train_dl, criterion, optimizer, device
            )
            v_loss, v_acc, _, _ = evaluate(model, val_dl, criterion, device)

            train_losses.append(t_loss)
            val_losses.append(v_loss)
            train_accs.append(t_acc)
            val_accs.append(v_acc)

            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            # MLflow per-epoch
            mlflow.log_metric("train/loss", t_loss, step=epoch)
            mlflow.log_metric("train/acc", t_acc, step=epoch)
            mlflow.log_metric("val/loss", v_loss, step=epoch)
            mlflow.log_metric("val/acc", v_acc, step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)

            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
                f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}  "
                f"lr={current_lr:.6f}"
            )

            # Early stopping
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(
                        f"  Early stopping at epoch {epoch} "
                        f"(patience={early_stopping_patience})"
                    )
                    break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Training curves plot
        curves_path = REPORTS_DIR / f"{run_tag}_training_curves.png"
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
        mlflow.log_artifact(str(curves_path), artifact_path="plots")

        # ---- Evaluate on val (for calibration) ----
        print("\n--- Calibration on val set ---")
        _, _, val_probs, val_labels = evaluate(model, val_dl, criterion, device)

        # Try each calibration method, pick the best
        best_calibrator = None
        best_cal_name = "none"
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

        print(f"  Selected calibration: {best_cal_name}")
        mlflow.log_param("calibration_method", best_cal_name)
        mlflow.log_metric("val/calibrated_auroc", best_cal_auroc)

        # ---- Evaluate on test set ----
        print("\n--- Evaluation on test set ---")
        _, _, test_probs_raw, test_labels = evaluate(
            model, test_dl, criterion, device
        )

        # Apply calibration if any
        if best_calibrator is not None:
            test_probs = best_calibrator.predict_proba(test_probs_raw)
        else:
            test_probs = test_probs_raw

        if len(np.unique(test_labels)) >= 2:
            test_metrics, opt_thr = compute_metrics(test_labels, test_probs)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test/{k}", v)
                print(f"  test/{k}: {v:.4f}")

            y_pred_test = (test_probs >= opt_thr).astype(int)

            # Plots
            roc_path = REPORTS_DIR / f"{run_tag}_test_roc.png"
            pr_path = REPORTS_DIR / f"{run_tag}_test_pr.png"
            cm_path = REPORTS_DIR / f"{run_tag}_test_cm.png"

            plot_roc(test_labels, test_probs, roc_path)
            plot_pr(test_labels, test_probs, pr_path)
            plot_confusion_matrix(test_labels, y_pred_test, cm_path)

            for p in [roc_path, pr_path, cm_path]:
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="plots/test")
        else:
            print("  [WARN] Only one class in test set; skipping metrics.")

        # ---- Threshold Sweep on test set ----
        print("\n--- Threshold Sweep ---")
        if len(np.unique(test_labels)) >= 2:
            best_thresholds, sweep_df = threshold_sweep(
                y_true=test_labels,
                probs=test_probs,
                t_low_range=t_low_range,
                t_high_range=t_high_range,
                optimize_for=optimize_for,
                max_heatmap_call_rate=max_heatmap_call_rate,
            )

            print(f"  Recommended T_low  : {best_thresholds['T_low']:.3f}")
            print(f"  Recommended T_high : {best_thresholds['T_high']:.3f}")
            print(f"  Recall             : {best_thresholds['recall']:.4f}")
            print(f"  Precision          : {best_thresholds['precision']:.4f}")
            print(f"  F1                 : {best_thresholds['f1']:.4f}")
            print(
                f"  Heatmap call rate  : {best_thresholds['heatmap_call_rate']:.4f}"
            )

            mlflow.log_metric("cascade/T_low", best_thresholds["T_low"])
            mlflow.log_metric("cascade/T_high", best_thresholds["T_high"])
            mlflow.log_metric("cascade/recall", best_thresholds["recall"])
            mlflow.log_metric("cascade/precision", best_thresholds["precision"])
            mlflow.log_metric("cascade/f1", best_thresholds["f1"])
            mlflow.log_metric(
                "cascade/heatmap_call_rate",
                best_thresholds["heatmap_call_rate"],
            )

            sweep_plot_path = REPORTS_DIR / f"{run_tag}_threshold_sweep.png"
            plot_threshold_sweep(sweep_df, best_thresholds, sweep_plot_path)
            mlflow.log_artifact(
                str(sweep_plot_path), artifact_path="plots/threshold_sweep"
            )

            # Save threshold recommendation as JSON artifact
            import json

            thr_json_path = REPORTS_DIR / f"{run_tag}_thresholds.json"
            thr_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(thr_json_path, "w") as f:
                json.dump(
                    {
                        "T_low": best_thresholds["T_low"],
                        "T_high": best_thresholds["T_high"],
                        "recall": best_thresholds["recall"],
                        "precision": best_thresholds["precision"],
                        "f1": best_thresholds["f1"],
                        "heatmap_call_rate": best_thresholds["heatmap_call_rate"],
                        "optimize_for": optimize_for,
                    },
                    f,
                    indent=2,
                )
            mlflow.log_artifact(str(thr_json_path), artifact_path="thresholds")
        else:
            print("  [WARN] Skipping threshold sweep (single class in test).")
            best_thresholds = {"T_low": 0.3, "T_high": 0.7}

        # ---- Save model + calibrator ----
        model_path = MODELS_DIR / f"{run_tag}_gate.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "gate_name": args.gate,
                "backbone": gate_cfg["backbone"],
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "T_low": best_thresholds["T_low"],
                "T_high": best_thresholds["T_high"],
            },
            model_path,
        )
        print(f"\nModel saved: {model_path}")
        mlflow.log_artifact(str(model_path), artifact_path="models")

        # Save calibrator
        if best_calibrator is not None:
            cal_path = MODELS_DIR / f"{run_tag}_calibrator.pkl"
            with open(cal_path, "wb") as f:
                pickle.dump(
                    {"method": best_cal_name, "calibrator": best_calibrator}, f
                )
            print(f"Calibrator saved: {cal_path}")
            mlflow.log_artifact(str(cal_path), artifact_path="models")

        # ---- Summary ----
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if len(np.unique(test_labels)) >= 2:
            print(
                f"  Test AUROC    : {test_metrics['auroc']:.4f}\n"
                f"  Test AUPRC    : {test_metrics['auprc']:.4f}\n"
                f"  Test F1       : {test_metrics['f1']:.4f}\n"
                f"  Test Recall   : {test_metrics['recall']:.4f}\n"
                f"  Test Precision: {test_metrics['precision']:.4f}\n"
                f"  Calibration   : {best_cal_name}\n"
                f"  Recommended T_low : {best_thresholds['T_low']:.3f}\n"
                f"  Recommended T_high: {best_thresholds['T_high']:.3f}"
            )
        print(f"\n  Model saved : {model_path}")
        print(f"  Reports in  : {REPORTS_DIR}")
        print("=" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()
