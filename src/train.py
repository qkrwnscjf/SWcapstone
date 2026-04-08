"""
PatchCore Heatmap Training Pipeline
====================================
Trains a PatchCore anomaly-detection model using only the normal training
split, evaluates on val_mix / test_mix / anchor_test_mix, and logs
everything to MLflow.

Usage
-----
    python src/train.py --round 1 --heatmap patchcore_r18
    python src/train.py --round 2 --heatmap patchcore_wrn50 --device cuda
"""

from __future__ import annotations

import argparse
import os
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
import torchvision.models as models
import torchvision.transforms as T
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
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
class SurfaceDataset(Dataset):
    """Dataset that loads images listed in a split CSV.

    For PatchCore training, all images are normal (label=0).
    For evaluation, label mapping:  normal -> 0, anomaly -> 1.
    """

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = self.label_map.get(row["label"], 0)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, img_path


def get_transforms(input_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """Extracts intermediate features from a pretrained backbone."""

    def __init__(self, backbone_name: str = "resnet18", device: str = "cpu") -> None:
        super().__init__()
        self.device = device

        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.layer_names = ["layer2", "layer3"]
        elif backbone_name == "wide_resnet50_2":
            backbone = models.wide_resnet50_2(
                weights=models.Wide_ResNet50_2_Weights.DEFAULT
            )
            self.layer_names = ["layer2", "layer3"]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        self.backbone.eval()
        self._features: Dict[str, torch.Tensor] = {}
        self._register_hooks()
        self.to(device)

    def _register_hooks(self) -> None:
        for name in self.layer_names:
            layer = dict(self.backbone.named_children())[name]
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self._features[name] = output

        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._features.clear()
        x = x.to(self.device)
        self.backbone(x)
        feats = []
        target_size = self._features[self.layer_names[0]].shape[2:]
        for name in self.layer_names:
            f = self._features[name]
            if f.shape[2:] != target_size:
                f = torch.nn.functional.interpolate(
                    f, size=target_size, mode="bilinear", align_corners=False
                )
            feats.append(f)
        combined = torch.cat(feats, dim=1)  # (B, C, H, W)
        return combined


# ---------------------------------------------------------------------------
# PatchCore model
# ---------------------------------------------------------------------------
class PatchCoreModel:
    """Simple PatchCore implementation: coreset memory bank + k-NN scoring."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        coreset_sampling_ratio: float = 0.10,
        num_neighbors: int = 9,
        device: str = "cpu",
    ) -> None:
        self.backbone_name = backbone_name
        self.coreset_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.device = device
        self.extractor = FeatureExtractor(backbone_name, device)
        self.memory_bank: Optional[torch.Tensor] = None  # (N_patches, C)

    # ----- fit -----
    def fit(self, dataloader: DataLoader) -> None:
        """Build the patch-level memory bank from normal images."""
        print("[PatchCore] Extracting features from normal training set ...")
        all_patches: List[torch.Tensor] = []
        for batch_idx, (imgs, _labels, _paths) in enumerate(dataloader):
            feats = self.extractor(imgs)  # (B, C, H, W)
            B, C, H, W = feats.shape
            patches = feats.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            all_patches.append(patches.cpu())
            if (batch_idx + 1) % 20 == 0:
                print(f"  processed {batch_idx + 1} batches ...")

        all_patches_t = torch.cat(all_patches, dim=0)
        print(f"[PatchCore] Total patches: {all_patches_t.shape[0]}")

        # Coreset subsampling (random for efficiency)
        n_total = all_patches_t.shape[0]
        n_coreset = max(1, int(n_total * self.coreset_ratio))
        indices = torch.randperm(n_total)[:n_coreset]
        self.memory_bank = all_patches_t[indices]
        print(f"[PatchCore] Coreset size: {self.memory_bank.shape[0]}")

    # ----- predict -----
    @torch.no_grad()
    def predict(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray]]:
        """Return (image_scores, labels, paths, heatmaps)."""
        assert self.memory_bank is not None, "Call fit() first"
        memory = self.memory_bank.to(self.device)

        all_scores: List[float] = []
        all_labels: List[int] = []
        all_paths: List[str] = []
        all_heatmaps: List[np.ndarray] = []

        for imgs, labels, paths in dataloader:
            feats = self.extractor(imgs)  # (B, C, H, W)
            B, C, H, W = feats.shape

            for i in range(B):
                patch = feats[i].permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
                # k-NN distances
                dists = torch.cdist(patch.to(self.device), memory)  # (H*W, M)
                knn_dists, _ = dists.topk(
                    self.num_neighbors, largest=False, dim=1
                )  # (H*W, k)
                patch_scores = knn_dists.mean(dim=1)  # (H*W,)

                # Image-level score = max patch score
                image_score = patch_scores.max().item()
                all_scores.append(image_score)
                all_labels.append(labels[i].item())
                all_paths.append(paths[i])

                # Heatmap
                hmap = patch_scores.cpu().numpy().reshape(H, W)
                hmap = gaussian_filter(hmap, sigma=4)
                all_heatmaps.append(hmap)

        return (
            np.array(all_scores),
            np.array(all_labels),
            all_paths,
            all_heatmaps,
        )

    # ----- save / load -----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "backbone_name": self.backbone_name,
            "coreset_ratio": self.coreset_ratio,
            "num_neighbors": self.num_neighbors,
            "memory_bank": self.memory_bank,
        }
        torch.save(state, path)
        print(f"[PatchCore] Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PatchCoreModel":
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            backbone_name=state["backbone_name"],
            coreset_sampling_ratio=state["coreset_ratio"],
            num_neighbors=state["num_neighbors"],
            device=device,
        )
        model.memory_bank = state["memory_bank"]
        return model


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray
) -> Tuple[dict, float]:
    """Compute AUROC, AUPRC, and optimal-threshold metrics.

    Returns (metrics_dict, optimal_threshold).
    """
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)

    # Optimal threshold via F1 on precision-recall curve
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds_pr[min(best_idx, len(thresholds_pr) - 1)])

    y_pred = (scores >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)

    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "optimal_threshold": optimal_threshold,
    }
    return metrics, optimal_threshold


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_roc(y_true: np.ndarray, scores: np.ndarray, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
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


def plot_pr(y_true: np.ndarray, scores: np.ndarray, save_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
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


def plot_heatmap_overlays(
    paths: List[str],
    heatmaps: List[np.ndarray],
    labels: np.ndarray,
    scores: np.ndarray,
    save_path: Path,
    n_samples: int = 8,
) -> None:
    """Save a grid of sample images with heatmap overlays."""
    n = min(n_samples, len(paths))
    if n == 0:
        return
    # Pick indices: half highest-scoring, half lowest-scoring
    sorted_idx = np.argsort(scores)
    top_idx = sorted_idx[-n // 2 :].tolist()
    bot_idx = sorted_idx[: n - n // 2].tolist()
    indices = bot_idx + top_idx

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, idx in enumerate(indices):
        img = Image.open(paths[idx]).convert("RGB").resize((224, 224))
        img_np = np.array(img)
        hmap = heatmaps[idx]
        # Resize heatmap to image size
        from scipy.ndimage import zoom

        h_scale = 224 / hmap.shape[0]
        w_scale = 224 / hmap.shape[1]
        hmap_resized = zoom(hmap, (h_scale, w_scale))

        label_str = "anomaly" if labels[idx] == 1 else "normal"
        axes[0, col].imshow(img_np)
        axes[0, col].set_title(f"{label_str}\nscore={scores[idx]:.3f}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(img_np)
        axes[1, col].imshow(hmap_resized, cmap="jet", alpha=0.5)
        axes[1, col].set_title("heatmap overlay", fontsize=8)
        axes[1, col].axis("off")

    fig.suptitle("Sample Heatmap Overlays", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluate helper
# ---------------------------------------------------------------------------
def evaluate_split(
    model: PatchCoreModel,
    csv_path: Path,
    transform: T.Compose,
    split_name: str,
    batch_size: int,
    num_workers: int,
    run_tag: str,
) -> dict:
    """Evaluate a PatchCore model on one split. Returns metrics dict."""
    print(f"\n--- Evaluating on {split_name} ({csv_path.name}) ---")
    ds = SurfaceDataset(csv_path, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    scores, labels, paths, heatmaps = model.predict(dl)

    # Handle case where only one class present
    if len(np.unique(labels)) < 2:
        print(f"  [WARN] Only one class in {split_name}; skipping metric computation.")
        return {}

    metrics, threshold = compute_metrics(labels, scores)
    y_pred = (scores >= threshold).astype(int)

    # Print
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Plots
    prefix = f"{run_tag}_{split_name}"
    roc_path = REPORTS_DIR / f"{prefix}_roc.png"
    pr_path = REPORTS_DIR / f"{prefix}_pr.png"
    cm_path = REPORTS_DIR / f"{prefix}_cm.png"
    hm_path = REPORTS_DIR / f"{prefix}_heatmaps.png"

    plot_roc(labels, scores, roc_path)
    plot_pr(labels, scores, pr_path)
    plot_confusion_matrix(labels, y_pred, cm_path)
    plot_heatmap_overlays(paths, heatmaps, labels, scores, hm_path)

    # Log to MLflow
    for k, v in metrics.items():
        mlflow.log_metric(f"{split_name}/{k}", v)
    for p in [roc_path, pr_path, cm_path, hm_path]:
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path=f"plots/{split_name}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PatchCore heatmap model training pipeline"
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Training round (1, 2, or 3)",
    )
    parser.add_argument(
        "--heatmap",
        type=str,
        required=True,
        choices=["patchcore_r18", "patchcore_wrn50"],
        help="Heatmap model variant",
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

    # Resolve heatmap model config
    heatmap_cfg = cfg["heatmap"]["models"][args.heatmap]
    backbone_name = heatmap_cfg["backbone"]
    input_size = heatmap_cfg["input_size"]
    if isinstance(input_size, list):
        input_size = input_size[0]
    coreset_ratio = heatmap_cfg["coreset_sampling_ratio"]
    num_neighbors = heatmap_cfg["num_neighbors"]
    batch_size = heatmap_cfg["batch_size"]

    # Device
    device = args.device
    if device is None:
        device = cfg.get("training", {}).get("device", "cpu")
    num_workers = cfg.get("training", {}).get("num_workers", 0)

    rnd = args.round
    run_tag = f"round{rnd}_{args.heatmap}"

    print("=" * 60)
    print(f"PatchCore Training Pipeline")
    print(f"  round       : {rnd}")
    print(f"  model       : {args.heatmap}")
    print(f"  backbone    : {backbone_name}")
    print(f"  device      : {device}")
    print(f"  batch_size  : {batch_size}")
    print(f"  coreset_ratio: {coreset_ratio}")
    print(f"  num_neighbors: {num_neighbors}")
    print(f"  seed        : {seed}")
    print("=" * 60)

    # ---- CSV paths ----
    splits_dir = Path(cfg.get("data", {}).get("splits_dir", str(SPLITS_DIR)))
    train_csv = splits_dir / f"round{rnd}_train_normal.csv"
    val_csv = splits_dir / f"round{rnd}_val_mix.csv"
    test_csv = splits_dir / f"round{rnd}_test_mix.csv"
    anchor_csv = splits_dir / "anchor_test_mix.csv"

    for csv_path in [train_csv, val_csv, test_csv, anchor_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    # ---- Transforms ----
    transform = get_transforms(input_size)

    # ---- MLflow ----
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    if not tracking_uri.startswith("file://") and not tracking_uri.startswith("http"):
        abs_path = (PROJECT_ROOT / tracking_uri).resolve()
        tracking_uri = abs_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = mlflow_cfg.get("experiment_name", "surface_defect_detection")
    experiment_name = f"{experiment_name}_round{rnd}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_tag):
        # Log params
        mlflow.log_param("round", rnd)
        mlflow.log_param("heatmap_model", args.heatmap)
        mlflow.log_param("backbone", backbone_name)
        mlflow.log_param("device", device)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("coreset_sampling_ratio", coreset_ratio)
        mlflow.log_param("num_neighbors", num_neighbors)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("seed", seed)

        # ---- Build model ----
        model = PatchCoreModel(
            backbone_name=backbone_name,
            coreset_sampling_ratio=coreset_ratio,
            num_neighbors=num_neighbors,
            device=device,
        )

        # ---- Train (fit memory bank) ----
        train_ds = SurfaceDataset(train_csv, transform=transform)
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        print(f"\nTraining on {len(train_ds)} normal images ...")
        model.fit(train_dl)
        mlflow.log_metric("train/num_images", len(train_ds))
        mlflow.log_metric(
            "train/memory_bank_size",
            model.memory_bank.shape[0] if model.memory_bank is not None else 0,
        )

        # ---- Evaluate ----
        eval_splits = [
            ("val_mix", val_csv),
            ("test_mix", test_csv),
            ("anchor_test_mix", anchor_csv),
        ]

        all_metrics: Dict[str, dict] = {}
        for split_name, csv_path in eval_splits:
            metrics = evaluate_split(
                model=model,
                csv_path=csv_path,
                transform=transform,
                split_name=split_name,
                batch_size=batch_size,
                num_workers=num_workers,
                run_tag=run_tag,
            )
            all_metrics[split_name] = metrics

        # ---- Save model ----
        model_path = MODELS_DIR / f"{run_tag}_patchcore.pt"
        model.save(model_path)
        mlflow.log_artifact(str(model_path), artifact_path="models")

        # ---- Summary ----
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for split_name, metrics in all_metrics.items():
            if metrics:
                auroc = metrics.get("auroc", float("nan"))
                f1 = metrics.get("f1", float("nan"))
                recall = metrics.get("recall", float("nan"))
                print(
                    f"  {split_name:20s}  AUROC={auroc:.4f}  "
                    f"F1={f1:.4f}  Recall={recall:.4f}"
                )
        print(f"\nModel saved : {model_path}")
        print(f"Reports in  : {REPORTS_DIR}")
        print("=" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()
