"""
PatchCore Heatmap Training Pipeline (v1/v2/v3 / round-tagged)
==============================================================
Trains a PatchCore anomaly-detection model on the normal training split
identified by `--tag` and evaluates on val_mix / test_mix.

Recovered from git commit e361cee:src/train.py and adapted:
  - MLflow optional (no-op if not installed)
  - --round int → --tag str (accepts v1/v2/v3 or round1/round2/round3)
  - splits/{tag}_train_normal.csv etc.
  - Anchor split optional (skipped if not present)
  - Saves models/{run_tag}_config.json alongside the .pt for reproducibility

Usage
-----
    python scripts/train_patchcore.py --tag v1 --heatmap patchcore_r18
    python scripts/train_patchcore.py --tag v2 --heatmap patchcore_r18 --device mps
"""

from __future__ import annotations

import argparse
import json
import os
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

# Use backend's production PatchCoreModel (sklearn NearestNeighbors-based).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from backend.src.heatmap_model import PatchCoreModel as BackendPatchCoreModel  # noqa: E402

# -- MLflow optional --------------------------------------------------------
try:
    import mlflow as _mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

class _MlflowStub:
    """No-op stub used when mlflow is not installed."""
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop
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
class SurfaceDataset(Dataset):
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
# PatchCore model — thin wrapper around backend.src.heatmap_model.PatchCoreModel
# (same code path the inference/serve layer uses, with sklearn NearestNeighbors
# for tractable evaluation on large memory banks)
# ---------------------------------------------------------------------------
class PatchCoreModel:
    """Wrapper exposing fit/predict(dataloader)/save matching the rest of this
    script while delegating to backend.src.heatmap_model.PatchCoreModel.

    Note: the wrapped model takes ``coreset_ratio`` and ``k_neighbors`` (note
    the renamed kwarg) and uses farthest-point coreset sampling + sklearn
    NearestNeighbors. ``predict(dataloader)`` here returns aggregate arrays
    in the same shape this script previously expected for downstream metrics.
    """

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
        self._inner = BackendPatchCoreModel(
            backbone_name=backbone_name,
            coreset_ratio=coreset_sampling_ratio,
            k_neighbors=num_neighbors,
            device=device,
        )

    @property
    def memory_bank(self):
        return self._inner._memory_bank

    def fit(self, dataloader: DataLoader) -> None:
        """Fit the memory bank.

        We monkey-patch the backend's `_coreset_subsample` for the duration of
        this fit() call so that random subsampling is used instead of greedy
        farthest-point sampling. Reason: at 8k+ training images the patch pool
        reaches 6.5M patches; farthest-point sampling on that is O(M·N) and
        takes hours of CPU. Random sampling produces a coreset of identical
        size with negligible accuracy loss for typical PatchCore use.
        """
        import logging as _logging
        if not _logging.getLogger().handlers:
            _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

        from backend.src import heatmap_model as _hm

        def _random_subsample(features, ratio, seed=42):
            n = features.shape[0]
            m = max(1, int(np.ceil(n * ratio)))
            if m >= n:
                return features.copy()
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, size=m, replace=False)
            return features[idx]

        original = _hm._coreset_subsample
        _hm._coreset_subsample = _random_subsample
        try:
            self._inner.fit(dataloader)
        finally:
            _hm._coreset_subsample = original

    def predict(self, dataloader: DataLoader):
        """Return (scores, labels, paths, heatmaps) in numpy.

        The wrapped predict_batch only takes images, so we reconstruct labels
        and paths from the dataloader's underlying dataset.
        """
        ds = dataloader.dataset

        # Run efficient batched prediction (sklearn NN)
        results = self._inner.predict_batch(dataloader)

        scores = np.array([r["anomaly_score"] for r in results], dtype=np.float64)
        heatmaps = [r["raw_score_heatmap"] for r in results]
        # Pull labels/paths from the underlying dataframe in dataloader order
        labels = ds.df["label"].map({"normal": 0, "anomaly": 1}).fillna(0).astype(int).values[: len(scores)]
        paths = ds.df["path"].values[: len(scores)]
        return scores, np.asarray(labels), list(paths), heatmaps

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Delegate to backend save to keep on-disk format compatible with
        # benchmark_pipeline.py (which loads via PatchCoreModel.load).
        self._inner.save(path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Tuple[dict, float]:
    if len(y_true) == 0:
        return {}, 0.5
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
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
def plot_roc(y_true, scores, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_pr(y_true, scores, save_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_cm(y_true, y_pred, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix"); fig.colorbar(im, ax=ax)
    classes = ["normal", "anomaly"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


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
    print(f"\n--- Evaluating on {split_name} ({csv_path.name}) ---")
    ds = SurfaceDataset(csv_path, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    scores, labels, paths, heatmaps = model.predict(dl)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        print(f"  [WARN] Only one class in {split_name}; skipping metrics.")
        return {}
    metrics, threshold = compute_metrics(labels, scores)
    y_pred = (scores >= threshold).astype(int)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    prefix = f"{run_tag}_{split_name}"
    plot_roc(labels, scores, REPORTS_DIR / f"{prefix}_roc.png")
    plot_pr(labels, scores, REPORTS_DIR / f"{prefix}_pr.png")
    plot_cm(labels, y_pred, REPORTS_DIR / f"{prefix}_cm.png")
    for k, v in metrics.items():
        mlflow.log_metric(f"{split_name}/{k}", v)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchCore heatmap training pipeline")
    p.add_argument("--tag", type=str, required=True,
                   help="Version tag (e.g. v1, v2, v3, round1, round2, round3). "
                        "CSVs read from splits/{tag}_train_normal.csv etc.")
    p.add_argument("--heatmap", type=str, required=True,
                   choices=["patchcore_r18", "patchcore_wrn50"],
                   help="Heatmap model variant")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                   help="Path to YAML config (default: configs/default.yaml)")
    p.add_argument("--device", type=str, default=None,
                   help="Device: mps | cpu | cuda (overrides config)")
    p.add_argument("--coreset-ratio", type=float, default=None,
                   help="Override coreset sampling ratio")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    heatmap_cfg = cfg["heatmap"]["models"][args.heatmap]
    backbone_name = heatmap_cfg["backbone"]
    input_size = heatmap_cfg["input_size"]
    if isinstance(input_size, list):
        input_size = input_size[0]
    coreset_ratio = (
        args.coreset_ratio if args.coreset_ratio is not None
        else heatmap_cfg["coreset_sampling_ratio"]
    )
    num_neighbors = heatmap_cfg["num_neighbors"]
    batch_size = heatmap_cfg["batch_size"]

    device = args.device or cfg.get("training", {}).get("device", "cpu")
    num_workers = cfg.get("training", {}).get("num_workers", 0)

    tag = args.tag
    run_tag = f"{tag}_{args.heatmap}"

    print("=" * 60)
    print("PatchCore Training Pipeline")
    print(f"  tag         : {tag}")
    print(f"  model       : {args.heatmap}")
    print(f"  backbone    : {backbone_name}")
    print(f"  device      : {device}")
    print(f"  batch_size  : {batch_size}")
    print(f"  coreset_ratio: {coreset_ratio}")
    print(f"  num_neighbors: {num_neighbors}")
    print(f"  seed        : {seed}")
    print(f"  mlflow      : {'enabled' if _MLFLOW_AVAILABLE else 'stub (not installed)'}")
    print("=" * 60)

    splits_dir = Path(cfg.get("data", {}).get("splits_dir", str(SPLITS_DIR)))
    if not splits_dir.is_absolute():
        splits_dir = PROJECT_ROOT / splits_dir
    if not splits_dir.exists() or not (splits_dir / f"{tag}_train_normal.csv").exists():
        # Fall back to local splits/ directory
        local = PROJECT_ROOT / "splits"
        if (local / f"{tag}_train_normal.csv").exists():
            print(f"  splits_dir from config ({splits_dir}) missing; falling back to {local}")
            splits_dir = local
    train_csv = splits_dir / f"{tag}_train_normal.csv"
    val_csv = splits_dir / f"{tag}_val_mix.csv"
    test_csv = splits_dir / f"{tag}_test_mix.csv"
    anchor_csv = splits_dir / "anchor_test_mix.csv"

    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    if not anchor_csv.exists():
        anchor_csv = None
        print(f"  (no anchor split — skipping anchor eval)")
    else:
        # Anchor split file exists, but its absolute paths might be Docker-only
        # (/app/data/...). Verify the first sample is reachable.
        try:
            first_row = pd.read_csv(anchor_csv, nrows=1).iloc[0]
            if not Path(first_row["path"]).exists():
                print(f"  (anchor split paths unreachable on local disk — skipping anchor eval)")
                anchor_csv = None
        except Exception:
            anchor_csv = None

    transform = get_transforms(input_size)

    if _MLFLOW_AVAILABLE:
        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
        if not tracking_uri.startswith("file://") and not tracking_uri.startswith("http"):
            tracking_uri = f"file://{PROJECT_ROOT / tracking_uri}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(f"surface_defect_detection_{tag}")

    with mlflow.start_run(run_name=run_tag):
        mlflow.log_param("tag", tag)
        mlflow.log_param("heatmap_model", args.heatmap)
        mlflow.log_param("backbone", backbone_name)
        mlflow.log_param("device", device)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("coreset_sampling_ratio", coreset_ratio)
        mlflow.log_param("num_neighbors", num_neighbors)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("seed", seed)

        model = PatchCoreModel(
            backbone_name=backbone_name,
            coreset_sampling_ratio=coreset_ratio,
            num_neighbors=num_neighbors,
            device=device,
        )

        train_ds = SurfaceDataset(train_csv, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"\nTraining on {len(train_ds)} normal images ...")
        model.fit(train_dl)
        mlflow.log_metric("train/num_images", len(train_ds))
        mlflow.log_metric("train/memory_bank_size",
                          model.memory_bank.shape[0] if model.memory_bank is not None else 0)

        eval_splits = [("val_mix", val_csv), ("test_mix", test_csv)]
        if anchor_csv is not None:
            eval_splits.append(("anchor_test_mix", anchor_csv))

        all_metrics: Dict[str, dict] = {}
        for split_name, csv_path in eval_splits:
            metrics = evaluate_split(
                model=model, csv_path=csv_path, transform=transform,
                split_name=split_name, batch_size=batch_size,
                num_workers=num_workers, run_tag=run_tag,
            )
            all_metrics[split_name] = metrics

        model_path = MODELS_DIR / f"{run_tag}_patchcore.pt"
        model.save(model_path)

        # Reproducibility config dump (next to the .pt)
        config_path = MODELS_DIR / f"{run_tag}_config.json"
        config_payload = {
            "tag": tag,
            "run_tag": run_tag,
            "model_kind": "patchcore",
            "model_variant": args.heatmap,
            "backbone": backbone_name,
            "device": device,
            "seed": seed,
            "batch_size": batch_size,
            "coreset_sampling_ratio": coreset_ratio,
            "num_neighbors": num_neighbors,
            "input_size": input_size,
            "splits_dir": str(splits_dir),
            "train_csv_path": str(train_csv),
            "val_csv_path": str(val_csv),
            "test_csv_path": str(test_csv),
            "train_n_images": len(train_ds),
            "memory_bank_size": int(model.memory_bank.shape[0]) if model.memory_bank is not None else 0,
            "metrics": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_metrics.items()},
            "git_sha": get_git_sha(),
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "mlflow_used": _MLFLOW_AVAILABLE,
        }
        config_path.write_text(json.dumps(config_payload, indent=2))
        print(f"[PatchCore] Config written to {config_path}")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for split_name, metrics in all_metrics.items():
            if metrics:
                auroc = metrics.get("auroc", float("nan"))
                f1 = metrics.get("f1", float("nan"))
                recall = metrics.get("recall", float("nan"))
                print(f"  {split_name:20s}  AUROC={auroc:.4f}  F1={f1:.4f}  Recall={recall:.4f}")
        print(f"\nModel saved : {model_path}")
        print(f"Config saved: {config_path}")
        print("=" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()
