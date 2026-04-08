"""
Gate Model -- Binary anomaly classifier (normal vs anomaly).

Serves as the first-stage gate in the anomaly detection pipeline.
Outputs p_gate(anomaly) in [0, 1] via sigmoid on a single logit.

Supported backbones:
    - EfficientNet-B0   (torchvision.models.efficientnet_b0)
    - MobileNetV3-Large (torchvision.models.mobilenet_v3_large)

Usage:
    from src.gate_model import GateModel

    gate = GateModel(backbone="efficientnet_b0")
    history = gate.train_model(train_loader, val_loader, config)
    p = gate.predict(image_tensor)
    gate.save("checkpoints/gate_round1.pt")
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE: int = 224
_VALID_BACKBONES = ("efficientnet_b0", "mobilenet_v3_large")


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class GateTrainConfig:
    """All hyper-parameters for training the Gate model."""

    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 7
    pos_weight: Optional[float] = None  # None => auto-calculate
    backbone: str = "efficientnet_b0"
    freeze_backbone_epochs: int = 3
    unfreeze_lr_factor: float = 0.1
    use_amp: bool = True  # mixed precision
    seed: int = 42
    # Cosine annealing
    T_max: Optional[int] = None  # defaults to epochs
    eta_min: float = 1e-6
    # Extra
    log_interval: int = 10  # batches between log prints


# ---------------------------------------------------------------------------
# Utility: device selection
# ---------------------------------------------------------------------------
def _resolve_device(device: Optional[str] = None) -> torch.device:
    """Pick the best available accelerator: cuda > mps > cpu."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Utility: auto pos_weight
# ---------------------------------------------------------------------------
def compute_pos_weight(loader: DataLoader) -> float:
    """Compute pos_weight = n_negative / n_positive from a DataLoader.

    Assumes the loader yields ``(images, labels)`` where labels are
    ``0`` (normal) or ``1`` (anomaly).

    Returns:
        float  --  ratio suitable for ``BCEWithLogitsLoss(pos_weight=...)``.
    """
    n_pos = 0
    n_neg = 0
    for _, labels in loader:
        labels_np = labels.cpu().numpy().flatten()
        n_pos += int((labels_np == 1).sum())
        n_neg += int((labels_np == 0).sum())
    if n_pos == 0:
        logger.warning("No positive (anomaly) samples found; pos_weight set to 1.0")
        return 1.0
    ratio = n_neg / n_pos
    logger.info(
        "Auto pos_weight: n_neg=%d, n_pos=%d, ratio=%.4f", n_neg, n_pos, ratio
    )
    return ratio


# ---------------------------------------------------------------------------
# Network builder
# ---------------------------------------------------------------------------
def _build_backbone(
    name: str, pretrained: bool = True
) -> Tuple[nn.Module, int]:
    """Return ``(feature_extractor, num_features)`` for the chosen backbone.

    The classifier head is removed so the returned module outputs a flat
    feature vector of size ``num_features``.
    """
    if name not in _VALID_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{name}'. Choose from {_VALID_BACKBONES}."
        )

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        num_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        return base, num_features

    # mobilenet_v3_large
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    base = models.mobilenet_v3_large(weights=weights)
    num_features = base.classifier[0].in_features
    base.classifier = nn.Identity()
    return base, num_features


# ---------------------------------------------------------------------------
# GateModel
# ---------------------------------------------------------------------------
class GateModel:
    """Binary gate classifier for anomaly detection.

    Parameters
    ----------
    backbone : str
        One of ``"efficientnet_b0"`` or ``"mobilenet_v3_large"``.
    pretrained : bool
        Load ImageNet-pretrained weights for the backbone.
    device : str or None
        Target device.  ``None`` triggers auto-detection (cuda > mps > cpu).

    Examples
    --------
    >>> gate = GateModel(backbone="efficientnet_b0")
    >>> prob = gate.predict(torch.randn(1, 3, 224, 224))
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.backbone_name = backbone
        self.device = _resolve_device(device)
        self._build_network(backbone, pretrained)
        self.calibrator: Optional[Any] = None  # sklearn calibrator
        self.threshold: float = 0.5
        self.training_history: List[Dict[str, float]] = []

    # -- internal -----------------------------------------------------------
    def _build_network(self, backbone: str, pretrained: bool) -> None:
        feature_extractor, num_features = _build_backbone(backbone, pretrained)
        self.feature_extractor = feature_extractor
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 1),
        )
        self.model = nn.Sequential(self.feature_extractor, self.head)
        self.model.to(self.device)

    # -- freezing helpers ---------------------------------------------------
    def _freeze_backbone(self) -> None:
        """Freeze all backbone (feature_extractor) parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[GateTrainConfig] = None,
    ) -> List[Dict[str, float]]:
        """Train the gate model.

        Parameters
        ----------
        train_loader : DataLoader
            Yields ``(images, labels)`` with labels in {0, 1}.
        val_loader : DataLoader
            Validation set in the same format.
        config : GateTrainConfig or None
            Training configuration.  Defaults are used when ``None``.

        Returns
        -------
        list[dict]
            Per-epoch training history with keys:
            ``epoch, train_loss, val_loss, val_acc, val_recall, val_precision,
            val_f1, lr, time_s``.
        """
        if config is None:
            config = GateTrainConfig()

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # -- pos_weight ------------------------------------------------------
        if config.pos_weight is not None:
            pw = config.pos_weight
        else:
            pw = compute_pos_weight(train_loader)
        pos_weight_tensor = torch.tensor([pw], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # -- optimiser & scheduler ------------------------------------------
        optimizer = AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        T_max = config.T_max if config.T_max is not None else config.epochs
        scheduler = CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=config.eta_min
        )

        # -- mixed precision ------------------------------------------------
        use_amp = config.use_amp and self.device.type in ("cuda",)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        amp_device_type = "cuda" if use_amp else "cpu"

        # -- early stopping -------------------------------------------------
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        history: List[Dict[str, float]] = []

        # -- optional backbone freeze at start ------------------------------
        if config.freeze_backbone_epochs > 0:
            self._freeze_backbone()
            logger.info(
                "Backbone frozen for the first %d epoch(s).",
                config.freeze_backbone_epochs,
            )

        for epoch in range(1, config.epochs + 1):
            # Unfreeze backbone after warmup
            if epoch == config.freeze_backbone_epochs + 1:
                self._unfreeze_backbone()
                # Lower backbone LR relative to head
                param_groups = [
                    {
                        "params": self.feature_extractor.parameters(),
                        "lr": config.lr * config.unfreeze_lr_factor,
                    },
                    {"params": self.head.parameters(), "lr": config.lr},
                ]
                optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=config.epochs - epoch + 1,
                    eta_min=config.eta_min,
                )
                logger.info("Backbone unfrozen at epoch %d.", epoch)

            t0 = time.time()

            # ---- train phase ------------------------------------------------
            train_loss = self._train_one_epoch(
                train_loader, criterion, optimizer, scaler, amp_device_type, config
            )

            # ---- val phase --------------------------------------------------
            val_metrics = self._validate(val_loader, criterion, amp_device_type)
            val_loss = val_metrics["loss"]

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_metrics["acc"],
                "val_recall": val_metrics["recall"],
                "val_precision": val_metrics["precision"],
                "val_f1": val_metrics["f1"],
                "lr": current_lr,
                "time_s": elapsed,
            }
            history.append(record)
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f  "
                "lr=%.2e  (%.1fs)",
                epoch,
                config.epochs,
                train_loss,
                val_loss,
                val_metrics["f1"],
                current_lr,
                elapsed,
            )

            # ---- early stopping check --------------------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d (patience=%d).",
                        epoch,
                        config.patience,
                    )
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Restored best model weights (val_loss=%.4f).", best_val_loss)

        self.training_history = history
        return history

    # -- single epoch helpers -----------------------------------------------
    def _train_one_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        amp_device_type: str,
        config: GateTrainConfig,
    ) -> float:
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast(device_type=amp_device_type):
                    logits = self.model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % config.log_interval == 0:
                logger.debug(
                    "  batch %d/%d  loss=%.4f",
                    batch_idx + 1,
                    len(loader),
                    loss.item(),
                )

        return running_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        amp_device_type: str,
    ) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        all_labels: List[np.ndarray] = []
        all_preds: List[np.ndarray] = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels_dev = labels.to(self.device, non_blocking=True).float().unsqueeze(1)

            logits = self.model(images)
            loss = criterion(logits, labels_dev)
            running_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.append(probs)
            all_labels.append(labels.numpy().flatten())

        all_labels_np = np.concatenate(all_labels)
        all_preds_np = np.concatenate(all_preds)
        pred_binary = (all_preds_np >= self.threshold).astype(int)

        tp = int(((pred_binary == 1) & (all_labels_np == 1)).sum())
        fp = int(((pred_binary == 1) & (all_labels_np == 0)).sum())
        fn = int(((pred_binary == 0) & (all_labels_np == 1)).sum())
        tn = int(((pred_binary == 0) & (all_labels_np == 0)).sum())

        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            2 * precision * recall / max(precision + recall, 1e-8)
        )

        return {
            "loss": running_loss / max(n_batches, 1),
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> float:
        """Return p_gate(anomaly) for a single image tensor.

        Parameters
        ----------
        image_tensor : Tensor
            Shape ``(1, 3, 224, 224)`` or ``(3, 224, 224)``.

        Returns
        -------
        float
            Probability of anomaly in [0, 1].
        """
        self.model.eval()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        # --- 이 부분을 수정하세요 ---
        # image_tensor를 모델이 있는 위치(self.device)로 강제 이동시킵니다.
        #image_tensor = image_tensor.to(self.device) 
        
        # 만약 self.model의 가중치가 CPU에 있다면 텐서도 CPU로 가야 합니다.
        # 가장 안전한 방법은 아래 한 줄을 추가하는 것입니다.
        model_device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(model_device)
        # -------------------------
        logit = self.model(image_tensor)
        prob = torch.sigmoid(logit).item()

        if self.calibrator is not None:
            prob = float(
                self.calibrator.predict(np.array([[prob]]))[0]
            )
            prob = float(np.clip(prob, 0.0, 1.0))
        return prob

    @torch.no_grad()
    def predict_batch(self, loader: DataLoader) -> np.ndarray:
        """Return an array of p_gate(anomaly) for every sample in *loader*.

        Parameters
        ----------
        loader : DataLoader
            Yields ``(images, ...)`` -- only the first element is used.

        Returns
        -------
        np.ndarray
            1-D array of probabilities, length = total samples in loader.
        """
        self.model.eval()
        all_probs: List[np.ndarray] = []

        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.append(probs)

        probs_np = np.concatenate(all_probs)

        if self.calibrator is not None:
            probs_np = self.calibrator.predict(probs_np.reshape(-1, 1)).flatten()
            probs_np = np.clip(probs_np, 0.0, 1.0)

        return probs_np

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------
    def calibrate(
        self,
        val_loader: DataLoader,
        method: str = "isotonic",
    ) -> "GateModel":
        """Post-hoc probability calibration using validation data.

        Parameters
        ----------
        val_loader : DataLoader
            Yields ``(images, labels)``.
        method : str
            ``"platt"`` for Platt scaling (logistic regression on logits)
            or ``"isotonic"`` for isotonic regression on probabilities.

        Returns
        -------
        GateModel
            ``self``, with the calibrator attached.

        Raises
        ------
        ImportError
            If scikit-learn is not installed.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for calibration. "
                "Install it with: pip install scikit-learn"
            )

        probs = self.predict_batch(val_loader)

        # Collect ground-truth labels
        all_labels: List[np.ndarray] = []
        for _, labels in val_loader:
            all_labels.append(labels.numpy().flatten())
        y_true = np.concatenate(all_labels)

        if method == "platt":
            # Platt scaling: logistic regression on raw probabilities
            lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
            lr.fit(probs.reshape(-1, 1), y_true)
            self.calibrator = lr
            logger.info("Platt scaling calibrator fitted.")
        elif method == "isotonic":
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(probs, y_true)
            self.calibrator = ir
            logger.info("Isotonic regression calibrator fitted.")
        else:
            raise ValueError(f"Unknown calibration method '{method}'. Use 'platt' or 'isotonic'.")

        return self

    # -----------------------------------------------------------------------
    # Threshold sweep
    # -----------------------------------------------------------------------
    def threshold_sweep(
        self,
        test_loader: DataLoader,
        thresholds: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """Sweep decision thresholds and compute binary metrics at each.

        Parameters
        ----------
        test_loader : DataLoader
            Yields ``(images, labels)``.
        thresholds : sequence of float or None
            Thresholds to evaluate.  Default: ``np.arange(0.0, 1.01, 0.01)``.

        Returns
        -------
        pd.DataFrame
            Columns: ``threshold, tp, fp, fn, tn, recall, precision, f1, fpr, accuracy``.
        """
        if thresholds is None:
            thresholds = np.arange(0.0, 1.01, 0.01)

        probs = self.predict_batch(test_loader)

        all_labels: List[np.ndarray] = []
        for _, labels in test_loader:
            all_labels.append(labels.numpy().flatten())
        y_true = np.concatenate(all_labels)

        rows: List[Dict[str, float]] = []
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())
            tn = int(((preds == 0) & (y_true == 0)).sum())

            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            fpr = fp / max(fp + tn, 1)
            accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

            rows.append(
                {
                    "threshold": round(float(t), 4),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "recall": round(recall, 6),
                    "precision": round(precision, 6),
                    "f1": round(f1, 6),
                    "fpr": round(fpr, 6),
                    "accuracy": round(accuracy, 6),
                }
            )

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Threshold recommendation
    # -----------------------------------------------------------------------
    @staticmethod
    def recommend_thresholds(
        sweep_df: pd.DataFrame,
        recall_floor: float = 0.95,
        fpr_ceiling: float = 0.10,
    ) -> Dict[str, Any]:
        """Recommend T_low (high-recall) and T_high (high-precision) thresholds.

        Parameters
        ----------
        sweep_df : pd.DataFrame
            Output of :meth:`threshold_sweep`.
        recall_floor : float
            Minimum acceptable recall for T_low.
        fpr_ceiling : float
            Maximum acceptable FPR for T_high.

        Returns
        -------
        dict
            ``{"T_low": float, "T_high": float, "T_best_f1": float,
            "T_low_metrics": dict, "T_high_metrics": dict,
            "T_best_f1_metrics": dict}``
        """
        df = sweep_df.copy()

        # T_best_f1: threshold maximising F1
        best_f1_idx = df["f1"].idxmax()
        t_best_f1 = df.loc[best_f1_idx, "threshold"]

        # T_low: lowest threshold where recall >= recall_floor
        high_recall = df[df["recall"] >= recall_floor]
        if len(high_recall) > 0:
            # Among those meeting recall constraint, pick the one with highest
            # precision (= highest threshold still meeting recall floor)
            t_low_idx = high_recall["precision"].idxmax()
            t_low = high_recall.loc[t_low_idx, "threshold"]
        else:
            # Fallback: just use the threshold with max recall
            t_low = df.loc[df["recall"].idxmax(), "threshold"]
            logger.warning(
                "No threshold achieves recall >= %.2f; using max-recall threshold.",
                recall_floor,
            )

        # T_high: highest threshold where FPR <= fpr_ceiling
        low_fpr = df[df["fpr"] <= fpr_ceiling]
        if len(low_fpr) > 0:
            # Among those meeting FPR constraint, pick highest recall
            t_high_idx = low_fpr["recall"].idxmax()
            t_high = low_fpr.loc[t_high_idx, "threshold"]
        else:
            t_high = df.loc[df["fpr"].idxmin(), "threshold"]
            logger.warning(
                "No threshold achieves FPR <= %.2f; using min-FPR threshold.",
                fpr_ceiling,
            )

        def _metrics_at(t: float) -> Dict[str, float]:
            row = df[df["threshold"] == t].iloc[0]
            return row.to_dict()

        return {
            "T_low": float(t_low),
            "T_high": float(t_high),
            "T_best_f1": float(t_best_f1),
            "T_low_metrics": _metrics_at(t_low),
            "T_high_metrics": _metrics_at(t_high),
            "T_best_f1_metrics": _metrics_at(t_best_f1),
        }

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Save model weights, config, calibrator, and threshold to disk.

        Parameters
        ----------
        path : str or Path
            File path for the checkpoint (e.g. ``"checkpoints/gate.pt"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "backbone_name": self.backbone_name,
            "threshold": self.threshold,
            "training_history": self.training_history,
        }

        if self.calibrator is not None:
            import pickle

            payload["calibrator_bytes"] = pickle.dumps(self.calibrator)
            payload["calibrator_type"] = type(self.calibrator).__name__

        torch.save(payload, path)
        logger.info("Gate model saved to %s", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "GateModel":
        """Load a previously saved GateModel checkpoint.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file.
        device : str or None
            Target device.

        Returns
        -------
        GateModel
        """
        path = Path(path)
        dev = _resolve_device(device)
        payload = torch.load(path, map_location=dev, weights_only=False)

        backbone_name = payload.get("backbone_name", "efficientnet_b0")
        instance = cls(backbone=backbone_name, pretrained=False, device=str(dev))

        # --- 가중치 키 매핑 ---
        # train_gate.py 저장 형식: features.xxx, classifier.1.weight
        # GateModel 구조: model = Sequential(feature_extractor[0], head[1])
        # 따라서: features.xxx → 0.features.xxx
        #         classifier.N.xxx → 1.N.xxx  (head는 model[1])
        state_dict = payload["model_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("features"):
                new_key = f"0.{k}"
            elif k.startswith("classifier"):
                # "classifier.1.weight" → "1.1.weight"
                sub_key = k[len("classifier."):]
                new_key = f"1.{sub_key}"
            else:
                new_key = k
            new_state_dict[new_key] = v
        instance.model.load_state_dict(new_state_dict, strict=False)
        # --- 키 매핑 끝 ---

        instance.threshold = payload.get("threshold", 0.5)
        instance.training_history = payload.get("training_history", [])

        # calibrator 로드 (1) .pt 내부에 저장된 경우
        if "calibrator_bytes" in payload:
            import pickle
            instance.calibrator = pickle.loads(payload["calibrator_bytes"])
            logger.info(
                "Loaded calibrator from .pt: %s", payload.get("calibrator_type", "unknown")
            )
        else:
            # calibrator 로드 (2) 별도 _calibrator.pkl 파일이 있는 경우 (train_gate.py 형식)
            import pickle
            import io
            from sklearn.isotonic import IsotonicRegression as _IR
            from sklearn.linear_model import LogisticRegression as _LR

            # train_gate.py가 __main__ 컨텍스트에서 저장했으므로 래퍼 클래스를 여기서 재정의
            class _IsotonicCalibrator:
                def __init__(self):
                    self.ir = _IR(out_of_bounds="clip")
                def fit(self, probs, labels):
                    self.ir.fit(probs, labels)
                def predict_proba(self, probs):
                    return self.ir.predict(probs)

            class _PlattCalibrator:
                def __init__(self):
                    self.lr = _LR(C=1e10, solver="lbfgs", max_iter=1000)
                def fit(self, probs, labels):
                    self.lr.fit(probs.reshape(-1, 1), labels)
                def predict_proba(self, probs):
                    return self.lr.predict_proba(probs.reshape(-1, 1))[:, 1]

            class _SafeUnpickler(pickle.Unpickler):
                """__main__에 저장된 train_gate 캘리브레이터 클래스를 안전하게 복원"""
                _CLASS_MAP = {
                    ("__main__", "IsotonicCalibrator"): _IsotonicCalibrator,
                    ("__main__", "PlattCalibrator"): _PlattCalibrator,
                }
                def find_class(self, module, name):
                    mapped = self._CLASS_MAP.get((module, name))
                    if mapped is not None:
                        return mapped
                    return super().find_class(module, name)

            cal_stem = Path(path).stem.replace("_gate", "_calibrator")
            cal_path = Path(path).parent / f"{cal_stem}.pkl"
            if cal_path.exists():
                with open(cal_path, "rb") as _f:
                    cal_data = _SafeUnpickler(_f).load()
                inner = cal_data.get("calibrator") if isinstance(cal_data, dict) else cal_data
                # 래퍼에서 sklearn 객체 추출
                if hasattr(inner, "ir"):
                    instance.calibrator = inner.ir
                    logger.info("Loaded IsotonicRegression calibrator from %s", cal_path)
                elif hasattr(inner, "lr"):
                    class _PlattAdapter:
                        def __init__(self, lr_model):
                            self._lr = lr_model
                        def predict(self, X):
                            return self._lr.predict_proba(
                                np.array(X).reshape(-1, 1)
                            )[:, 1]
                    instance.calibrator = _PlattAdapter(inner.lr)
                    logger.info("Loaded Platt calibrator from %s", cal_path)
                else:
                    instance.calibrator = inner
                    logger.info("Loaded calibrator from %s", cal_path)

        logger.info("Gate model loaded from %s (backbone=%s)", path, backbone_name)
        return instance
        '''
        path = Path(path)
        dev = _resolve_device(device)
        payload = torch.load(path, map_location=dev, weights_only=False)

        backbone_name = payload.get("backbone_name", "efficientnet_b0")
        instance = cls(backbone=backbone_name, pretrained=False, device=str(dev))
        #수정전
        #instance.model.load_state_dict(payload["model_state_dict"])
        # 수정 후
        state_dict = payload["model_state_dict"]
        # strict=False 옵션을 주면 미세한 레이어 이름 차이는 무시하고 값만 주입합니다.
        instance.model.load_state_dict(state_dict, strict=False)
        instance.threshold = payload.get("threshold", 0.5)
        instance.training_history = payload.get("training_history", [])

        if "calibrator_bytes" in payload:
            import pickle

            instance.calibrator = pickle.loads(payload["calibrator_bytes"])
            logger.info(
                "Loaded calibrator: %s", payload.get("calibrator_type", "unknown")
            )

        logger.info("Gate model loaded from %s (backbone=%s)", path, backbone_name)
        return instance
        '''

    # -----------------------------------------------------------------------
    # Convenience / repr
    # -----------------------------------------------------------------------
    def set_threshold(self, t: float) -> None:
        """Set the decision threshold used by :meth:`_validate`."""
        if not 0.0 <= t <= 1.0:
            raise ValueError("Threshold must be in [0, 1].")
        self.threshold = t

    @property
    def num_parameters(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"GateModel(backbone={self.backbone_name!r}, "
            f"device={self.device}, "
            f"params={self.num_parameters:,}, "
            f"threshold={self.threshold:.4f})"
        )
