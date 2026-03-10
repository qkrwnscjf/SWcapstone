#!/usr/bin/env python3
"""
Standalone cascade evaluation module.

Provides CascadeEvaluator -- a class that takes a trained gate model and a
trained heatmap model and evaluates the full Gate -> Heatmap cascade across
configurable threshold pairs (T_low, T_high).

Main API:
    evaluator = CascadeEvaluator(gate_model, heatmap_model, device)
    metrics   = evaluator.evaluate(test_loader, anchor_loader, T_low, T_high)
    df        = evaluator.sweep_cascade(test_loader, T_low_range, T_high_range)
    report    = evaluator.generate_report(metrics)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project imports -- expected interface:
#   gate_model.gate_predict(model, tensor)       -> float (p_anomaly in [0,1])
#   heatmap_model.heatmap_predict(model, tensor) -> float (anomaly score)
# ---------------------------------------------------------------------------
#from src.gate_model import gate_predict
#from src.heatmap_model import heatmap_predict

# 기존 import 구문 수정
from src.gate_model import GateModel
from src.heatmap_model import PatchCoreModel

def gate_predict(model, tensor):
    # GateModel.predict는 float 확률을 바로 반환함
    return model.predict(tensor)

def heatmap_predict(model, tensor):
    # PatchCoreModel.predict는 Dict를 반환하므로 anomaly_score만 추출
    result = model.predict(tensor)
    return result["anomaly_score"]

logger = logging.getLogger("cascade_eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Data class for per-sample prediction record
# ---------------------------------------------------------------------------
@dataclass
class _SampleRecord:
    """Internal record for a single test sample."""
    label: int                       # ground-truth: 0=normal, 1=anomaly
    gate_score: float                # p_gate(anomaly)
    heatmap_score: Optional[float]   # None when heatmap was not called
    heatmap_called: bool
    cascade_pred: int                # final binary prediction
    gate_latency_ms: float
    heatmap_latency_ms: float
    total_latency_ms: float


# ---------------------------------------------------------------------------
# CascadeEvaluator
# ---------------------------------------------------------------------------
class CascadeEvaluator:
    """
    Evaluate a Gate -> Heatmap cascade on a test DataLoader.

    Parameters
    ----------
    gate_model : torch.nn.Module
        Trained gate (binary classifier) model.
    heatmap_model : Any
        Trained heatmap (e.g. PatchCore) model / object.
    device : torch.device | str
        Computation device.
    heatmap_threshold : float
        Decision boundary for the heatmap anomaly score (default 0.5).
    """

    def __init__(
        self,
        gate_model: torch.nn.Module,
        heatmap_model: Any,
        device: torch.device | str = "cpu",
        heatmap_threshold: float = 0.5,
    ) -> None:
        self.gate_model = gate_model
        self.heatmap_model = heatmap_model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.heatmap_threshold = heatmap_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _should_call_heatmap(
        p_gate: float,
        t_low: float,
        t_high: float,
        override: bool = False,
    ) -> bool:
        """Mirror the cascade decision logic from serve.py."""
        if override:
            return True
        if p_gate <= t_low:
            return False
        return True  # uncertain zone or p_gate >= t_high

    def _run_cascade_single(
        self,
        tensor: torch.Tensor,
        t_low: float,
        t_high: float,
        override: bool = False,
    ) -> Tuple[float, Optional[float], bool, int, float, float, float]:
        """
        Run the cascade on a single sample tensor [1, C, H, W].

        Returns
        -------
        gate_score, heatmap_score, heatmap_called, cascade_pred,
        gate_latency_ms, heatmap_latency_ms, total_latency_ms
        """
        t_start = time.perf_counter()

        # Gate inference
        t_gate_start = time.perf_counter()
        gate_score = gate_predict(self.gate_model, tensor)
        t_gate_end = time.perf_counter()
        gate_latency_ms = (t_gate_end - t_gate_start) * 1000.0

        # Cascade decision
        call_heatmap = self._should_call_heatmap(gate_score, t_low, t_high, override)

        heatmap_score: Optional[float] = None
        heatmap_latency_ms: float = 0.0

        if call_heatmap:
            t_hm_start = time.perf_counter()
            heatmap_score = heatmap_predict(self.heatmap_model, tensor)
            t_hm_end = time.perf_counter()
            heatmap_latency_ms = (t_hm_end - t_hm_start) * 1000.0

        # Final prediction
        if heatmap_score is not None:
            cascade_pred = int(heatmap_score >= self.heatmap_threshold)
        else:
            # Gate-only exit -> predict normal
            cascade_pred = 0

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0

        return (
            gate_score,
            heatmap_score,
            call_heatmap,
            cascade_pred,
            gate_latency_ms,
            heatmap_latency_ms,
            total_latency_ms,
        )

    def _collect_records(
        self,
        loader: DataLoader,
        t_low: float,
        t_high: float,
        override: bool = False,
    ) -> List[_SampleRecord]:
        """Iterate over a DataLoader and collect per-sample records."""
        records: List[_SampleRecord] = []
        #self.gate_model.eval()

        # [수정(2/20)] GateModel 내부 모델 eval 모드 전환
        if hasattr(self.gate_model, 'model'):
            self.gate_model.model.eval()
        
        # [수정(2/20)] HeatmapModel 내부 feature_extractor eval 모드 전환
        if hasattr(self.heatmap_model, '_feature_extractor'):
            self.heatmap_model._feature_extractor.eval()

        with torch.no_grad():
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(self.device)
                # Process sample-by-sample for accurate per-sample latency
                for i in range(batch_images.size(0)):
                    tensor = batch_images[i].unsqueeze(0)  # [1, C, H, W]
                    label = int(batch_labels[i].item())

                    (
                        gate_score,
                        heatmap_score,
                        heatmap_called,
                        cascade_pred,
                        gate_lat,
                        hm_lat,
                        total_lat,
                    ) = self._run_cascade_single(tensor, t_low, t_high, override)

                    records.append(
                        _SampleRecord(
                            label=label,
                            gate_score=gate_score,
                            heatmap_score=heatmap_score,
                            heatmap_called=heatmap_called,
                            cascade_pred=cascade_pred,
                            gate_latency_ms=gate_lat,
                            heatmap_latency_ms=hm_lat,
                            total_latency_ms=total_lat,
                        )
                    )

        return records

    @staticmethod
    def _compute_metrics(records: List[_SampleRecord]) -> Dict[str, Any]:
        """Compute aggregate metrics from sample records."""
        n = len(records)
        if n == 0:
            return {"error": "no samples"}

        y_true = np.array([r.label for r in records])
        y_pred = np.array([r.cascade_pred for r in records])
        gate_scores = np.array([r.gate_score for r in records])

        heatmap_calls = sum(1 for r in records if r.heatmap_called)
        gate_only_exits = n - heatmap_calls
        heatmap_call_rate = heatmap_calls / n

        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # AUROC -- use gate_scores for samples that only went through the
        # gate, and heatmap_scores where available, to build a combined
        # anomaly score vector for ROC computation.
        combined_scores = np.array([
            r.heatmap_score if r.heatmap_score is not None else r.gate_score
            for r in records
        ])

        try:
            cascade_auroc = float(roc_auc_score(y_true, combined_scores))
        except ValueError:
            # Single-class edge case
            cascade_auroc = float("nan")

        # Gate-only AUROC
        try:
            gate_auroc = float(roc_auc_score(y_true, gate_scores))
        except ValueError:
            gate_auroc = float("nan")

        # Latency statistics
        gate_lats = [r.gate_latency_ms for r in records]
        hm_lats = [r.heatmap_latency_ms for r in records if r.heatmap_called]
        total_lats = [r.total_latency_ms for r in records]

        return {
            "total_predictions": n,
            "gate_only_exits": gate_only_exits,
            "heatmap_calls": heatmap_calls,
            "heatmap_call_rate": round(heatmap_call_rate, 4),
            # Classification
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "cascade_auroc": round(cascade_auroc, 4),
            "gate_auroc": round(gate_auroc, 4),
            "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
            # Latency (ms)
            "avg_gate_latency_ms": round(float(np.mean(gate_lats)), 2),
            "avg_heatmap_latency_ms": (
                round(float(np.mean(hm_lats)), 2) if hm_lats else 0.0
            ),
            "avg_total_latency_ms": round(float(np.mean(total_lats)), 2),
            "p50_total_latency_ms": round(float(np.percentile(total_lats, 50)), 2),
            "p95_total_latency_ms": round(float(np.percentile(total_lats, 95)), 2),
            "p99_total_latency_ms": round(float(np.percentile(total_lats, 99)), 2),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        test_loader: DataLoader,
        anchor_loader: Optional[DataLoader] = None,
        T_low: float = 0.3,
        T_high: float = 0.7,
        override: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate the cascade on a test DataLoader.

        Parameters
        ----------
        test_loader : DataLoader
            Yields (images: Tensor, labels: Tensor) batches.
            Labels: 0 = normal, 1 = anomaly.
        anchor_loader : DataLoader | None
            Optional anchor set for additional evaluation.
        T_low : float
            Lower gate threshold -- below this, predict normal (skip heatmap).
        T_high : float
            Upper gate threshold -- above this, call heatmap.
        override : bool
            If True, always call heatmap (simulates override flags).

        Returns
        -------
        dict
            Metrics dictionary with classification scores, call rate, latency.
        """
        logger.info(
            "Evaluating cascade  T_low=%.3f  T_high=%.3f  override=%s",
            T_low, T_high, override,
        )

        records = self._collect_records(test_loader, T_low, T_high, override)
        result = self._compute_metrics(records)
        result["T_low"] = T_low
        result["T_high"] = T_high
        result["override"] = override

        # Optionally evaluate on anchor set
        if anchor_loader is not None:
            logger.info("Evaluating cascade on anchor set ...")
            anchor_records = self._collect_records(
                anchor_loader, T_low, T_high, override
            )
            anchor_metrics = self._compute_metrics(anchor_records)
            result["anchor"] = anchor_metrics

        logger.info(
            "Cascade results  AUROC=%.4f  Recall=%.4f  F1=%.4f  "
            "CallRate=%.4f  AvgLatency=%.1fms",
            result["cascade_auroc"],
            result["recall"],
            result["f1"],
            result["heatmap_call_rate"],
            result["avg_total_latency_ms"],
        )

        return result

    def sweep_cascade(
        self,
        test_loader: DataLoader,
        T_low_range: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
        T_high_range: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
        anchor_loader: Optional[DataLoader] = None,
    ) -> pd.DataFrame:
        """
        Sweep over all (T_low, T_high) combinations and collect metrics.

        Only evaluates combinations where T_low <= T_high to avoid invalid
        configurations.

        Parameters
        ----------
        test_loader : DataLoader
            Test data loader.
        T_low_range : sequence of float
            T_low values to sweep.
        T_high_range : sequence of float
            T_high values to sweep.
        anchor_loader : DataLoader | None
            Optional anchor set loader.

        Returns
        -------
        pd.DataFrame
            One row per valid (T_low, T_high) combination with all metrics.
        """
        rows: List[Dict[str, Any]] = []
        combos = [
            (tl, th)
            for tl, th in product(T_low_range, T_high_range)
            if tl <= th
        ]
        logger.info(
            "Sweeping %d cascade threshold combinations ...", len(combos)
        )

        for idx, (tl, th) in enumerate(combos, 1):
            logger.info(
                "[%d/%d] T_low=%.2f  T_high=%.2f", idx, len(combos), tl, th
            )
            m = self.evaluate(
                test_loader,
                anchor_loader=anchor_loader,
                T_low=tl,
                T_high=th,
            )
            rows.append(m)

        df = pd.DataFrame(rows)

        # Sort by primary optimization target (recall) then by call rate
        df = df.sort_values(
            by=["recall", "heatmap_call_rate"],
            ascending=[False, True],
        ).reset_index(drop=True)

        logger.info("Sweep complete. Best by recall:\n%s", df.head(5).to_string())
        return df

    @staticmethod
    def generate_report(metrics_dict: Dict[str, Any]) -> str:
        """
        Generate a human-readable report string from a metrics dictionary.

        Parameters
        ----------
        metrics_dict : dict
            Output of ``evaluate()``.

        Returns
        -------
        str
            Formatted multi-line report.
        """
        lines = [
            "=" * 64,
            "  CASCADE EVALUATION REPORT",
            "=" * 64,
            "",
            f"  Thresholds           T_low = {metrics_dict.get('T_low', 'N/A')}    "
            f"T_high = {metrics_dict.get('T_high', 'N/A')}",
            f"  Override active      {metrics_dict.get('override', False)}",
            "",
            "--- Prediction Summary ---",
            f"  Total predictions    {metrics_dict.get('total_predictions', 0)}",
            f"  Gate-only exits      {metrics_dict.get('gate_only_exits', 0)}",
            f"  Heatmap calls        {metrics_dict.get('heatmap_calls', 0)}",
            f"  Heatmap call rate    {metrics_dict.get('heatmap_call_rate', 0):.2%}",
            "",
            "--- Classification Metrics ---",
            f"  Accuracy             {metrics_dict.get('accuracy', 0):.4f}",
            f"  Precision            {metrics_dict.get('precision', 0):.4f}",
            f"  Recall               {metrics_dict.get('recall', 0):.4f}",
            f"  F1 Score             {metrics_dict.get('f1', 0):.4f}",
            f"  Cascade AUROC        {metrics_dict.get('cascade_auroc', 0):.4f}",
            f"  Gate-only AUROC      {metrics_dict.get('gate_auroc', 0):.4f}",
            "",
            "--- Confusion Matrix ---",
            "                     Pred Normal   Pred Anomaly",
        ]

        cm = metrics_dict.get("confusion_matrix", [[0, 0], [0, 0]])
        lines.append(
            f"  Actual Normal      {cm[0][0]:>10d}   {cm[0][1]:>12d}"
        )
        lines.append(
            f"  Actual Anomaly     {cm[1][0]:>10d}   {cm[1][1]:>12d}"
        )

        lines += [
            "",
            "--- Latency (ms) ---",
            f"  Avg gate latency     {metrics_dict.get('avg_gate_latency_ms', 0):.2f}",
            f"  Avg heatmap latency  {metrics_dict.get('avg_heatmap_latency_ms', 0):.2f}",
            f"  Avg total latency    {metrics_dict.get('avg_total_latency_ms', 0):.2f}",
            f"  P50 total latency    {metrics_dict.get('p50_total_latency_ms', 0):.2f}",
            f"  P95 total latency    {metrics_dict.get('p95_total_latency_ms', 0):.2f}",
            f"  P99 total latency    {metrics_dict.get('p99_total_latency_ms', 0):.2f}",
        ]

        # Anchor set section (if present)
        if "anchor" in metrics_dict:
            a = metrics_dict["anchor"]
            lines += [
                "",
                "--- Anchor Set Metrics ---",
                f"  Accuracy             {a.get('accuracy', 0):.4f}",
                f"  Recall               {a.get('recall', 0):.4f}",
                f"  F1 Score             {a.get('f1', 0):.4f}",
                f"  Cascade AUROC        {a.get('cascade_auroc', 0):.4f}",
                f"  Heatmap call rate    {a.get('heatmap_call_rate', 0):.2%}",
            ]

        lines += ["", "=" * 64]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point -- allows `python -m src.cascade` for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path

    #from src.gate_model import load_gate_model (2/20 수정)
    #from src.heatmap_model import load_heatmap_model (2/20 수정)

    from src.gate_model import GateModel
    from src.heatmap_model import PatchCoreModel

    parser = argparse.ArgumentParser(description="Run cascade evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to test split CSV (columns: path, label)",
    )
    parser.add_argument(
        "--anchor-csv",
        type=str,
        default=None,
        help="Optional anchor split CSV",
    )
    parser.add_argument(
        "--gate-model",
        type=str,
        default="models/gate_model.pt",
        help="Path to gate model checkpoint",
    )
    parser.add_argument(
        "--heatmap-model",
        type=str,
        default="models/heatmap_model.pt",
        help="Path to heatmap model checkpoint",
    )
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    parser.add_argument("--output", type=str, default=None, help="Output CSV for sweep")
    args = parser.parse_args()

    # Load config
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Device
    device_str = cfg.get("training", {}).get("device", "cpu")
    device = torch.device(device_str)

    # Load models
    #gate = load_gate_model(args.gate_model, device) (2/20)
    #heatmap = load_heatmap_model(args.heatmap_model, device) (2/20)

    # [수정(2/20)] GateModel 로드
    gate = GateModel.load(args.gate_model, device=device_str)
    
    # [수정(2/20)] PatchCoreModel 로드 (heatmap_model.py의 PatchCoreModel.load 사용)
    from src.heatmap_model import PatchCoreModel
    heatmap = PatchCoreModel.load(args.heatmap_model, device=device_str)

    # Build data loaders -- minimal CSV-based loader for standalone use
    from torchvision import transforms as T
    from torch.utils.data import Dataset

    class _CSVImageDataset(Dataset):
        """Lightweight CSV-backed dataset for evaluation."""

        def __init__(self, csv_path: str) -> None:
            import csv
            with open(csv_path) as fh:
                reader = csv.DictReader(fh)
                self.rows = list(reader)
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            from PIL import Image
            row = self.rows[idx]
            img = Image.open(row["path"]).convert("RGB")
            tensor = self.transform(img)
            label = 1 if row["label"] == "anomaly" else 0
            return tensor, label

    test_ds = _CSVImageDataset(args.test_csv)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    anchor_loader = None
    if args.anchor_csv:
        anchor_ds = _CSVImageDataset(args.anchor_csv)
        anchor_loader = DataLoader(anchor_ds, batch_size=32, shuffle=False, num_workers=0)

    # Evaluate
    evaluator = CascadeEvaluator(gate, heatmap, device=device)

    cascade_cfg = cfg.get("cascade", {})
    t_low = cascade_cfg.get("T_low", 0.3)
    t_high = cascade_cfg.get("T_high", 0.7)

    if args.sweep:
        sweep_cfg = cfg.get("threshold_sweep", {})
        t_low_range = sweep_cfg.get("T_low_range", [0.1, 0.2, 0.3, 0.4, 0.5])
        t_high_range = sweep_cfg.get("T_high_range", [0.5, 0.6, 0.7, 0.8, 0.9])

        df = evaluator.sweep_cascade(
            test_loader,
            T_low_range=t_low_range,
            T_high_range=t_high_range,
            anchor_loader=anchor_loader,
        )
        print("\n--- Sweep Results (top 10 by recall) ---")
        print(df.head(10).to_string())

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved full sweep to {args.output}")
    else:
        result = evaluator.evaluate(
            test_loader,
            anchor_loader=anchor_loader,
            T_low=t_low,
            T_high=t_high,
        )
        report = CascadeEvaluator.generate_report(result)
        print(report)
