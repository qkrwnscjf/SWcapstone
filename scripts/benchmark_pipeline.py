#!/usr/bin/env python3
"""
Benchmark Baseline (PatchCore-only) vs Gate-Cascade pipelines
across multiple dataset versions (e.g. v1, v2, v3 or round1/2/3).

For each (version, pipeline) pair, computes:
  * Accuracy / Precision / Recall / F1
  * Inference time per image: mean, std, median, p50, p95, total
  * Heatmap call rate (cascade only)

Usage:
  python scripts/benchmark_pipeline.py \
      --versions v1 v2 v3 \
      --test-csv splits/round1_test_mix.csv splits/round2_test_mix.csv splits/round3_test_mix.csv \
      --gate-model models/round1_effnetb0_gate.pt models/round2_effnetb0_gate.pt models/round3_effnetb0_gate.pt \
      --gate-calib models/round1_effnetb0_calibrator.pkl models/round2_effnetb0_calibrator.pkl models/round3_effnetb0_calibrator.pkl \
      --patchcore-model models/round1_patchcore_r18_patchcore.pt models/round2_patchcore_r18_patchcore.pt models/round3_patchcore_r18_patchcore.pt \
      --data-root-old /app/data --data-root-new ./data \
      --t-low 0.3 --t-high 0.7 \
      --device cpu \
      --output reports/benchmark_v1v2v3.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.gate_model import GateModel
from backend.src.heatmap_model import PatchCoreModel


class IsotonicCalibrator:
    """Stub matching the calibrator class pickled into models/*_calibrator.pkl.

    The training script defined a user-class with attributes
        - ``method``: 'isotonic' or 'platt'
        - ``ir``: ``sklearn.isotonic.IsotonicRegression`` (or LR for platt)

    GateModel.predict and predict_batch call ``calibrator.predict(arr)``
    expecting a numpy array of the same shape back. Unpickling needs the
    class on this module's namespace.
    """

    def __init__(self) -> None:
        self.method: Optional[str] = None
        self.ir = None
        self.calibrator = None

    def predict(self, x):
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        model = self.ir if self.ir is not None else self.calibrator
        if model is None:
            return arr
        out = model.predict(arr)
        return np.asarray(out).reshape(-1)


# Register on __main__ as well — that is the module path embedded in the pickle.
sys.modules["__main__"].IsotonicCalibrator = IsotonicCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("benchmark")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform(input_size: int = 224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def remap_path(path: str, old_prefix: Optional[str], new_prefix: Optional[str]) -> str:
    if old_prefix and new_prefix and path.startswith(old_prefix):
        return new_prefix + path[len(old_prefix):]
    return path


def load_split(csv_path: str, old_prefix: Optional[str], new_prefix: Optional[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapped = remap_path(row["path"], old_prefix, new_prefix)
            samples.append({
                "path": mapped,
                "label": 1 if row["label"] == "anomaly" else 0,
                "dataset_type": row.get("dataset_type", ""),
                "defect_type": row.get("defect_type", ""),
            })
    return samples


def filter_existing(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = [s for s in samples if os.path.exists(s["path"])]
    missing = len(samples) - len(kept)
    if missing > 0:
        logger.warning("%d/%d samples missing on disk and will be skipped", missing, len(samples))
    return kept


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "p95": 0.0, "total": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "total": float(arr.sum()),
    }


def classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    y_t = np.array(y_true, dtype=np.int32)
    y_p = np.array(y_pred, dtype=np.int32)
    tp = int(((y_p == 1) & (y_t == 1)).sum())
    fp = int(((y_p == 1) & (y_t == 0)).sum())
    fn = int(((y_p == 0) & (y_t == 1)).sum())
    tn = int(((y_p == 0) & (y_t == 0)).sum())
    n = max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / n
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def find_baseline_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Pick the threshold (over PatchCore raw anomaly scores) that maximises F1."""
    if scores.size == 0:
        return 0.0, 0.0
    candidates = np.unique(np.concatenate([
        np.linspace(scores.min(), scores.max(), 101),
        np.quantile(scores, np.linspace(0.0, 1.0, 21)),
    ]))
    best_t, best_f1 = float(candidates[0]), -1.0
    for t in candidates:
        preds = (scores >= t).astype(np.int32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def evaluate_baseline(
    samples: List[Dict[str, Any]],
    patchcore: PatchCoreModel,
    transform,
    device: str,
    fixed_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """PatchCore-only pipeline. Threshold is selected on the test set itself
    if not supplied (oracle upper bound) — pass fixed_threshold to evaluate
    deterministically."""
    scores: List[float] = []
    latencies: List[float] = []
    labels: List[int] = []

    for s in samples:
        try:
            image = Image.open(s["path"]).convert("RGB")
        except Exception as e:
            logger.warning("skip %s: %s", s["path"], e)
            continue
        tensor = transform(image).unsqueeze(0)

        t0 = time.perf_counter()
        out = patchcore.predict(tensor.squeeze(0))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize() if hasattr(torch.mps, "synchronize") else None
        dt = (time.perf_counter() - t0) * 1000.0  # ms

        scores.append(float(out["anomaly_score"]))
        latencies.append(dt)
        labels.append(s["label"])

    scores_np = np.array(scores)
    labels_np = np.array(labels)

    if fixed_threshold is None:
        threshold, _ = find_baseline_threshold(scores_np, labels_np)
    else:
        threshold = fixed_threshold

    preds = (scores_np >= threshold).astype(np.int32).tolist()
    metrics = classification_metrics(labels, preds)
    metrics["threshold"] = threshold
    metrics["latency_ms"] = stats(latencies)
    metrics["pipeline"] = "baseline_patchcore"
    return metrics


def evaluate_cascade(
    samples: List[Dict[str, Any]],
    gate: GateModel,
    patchcore: PatchCoreModel,
    transform,
    device: str,
    t_low: float,
    t_high: float,
    patchcore_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Gate → PatchCore cascade.

    Decision rules:
      * p_gate < T_low  -> normal (no heatmap call)
      * p_gate > T_high -> anomaly (no heatmap call)
      * T_low <= p_gate <= T_high -> uncertain → call PatchCore, use its
        threshold to decide.
    """
    latencies: List[float] = []
    preds: List[int] = []
    labels: List[int] = []
    heatmap_calls = 0

    # We need a PatchCore decision threshold. If not provided, compute one
    # from images falling into the uncertain band by sweeping F1.
    pc_scores_uncertain: List[float] = []
    pc_labels_uncertain: List[float] = []
    decisions: List[Tuple[str, float, Optional[float]]] = []  # (route, p_gate, pc_score)

    # First pass: route + collect PatchCore scores for uncertain samples.
    for s in samples:
        try:
            image = Image.open(s["path"]).convert("RGB")
        except Exception as e:
            logger.warning("skip %s: %s", s["path"], e)
            continue
        tensor = transform(image)

        t0 = time.perf_counter()
        p_gate = gate.predict(tensor)
        pc_score: Optional[float] = None
        if p_gate < t_low:
            route = "normal_short_circuit"
        elif p_gate > t_high:
            route = "anomaly_short_circuit"
        else:
            route = "heatmap"
            out = patchcore.predict(tensor)
            pc_score = float(out["anomaly_score"])
            heatmap_calls += 1
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0

        latencies.append(dt)
        labels.append(s["label"])
        decisions.append((route, p_gate, pc_score))

        if route == "heatmap":
            pc_scores_uncertain.append(pc_score)
            pc_labels_uncertain.append(s["label"])

    # Determine PatchCore threshold for the uncertain band.
    if patchcore_threshold is None and pc_scores_uncertain:
        thr, _ = find_baseline_threshold(
            np.array(pc_scores_uncertain), np.array(pc_labels_uncertain)
        )
    else:
        thr = patchcore_threshold if patchcore_threshold is not None else 0.0

    for route, p_gate, pc_score in decisions:
        if route == "normal_short_circuit":
            preds.append(0)
        elif route == "anomaly_short_circuit":
            preds.append(1)
        else:
            preds.append(int(pc_score >= thr))

    metrics = classification_metrics(labels, preds)
    metrics["t_low"] = t_low
    metrics["t_high"] = t_high
    metrics["patchcore_threshold"] = thr
    metrics["heatmap_call_rate"] = heatmap_calls / max(len(labels), 1)
    metrics["latency_ms"] = stats(latencies)
    metrics["pipeline"] = "gate_cascade"
    return metrics


def run_version(
    version_name: str,
    test_csv: str,
    gate_model_path: str,
    gate_calib_path: Optional[str],
    patchcore_model_path: str,
    args,
    transform,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("Evaluating version: %s", version_name)
    logger.info("  test_csv:        %s", test_csv)
    logger.info("  gate_model:      %s", gate_model_path)
    logger.info("  gate_calibrator: %s", gate_calib_path)
    logger.info("  patchcore_model: %s", patchcore_model_path)

    samples = load_split(test_csv, args.data_root_old, args.data_root_new)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(samples)
        logger.info("  shuffled samples (seed=%d)", args.seed)
    if args.limit:
        samples = samples[: args.limit]
    samples = filter_existing(samples)
    n_anom = sum(1 for s in samples if s["label"] == 1)
    n_norm = len(samples) - n_anom
    logger.info("  class distribution: anomaly=%d  normal=%d", n_anom, n_norm)
    if not samples:
        logger.error("No samples available for version %s; skipping.", version_name)
        return {"version": version_name, "error": "no samples"}
    logger.info("  loaded %d samples", len(samples))

    # Load models fresh per version
    gate = GateModel.load(gate_model_path, device=args.device)
    if gate_calib_path and os.path.exists(gate_calib_path) and gate.calibrator is None:
        with open(gate_calib_path, "rb") as f:
            payload = pickle.load(f)
        # Training stored either {method, calibrator} dict or a calibrator instance.
        if isinstance(payload, dict) and "calibrator" in payload:
            gate.calibrator = payload["calibrator"]
        else:
            gate.calibrator = payload
        logger.info("  attached external calibrator from %s (%s)", gate_calib_path, type(gate.calibrator).__name__)

    # Load PatchCore on CPU first because the saved memory_bank is a torch
    # tensor that breaks sklearn NearestNeighbors.fit on a non-CPU device.
    patchcore = PatchCoreModel.load(patchcore_model_path, device="cpu")
    if torch.is_tensor(patchcore._memory_bank):
        patchcore._memory_bank = patchcore._memory_bank.detach().cpu().numpy()
        from sklearn.neighbors import NearestNeighbors
        patchcore._nn_index = NearestNeighbors(
            n_neighbors=patchcore.k_neighbors, metric="euclidean", algorithm="auto"
        ).fit(patchcore._memory_bank)
    if args.device and args.device != "cpu":
        patchcore._feature_extractor.to(args.device)
        patchcore.device = args.device

    runs_baseline: List[Dict[str, Any]] = []
    runs_cascade: List[Dict[str, Any]] = []

    for r in range(args.runs):
        logger.info("  run %d/%d", r + 1, args.runs)
        b = evaluate_baseline(samples, patchcore, transform, args.device)
        c = evaluate_cascade(
            samples, gate, patchcore, transform, args.device,
            t_low=args.t_low, t_high=args.t_high,
        )
        runs_baseline.append(b)
        runs_cascade.append(c)

    def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = ["accuracy", "precision", "recall", "f1"]
        agg = {k: stats([r[k] for r in runs]) for k in keys}
        agg["latency_ms_mean_per_image"] = stats([r["latency_ms"]["mean"] for r in runs])
        agg["latency_ms_total"] = stats([r["latency_ms"]["total"] for r in runs])
        agg["latency_ms_std_per_image_avg"] = float(np.mean([r["latency_ms"]["std"] for r in runs]))
        return agg

    return {
        "version": version_name,
        "n_samples": len(samples),
        "runs": args.runs,
        "baseline": {
            "per_run": runs_baseline,
            "aggregate": aggregate(runs_baseline),
        },
        "gate_cascade": {
            "per_run": runs_cascade,
            "aggregate": aggregate(runs_cascade),
        },
    }


def write_markdown(results: List[Dict[str, Any]], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# Pipeline Benchmark: Baseline (PatchCore) vs Gate-Cascade")
    lines.append("")
    lines.append("| version | n | pipeline | accuracy (mean ± std) | precision | recall | F1 | latency/img ms (mean) | latency std | total ms |")
    lines.append("|---|---:|---|---|---|---|---|---:|---:|---:|")
    for r in results:
        if "error" in r:
            lines.append(f"| {r['version']} | - | error | {r['error']} | | | | | | |")
            continue
        for kind in ("baseline", "gate_cascade"):
            agg = r[kind]["aggregate"]
            acc = agg["accuracy"]; pr = agg["precision"]; rc = agg["recall"]; f1 = agg["f1"]
            lat = agg["latency_ms_mean_per_image"]
            lat_std = agg["latency_ms_std_per_image_avg"]
            tot = agg["latency_ms_total"]
            lines.append(
                f"| {r['version']} | {r['n_samples']} | {kind} "
                f"| {acc['mean']:.4f} ± {acc['std']:.4f} "
                f"| {pr['mean']:.4f} ± {pr['std']:.4f} "
                f"| {rc['mean']:.4f} ± {rc['std']:.4f} "
                f"| {f1['mean']:.4f} ± {f1['std']:.4f} "
                f"| {lat['mean']:.2f} "
                f"| {lat_std:.2f} "
                f"| {tot['mean']:.0f} |"
            )
    lines.append("")
    lines.append("> latency std는 각 run 안에서의 per-image latency 표준편차의 평균이며,")
    lines.append("> accuracy/precision/recall/F1의 std는 run 간의 표준편차입니다.")
    out_md.write_text("\n".join(lines) + "\n")
    logger.info("Markdown report written to %s", out_md)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--versions", nargs="+", required=True, help="Version labels (e.g. v1 v2 v3).")
    p.add_argument("--test-csv", nargs="+", required=True, help="One CSV per version.")
    p.add_argument("--gate-model", nargs="+", required=True, help="One gate model per version.")
    p.add_argument("--gate-calib", nargs="*", default=[], help="Optional calibrator per version.")
    p.add_argument("--patchcore-model", nargs="+", required=True, help="One patchcore model per version.")
    p.add_argument("--data-root-old", default="/app/data", help="Path prefix in CSV that should be remapped.")
    p.add_argument("--data-root-new", default=None, help="Local replacement for --data-root-old.")
    p.add_argument("--t-low", type=float, default=0.3)
    p.add_argument("--t-high", type=float, default=0.7)
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto if omitted).")
    p.add_argument("--runs", type=int, default=3, help="Repeated runs to compute mean/std.")
    p.add_argument("--limit", type=int, default=0, help="Limit samples per version (0 = all).")
    p.add_argument("--shuffle", action="store_true", help="Shuffle CSV rows before applying --limit (recommended when CSV is class-grouped).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for --shuffle.")
    p.add_argument("--output", default="reports/benchmark_pipeline.json")
    args = p.parse_args()

    n = len(args.versions)
    for name, lst in [("test-csv", args.test_csv), ("gate-model", args.gate_model), ("patchcore-model", args.patchcore_model)]:
        if len(lst) != n:
            p.error(f"--{name} must have exactly {n} entries (one per version)")
    if args.gate_calib and len(args.gate_calib) != n:
        p.error(f"--gate-calib must have exactly {n} entries (or be omitted)")

    transform = get_eval_transform(224)

    results: List[Dict[str, Any]] = []
    for i, version in enumerate(args.versions):
        gc = args.gate_calib[i] if i < len(args.gate_calib) else None
        try:
            r = run_version(
                version_name=version,
                test_csv=args.test_csv[i],
                gate_model_path=args.gate_model[i],
                gate_calib_path=gc,
                patchcore_model_path=args.patchcore_model[i],
                args=args,
                transform=transform,
            )
        except Exception as e:
            logger.exception("version %s failed: %s", version, e)
            r = {"version": version, "error": str(e)}
        results.append(r)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "results": results,
    }, indent=2, default=str))
    logger.info("JSON results written to %s", out)

    write_markdown(results, out.with_suffix(".md"))

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"[{r['version']}] ERROR: {r['error']}")
            continue
        for kind in ("baseline", "gate_cascade"):
            a = r[kind]["aggregate"]
            print(
                f"[{r['version']:<6}] {kind:<14}  "
                f"acc={a['accuracy']['mean']:.4f}±{a['accuracy']['std']:.4f}  "
                f"f1={a['f1']['mean']:.4f}±{a['f1']['std']:.4f}  "
                f"lat={a['latency_ms_mean_per_image']['mean']:.2f}±{a['latency_ms_std_per_image_avg']:.2f} ms"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
