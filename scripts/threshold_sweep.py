#!/usr/bin/env python3
"""
Agent-Gate: Threshold sweep and calibration script.
Sweeps T_low and T_high for gate model and evaluates cascade performance.
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, recall_score, precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def sweep_gate_thresholds(scores, labels, thresholds=None):
    """
    Sweep gate threshold and compute metrics at each point.

    Args:
        scores: np.array of p_gate(anomaly) scores
        labels: np.array of true labels (0=normal, 1=anomaly)
        thresholds: list of thresholds to try

    Returns:
        DataFrame with columns: threshold, recall, precision, f1, fpr, accuracy,
                                 heatmap_call_rate (fraction above threshold)
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    results = []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
        heatmap_call_rate = np.sum(preds == 1) / len(preds) if len(preds) > 0 else 0.0

        results.append({
            'threshold': round(t, 3),
            'recall': round(recall, 4),
            'precision': round(precision, 4),
            'f1': round(f1, 4),
            'fpr': round(fpr, 4),
            'accuracy': round(accuracy, 4),
            'heatmap_call_rate': round(heatmap_call_rate, 4),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        })

    return pd.DataFrame(results)


def recommend_thresholds(sweep_df, min_recall=0.90, max_call_rate=0.50):
    """
    Recommend T_low and T_high from sweep results.

    Strategy:
    - T_high: highest threshold where recall >= min_recall
    - T_low: lowest threshold where the model is confident enough about normalcy
      (i.e., samples below T_low are very likely normal)
    - Between T_low and T_high: uncertain zone -> call heatmap

    Args:
        sweep_df: DataFrame from sweep_gate_thresholds
        min_recall: minimum anomaly recall target
        max_call_rate: max acceptable heatmap call rate

    Returns:
        dict with T_low, T_high, expected metrics
    """
    # Find T_high: highest threshold maintaining recall >= min_recall
    high_candidates = sweep_df[sweep_df['recall'] >= min_recall]
    if len(high_candidates) == 0:
        # If no threshold achieves min_recall, use the one with best recall
        best_idx = sweep_df['recall'].idxmax()
        T_high = sweep_df.loc[best_idx, 'threshold']
        print(f"WARNING: No threshold achieves recall >= {min_recall}. "
              f"Using best recall threshold: {T_high}")
    else:
        T_high = high_candidates['threshold'].max()

    # T_low: find threshold where FPR is very low (< 0.05) meaning
    # samples below this are very likely normal
    # Use the threshold that gives best F1 as a reference
    best_f1_idx = sweep_df['f1'].idxmax()
    best_f1_thresh = sweep_df.loc[best_f1_idx, 'threshold']

    # T_low should be lower than T_high
    # Strategy: T_low = threshold where 95%+ of normals are below it
    low_candidates = sweep_df[
        (sweep_df['threshold'] < T_high) &
        (sweep_df['fpr'] <= 0.10)
    ]
    if len(low_candidates) > 0:
        T_low = low_candidates['threshold'].max()
    else:
        T_low = max(0.1, T_high - 0.3)

    # Ensure T_low < T_high
    if T_low >= T_high:
        T_low = max(0.05, T_high - 0.2)

    # Calculate expected heatmap call rate at these thresholds
    # Heatmap is called when p_gate >= T_low (uncertain + positive)
    t_low_row = sweep_df.iloc[(sweep_df['threshold'] - T_low).abs().argsort()[:1]]
    expected_call_rate = t_low_row['heatmap_call_rate'].values[0]

    return {
        'T_low': round(T_low, 3),
        'T_high': round(T_high, 3),
        'best_f1_threshold': round(best_f1_thresh, 3),
        'expected_heatmap_call_rate': round(expected_call_rate, 4),
        'recall_at_T_high': round(
            sweep_df.iloc[(sweep_df['threshold'] - T_high).abs().argsort()[:1]]['recall'].values[0], 4),
        'precision_at_T_high': round(
            sweep_df.iloc[(sweep_df['threshold'] - T_high).abs().argsort()[:1]]['precision'].values[0], 4),
    }


def plot_threshold_sweep(sweep_df, save_path, title="Gate Threshold Sweep"):
    """Plot threshold sweep results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Recall, Precision, F1 vs Threshold
    ax = axes[0]
    ax.plot(sweep_df['threshold'], sweep_df['recall'], 'r-', label='Recall', linewidth=2)
    ax.plot(sweep_df['threshold'], sweep_df['precision'], 'b-', label='Precision', linewidth=2)
    ax.plot(sweep_df['threshold'], sweep_df['f1'], 'g--', label='F1', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Heatmap Call Rate vs Threshold
    ax = axes[1]
    ax.plot(sweep_df['threshold'], sweep_df['heatmap_call_rate'], 'purple', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Heatmap Call Rate')
    ax.set_title('Heatmap Call Rate vs Threshold')
    ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='50% budget')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Recall vs Heatmap Call Rate (trade-off)
    ax = axes[2]
    ax.plot(sweep_df['heatmap_call_rate'], sweep_df['recall'], 'ko-', markersize=2)
    ax.set_xlabel('Heatmap Call Rate')
    ax.set_ylabel('Anomaly Recall')
    ax.set_title('Recall vs Call Rate Trade-off')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved threshold sweep plot: {save_path}")


def calibrate_scores_isotonic(val_scores, val_labels, test_scores):
    """
    Calibrate gate scores using isotonic regression.

    Args:
        val_scores: validation set scores
        val_labels: validation set labels
        test_scores: test set scores to calibrate

    Returns:
        calibrated test scores
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(val_scores, val_labels)
    calibrated = iso.predict(test_scores)
    return calibrated, iso


def plot_calibration_curve(labels, scores_before, scores_after, save_path, n_bins=10):
    """Plot calibration curves before and after calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, scores, title in zip(axes, [scores_before, scores_after],
                                  ['Before Calibration', 'After Calibration']):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = []
        bin_true = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (scores >= lo) & (scores < hi)
            if mask.sum() > 0:
                bin_means.append(scores[mask].mean())
                bin_true.append(labels[mask].mean())
            else:
                bin_means.append(np.nan)
                bin_true.append(np.nan)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.plot(bin_means, bin_true, 'ro-', label='Model')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plot: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold sweep for gate model")
    parser.add_argument('--round', type=int, default=1, help='Round number')
    parser.add_argument('--gate', type=str, default='effnetb0', choices=['effnetb0', 'mnv3_large'])
    parser.add_argument('--scores-file', type=str, help='Path to saved gate scores (npz)')
    parser.add_argument('--out', type=str, default=os.path.join(PROJECT_ROOT, 'artifacts'))
    parser.add_argument('--assets', type=str, default=os.path.join(PROJECT_ROOT, 'reports', 'assets'))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.assets, exist_ok=True)

    if args.scores_file and os.path.exists(args.scores_file):
        data = np.load(args.scores_file)
        scores = data['scores']
        labels = data['labels']
    else:
        print("No scores file provided. Run train_gate.py first to generate scores.")
        sys.exit(1)

    # Sweep
    sweep_df = sweep_gate_thresholds(scores, labels)
    sweep_path = os.path.join(args.out, f"threshold_sweep_round{args.round}_{args.gate}.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved sweep results: {sweep_path}")

    # Recommend
    recs = recommend_thresholds(sweep_df)
    print(f"\nRecommended thresholds:")
    print(json.dumps(recs, indent=2))

    rec_path = os.path.join(args.out, f"recommended_thresholds_round{args.round}_{args.gate}.json")
    with open(rec_path, 'w') as f:
        json.dump(recs, f, indent=2)

    # Plot
    plot_path = os.path.join(args.assets, f"threshold_sweep_round{args.round}_{args.gate}.png")
    plot_threshold_sweep(sweep_df, plot_path,
                         title=f"Gate Threshold Sweep - Round {args.round} ({args.gate})")

    print("\nDone!")
