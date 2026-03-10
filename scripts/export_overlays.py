#!/usr/bin/env python3
"""
Agent-Heatmap: Export heatmap overlay visualizations.
Generates overlay images for anomaly samples showing PatchCore heatmaps.
"""
import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def create_heatmap_overlay(original_image, heatmap, alpha=0.4):
    """
    Create a heatmap overlay on the original image.

    Args:
        original_image: numpy array HxWx3 (RGB, 0-255)
        heatmap: numpy array HxW (0-1 normalized)
        alpha: overlay transparency

    Returns:
        overlay: numpy array HxWx3
    """
    # Resize heatmap to match image
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (1 - alpha) * original_image.astype(np.float32) + alpha * heatmap_colored.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def create_comparison_figure(original, heatmap, overlay, score, path_info, save_path):
    """
    Create a side-by-side comparison figure.

    Args:
        original: HxWx3 RGB image
        heatmap: HxW normalized heatmap
        overlay: HxWx3 overlay image
        score: anomaly score
        path_info: string describing the image source
        save_path: where to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'Heatmap (score: {score:.4f})', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')

    plt.suptitle(f'PatchCore Anomaly Detection\n{path_info}', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def export_overlays_from_results(results, output_dir, max_samples=20):
    """
    Export overlay visualizations from prediction results.

    Args:
        results: list of dicts with keys: path, original, heatmap, score, label
        output_dir: directory to save overlays
        max_samples: maximum number of overlays to generate
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort by score descending (most anomalous first)
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Take top anomalies and some normals
    anomalies = [r for r in sorted_results if r.get('label', 1) == 1][:max_samples // 2]
    normals = [r for r in sorted_results if r.get('label', 0) == 0][:max_samples // 2]

    for i, result in enumerate(anomalies + normals):
        label_str = 'anomaly' if result.get('label', 1) == 1 else 'normal'
        overlay = create_heatmap_overlay(result['original'], result['heatmap'])

        filename = f"overlay_{label_str}_{i:03d}.png"
        save_path = os.path.join(output_dir, filename)

        path_info = os.path.basename(result.get('path', 'unknown'))
        create_comparison_figure(
            result['original'],
            result['heatmap'],
            overlay,
            result['score'],
            f"{label_str} | {path_info}",
            save_path,
        )
        print(f"  Saved: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export heatmap overlays")
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--results-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'artifacts'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(PROJECT_ROOT, 'reports', 'assets'))
    parser.add_argument('--max-samples', type=int, default=20)
    args = parser.parse_args()

    results_path = os.path.join(args.results_dir, f"heatmap_results_round{args.round}.npz")
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Run train.py first to generate heatmap results.")
        sys.exit(1)

    data = np.load(results_path, allow_pickle=True)
    results = data['results'].tolist()

    print(f"Exporting overlays for {len(results)} results...")
    export_overlays_from_results(results, args.out, max_samples=args.max_samples)
    print("Done!")
