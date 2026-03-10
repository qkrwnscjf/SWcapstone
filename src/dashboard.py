#!/usr/bin/env python3
"""
Agent-Integration: Simple dashboard for monitoring cascade pipeline.
Reads metrics from MLflow and displays summary.
Can be run as a standalone script or imported.
"""
import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_metrics_from_artifacts(artifacts_dir):
    """Load metrics from artifact JSON files."""
    metrics = {}
    for fname in os.listdir(artifacts_dir):
        if fname.endswith('.json'):
            with open(os.path.join(artifacts_dir, fname), 'r') as f:
                data = json.load(f)
                metrics[fname.replace('.json', '')] = data
    return metrics


def generate_dashboard_text(metrics, round_num=1):
    """Generate a text-based dashboard summary."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  ANOMALY DETECTION PIPELINE DASHBOARD - Round {round_num}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Gate metrics
    for key in sorted(metrics.keys()):
        if 'gate' in key.lower() or 'threshold' in key.lower():
            lines.append(f"\n--- {key} ---")
            data = metrics[key]
            if isinstance(data, dict):
                for k, v in data.items():
                    lines.append(f"  {k}: {v}")

    # Heatmap metrics
    for key in sorted(metrics.keys()):
        if 'heatmap' in key.lower() or 'patchcore' in key.lower():
            lines.append(f"\n--- {key} ---")
            data = metrics[key]
            if isinstance(data, dict):
                for k, v in data.items():
                    lines.append(f"  {k}: {v}")

    # Cascade metrics
    for key in sorted(metrics.keys()):
        if 'cascade' in key.lower():
            lines.append(f"\n--- {key} ---")
            data = metrics[key]
            if isinstance(data, dict):
                for k, v in data.items():
                    lines.append(f"  {k}: {v}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def generate_html_dashboard(metrics, round_num=1, output_path=None):
    """Generate an HTML dashboard."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Detection Dashboard - Round {round_num}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h2 {{ color: #007bff; margin-top: 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .metric-value {{ font-size: 1.2em; font-weight: bold; color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔍 Anomaly Detection Pipeline - Round {round_num}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

    for key in sorted(metrics.keys()):
        data = metrics[key]
        html += f'<div class="card"><h2>{key}</h2><table>'
        if isinstance(data, dict):
            for k, v in data.items():
                css_class = ''
                if isinstance(v, (int, float)):
                    if 'recall' in k.lower() and v < 0.8:
                        css_class = 'danger'
                    elif 'rate' in k.lower() and v > 0.5:
                        css_class = 'warning'
                html += f'<tr><th>{k}</th><td class="{css_class}">{v}</td></tr>'
        html += '</table></div>'

    html += """
</div>
</body>
</html>"""

    if output_path:
        with open(output_path, 'w') as f:
            f.write(html)
        print(f"Dashboard saved: {output_path}")

    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pipeline dashboard")
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--artifacts-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'artifacts'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--format', choices=['text', 'html'], default='text')
    args = parser.parse_args()

    if not os.path.exists(args.artifacts_dir):
        print(f"Artifacts directory not found: {args.artifacts_dir}")
        sys.exit(1)

    metrics = load_metrics_from_artifacts(args.artifacts_dir)

    if args.format == 'html':
        output_path = args.output or os.path.join(PROJECT_ROOT, 'reports', f'dashboard_round{args.round}.html')
        generate_html_dashboard(metrics, args.round, output_path)
    else:
        text = generate_dashboard_text(metrics, args.round)
        print(text)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(text)
