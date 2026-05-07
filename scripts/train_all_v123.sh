#!/bin/bash
# Sequential training of v1, v2, v3 — same hyperparameters across versions.
#
# Hyperparameters chosen from smoke run:
#   PatchCore: coreset_ratio=0.005 (gives ~32K memory bank for v1)
#   Gate: epochs=3, batch=16, lr=1e-3, AdamW, cosine scheduler
#
# Output naming:
#   models/v{N}_patchcore_r18_patchcore.pt
#   models/v{N}_effnetb0_gate.pt
#   models/v{N}_effnetb0_calibrator.pkl
#   models/v{N}_*_config.json
#   logs/v{N}_{patchcore,gate}.log

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs models

VERSIONS=("v1" "v2" "v3")

for V in "${VERSIONS[@]}"; do
  echo "=================================================================="
  echo "  $V — PatchCore"
  echo "=================================================================="
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/train_patchcore.py \
    --tag "$V" \
    --heatmap patchcore_r18 \
    --device mps \
    --coreset-ratio 0.005 \
    2>&1 | tee "logs/${V}_patchcore.log"

  echo "=================================================================="
  echo "  $V — Gate"
  echo "=================================================================="
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/train_gate.py \
    --tag "$V" \
    --gate effnetb0 \
    --device mps \
    --epochs 3 \
    --batch_size 16 \
    --lr 1e-3 \
    --optimizer AdamW \
    --weight_decay 1e-4 \
    --scheduler cosine \
    2>&1 | tee "logs/${V}_gate.log"

  echo ""
  echo "  $V complete. Models:"
  ls -la "models/${V}_"* 2>/dev/null || echo "    (no models found — check logs)"
  echo ""
done

echo "=================================================================="
echo "All versions trained. Run benchmark with:"
echo "  scripts/eval_v123.sh"
echo "=================================================================="
