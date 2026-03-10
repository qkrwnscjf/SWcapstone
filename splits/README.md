# Data Split Policy

## Overview
3-round incremental split strategy for anomaly detection pipeline.

## Hierarchy
```
dataset_type / defect_type / label
```

| Dataset  | Defect Types | Labels | Notes |
|----------|-------------|--------|-------|
| Kolektor | surface_defect | normal, anomaly | Industrial metal surface |
| MVTec    | grid, metal_nut, tile | normal, anomaly | Benchmark subset |
| NEU      | Crazing, Inclusion, Patches, Pitted, Rolled, Scratches | anomaly only | **No normal images** |

## Split Strategy

### Anchor Set (Fixed)
- **anchor_test_mix.csv**: 20% of Kolektor + MVTec, stratified by (dataset_type, defect_type, label). Contains both normal and anomaly. Used as a **fixed benchmark** across all rounds.
- **anchor_neu_test.csv**: 20% of NEU (anomaly-only). Used for **sanity checks and overlay visualization only**. No AUROC claims (single-class).

### Round Pools
Remaining 80% split into 3 equal pools (stratified):
- **round1_pool.csv**: ~1/3 of remaining data
- **round2_pool.csv**: ~1/3
- **round3_pool.csv**: ~1/3 (+ remainder)

### Per-Round Splits (Cumulative)
Each round uses **cumulative** data (round N includes pools 1..N):

#### Heatmap Splits (PatchCore)
| Split | Content | Purpose |
|-------|---------|---------|
| `roundN_train_normal.csv` | 70% of cumulative normals | PatchCore training (normal-only) |
| `roundN_val_mix.csv` | 15% normals + 30% anomalies | Validation (mixed) |
| `roundN_test_mix.csv` | 15% normals + 70% anomalies | Testing (mixed) |

#### Gate Splits (Binary Classifier)
| Split | Content | Purpose |
|-------|---------|---------|
| `roundN_train_gate_mix.csv` | 70% normal + 70% anomaly | Gate training (both classes required) |
| `roundN_val_gate.csv` | 15% normal + 15% anomaly | Gate validation + calibration |
| `roundN_test_gate.csv` | 15% normal + 15% anomaly | Gate testing + threshold sweep |

## CSV Columns
```
path, dataset_type, defect_type, label, round, split
```

## Validation Guarantees
- ✅ No duplicates across heatmap splits within each round
- ✅ No overlap between anchor and any round pool
- ✅ train_normal contains ONLY normal samples
- ✅ Gate train contains BOTH normal and anomaly
- ✅ All file paths verified to exist on disk
- ✅ Stratified by (dataset_type, defect_type, label)

## Global Seed
All splits generated with **seed=42** for reproducibility.

## Important Notes
- **NEU data is anomaly-only**: NEU samples appear only in anomaly portions of splits and in anchor_neu_test.
- **No AUROC on anchor_neu_test**: Single-class set; use only for qualitative sanity checks.
- **Rounds are cumulative**: Round 2 training includes Round 1 + Round 2 pool data.
