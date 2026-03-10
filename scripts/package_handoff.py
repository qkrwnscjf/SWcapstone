#!/usr/bin/env python3
"""
Agent-DataOps + Integration: Package handoff datasets for MLOps owner.
Creates zip files for Round 2 and Round 3 with the expected data/raw/ structure.
"""
import os
import sys
import csv
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS_DIR = PROJECT_ROOT / "splits"
HANDOFF_DIR = PROJECT_ROOT / "handoff" / "mlops_owner"


def package_round(round_num, output_zip_path):
    """
    Package data for a given round into a zip with the MLOps-expected structure:

    data/raw/{dataset_type}/{defect_type}/{label}/image.ext
    splits/roundN_*.csv
    RUN_INSTRUCTIONS.txt
    """
    temp_dir = HANDOFF_DIR / f"_temp_round{round_num}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Create directory structure
    data_dir = temp_dir / "data" / "raw"
    splits_out = temp_dir / "splits"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_out.mkdir(parents=True, exist_ok=True)

    # Collect all CSVs for this round
    round_csvs = list(SPLITS_DIR.glob(f"round{round_num}_*.csv"))
    anchor_csvs = list(SPLITS_DIR.glob("anchor_*.csv"))

    # Copy relevant split CSVs
    for csv_file in round_csvs + anchor_csvs:
        shutil.copy2(csv_file, splits_out / csv_file.name)

    # Copy actual image files into data/raw structure
    all_paths = set()
    for csv_file in round_csvs:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_paths.add((row['path'], row['dataset_type'], row['defect_type'], row['label']))

    print(f"  Round {round_num}: {len(all_paths)} unique images")

    # Rewrite CSVs with relative paths
    for csv_file in round_csvs:
        rows = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create relative path inside package
                ext = os.path.splitext(row['path'])[1]
                fname = os.path.basename(row['path'])
                rel_path = f"data/raw/{row['dataset_type']}/{row['defect_type']}/{row['label']}/{fname}"
                row_new = dict(row)
                row_new['path'] = rel_path
                rows.append(row_new)

        # Write updated CSV to package
        with open(splits_out / csv_file.name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["path", "dataset_type", "defect_type", "label", "round", "split"])
            writer.writeheader()
            writer.writerows(rows)

    # Also rewrite anchor CSVs with relative paths
    for csv_file in anchor_csvs:
        rows = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = os.path.basename(row['path'])
                rel_path = f"data/raw/{row['dataset_type']}/{row['defect_type']}/{row['label']}/{fname}"
                row_new = dict(row)
                row_new['path'] = rel_path
                rows.append(row_new)

        with open(splits_out / csv_file.name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["path", "dataset_type", "defect_type", "label", "round", "split"])
            writer.writeheader()
            writer.writerows(rows)

    # Copy image files
    for path, ds, dt, label in all_paths:
        fname = os.path.basename(path)
        dst_dir = data_dir / ds / dt / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / fname
        if not dst.exists():
            shutil.copy2(path, dst)

    # Also copy anchor images
    for csv_file in anchor_csvs:
        with open(SPLITS_DIR / csv_file.name, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = os.path.basename(row['path'])
                dst_dir = data_dir / row['dataset_type'] / row['defect_type'] / row['label']
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / fname
                if not dst.exists():
                    shutil.copy2(row['path'], dst)

    # Write RUN_INSTRUCTIONS.txt
    instructions = f"""# Round {round_num} Dataset - Run Instructions

## Setup
1. Extract this zip to your project root
2. Ensure the data/ directory is at the project root level

## Directory Structure
```
data/raw/
  Kolektor/surface_defect/{{normal,anomaly}}/
  MVTec/{{grid,metal_nut,tile}}/{{normal,anomaly}}/
  NEU/{{Crazing,Inclusion,Patches,Pitted,Rolled,Scratches}}/anomaly/
splits/
  round{round_num}_train_normal.csv
  round{round_num}_val_mix.csv
  round{round_num}_test_mix.csv
  round{round_num}_train_gate_mix.csv
  round{round_num}_val_gate.csv
  round{round_num}_test_gate.csv
  anchor_test_mix.csv
  anchor_neu_test.csv
```

## Commands
```bash
# Train PatchCore heatmap model
python src/train.py --round {round_num} --heatmap patchcore_r18 --device mps

# Train Gate model (EfficientNet-B0)
python src/train_gate.py --round {round_num} --gate effnetb0 --device mps

# Run threshold sweep
python scripts/threshold_sweep.py --round {round_num} --gate effnetb0

# Run cascade evaluation (after both models trained)
python src/run_cascade.py --round {round_num} --gate effnetb0 --heatmap patchcore_r18
```

## Expected Outputs
- models/gate_round{round_num}_effnetb0.pth
- models/heatmap_round{round_num}_patchcore_r18.pkl
- artifacts/threshold_sweep_round{round_num}_effnetb0.csv
- artifacts/cascade_metrics_round{round_num}.json
- reports/assets/*.png (plots and overlays)

## Notes
- CSV paths are relative to the package root
- NEU dataset is anomaly-only; do not compute AUROC on anchor_neu_test alone
- PatchCore trains on normal-only (train_normal split)
- Gate trains on mixed data (train_gate_mix split)
"""
    with open(temp_dir / "RUN_INSTRUCTIONS.txt", 'w') as f:
        f.write(instructions)

    # Create zip
    print(f"  Creating zip: {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zf.write(file_path, arcname)

    # Cleanup temp
    shutil.rmtree(temp_dir)
    zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"  Done! Size: {zip_size_mb:.1f} MB")


if __name__ == "__main__":
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)

    print("Packaging Round 2 dataset...")
    package_round(2, HANDOFF_DIR / "round2_dataset.zip")

    print("\nPackaging Round 3 dataset...")
    package_round(3, HANDOFF_DIR / "round3_dataset.zip")

    print("\n✅ All handoff packages created!")
