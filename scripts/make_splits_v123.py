#!/usr/bin/env python3
"""
Per-version stratified split generator for data/split_versions/v{1,2,3}/.

Each version has its own pool (pretrain ∪ additional) and gets independent
70/15/15 train/val/test splits — versions are NOT cumulative (unlike round-based
make_splits.py).

Outputs (per version):
  splits/v{N}_train_normal.csv     normal-only, 70%
  splits/v{N}_val_mix.csv          15% normals + 30% anomalies
  splits/v{N}_test_mix.csv         15% normals + 70% anomalies
  splits/v{N}_train_gate_mix.csv   70% stratified by label (gate train)
  splits/v{N}_val_gate.csv         15% stratified
  splits/v{N}_test_gate.csv        15% stratified
  splits/v{N}_pool.csv             full pre-split pool (audit artifact)

CSV schema: path,dataset_type,defect_type,label,round,split
  (the "round" column carries the version tag (v1/v2/v3) for traceability —
   downstream loaders ignore its value.)

Stratification key: (version, label). Seed=42.

Usage:
  python scripts/make_splits_v123.py --versions v1 v2 v3 --out-dir splits
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42


def collect_records(version_root: Path, version_tag: str) -> list[dict]:
    """Walk pretrain/ ∪ additional/ for one version, return record dicts."""
    records: list[dict] = []
    for sub in ("pretrain", "additional"):
        for cls in ("anomaly", "normal"):
            d = version_root / sub / cls
            if not d.exists():
                print(f"  !! missing {d}")
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in EXTS:
                    records.append({
                        "path": str(p.resolve()),
                        "dataset_type": "Kolektor",
                        "defect_type": "surface_defect",
                        "label": cls,
                        "version": version_tag,
                        "subset": sub,
                    })
    return records


def split_for_version(items: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """Adapted from make_splits.py:split_for_round — same 70/15/15 logic.

    Heatmap splits: train_normal (70% normal), val_mix (15% normal + 30% anom),
                    test_mix (15% normal + 70% anom)
    Gate splits:    70/15/15 stratified by label.
    """
    normals = [r for r in items if r["label"] == "normal"]
    anomalies = [r for r in items if r["label"] == "anomaly"]
    rng.shuffle(normals)
    rng.shuffle(anomalies)

    n_train_n = int(len(normals) * 0.70)
    n_val_n = int(len(normals) * 0.15)
    train_normal = normals[:n_train_n]
    val_normals = normals[n_train_n:n_train_n + n_val_n]
    test_normals = normals[n_train_n + n_val_n:]

    n_val_a = max(1, int(len(anomalies) * 0.30))
    val_anomalies = anomalies[:n_val_a]
    test_anomalies = anomalies[n_val_a:]

    val_mix = val_normals + val_anomalies
    test_mix = test_normals + test_anomalies

    # Gate splits — stratified 70/15/15 by label
    gate_normals = list(normals)
    gate_anomalies = list(anomalies)
    rng.shuffle(gate_normals)
    rng.shuffle(gate_anomalies)
    gn_train = int(len(gate_normals) * 0.70)
    gn_val = int(len(gate_normals) * 0.15)
    ga_train = int(len(gate_anomalies) * 0.70)
    ga_val = int(len(gate_anomalies) * 0.15)

    train_gate = gate_normals[:gn_train] + gate_anomalies[:ga_train]
    val_gate = gate_normals[gn_train:gn_train + gn_val] + gate_anomalies[ga_train:ga_train + ga_val]
    test_gate = gate_normals[gn_train + gn_val:] + gate_anomalies[ga_train + ga_val:]

    rng.shuffle(train_gate)
    rng.shuffle(val_gate)
    rng.shuffle(test_gate)

    return {
        "train_normal": train_normal,
        "val_mix": val_mix,
        "test_mix": test_mix,
        "train_gate_mix": train_gate,
        "val_gate": val_gate,
        "test_gate": test_gate,
    }


def write_csv(items: list[dict], version_tag: str, split_name: str, filepath: Path) -> None:
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "dataset_type", "defect_type", "label", "round", "split"])
        w.writeheader()
        for r in items:
            w.writerow({
                "path": r["path"],
                "dataset_type": r["dataset_type"],
                "defect_type": r["defect_type"],
                "label": r["label"],
                "round": version_tag,
                "split": split_name,
            })


def validate_version(out_dir: Path, version_tag: str) -> None:
    """Sanity checks: no overlap, train_normal is normal-only, gate has both classes, paths exist."""
    files = {
        s: out_dir / f"{version_tag}_{s}.csv"
        for s in ("train_normal", "val_mix", "test_mix",
                 "train_gate_mix", "val_gate", "test_gate")
    }
    sets = {
        s: set(r["path"] for r in csv.DictReader(open(p)))
        for s, p in files.items()
    }

    # Heatmap split disjointness
    for a, b in (("train_normal", "val_mix"),
                 ("train_normal", "test_mix"),
                 ("val_mix", "test_mix")):
        overlap = sets[a] & sets[b]
        assert not overlap, f"{version_tag}: overlap between {a} and {b} ({len(overlap)} paths)"
    print(f"  {version_tag} heatmap splits: no overlap ✓")

    # Gate split disjointness
    for a, b in (("train_gate_mix", "val_gate"),
                 ("train_gate_mix", "test_gate"),
                 ("val_gate", "test_gate")):
        overlap = sets[a] & sets[b]
        assert not overlap, f"{version_tag}: overlap between {a} and {b} ({len(overlap)} paths)"
    print(f"  {version_tag} gate splits: no overlap ✓")

    # train_normal must be normal-only
    rows = list(csv.DictReader(open(files["train_normal"])))
    bad = [r for r in rows if r["label"] != "normal"]
    assert not bad, f"{version_tag}: train_normal contains {len(bad)} non-normal rows"
    print(f"  {version_tag} train_normal: normal-only ✓")

    # Gate training must contain both classes
    rows = list(csv.DictReader(open(files["train_gate_mix"])))
    labels = set(r["label"] for r in rows)
    assert labels == {"normal", "anomaly"}, f"{version_tag}: train_gate_mix labels={labels}"
    print(f"  {version_tag} train_gate_mix: both classes ✓")

    # Path existence on disk
    missing = 0
    for s, p in files.items():
        for r in csv.DictReader(open(p)):
            if not os.path.exists(r["path"]):
                missing += 1
                if missing <= 3:
                    print(f"    MISSING: {r['path']}")
    if missing == 0:
        print(f"  {version_tag} all paths exist ✓")
    else:
        print(f"  {version_tag} WARNING: {missing} missing paths!")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data", help="dataset root containing split_versions/")
    ap.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for v in args.versions:
        print(f"\n=== {v} ===")
        version_root = data_root / "split_versions" / v
        if not version_root.exists():
            print(f"  !! version dir not found: {version_root}")
            continue

        records = collect_records(version_root, v)
        n_anom = sum(1 for r in records if r["label"] == "anomaly")
        n_norm = sum(1 for r in records if r["label"] == "normal")
        print(f"  pool: total={len(records)}  anomaly={n_anom}  normal={n_norm}")

        # Save full pool snapshot for audit
        write_csv(records, v, f"{v}_pool", out_dir / f"{v}_pool.csv")

        splits = split_for_version(records, rng)
        for split_name, items in splits.items():
            fname = f"{v}_{split_name}.csv"
            write_csv(items, v, split_name, out_dir / fname)
            n_a = sum(1 for r in items if r["label"] == "anomaly")
            n_n = sum(1 for r in items if r["label"] == "normal")
            print(f"  {fname}: total={len(items)}  anomaly={n_a}  normal={n_n}")

        print(f"  -- validating {v} --")
        validate_version(out_dir, v)

    print("\n✅ All splits generated and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
