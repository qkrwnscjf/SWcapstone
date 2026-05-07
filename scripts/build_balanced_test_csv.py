#!/usr/bin/env python3
"""Build BALANCED test CSVs from split_versions/v{1,2,3}/additional/{anomaly,normal}/.

Each version's CSV has equal numbers of anomaly and normal samples, so accuracy
and FP rate are both well-defined.

Output columns: path,dataset_type,defect_type,label,round,split
"""
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect(class_dir: Path) -> list[Path]:
    if not class_dir.exists():
        return []
    return sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="dataset root containing split_versions/")
    ap.add_argument("--out-dir", required=True, help="output directory for CSVs")
    ap.add_argument("--sub", default="additional", help="subdir under each version (default: additional)")
    ap.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    ap.add_argument("--per-class", type=int, default=200, help="samples per class per version")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for v in args.versions:
        rows = []
        per_class = args.per_class
        for cls in ("anomaly", "normal"):
            d = root / f"split_versions/{v}/{args.sub}/{cls}"
            files = collect(d)
            if not files:
                print(f"!! missing or empty: {d}")
                continue
            if len(files) < per_class:
                print(f"   {v}/{cls}: only {len(files)} files (< {per_class}); using all")
                chosen = files
            else:
                chosen = rng.sample(files, per_class)
            for p in chosen:
                rows.append({
                    "path": str(p),
                    "dataset_type": "Kolektor",
                    "defect_type": "surface_defect",
                    "label": cls,
                    "round": v,
                    "split": f"{args.sub}_balanced_test",
                })
        rng.shuffle(rows)

        out_csv = out_dir / f"{v}_balanced_test.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "dataset_type", "defect_type", "label", "round", "split"])
            w.writeheader()
            w.writerows(rows)
        n_anom = sum(1 for r in rows if r["label"] == "anomaly")
        n_norm = sum(1 for r in rows if r["label"] == "normal")
        print(f"  {out_csv.name}: total={len(rows)}  anomaly={n_anom}  normal={n_norm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
