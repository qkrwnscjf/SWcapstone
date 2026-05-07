#!/usr/bin/env python3
"""Build test CSVs from split_versions/v{1,2,3}/additional/{anomaly,normal}/.

Output columns: path,dataset_type,defect_type,label,round,split
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main(root: str, out_dir: str, sub: str = "additional") -> int:
    root_p = Path(root).resolve()
    out_p = Path(out_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    for v in ("v1", "v2", "v3"):
        rows = []
        for cls in ("anomaly", "normal"):
            d = root_p / f"split_versions/{v}/{sub}/{cls}"
            if not d.exists():
                print(f"!! missing {d}")
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in EXTS:
                    rows.append({
                        "path": str(p),
                        "dataset_type": "Kolektor",
                        "defect_type": "surface_defect",
                        "label": cls,
                        "round": v,
                        "split": f"{sub}_test",
                    })
        out_csv = out_p / f"{v}_{sub}_test.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "dataset_type", "defect_type", "label", "round", "split"])
            w.writeheader()
            w.writerows(rows)
        n_anom = sum(1 for r in rows if r["label"] == "anomaly")
        n_norm = sum(1 for r in rows if r["label"] == "normal")
        print(f"  {out_csv.name}: total={len(rows)}  anomaly={n_anom}  normal={n_norm}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: build_test_csv.py <data_root> <out_dir> [sub=additional]")
        sys.exit(2)
    sub = sys.argv[3] if len(sys.argv) > 3 else "additional"
    sys.exit(main(sys.argv[1], sys.argv[2], sub))
