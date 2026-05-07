#!/usr/bin/env python3
"""Inspect a downloaded dataset directory: print structure & label counts.

Usage: python scripts/inspect_dataset.py <data_root>
"""
from __future__ import annotations
import os
import sys
from collections import Counter
from pathlib import Path


def main(root: str) -> int:
    root_p = Path(root)
    if not root_p.exists():
        print(f"NOT FOUND: {root_p}")
        return 1

    print(f"# Inspect: {root_p.resolve()}")
    print()

    # Top 3 levels
    print("## Top-level structure (depth ≤ 3)")
    for dirpath, dirnames, filenames in os.walk(root_p):
        depth = len(Path(dirpath).relative_to(root_p).parts)
        if depth > 3:
            dirnames[:] = []
            continue
        rel = Path(dirpath).relative_to(root_p)
        indent = "  " * depth
        print(f"{indent}{rel if str(rel) != '.' else root_p.name}/  ({len(filenames)} files, {len(dirnames)} dirs)")
    print()

    # Look for split CSVs
    csvs = sorted(root_p.rglob("*.csv"))
    print(f"## CSV files found: {len(csvs)}")
    for c in csvs[:30]:
        print(f"  {c.relative_to(root_p)}  ({c.stat().st_size} B)")
    print()

    # Look for v1/v2/v3 dirs
    print("## v1/v2/v3 detection")
    for v in ("v1", "v2", "v3"):
        hits = list(root_p.rglob(v))
        hits = [h for h in hits if h.is_dir()]
        if hits:
            print(f"  {v}: {hits[0]}")
        else:
            print(f"  {v}: NOT FOUND")
    print()

    # Image counts by extension under each top-level dir
    print("## Image counts by top-level subdir")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for sub in sorted(p for p in root_p.iterdir() if p.is_dir()):
        cnt = Counter()
        for p in sub.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                cnt[p.suffix.lower()] += 1
        total = sum(cnt.values())
        print(f"  {sub.name}: {total} images  ({dict(cnt)})")
    print()

    # Sample paths from CSVs
    if csvs:
        import csv as csvlib
        print("## Sample rows from each CSV (first 3)")
        for c in csvs[:5]:
            print(f"--- {c.relative_to(root_p)}")
            try:
                with open(c) as f:
                    reader = csvlib.reader(f)
                    for i, row in enumerate(reader):
                        if i >= 3:
                            break
                        print("   ", row)
            except Exception as e:
                print(f"  (error reading: {e})")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
