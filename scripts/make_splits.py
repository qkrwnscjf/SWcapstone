#!/usr/bin/env python3
"""
Agent-DataOps: 3-round stratified split generator.

Policy:
  - Anchor set: fixed 20% of (Kolektor + MVTec) stratified by (dataset_type, defect_type, label).
    NEU anchor: 20% of NEU (anomaly-only), used for sanity checks only.
  - Remaining 80% split into round1/round2/round3 pools (1/3 each).
  - Per round, incremental train/val/test:
      Heatmap: train_normal (normal-only), val_mix (small mix), test_mix (mix)
      Gate:    train_gate_mix, val_gate, test_gate (must include anomalies)
  - Columns: path,dataset_type,defect_type,label,round,split

Global seed: 42
"""
import os, sys, csv, random, collections, json
from pathlib import Path

SEED = 42
random.seed(SEED)

BASE_DATA = Path("/Users/danghyeonsong/HYUE/4-1/final/final_dataset/dataset_type")
OUT_DIR = Path("/Users/danghyeonsong/HYUE/4-1/final/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Scan all images ──────────────────────────────────────────────
records = []
for root, dirs, files in os.walk(BASE_DATA):
    for f in files:
        if not f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
            continue
        full = os.path.join(root, f)
        rel = os.path.relpath(root, BASE_DATA)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue
        dataset_type = parts[0]
        defect_type = parts[2]
        label = parts[3]  # normal or anomaly
        # For MVTec anomaly sub-types, flatten to parent defect_type
        records.append({
            "path": full,
            "dataset_type": dataset_type,
            "defect_type": defect_type,
            "label": label,
        })

print(f"Total images scanned: {len(records)}")

# ── 2. Group by stratification key ──────────────────────────────────
def strat_key(r):
    return (r["dataset_type"], r["defect_type"], r["label"])

groups = collections.defaultdict(list)
for r in records:
    groups[strat_key(r)].append(r)

for k in groups:
    random.shuffle(groups[k])

print(f"Stratification groups: {len(groups)}")
for k in sorted(groups):
    print(f"  {k}: {len(groups[k])}")

# ── 3. Split: anchor 20% vs pool 80% ───────────────────────────────
anchor = []
pool = []
for k, items in groups.items():
    n = len(items)
    n_anchor = max(1, round(n * 0.20))
    anchor.extend(items[:n_anchor])
    pool.extend(items[n_anchor:])

print(f"\nAnchor: {len(anchor)}, Pool: {len(pool)}")

# ── 4. Pool -> round1 / round2 / round3 (1/3 each, stratified) ─────
pool_groups = collections.defaultdict(list)
for r in pool:
    pool_groups[strat_key(r)].append(r)

round_pools = {1: [], 2: [], 3: []}
for k, items in pool_groups.items():
    random.shuffle(items)
    n = len(items)
    n1 = n // 3
    n2 = n1
    # n3 gets remainder
    round_pools[1].extend(items[:n1])
    round_pools[2].extend(items[n1:n1+n2])
    round_pools[3].extend(items[n1+n2:])

for rnd in [1,2,3]:
    print(f"Round {rnd} pool: {len(round_pools[rnd])}")

# ── 5. Per-round splits ────────────────────────────────────────────
# Heatmap splits: train_normal (normal-only 70%), val_mix (15%), test_mix (15%)
# Gate splits:    train_gate_mix (70%), val_gate (15%), test_gate (15%)
# Both use the cumulative pool up to that round.

def split_for_round(cumulative_items, round_num):
    """Generate heatmap + gate splits for a given round."""
    # Separate normal and anomaly
    normals = [r for r in cumulative_items if r["label"] == "normal"]
    anomalies = [r for r in cumulative_items if r["label"] == "anomaly"]

    random.shuffle(normals)
    random.shuffle(anomalies)

    # -- Heatmap splits --
    # train_normal: 70% of normals (normal-only)
    # val_mix: 15% normals + 30% anomalies
    # test_mix: 15% normals + 70% anomalies
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

    # -- Gate splits (need both classes) --
    # Combine all and stratified split 70/15/15
    all_items = normals + anomalies
    random.shuffle(all_items)

    # Stratified by label for gate
    gate_normals = list(normals)
    gate_anomalies = list(anomalies)
    random.shuffle(gate_normals)
    random.shuffle(gate_anomalies)

    gn_train = int(len(gate_normals) * 0.70)
    gn_val = int(len(gate_normals) * 0.15)
    ga_train = int(len(gate_anomalies) * 0.70)
    ga_val = int(len(gate_anomalies) * 0.15)

    train_gate = gate_normals[:gn_train] + gate_anomalies[:ga_train]
    val_gate = gate_normals[gn_train:gn_train+gn_val] + gate_anomalies[ga_train:ga_train+ga_val]
    test_gate = gate_normals[gn_train+gn_val:] + gate_anomalies[ga_train+ga_val:]

    random.shuffle(train_gate)
    random.shuffle(val_gate)
    random.shuffle(test_gate)

    return {
        "train_normal": train_normal,
        "val_mix": val_mix,
        "test_mix": test_mix,
        "train_gate_mix": train_gate,
        "val_gate": val_gate,
        "test_gate": test_gate,
    }

def write_csv(items, round_num, split_name, filepath):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path","dataset_type","defect_type","label","round","split"])
        writer.writeheader()
        for r in items:
            writer.writerow({
                "path": r["path"],
                "dataset_type": r["dataset_type"],
                "defect_type": r["defect_type"],
                "label": r["label"],
                "round": round_num,
                "split": split_name,
            })

# Write anchor sets
anchor_km = [r for r in anchor if r["dataset_type"] != "NEU"]
anchor_neu = [r for r in anchor if r["dataset_type"] == "NEU"]

# anchor_test_mix: Kolektor+MVTec anchor (has both normal and anomaly)
write_csv(anchor_km, "anchor", "anchor_test_mix",
          OUT_DIR / "anchor_test_mix.csv")
print(f"anchor_test_mix: {len(anchor_km)} (normal: {sum(1 for r in anchor_km if r['label']=='normal')}, anomaly: {sum(1 for r in anchor_km if r['label']=='anomaly')})")

# anchor_neu_test: NEU anchor (anomaly-only)
if anchor_neu:
    write_csv(anchor_neu, "anchor", "anchor_neu_test",
              OUT_DIR / "anchor_neu_test.csv")
    print(f"anchor_neu_test: {len(anchor_neu)} (all anomaly)")

# Write round pool CSVs
for rnd in [1,2,3]:
    write_csv(round_pools[rnd], rnd, f"round{rnd}_pool",
              OUT_DIR / f"round{rnd}_pool.csv")

# Generate per-round splits (cumulative)
cumulative = []
for rnd in [1,2,3]:
    cumulative.extend(round_pools[rnd])
    splits = split_for_round(cumulative, rnd)
    for split_name, items in splits.items():
        fname = f"round{rnd}_{split_name}.csv"
        write_csv(items, rnd, split_name, OUT_DIR / fname)
        n_normal = sum(1 for r in items if r["label"] == "normal")
        n_anomaly = sum(1 for r in items if r["label"] == "anomaly")
        print(f"  round{rnd}_{split_name}: {len(items)} (normal: {n_normal}, anomaly: {n_anomaly})")

# ── 6. Validation checks ───────────────────────────────────────────
print("\n=== VALIDATION ===")

# Check no duplicates across heatmap splits within same round
for rnd in [1,2,3]:
    tn = set(r["path"] for r in csv.DictReader(open(OUT_DIR / f"round{rnd}_train_normal.csv")))
    vm = set(r["path"] for r in csv.DictReader(open(OUT_DIR / f"round{rnd}_val_mix.csv")))
    tm = set(r["path"] for r in csv.DictReader(open(OUT_DIR / f"round{rnd}_test_mix.csv")))

    assert len(tn & vm) == 0, f"Round {rnd}: overlap train_normal & val_mix"
    assert len(tn & tm) == 0, f"Round {rnd}: overlap train_normal & test_mix"
    assert len(vm & tm) == 0, f"Round {rnd}: overlap val_mix & test_mix"
    print(f"  Round {rnd} heatmap: no duplicates ✓")

    # Check train_normal is normal-only
    tn_rows = list(csv.DictReader(open(OUT_DIR / f"round{rnd}_train_normal.csv")))
    anomalies_in_tn = [r for r in tn_rows if r["label"] == "anomaly"]
    assert len(anomalies_in_tn) == 0, f"Round {rnd}: anomaly in train_normal!"
    print(f"  Round {rnd} train_normal: normal-only ✓")

    # Check gate splits have both classes
    tg = list(csv.DictReader(open(OUT_DIR / f"round{rnd}_train_gate_mix.csv")))
    labels = set(r["label"] for r in tg)
    assert "normal" in labels and "anomaly" in labels, f"Round {rnd}: gate train single-class!"
    print(f"  Round {rnd} gate train: both classes ✓")

# Check all paths exist
all_csvs = list(OUT_DIR.glob("*.csv"))
missing = 0
for csvf in all_csvs:
    for row in csv.DictReader(open(csvf)):
        if not os.path.exists(row["path"]):
            missing += 1
            print(f"  MISSING: {row['path']}")
if missing == 0:
    print("  All paths exist ✓")
else:
    print(f"  WARNING: {missing} paths missing!")

# Check anchor does not overlap with any round pool
anchor_paths = set(r["path"] for r in anchor)
for rnd in [1,2,3]:
    pool_paths = set(r["path"] for r in csv.DictReader(open(OUT_DIR / f"round{rnd}_pool.csv")))
    overlap = anchor_paths & pool_paths
    assert len(overlap) == 0, f"Anchor overlaps with round {rnd} pool!"
    print(f"  Anchor vs Round {rnd} pool: no overlap ✓")

print("\n✅ All validations passed!")
