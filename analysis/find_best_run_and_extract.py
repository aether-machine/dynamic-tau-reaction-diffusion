#!/usr/bin/env python3
"""
find_best_run_and_extract.py

Run from project root. Requires: pandas, Pillow (PIL), matplotlib (optional).

Outputs:
 - plots/proto_life_v3/runs_summary.csv
 - plots/proto_life_v3/best_run_snaps/ (first/mid/last B & tau)
 - plots/proto_life_v3/figure3_oscillon_candidate.png (3-panel)
 - plots/proto_life_v3/cross_section_B.png
"""

import os, glob, json, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BASE = Path("outputs/dynamic_tau_v3")
OUT = Path("plots/proto_life_v3")
OUT.mkdir(parents=True, exist_ok=True)

runs = []
for d in sorted(BASE.glob("*")):
    if (d / "metrics.csv").exists() and ((d / "summary.json").exists() or (d / "meta.json").exists()):
        runs.append(d)

if not runs:
    raise SystemExit("No runs found under outputs/dynamic_tau_v3")

summary_rows = []
for r in runs:
    metrics = pd.read_csv(r / "metrics.csv")
    # inspect columns, prefer column named 'coherence'
    if 'coherence' not in metrics.columns:
        # try to find close name
        print("Warning: 'coherence' column missing in", r)
    mean_coh = metrics['coherence'].mean() if 'coherence' in metrics.columns else np.nan
    max_coh = metrics['coherence'].max() if 'coherence' in metrics.columns else np.nan
    std_coh = metrics['coherence'].std() if 'coherence' in metrics.columns else np.nan
    mean_ent = metrics['entropy'].mean() if 'entropy' in metrics.columns else np.nan
    mean_auto = metrics['autocat'].mean() if 'autocat' in metrics.columns else np.nan

    # load cfg
    cfg = {}
    for fname in ("summary.json", "meta.json"):
        p = r / fname
        if p.exists():
            try:
                meta = json.load(open(p))
                cfg = meta.get("cfg", meta)
            except Exception:
                cfg = {}
            break

    summary_rows.append({
        "run_dir": str(r),
        "alpha": cfg.get("alpha"),
        "beta": cfg.get("beta"),
        "feed": cfg.get("feed"),
        "kill": cfg.get("kill"),
        "mean_coherence": mean_coh,
        "max_coherence": max_coh,
        "std_coherence": std_coh,
        "mean_entropy": mean_ent,
        "mean_autocat": mean_auto
    })

df_summary = pd.DataFrame(summary_rows)
df_summary = df_summary.sort_values("mean_coherence", ascending=False).reset_index(drop=True)
df_summary.to_csv(OUT / "runs_summary.csv", index=False)
print("Wrote runs_summary.csv with", len(df_summary), "runs.")

# pick best run by mean_coherence
best = df_summary.iloc[0]
best_dir = Path(best['run_dir'])
print("Best run:", best_dir)
# copy representative snapshots: first, mid, last
def pick_and_copy(pattern, dest_prefix):
    snaps = sorted(best_dir.glob(pattern))
    if not snaps:
        return []
    n = len(snaps)
    idxs = [0] if n==1 else [0, n//2, n-1]
    copied = []
    dest = OUT / "best_run_snaps"
    dest.mkdir(parents=True, exist_ok=True)
    for i in idxs:
        src = snaps[i]
        dst = dest / f"{dest_prefix}_{i:02d}{src.suffix}"
        shutil.copy(src, dst)
        copied.append(dst)
    return copied

b_copied = pick_and_copy("B_snapshot_*.png", "B")
if not b_copied:
    b_copied = pick_and_copy("snapshot_*.png", "B")
tau_copied = pick_and_copy("tau_snapshot_*.png", "tau")

print("Copied snapshots:", len(b_copied), "B and", len(tau_copied), "tau (if present).")

# create 3-panel for candidate oscillation division (FIGURE 3)
if b_copied:
    imgs = [Image.open(p).convert("RGB") for p in b_copied]
    w,h = imgs[0].size
    canvas = Image.new("RGB", (w*3, h))
    for i,im in enumerate(imgs):
        canvas.paste(im, (i*w, 0))
    panel_path = OUT / "figure3_oscillon_candidate.png"
    canvas.save(panel_path)
    print("Saved", panel_path)

# create cross-section of final B snapshot (centerline) for structure detection
final_b = sorted(best_dir.glob("B_snapshot_*.png"))
if not final_b:
    final_b = sorted(best_dir.glob("snapshot_*.png"))
if final_b:
    arr = np.array(Image.open(final_b[-1]).convert("L"), dtype=float)
    center = arr[arr.shape[0]//2, :]
    plt.figure(figsize=(10,3))
    plt.plot(center, lw=1.2)
    plt.title("Centerline cross-section of final B snapshot")
    plt.xlabel("x")
    plt.ylabel("B intensity")
    plt.tight_layout()
    cs_path = OUT / "cross_section_B.png"
    plt.savefig(cs_path)
    plt.close()
    print("Saved cross section to", cs_path)

print("Analysis artifacts saved in", OUT)
