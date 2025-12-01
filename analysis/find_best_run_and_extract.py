#!/usr/bin/env python3
"""
proto_life_results_figures_v3.py

Reads all runs under:
    outputs/dynamic_tau_v3/*/

Generates:
 - Phase map (alpha × beta)
 - Coherence vs time
 - Entropy vs time
 - Coherence–entropy portrait
 - Autocatalysis–coherence scatter
 - Best-run montage (B, tau, N)
 - Cross-sections to detect tubular growth
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE = os.path.join(PROJECT_ROOT, "outputs", "dynamic_tau_v3")
OUTDIR = os.path.join(PROJECT_ROOT, "plots", "proto_life_v3")
os.makedirs(OUTDIR, exist_ok=True)

print("Using BASE:", BASE)
print("Saving to:", OUTDIR)


# -----------------------------------------------------------
# Load all runs
# -----------------------------------------------------------
def load_all_runs():
    run_dirs = sorted(glob.glob(os.path.join(BASE, "*")))
    if not run_dirs:
        raise RuntimeError(f"No runs found under {BASE}")

    rows = []
    for rd in run_dirs:
        met = os.path.join(rd, "metrics.csv")
        summ = os.path.join(rd, "summary.json")
        if not os.path.exists(met) or not os.path.exists(summ):
            continue

        try:
            with open(summ, "r") as f:
                s = json.load(f)

            cfg = s.get("cfg", None)
            if cfg is None:
                continue

            df = pd.read_csv(met)
            df["alpha"] = cfg.get("alpha")
            df["beta"] = cfg.get("beta")
            df["gamma"] = cfg.get("gamma")
            df["feed"] = cfg.get("feed")
            df["kill"] = cfg.get("kill")
            df["run_dir"] = rd
            rows.append(df)

        except Exception as e:
            print("Error reading", rd, e)

    if not rows:
        raise RuntimeError("No valid runs loaded.")

    return pd.concat(rows, ignore_index=True)


print("\nLoading runs...")
df = load_all_runs()
print("Loaded", len(df), "rows from", df["run_dir"].nunique(), "runs.\n")


# -----------------------------------------------------------
# 1. Coherence vs time (ensemble mean)
# -----------------------------------------------------------
plt.figure(figsize=(8,5))
mean_c = df.groupby("time")["coherence"].mean()
plt.plot(mean_c.index, mean_c.values, lw=2)
plt.title("Mean Coherence vs Time")
plt.xlabel("Time")
plt.ylabel("Coherence")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "coherence_mean.png"))
plt.close()


# -----------------------------------------------------------
# 2. Phase map (alpha × beta)
# -----------------------------------------------------------
phase = df.groupby(["alpha","beta"])["coherence"].mean().unstack()

plt.figure(figsize=(6,5))
plt.imshow(phase, origin="lower", cmap="viridis", aspect="auto")
plt.xticks(range(len(phase.columns)), phase.columns)
plt.yticks(range(len(phase.index)), phase.index)
plt.xlabel("beta")
plt.ylabel("alpha")
plt.title("Mean Coherence (alpha × beta)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "phase_alpha_beta.png"))
plt.close()


# -----------------------------------------------------------
# 3. Entropy vs Coherence
# -----------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(df["coherence"], df["entropy"], s=5, alpha=0.4)
plt.xlabel("Coherence")
plt.ylabel("Entropy")
plt.title("Coherence vs Entropy")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "coherence_vs_entropy.png"))
plt.close()


# -----------------------------------------------------------
# 4. Coherence vs Autocatalysis
# -----------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(df["coherence"], df["autocat"], s=5, alpha=0.4)
plt.xlabel("Coherence")
plt.ylabel("Autocatalysis")
plt.title("Autocatalysis vs Coherence")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "autocat_vs_coherence.png"))
plt.close()


# -----------------------------------------------------------
# 5. Best run by coherence → montage
# -----------------------------------------------------------
best_run = df.groupby("run_dir")["coherence"].mean().idxmax()
print("Best run:", best_run)

def montage(run_dir, pattern, outpath):
    snaps = sorted(glob.glob(os.path.join(run_dir, pattern)))
    snaps = snaps[:9]
    if not snaps:
        return
    imgs = [Image.open(s) for s in snaps]
    w, h = imgs[0].size
    cols = 3
    rows = (len(imgs)+cols-1)//cols
    canvas = Image.new("RGB", (cols*w, rows*h))
    for i, im in enumerate(imgs):
        canvas.paste(im, ((i%cols)*w, (i//cols)*h))
    canvas.save(outpath)
    print("Saved:", outpath)

montage(best_run, "B_snapshot_*.png",
        os.path.join(OUTDIR, "best_B_montage.png"))
montage(best_run, "tau_snapshot_*.png",
        os.path.join(OUTDIR, "best_tau_montage.png"))
montage(best_run, "N_snapshot_*.png",
        os.path.join(OUTDIR, "best_N_montage.png"))


# -----------------------------------------------------------
# 6. Cross-section extraction (to detect tubular growth)
# -----------------------------------------------------------
import numpy as np

# choose final snapshot
last_B = sorted(glob.glob(os.path.join(best_run, "B_snapshot_*.png")))[-1]
img = np.array(Image.open(last_B).convert("L")).astype(float)

centerline = img[img.shape[0]//2, :]

plt.figure(figsize=(10,4))
plt.plot(centerline, lw=1.5)
plt.title("Centerline Cross-Section of B (final frame)")
plt.xlabel("x")
plt.ylabel("B concentration")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "cross_section_B.png"))
plt.close()

print("\nAll plots generated successfully.")
