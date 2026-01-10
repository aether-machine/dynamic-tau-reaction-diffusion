#!/usr/bin/env python3
"""analyze_search_v6.py

Quick analysis for BO/QD search outputs produced by run_search_v6.

- Loads boqd_log.csv (or scans run dirs) and summarizes distributions
- Builds a QD occupancy / best-score heatmap from (desc_1, desc_2)
- Writes:
    - analysis_quantiles.json
    - top25_by_score.csv
    - qd_elites.json
    - qd_map.png
    - coverage_over_time.png
    - qd_counts.png
    - descriptor_scatter.png

This version includes a small robustness patch:
- If desc_1 / desc_2 contain NaN/inf, they are safely mapped to 0.0 for binning
  (prevents RuntimeWarning: invalid value encountered in cast).

Usage:
  python analyze_search_v6.py --run_dir outputs/dynamic_tau_v6_search_tuned
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Root directory of a V6 search run (contains boqd_log.csv)")
    p.add_argument("--bins", type=int, default=12, help="QD bins per axis (same as --qd_bins used in the sweep)")
    p.add_argument("--out_prefix", type=str, default="", help="Optional filename prefix for outputs")
    return p.parse_args()


def safe_quantiles(x: np.ndarray, qs):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {str(q): float("nan") for q in qs}
    return {str(q): float(np.quantile(x, q)) for q in qs}


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    bins = int(args.bins)

    log_path = run_dir / "boqd_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find boqd_log.csv under: {run_dir}")

    df = pd.read_csv(log_path)
    if df.empty:
        raise RuntimeError(f"{log_path} is empty")

    # Expected columns (best effort):
    # score, iou, reorg, osc, desc_1, desc_2, method, run_dir, step_idx, etc.
    # We'll proceed as long as we have score + desc columns.
    for col in ["score", "desc_1", "desc_2"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column '{col}' in {log_path}")

    # -------------------------
    # Quantiles summary
    # -------------------------
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    out_quant = {
        "n_rows": int(len(df)),
        "score": safe_quantiles(df["score"].to_numpy(dtype=float), quantiles),
    }
    for k in ["iou", "reorg", "osc", "tau_var", "tau_grad2"]:
        if k in df.columns:
            out_quant[k] = safe_quantiles(df[k].to_numpy(dtype=float), quantiles)

    quant_path = run_dir / f"{args.out_prefix}analysis_quantiles.json"
    with open(quant_path, "w") as f:
        json.dump(out_quant, f, indent=2)
    print(f"Wrote {quant_path}")

    # -------------------------
    # Top by score
    # -------------------------
    top = df.sort_values("score", ascending=False).head(25).copy()
    top_path = run_dir / f"{args.out_prefix}top25_by_score.csv"
    top.to_csv(top_path, index=False)
    print(f"Wrote {top_path}")

    # -------------------------
    # QD map (best score per bin)
    # -------------------------
    # NOTE: descriptors are assumed in [0,1) or at least bounded.
    # We clip to [0, 0.999999] for stable binning.
    d1 = df["desc_1"].to_numpy(dtype=float)
    d2 = df["desc_2"].to_numpy(dtype=float)
    # Handle missing / non-finite descriptors (can occur if a run failed mid-metrics)
    bad = (~np.isfinite(d1)) | (~np.isfinite(d2))
    if np.any(bad):
        n_bad = int(np.sum(bad))
        print(f"[WARN] {n_bad} rows have non-finite descriptors; treating as 0 for binning.")
    d1 = np.nan_to_num(d1, nan=0.0, posinf=0.999999, neginf=0.0)
    d2 = np.nan_to_num(d2, nan=0.0, posinf=0.999999, neginf=0.0)
    d1 = np.clip(d1, 0.0, 0.999999)
    d2 = np.clip(d2, 0.0, 0.999999)
    i1 = np.minimum((d1 * bins).astype(np.int64), bins - 1)
    i2 = np.minimum((d2 * bins).astype(np.int64), bins - 1)

    scores = df["score"].to_numpy(dtype=float)
    best = np.full((bins, bins), -np.inf, dtype=float)
    elite_idx = np.full((bins, bins), -1, dtype=int)

    for idx in range(len(df)):
        a = i1[idx]
        b = i2[idx]
        s = scores[idx]
        if not np.isfinite(s):
            continue
        if s > best[a, b]:
            best[a, b] = s
            elite_idx[a, b] = idx

    # occupancy
    occ = np.isfinite(best) & (best > -np.inf)
    n_occ = int(np.sum(occ))

    # Save elites json
    elites = []
    for a in range(bins):
        for b in range(bins):
            idx = elite_idx[a, b]
            if idx >= 0:
                row = df.iloc[idx].to_dict()
                row["bin_i"] = int(a)
                row["bin_j"] = int(b)
                elites.append(row)

    elites_path = run_dir / f"{args.out_prefix}qd_elites.json"
    with open(elites_path, "w") as f:
        json.dump(elites, f, indent=2, default=str)
    print(f"Wrote {elites_path} ({len(elites)} elites, {n_occ}/{bins*bins} occupied)")

    # Heatmap
    # Replace -inf with NaN for plotting
    plot_map = best.copy()
    plot_map[~occ] = np.nan

    plt.figure(figsize=(7, 6))
    plt.imshow(plot_map.T, origin="lower", aspect="auto")
    plt.colorbar(label="best score in bin")
    plt.title(f"QD map (occupied: {n_occ}/{bins*bins})")
    plt.xlabel("desc_1 bin")
    plt.ylabel("desc_2 bin")
    plt.tight_layout()
    map_path = run_dir / f"{args.out_prefix}qd_map.png"
    plt.savefig(map_path, dpi=150)
    plt.close()
    print(f"Wrote {map_path}")

    # -------------------------
    # Coverage over time (if step index exists)
    # -------------------------
    if "t" in df.columns:
        tcol = "t"
    elif "step" in df.columns:
        tcol = "step"
    elif "iter" in df.columns:
        tcol = "iter"
    elif "idx" in df.columns:
        tcol = "idx"
    else:
        tcol = None

    if tcol is not None:
        dfx = df.copy()
        dfx = dfx.sort_values(tcol)
        d1x = np.nan_to_num(dfx["desc_1"].to_numpy(dtype=float), nan=0.0, posinf=0.999999, neginf=0.0)
        d2x = np.nan_to_num(dfx["desc_2"].to_numpy(dtype=float), nan=0.0, posinf=0.999999, neginf=0.0)
        d1x = np.clip(d1x, 0.0, 0.999999)
        d2x = np.clip(d2x, 0.0, 0.999999)
        i1x = np.minimum((d1x * bins).astype(np.int64), bins - 1)
        i2x = np.minimum((d2x * bins).astype(np.int64), bins - 1)

        seen = np.zeros((bins, bins), dtype=bool)
        cov = []
        ts = dfx[tcol].to_numpy()
        for k in range(len(dfx)):
            seen[i1x[k], i2x[k]] = True
            cov.append(float(np.sum(seen)) / float(bins * bins))

        plt.figure(figsize=(8, 4))
        plt.plot(ts, cov)
        plt.xlabel(tcol)
        plt.ylabel("coverage fraction")
        plt.title("QD coverage over time")
        plt.tight_layout()
        cov_path = run_dir / f"{args.out_prefix}coverage_over_time.png"
        plt.savefig(cov_path, dpi=150)
        plt.close()
        print(f"Wrote {cov_path}")

    # -------------------------
    # Descriptor scatter
    # -------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(d1, d2, s=8, alpha=0.6)
    plt.xlabel("desc_1")
    plt.ylabel("desc_2")
    plt.title("Descriptor scatter")
    plt.tight_layout()
    scat_path = run_dir / f"{args.out_prefix}descriptor_scatter.png"
    plt.savefig(scat_path, dpi=150)
    plt.close()
    print(f"Wrote {scat_path}")

    # -------------------------
    # QD bin counts
    # -------------------------
    counts = np.zeros((bins, bins), dtype=int)
    for a, b in zip(i1, i2):
        counts[a, b] += 1

    plt.figure(figsize=(7, 6))
    plt.imshow(counts.T, origin="lower", aspect="auto")
    plt.colorbar(label="count")
    plt.title("QD bin counts")
    plt.xlabel("desc_1 bin")
    plt.ylabel("desc_2 bin")
    plt.tight_layout()
    cnt_path = run_dir / f"{args.out_prefix}qd_counts.png"
    plt.savefig(cnt_path, dpi=150)
    plt.close()
    print(f"Wrote {cnt_path}")

    print("[OK] analysis complete")


if __name__ == "__main__":
    main()
