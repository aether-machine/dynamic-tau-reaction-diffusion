#!/usr/bin/env python3
"""
analyze_proto_life_v5.py

Flexible analyser for dynamic_tau_v5 sweeps.

Features:
 - Accept one or more summary CSVs via --csv
   (e.g. runs_summary_v5.csv, runs_summary_v5_qridge.csv, etc.)
 - Adds proto_life_score_v5 if missing
 - Tags each run with a sweep_label based on the CSV name (or --label)
 - Produces:
     * augmented CSV(s) with proto_life_score_v5
     * global scatter plots
     * top-run montages (B first/mid/last + τ last) for chosen sweep

Typical usage:

  # Global v5 sweep (default path)
  python analyze_proto_life_v5.py

  # Explicit CSV
  python analyze_proto_life_v5.py --csv plots/proto_life_v5/runs_summary_v5.csv

  # Q-ridge sweep only
  python analyze_proto_life_v5.py --csv plots/proto_life_v5/runs_summary_v5_qridge.csv --top-n 16

  # Compare global + qridge, montages for qridge only
  python analyze_proto_life_v5.py \\
      --csv plots/proto_life_v5/runs_summary_v5.csv \\
      --csv plots/proto_life_v5/runs_summary_v5_qridge.csv \\
      --montage-label qridge
"""

import argparse
from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_png_gray(path):
    arr = np.array(Image.open(path).convert("L"), dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def z_score(series: pd.Series):
    s = series.astype(float)
    m = s.mean()
    v = s.var()
    if np.isnan(v) or v <= 0:
        return pd.Series(np.zeros_like(s), index=s.index)
    return (s - m) / np.sqrt(v)


# ------------------------------------------------------------
# Proto-life score
# ------------------------------------------------------------

def add_proto_life_score(df: pd.DataFrame) -> pd.DataFrame:
    # if it already exists, just return
    if "proto_life_score_v5" in df.columns:
        return df.copy()

    required = ["mean_coherence", "mean_entropy", "maintenance_iou",
                "coherence_slope", "tau_var_final"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for proto_life_score_v5: {missing}")

    z_coh = z_score(df["mean_coherence"])
    z_entropy = z_score(df["mean_entropy"])
    z_maint = z_score(df["maintenance_iou"])
    z_slope = z_score(df["coherence_slope"])
    z_tauvar = z_score(df["tau_var_final"])

    proto_score = (
        z_coh
        + z_maint
        + 0.5 * z_slope
        - 0.5 * z_entropy
        - 0.5 * z_tauvar
    )

    df = df.copy()
    df["proto_life_score_v5"] = proto_score
    return df


# ------------------------------------------------------------
# Global scatter plots
# ------------------------------------------------------------

def plot_scatter_mean_coh_vs_entropy(df, outdir):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        df["mean_coherence"],
        df["mean_entropy"],
        c=df["maintenance_iou"],
        s=20,
        alpha=0.7
    )
    plt.xlabel("Mean coherence")
    plt.ylabel("Mean entropy")
    plt.title("Mean Coherence vs Mean Entropy\ncolor = maintenance IoU")
    cbar = plt.colorbar(sc)
    cbar.set_label("maintenance IoU")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_mean_coh_vs_entropy.png", dpi=200)
    plt.close()


def plot_scatter_coh_slope_vs_maintenance(df, outdir):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        df["coherence_slope"],
        df["maintenance_iou"],
        c=df["mean_entropy"],
        s=20,
        alpha=0.7
    )
    plt.xlabel("Coherence slope")
    plt.ylabel("Maintenance IoU")
    plt.title("Coherence growth vs Maintenance\ncolor = mean entropy")
    cbar = plt.colorbar(sc)
    cbar.set_label("mean entropy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_coh_slope_vs_maintenance.png", dpi=200)
    plt.close()


def plot_scatter_maintenance_vs_tauvar(df, outdir):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        df["tau_var_final"],
        df["maintenance_iou"],
        c=df["proto_life_score_v5"],
        s=20,
        alpha=0.7
    )
    plt.xlabel("Final τ variance")
    plt.ylabel("Maintenance IoU")
    plt.title("Maintenance vs τ structure\ncolor = proto_life_score_v5")
    cbar = plt.colorbar(sc)
    cbar.set_label("proto_life_score_v5")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_maintenance_vs_tauvar.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# Per-run montages
# ------------------------------------------------------------

def find_snapshots(run_dir: Path):
    B_snaps = sorted(run_dir.glob("B_snapshot_*.png"))
    if not B_snaps:
        B_snaps = sorted(run_dir.glob("snapshot_*.png"))
    tau_snaps = sorted(run_dir.glob("tau_snapshot_*.png"))
    return B_snaps, tau_snaps


def choose_indices(n, k=3):
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    return [0, n // 2, n - 1]


def create_run_montage(run_dir: Path, outpath: Path, score: float = None, rank: int = None):
    B_snaps, tau_snaps = find_snapshots(run_dir)
    if not B_snaps or not tau_snaps:
        print("Skipping montage (missing snapshots) for", run_dir)
        return

    b_idxs = choose_indices(len(B_snaps), k=3)
    b_imgs = [load_png_gray(B_snaps[i]) for i in b_idxs]
    tau_last = load_png_gray(tau_snaps[-1])

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    titles = [f"B {B_snaps[i].name}" for i in b_idxs]

    for ax, img, title in zip(axes[:len(b_imgs)], b_imgs, titles):
        ax.imshow(img, cmap="magma", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    ax_tau = axes[-1]
    im = ax_tau.imshow(tau_last, cmap="viridis", interpolation="nearest")
    ax_tau.set_title(tau_snaps[-1].name, fontsize=9)
    ax_tau.axis("off")

    cbar = fig.colorbar(im, ax=ax_tau, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    hash_part = run_dir.name
    title_parts = [f"Run {hash_part}"]
    if rank is not None:
        title_parts.insert(0, f"Rank {rank}")
    if score is not None:
        title_parts.append(f"score={score:.2f}")
    fig.suptitle(" | ".join(title_parts), fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outpath, dpi=200)
    plt.close()


def create_top_run_montages(df: pd.DataFrame, outdir: Path, top_n: int = 12, label: str = None):
    df_sorted = df.sort_values("proto_life_score_v5", ascending=False).reset_index(drop=True)
    n = min(top_n, len(df_sorted))
    for rank in range(n):
        row = df_sorted.iloc[rank]
        run_dir = Path(row["run_dir"])
        score = float(row["proto_life_score_v5"])
        hash_part = run_dir.name

        tag = f"{label}_" if label else ""
        outpath = outdir / f"{tag}top_run_{rank+1:02d}_{hash_part}.png"
        print(f"Creating montage for rank {rank+1}, run {run_dir}")
        create_run_montage(run_dir, outpath, score=score, rank=rank+1)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        action="append",
        help="Path to a runs_summary CSV. Can be given multiple times. "
             "Default: plots/proto_life_v5/runs_summary_v5.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of top runs to create montages for (per chosen sweep).",
    )
    parser.add_argument(
        "--montage-label",
        type=str,
        default=None,
        help="If multiple CSVs are provided, only make montages for rows "
             "with sweep_label equal to this string.",
    )
    args = parser.parse_args()

    # Default CSV if none specified
    if not args.csv:
        default_csv = Path("plots") / "proto_life_v5" / "runs_summary_v5.csv"
        if not default_csv.exists():
            raise SystemExit(f"No --csv given and default {default_csv} not found.")
        csv_paths = [default_csv]
    else:
        csv_paths = [Path(p) for p in args.csv]

    all_dfs = []
    for cpath in csv_paths:
        if not cpath.exists():
            print(f"Warning: CSV not found: {cpath}, skipping.")
            continue
        df = pd.read_csv(cpath)
        df = add_proto_life_score(df)

        # Label by filename stem
        label = cpath.stem  # e.g. "runs_summary_v5", "runs_summary_v5_qridge"
        df["sweep_label"] = label

        # Save back an augmented CSV next to original
        out_aug = cpath.parent / f"{label}_with_score.csv"
        df.to_csv(out_aug, index=False)
        print("Augmented CSV written to", out_aug)

        all_dfs.append(df)

    if not all_dfs:
        raise SystemExit("No valid CSVs to analyse.")

    # Merge all sweeps for global plots
    df_merged = pd.concat(all_dfs, ignore_index=True)
    outdir = ensure_dir(Path("plots") / "proto_life_v5" / "analysis_v5")

    # Global scatter plots (all sweeps together)
    print("Generating global scatter plots across all provided CSVs...")
    plot_scatter_mean_coh_vs_entropy(df_merged, outdir)
    plot_scatter_coh_slope_vs_maintenance(df_merged, outdir)
    plot_scatter_maintenance_vs_tauvar(df_merged, outdir)

    # Decide which sweep to use for montages
    if args.montage_label is not None:
        df_for_montage = df_merged[df_merged["sweep_label"] == args.montage_label]
        montage_tag = args.montage_label
    else:
        # Use the first CSV's label
        first_label = all_dfs[0]["sweep_label"].iloc[0]
        df_for_montage = df_merged[df_merged["sweep_label"] == first_label]
        montage_tag = first_label

    print(f"Creating top-run montages for sweep_label='{montage_tag}'")
    create_top_run_montages(df_for_montage, outdir, top_n=args.top_n, label=montage_tag)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
