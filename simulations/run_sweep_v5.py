#!/usr/bin/env python3

"""
run_sweep_v5.py

Random parameter sweep driver for dynamic_tau_v5.py.

Features:
 - Random sampling in a broad parameter space
 - Parallel execution with multiprocessing
 - Each run stored under: outputs/dynamic_tau_v5/<hash>/
 - Post-hoc analysis to build a runs_summary_v5.csv with:
     * mean/max/std coherence
     * mean entropy, mean energy, mean autocat
     * coherence slope
     * maintenance IoU (first/mid/last B snapshots)
     * tau_var_final, tau_grad2_final from final tau snapshot
"""

import os
import json
import hashlib
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dynamic_tau_v5 import run_simulation


# ------------------------------------------------------------
# Config sampling
# ------------------------------------------------------------

def sample_uniform(a, b):
    return float(np.random.uniform(a, b))


def sample_log_uniform(a, b):
    """
    Log-uniform between a and b (both > 0).
    """
    return float(np.exp(np.random.uniform(np.log(a), np.log(b))))


def build_random_config():
    """
    Build a single random configuration for dynamic_tau_v5.
    You can widen/narrow ranges here as desired.
    """

    # Base grid / timing
    nx = ny = 150
    cfg = {
        "nx": nx,
        "ny": ny,
        "dx": 1.0,
        "dy": 1.0,
        "dt": 0.01,
        "steps": 4000,
        "snap_every": 200,
        "log_every": 20,
        "Da": 0.16,
        "Db": 0.08,
    }

    # Gray–Scott parameters (slightly expanded vs your v4 ranges)
    cfg["feed"] = sample_uniform(0.02, 0.05)
    cfg["kill"] = sample_uniform(0.045, 0.075)

    # τ single-species baseline (will be used if num_tau_species == 1)
    cfg["tau0"] = 1.0
    cfg["tau_min"] = 0.2
    cfg["tau_max"] = 3.0

    # Memory → τ couplings
    cfg["alpha"] = sample_uniform(0.0, 0.05)    # memory feedback
    cfg["beta"] = sample_uniform(0.001, 0.02)   # relaxation
    cfg["gamma"] = sample_uniform(0.0, 0.01)    # nutrient coupling

    # τ geometry & noise
    cfg["kappa_tau"] = sample_uniform(0.0, 0.05)   # τ diffusion
    cfg["tau_noise"] = sample_uniform(0.0, 0.02)   # τ stochasticity

    # Nutrient baseline
    cfg["N0"] = 1.0
    cfg["nutrient_use"] = sample_uniform(0.005, 0.03)

    # Memory: decide randomly whether to use single or multiscale
    use_multiscale = np.random.rand() < 0.5
    cfg["use_multiscale_memory"] = bool(use_multiscale)

    if use_multiscale:
        # Fast memory: short timescale
        cfg["memory_fast_decay"] = sample_uniform(0.05, 0.3)
        # Slow memory: long timescale
        cfg["memory_slow_decay"] = sample_uniform(0.001, 0.02)

        # Weights: normalised later by user logic, but keep them in [0,1]
        w_fast = sample_uniform(0.3, 0.8)
        w_slow = 1.0 - w_fast
        cfg["memory_fast_weight"] = w_fast
        cfg["memory_slow_weight"] = w_slow

        # Single-memory decay not used, but set to a default
        cfg["memory_decay"] = 0.01
    else:
        cfg["memory_decay"] = sample_uniform(0.002, 0.05)

    # Nutrient module: diffusive or simple
    use_diffusive_nutrient = np.random.rand() < 0.5
    cfg["use_diffusive_nutrient"] = bool(use_diffusive_nutrient)
    if use_diffusive_nutrient:
        cfg["Dn"] = sample_uniform(0.01, 0.2)
        cfg["nutrient_feed_rate"] = sample_uniform(0.002, 0.05)

    # Multi-τ species: 1 or 2
    if np.random.rand() < 0.3:
        num_tau = 2
    else:
        num_tau = 1
    cfg["num_tau_species"] = num_tau

    if num_tau == 1:
        # all single-τ params already set above
        pass
    else:
        # Build parameter lists for two τ species.
        # You can tweak these ranges / relationships later.
        tau0_list = [1.0, 1.0]
        alpha_list = [
            cfg["alpha"],                      # τ1 close to global alpha
            sample_uniform(0.0, 0.05),         # τ2 independent
        ]
        beta_list = [
            cfg["beta"],
            sample_uniform(0.001, 0.03),
        ]
        gamma_list = [
            cfg["gamma"],
            sample_uniform(0.0, 0.01),
        ]
        kappa_list = [
            cfg["kappa_tau"],
            sample_uniform(0.0, 0.05),
        ]
        tau_min_list = [cfg["tau_min"], cfg["tau_min"]]
        tau_max_list = [cfg["tau_max"], cfg["tau_max"]]
        mix_w = np.array([sample_uniform(0.3, 0.7), 0.0])
        mix_w[1] = 1.0 - mix_w[0]

        cfg["tau0_list"] = tau0_list
        cfg["alpha_list"] = alpha_list
        cfg["beta_list"] = beta_list
        cfg["gamma_list"] = gamma_list
        cfg["kappa_tau_list"] = kappa_list
        cfg["tau_min_list"] = tau_min_list
        cfg["tau_max_list"] = tau_max_list
        cfg["tau_mix_weights"] = mix_w.tolist()
        cfg["save_tau_species"] = True

    # Initial condition / noise
    cfg["seed_radius"] = 10
    cfg["noise"] = 0.02

    return cfg


# ------------------------------------------------------------
# Worker
# ------------------------------------------------------------

def config_hash(cfg):
    """
    Deterministic short hash for a config for directory naming.
    """
    s = json.dumps(cfg, sort_keys=True)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:10]


def worker(args):
    """
    Worker function for multiprocessing.
    """
    idx, cfg, base_outdir = args
    h = config_hash(cfg)
    outdir = os.path.join(base_outdir, h)
    print(f"→ [{idx}] Running {outdir}")
    run_simulation(cfg, outdir)
    print(f"✓ [{idx}] Done {outdir}")
    return outdir


# ------------------------------------------------------------
# Post-hoc analysis helpers
# ------------------------------------------------------------

def load_metrics(run_dir):
    metrics_path = Path(run_dir) / "metrics.csv"
    if not metrics_path.exists():
        return None
    try:
        df = pd.read_csv(metrics_path)
        return df
    except Exception as e:
        print("Error reading metrics for", run_dir, ":", e)
        return None


def compute_coherence_stats(df):
    coh = df["coherence"].values
    ent = df["entropy"].values
    energy = df["energy"].values
    auto = df["autocat"].values
    t = df["time"].values

    mean_coh = float(np.mean(coh))
    max_coh = float(np.max(coh))
    std_coh = float(np.std(coh))
    mean_ent = float(np.mean(ent))
    mean_energy = float(np.mean(energy))
    mean_auto = float(np.mean(auto))

    if len(t) > 1:
        coh_slope = float((coh[-1] - coh[0]) / (t[-1] - t[0]))
    else:
        coh_slope = 0.0

    # correlation between coherence and entropy
    if len(coh) > 1:
        corr_C_S = float(np.corrcoef(coh, ent)[0, 1])
    else:
        corr_C_S = np.nan

    return {
        "mean_coherence": mean_coh,
        "max_coherence": max_coh,
        "std_coherence": std_coh,
        "mean_entropy": mean_ent,
        "mean_energy": mean_energy,
        "mean_autocat": mean_auto,
        "coherence_slope": coh_slope,
        "corr_coh_entropy": corr_C_S,
    }


def load_png_gray(path):
    arr = np.array(Image.open(path).convert("L"), dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def compute_maintenance_iou(run_dir, threshold=0.3):
    """
    Compute a simple boundary persistence metric (IoU)
    between mid and final B snapshots.
    """
    p = Path(run_dir)
    snaps = sorted(p.glob("B_snapshot_*.png"))
    if len(snaps) < 2:
        return np.nan

    # pick first, middle, last for potential later use; IoU on mid vs last
    n = len(snaps)
    mid_idx = n // 2
    first = snaps[0]
    mid = snaps[mid_idx]
    last = snaps[-1]

    arr_mid = load_png_gray(mid)
    arr_last = load_png_gray(last)

    mask_mid = arr_mid > threshold
    mask_last = arr_last > threshold

    inter = np.logical_and(mask_mid, mask_last).sum()
    union = np.logical_or(mask_mid, mask_last).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def compute_tau_structure(run_dir):
    """
    Compute simple τ structure metrics from final tau_snapshot_*.png:
      - variance of τ
      - mean squared gradient (roughness)
    """
    p = Path(run_dir)
    snaps = sorted(p.glob("tau_snapshot_*.png"))
    if not snaps:
        return np.nan, np.nan

    last = snaps[-1]
    arr = load_png_gray(last)

    tau_var = float(np.var(arr))

    # gradient squared
    gy, gx = np.gradient(arr)
    grad2 = gx**2 + gy**2
    tau_grad2 = float(np.mean(grad2))

    return tau_var, tau_grad2


def summarize_runs(run_dirs, summary_outdir):
    """
    Build a runs_summary_v5.csv from a list of run_dirs.
    """
    rows = []

    for rd in run_dirs:
        rd = Path(rd)
        meta_path = rd / "meta.json"
        if not meta_path.exists():
            print("Skipping (no meta.json):", rd)
            continue

        try:
            cfg = json.load(open(meta_path, "r"))
        except Exception as e:
            print("Error reading meta for", rd, ":", e)
            continue

        df = load_metrics(rd)
        if df is None or len(df) == 0:
            print("Skipping (no metrics):", rd)
            continue

        stats = compute_coherence_stats(df)
        iou = compute_maintenance_iou(rd)
        tau_var, tau_grad2 = compute_tau_structure(rd)

        row = {
            "run_dir": str(rd),
            "alpha": cfg.get("alpha"),
            "beta": cfg.get("beta"),
            "gamma": cfg.get("gamma"),
            "feed": cfg.get("feed"),
            "kill": cfg.get("kill"),
            "kappa_tau": cfg.get("kappa_tau"),
            "tau_noise": cfg.get("tau_noise"),
            "use_multiscale_memory": cfg.get("use_multiscale_memory", False),
            "use_diffusive_nutrient": cfg.get("use_diffusive_nutrient", False),
            "num_tau_species": cfg.get("num_tau_species", 1),
            "nutrient_use": cfg.get("nutrient_use"),
            "noise": cfg.get("noise"),
            "maintenance_iou": iou,
            "tau_var_final": tau_var,
            "tau_grad2_final": tau_grad2,
        }
        row.update(stats)
        rows.append(row)

    if not rows:
        print("No valid runs to summarise.")
        return None

    summary_df = pd.DataFrame(rows)
    summary_outdir = Path(summary_outdir)
    summary_outdir.mkdir(parents=True, exist_ok=True)
    out_csv = summary_outdir / "runs_summary_v5.csv"
    summary_df.to_csv(out_csv, index=False)
    print("Wrote", out_csv, "with", len(summary_df), "runs.")
    return out_csv


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200,
                        help="Number of random configurations to run.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes.")
    parser.add_argument("--base_outdir", type=str,
                        default="outputs/dynamic_tau_v5",
                        help="Base directory for simulation outputs.")
    parser.add_argument("--no_summarize", action="store_true",
                        help="Skip post-hoc summary CSV.")
    args = parser.parse_args()

    np.random.seed()  # each process will get different states anyway

    base_outdir = args.base_outdir
    os.makedirs(base_outdir, exist_ok=True)

    # Build job list
    jobs = []
    for i in range(args.runs):
        cfg = build_random_config()
        jobs.append((i, cfg, base_outdir))

    print(f"Total simulations: {len(jobs)}")
    run_dirs = []

    if args.workers > 1:
        with Pool(args.workers) as P:
            for outdir in P.map(worker, jobs):
                run_dirs.append(outdir)
    else:
        for j in jobs:
            outdir = worker(j)
            run_dirs.append(outdir)

    print("\nSweep complete.")

    if not args.no_summarize:
        # store summaries under plots/proto_life_v5
        summary_dir = Path("plots") / "proto_life_v5"
        summarize_runs(run_dirs, summary_dir)


if __name__ == "__main__":
    main()
