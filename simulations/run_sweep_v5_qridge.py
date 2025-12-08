#!/usr/bin/env python3

"""
run_sweep_v5_qridge.py

Focused parameter sweep for dynamic_tau_v5.py around a "Q-ridge" regime
where v5 has already shown strong proto-life behaviour.

Differences from run_sweep_v5.py:
 - Narrowed parameter ranges for feed, kill, alpha, beta, gamma, kappa_tau, tau_noise
 - Same modules (multiscale memory, diffusive nutrient, multi-τ) but sampled
   within this narrower strip
 - Uses the same summary logic (coherence, entropy, maintenance IoU, τ-structure)

Outputs:
 - Simulations under: outputs/dynamic_tau_v5_qridge/<hash>/
 - Summary CSV under: plots/proto_life_v5/ (re-uses same analysis path)
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
# Narrow-range config sampling (Q-ridge region)
# ------------------------------------------------------------

def sample_uniform(a, b):
    return float(np.random.uniform(a, b))


def build_qridge_config():
    """
    Build a single configuration sampled from a narrower "Q-ridge" region
    inferred from previous v5 sweeps.

    Philosophy:
      - feed: moderately high
      - kill: high (harsh pruning, but with feed)
      - alpha: moderate
      - beta: small–moderate
      - gamma: small–moderate (τ–nutrient coupling)
      - kappa_tau: nonzero, moderate
      - tau_noise: small
      - modules: same as v5, but we only explore local toggles
    """

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

    # Focused Gray–Scott region
    cfg["feed"] = sample_uniform(0.038, 0.048)   # moderately high
    cfg["kill"] = sample_uniform(0.068, 0.076)   # high

    # τ single-species baseline
    cfg["tau0"] = 1.0
    cfg["tau_min"] = 0.2
    cfg["tau_max"] = 3.0

    # Memory → τ couplings (moderate bands)
    cfg["alpha"] = sample_uniform(0.015, 0.045)   # memory feedback
    cfg["beta"] = sample_uniform(0.003, 0.015)    # relaxation
    cfg["gamma"] = sample_uniform(0.002, 0.010)   # nutrient coupling

    # τ geometry & noise
    cfg["kappa_tau"] = sample_uniform(0.01, 0.05)   # τ diffusion
    cfg["tau_noise"] = sample_uniform(0.0, 0.015)   # small jitter

    # Nutrient
    cfg["N0"] = 1.0
    cfg["nutrient_use"] = sample_uniform(0.008, 0.025)

    # Memory: bias towards multiscale sometimes, but keep single-memory common
    use_multiscale = np.random.rand() < 0.4   # 40% multiscale, 60% single
    cfg["use_multiscale_memory"] = bool(use_multiscale)

    if use_multiscale:
        cfg["memory_fast_decay"] = sample_uniform(0.05, 0.25)
        cfg["memory_slow_decay"] = sample_uniform(0.0015, 0.015)
        w_fast = sample_uniform(0.4, 0.8)
        w_slow = 1.0 - w_fast
        cfg["memory_fast_weight"] = w_fast
        cfg["memory_slow_weight"] = w_slow
        cfg["memory_decay"] = 0.01   # not used, but set sensibly
    else:
        cfg["memory_decay"] = sample_uniform(0.004, 0.03)

    # Nutrient module: 50/50 mix of diffusive vs local depletion
    use_diffusive_nutrient = np.random.rand() < 0.5
    cfg["use_diffusive_nutrient"] = bool(use_diffusive_nutrient)
    if use_diffusive_nutrient:
        cfg["Dn"] = sample_uniform(0.02, 0.15)
        cfg["nutrient_feed_rate"] = sample_uniform(0.005, 0.04)

    # Multi-τ species: mostly single-τ, but some 2-τ for structure variations
    if np.random.rand() < 0.25:
        num_tau = 2
    else:
        num_tau = 1
    cfg["num_tau_species"] = num_tau

    if num_tau == 1:
        pass
    else:
        tau0_list = [1.0, 1.0]
        alpha_list = [
            cfg["alpha"],
            sample_uniform(0.0, 0.045),
        ]
        beta_list = [
            cfg["beta"],
            sample_uniform(0.003, 0.02),
        ]
        gamma_list = [
            cfg["gamma"],
            sample_uniform(0.0, 0.010),
        ]
        kappa_list = [
            cfg["kappa_tau"],
            sample_uniform(0.01, 0.06),
        ]
        tau_min_list = [cfg["tau_min"], cfg["tau_min"]]
        tau_max_list = [cfg["tau_max"], cfg["tau_max"]]

        mix_w = np.array([sample_uniform(0.4, 0.8), 0.0])
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

    # tag for later filtering
    cfg["regime_tag"] = "qridge_v5"

    return cfg


# ------------------------------------------------------------
# Worker logic
# ------------------------------------------------------------

def config_hash(cfg):
    s = json.dumps(cfg, sort_keys=True)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:10]


def worker(args):
    idx, cfg, base_outdir = args
    h = config_hash(cfg)
    outdir = os.path.join(base_outdir, h)
    print(f"→ [{idx}] Running {outdir}")
    run_simulation(cfg, outdir)
    print(f"✓ [{idx}] Done {outdir}")
    return outdir


# ------------------------------------------------------------
# Summary (re-uses same metrics as v5 sweep)
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
    p = Path(run_dir)
    snaps = sorted(p.glob("B_snapshot_*.png"))
    if len(snaps) < 2:
        return np.nan

    n = len(snaps)
    mid_idx = n // 2
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
    p = Path(run_dir)
    snaps = sorted(p.glob("tau_snapshot_*.png"))
    if not snaps:
        return np.nan, np.nan

    last = snaps[-1]
    arr = load_png_gray(last)

    tau_var = float(np.var(arr))
    gy, gx = np.gradient(arr)
    grad2 = gx**2 + gy**2
    tau_grad2 = float(np.mean(grad2))

    return tau_var, tau_grad2


def summarize_runs(run_dirs, summary_outdir):
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
            "regime_tag": cfg.get("regime_tag", "qridge_v5"),
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
    out_csv = summary_outdir / "runs_summary_v5_qridge.csv"
    summary_df.to_csv(out_csv, index=False)
    print("Wrote", out_csv, "with", len(summary_df), "runs.")
    return out_csv


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200,
                        help="Number of Q-ridge configurations to run.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes.")
    parser.add_argument("--base_outdir", type=str,
                        default="outputs/dynamic_tau_v5_qridge",
                        help="Base directory for Q-ridge simulation outputs.")
    parser.add_argument("--no_summarize", action="store_true",
                        help="Skip summary CSV.")
    args = parser.parse_args()

    np.random.seed()

    base_outdir = args.base_outdir
    os.makedirs(base_outdir, exist_ok=True)

    jobs = []
    for i in range(args.runs):
        cfg = build_qridge_config()
        jobs.append((i, cfg, base_outdir))

    print(f"Total Q-ridge simulations: {len(jobs)}")
    run_dirs = []

    if args.workers > 1:
        with Pool(args.workers) as P:
            for outdir in P.map(worker, jobs):
                run_dirs.append(outdir)
    else:
        for j in jobs:
            outdir = worker(j)
            run_dirs.append(outdir)

    print("\nQ-ridge sweep complete.")

    if not args.no_summarize:
        summary_dir = Path("plots") / "proto_life_v5"
        summarize_runs(run_dirs, summary_dir)


if __name__ == "__main__":
    main()
