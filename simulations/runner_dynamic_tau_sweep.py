#!/usr/bin/env python3
"""
runner_dynamic_tau_sweep.py

Parameter sweep runner for dynamic-tau reaction-diffusion experiments.

Saves:
 - outputs/<expname>/<param_hash>/B_snapshot_*.png
 - outputs/<expname>/<param_hash>/tau_snapshot_*.png
 - outputs/<expname>/<param_hash>/metrics.csv
 - outputs/<expname>/<param_hash>/meta.json

Usage:
    python runner_dynamic_tau_sweep.py --workers 4

Notes:
 - Designed to be robust and conservative with memory.
 - Each run is independent; ensemble runs are reproducible by providing --seed-base.
"""

import os
import json
import math
import argparse
import hashlib
import itertools
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-friendly backend
import matplotlib.pyplot as plt

# -------------------------
# Simulation function (based on your Simulation 4)
# -------------------------
def run_simulation(cfg):
    """
    Run one simulation with parameters in cfg (dict).
    Returns path to output directory and a metrics summary dict.
    """
    # Setup RNG
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)

    # Build unique param hash for folder name
    param_string = json.dumps({k: cfg[k] for k in sorted(cfg.keys())}, sort_keys=True)
    param_hash = hashlib.sha1(param_string.encode("utf-8")).hexdigest()[:10]
    outdir = os.path.join(cfg["outbase"], cfg["expname"], param_hash)
    os.makedirs(outdir, exist_ok=True)

    # Save meta
    meta = {
        "created": datetime.utcnow().isoformat()+"Z",
        "param_hash": param_hash,
        "cfg": cfg,
        "param_string": param_string,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Grid and time parameters (kept modest; tweak in cfg)
    nx = cfg.get("nx", 120)
    ny = cfg.get("ny", 120)
    dx = cfg.get("dx", 1.0)
    dy = cfg.get("dy", 1.0)
    dt = cfg.get("dt", 0.01)
    steps = cfg.get("steps", 3000)
    snap_every = cfg.get("snap_every", 300)

    # RD parameters
    Da = cfg.get("Da", 0.16)
    Db = cfg.get("Db", 0.08)
    feed = cfg["feed"]
    kill = cfg["kill"]

    # tau dynamics params
    tau0 = cfg.get("tau0", 1.0)
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    tau_min = cfg.get("tau_min", 0.2)
    tau_max = cfg.get("tau_max", 3.0)

    # Initialize fields
    A = np.ones((ny, nx), dtype=float)
    B = np.zeros((ny, nx), dtype=float)

    # initial central perturbation block
    r = cfg.get("seed_radius", 10)
    cy = ny // 2
    cx = nx // 2
    A[cy-r:cy+r, cx-r:cx+r] = 0.5
    B[cy-r:cy+r, cx-r:cx+r] = 0.25

    # noise
    noise_amp = cfg.get("noise_amp", 0.02)
    A += noise_amp * np.random.rand(ny, nx)
    B += noise_amp * np.random.rand(ny, nx)

    # tau initial
    tau = tau0 * np.ones((ny, nx), dtype=float)
    # optional seeded tau hot-spot
    if cfg.get("seed_tau_hotspot", False):
        hx = cfg.get("hot_x", nx//3)
        hy = cfg.get("hot_y", ny//3)
        hr = cfg.get("hot_r", 6)
        tau[hy-hr:hy+hr, hx-hr:hx+hr] = tau0 + cfg.get("hot_delta", 0.8)

    # Laplacian helper
    def laplacian(Z):
        return (
            -4*Z
            + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
            + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
        ) / (dx*dy)

    # metrics collectors
    times = []
    coherence_ts = []
    energy_ts = []
    entropy_ts = []
    autocat_ts = []

    def shannon_entropy(field):
        flat = field.flatten().astype(float)
        eps = 1e-12
        p = flat - flat.min()
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / (s + eps)
        p = np.maximum(p, eps)
        return -np.sum(p * np.log(p))

    # Main loop
    for t in range(steps):
        lapA = laplacian(A)
        lapB = laplacian(B)
        reaction = A * (B**2)
        dA = Da * lapA - reaction + feed * (1 - A)
        dB = Db * lapB + reaction - (kill + feed) * B

        A += (dA * dt * tau)
        B += (dB * dt * tau)

        # numerical stability
        np.clip(A, 0.0, 2.0, out=A)
        np.clip(B, 0.0, 2.0, out=B)

        # compute activity S (abs(reaction) + 0.5*|grad B|)
        gradBx = (np.roll(B, -1, axis=1) - np.roll(B, 1, axis=1)) / (2*dx)
        gradBy = (np.roll(B, -1, axis=0) - np.roll(B, 1, axis=0)) / (2*dy)
        gradB_mag = np.sqrt(gradBx**2 + gradBy**2)
        S_activity = np.abs(reaction) + 0.5 * gradB_mag

        # update tau slowly
        tau += dt * (alpha * S_activity - beta * (tau - tau0))
        np.clip(tau, tau_min, tau_max, out=tau)

        # logging
        if (t % cfg.get("log_every", 10)) == 0:
            M = A + 1j * B
            coherence = np.mean(np.abs(M)**2)
            energy = 0.5 * np.sum(A**2 + B**2) / (nx*ny)
            entropy = shannon_entropy(B)
            autocat = np.mean(reaction)
            times.append(t * dt)
            coherence_ts.append(coherence)
            energy_ts.append(energy)
            entropy_ts.append(entropy)
            autocat_ts.append(autocat)

        # snapshots
        if (t % snap_every) == 0:
            idx = t // snap_every
            fnB = os.path.join(outdir, f"B_snapshot_{idx:04d}.png")
            fntau = os.path.join(outdir, f"tau_snapshot_{idx:04d}.png")

            fig, ax = plt.subplots(figsize=(4,4), dpi=120)
            ax.imshow(B, cmap='magma', origin='lower')
            ax.set_title(f"B t={t*dt:.2f}")
            ax.axis('off')
            plt.tight_layout()
            fig.savefig(fnB)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4,4), dpi=120)
            ax.imshow(tau, cmap='viridis', origin='lower')
            ax.set_title(f"tau t={t*dt:.2f}")
            ax.axis('off')
            plt.tight_layout()
            fig.savefig(fntau)
            plt.close(fig)

    # Save metrics CSV
    import csv
    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "coherence", "energy", "entropy", "autocat"])
        for i in range(len(times)):
            writer.writerow([times[i], coherence_ts[i], energy_ts[i], entropy_ts[i], autocat_ts[i]])

    # Save summary JSON
    summary = {
        "param_hash": param_hash,
        "cfg": cfg,
        "metrics_file": "metrics.csv",
        "snapshots": sorted([f for f in os.listdir(outdir) if f.startswith("B_snapshot_")]),
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return outdir, summary

# -------------------------
# Worker wrapper for Pool
# -------------------------
def worker_run(cfg):
    try:
        outdir, summary = run_simulation(cfg)
        print(f"[OK] Done: {outdir}")
        return {"status": "ok", "outdir": outdir, "summary": summary}
    except Exception as e:
        print(f"[ERR] Exception for cfg {cfg}: {e}")
        return {"status": "error", "err": str(e), "cfg": cfg}

# -------------------------
# CLI and grid builder
# -------------------------
def build_param_grid(alpha_vals, beta_vals, feed_vals, kill_vals, ensemble=3, seed_base=1000):
    grid = []
    for a, b, f, k in itertools.product(alpha_vals, beta_vals, feed_vals, kill_vals):
        for s in range(ensemble):
            cfg = {
                "expname": "dynamic_tau_sweep",
                "outbase": "outputs",
                "seed": seed_base + s,
                # simulation grid size/time (safe defaults)
                "nx": 120, "ny": 120, "dx": 1.0, "dy": 1.0,
                "dt": 0.01, "steps": 3000, "snap_every": 300,
                "log_every": 10,
                "Da": 0.16, "Db": 0.08,
                "feed": f, "kill": k,
                "tau0": 1.0, "alpha": a, "beta": b, "tau_min": 0.2, "tau_max": 3.0,
                "seed_radius": 10, "noise_amp": 0.02,
                "seed_tau_hotspot": False,
            }
            grid.append(cfg)
    return grid

def parse_args():
    parser = argparse.ArgumentParser(description="Parameter sweep runner for dynamic-tau RD")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (multiprocessing Pool)")
    parser.add_argument("--ensemble", type=int, default=3, help="Ensemble size per parameter tuple")
    parser.add_argument("--seed-base", type=int, default=1000, help="Base seed for ensembles")
    parser.add_argument("--outdir", type=str, default="outputs", help="Base output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    # Example parameter ranges (tweak as desired)
    alpha_vals = [0.0, 0.01, 0.02, 0.04]
    beta_vals = [0.001, 0.005, 0.01]
    feed_vals = [0.03, 0.035, 0.04]
    kill_vals = [0.055, 0.065]
    ensemble = args.ensemble

    # Build grid
    grid = build_param_grid(alpha_vals, beta_vals, feed_vals, kill_vals, ensemble=ensemble, seed_base=args.seed_base)
    print(f"Total runs to execute: {len(grid)}")

    # Update outbase & expname inline (optional)
    for i, cfg in enumerate(grid):
        cfg["outbase"] = args.outdir
        cfg["expname"] = "dynamic_tau_sweep"

    # Run with multiprocessing Pool
    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            results = pool.map(worker_run, grid)
    else:
        results = [worker_run(cfg) for cfg in grid]

    # Summarize results into a master index
    master_index = []
    for r in results:
        if r.get("status") == "ok":
            outdir = r["outdir"]
            summary = r["summary"]
            master_index.append({"outdir": outdir, "summary": summary})
        else:
            master_index.append(r)

    idx_path = os.path.join(args.outdir, "dynamic_tau_sweep_index.json")
    with open(idx_path, "w") as fh:
        json.dump(master_index, fh, indent=2)

    print("Sweep complete. Index saved to:", idx_path)

if __name__ == "__main__":
    main()
