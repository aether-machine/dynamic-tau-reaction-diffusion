#!/usr/bin/env python3

"""
run_sweep_v3.py
Runs parameter sweep for dynamic_tau_v3 simulation.
"""

import os, json, hashlib
import argparse
from multiprocessing import Pool
from datetime import datetime
from dynamic_tau_v3 import run_simulation

# ------------------------------------------------------------
# Build configs
# ------------------------------------------------------------
def build_configs():
    alpha_vals = [0.0, 0.01, 0.02, 0.04]
    beta_vals  = [0.001, 0.005, 0.01]
    gamma_vals = [0.0, 0.005]
    feed_vals  = [0.03, 0.035]
    kill_vals  = [0.055, 0.065]

    configs = []
    for a in alpha_vals:
        for b in beta_vals:
            for g in gamma_vals:
                for f in feed_vals:
                    for k in kill_vals:
                        cfg = {
                            "nx": 150, "ny": 150,
                            "dx": 1.0, "dy": 1.0,
                            "dt": 0.01,
                            "steps": 4000,
                            "snap_every": 200,
                            "log_every": 20,
                            "Da": 0.16, "Db": 0.08,
                            "feed": f, "kill": k,
                            "alpha": a, "beta": b, "gamma": g,
                            "tau0": 1.0, "tau_min": 0.2, "tau_max": 3.0,
                            "N0": 1.0,
                            "nutrient_use": 0.01,
                            "memory_decay": 0.01,
                            "seed_radius": 10,
                            "noise": 0.02
                        }
                        configs.append(cfg)
    return configs

# ------------------------------------------------------------
# Worker
# ------------------------------------------------------------
def worker(cfg):
    # Build unique hash
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    outdir = os.path.join("outputs", "dynamic_tau_v3", h)
    print("→ Running", outdir)
    run_simulation(cfg, outdir)
    print("✓ Done", outdir)
    return outdir

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    cfgs = build_configs()
    print(f"Total simulations: {len(cfgs)}")

    os.makedirs("outputs/dynamic_tau_v3", exist_ok=True)

    if args.workers > 1:
        with Pool(args.workers) as P:
            P.map(worker, cfgs)
    else:
        for c in cfgs:
            worker(c)

    print("\nSweep complete.")

if __name__ == "__main__":
    main()
