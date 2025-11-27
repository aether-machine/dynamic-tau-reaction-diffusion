#!/usr/bin/env python3
"""
Corrected analyzer for dynamic_tau_sweep
Matches the ACTUAL structure of dynamic_tau_sweep_index.json
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))          # /analysis
ROOT = os.path.abspath(os.path.join(BASE, ".."))           # /time-density
SWEEP_DIR = os.path.join(ROOT, "simulations", "outputs")   # where index + runs live
OUTPUT_DIR = os.path.join(SWEEP_DIR, "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_index():
    index_path = os.path.join(SWEEP_DIR, "dynamic_tau_sweep_index.json")
    print("Loading index:", index_path)

    if not os.path.exists(index_path):
        raise FileNotFoundError("Index file not found:", index_path)

    with open(index_path) as f:
        data = json.load(f)

    print("Entries found in index:", len(data))
    return data


def load_run(entry):
    """Load metrics + parameters from a single entry."""
    outdir = entry["outdir"]
    summary = entry["summary"]
    cfg = summary["cfg"]

    metrics_path = os.path.join(SWEEP_DIR, outdir.replace("outputs/", ""), "metrics.csv")
    # Example: simulations/outputs/dynamic_tau_sweep/4168.../metrics.csv

    # Normalize full path
    metrics_path = os.path.join(ROOT, "simulations", metrics_path)

    if not os.path.exists(metrics_path):
        print("WARNING: Missing metrics for", metrics_path)
        return None

    df = pd.read_csv(metrics_path)
    # Attach parameters for filtering/analysis
    for k, v in cfg.items():
        df[k] = v

    df["outdir"] = outdir
    return df


def main():
    index = load_index()

    all_dfs = []
    for entry in index:
        df = load_run(entry)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No valid runs loaded — check file paths.")

    df = pd.concat(all_dfs, ignore_index=True)
    print("Loaded dataframe shape:", df.shape)

    # Save full dataframe
    df.to_csv(os.path.join(OUTPUT_DIR, "combined_metrics.csv"), index=False)
    print("Saved combined metrics.")

    # ----------------------
    # BASIC SUMMARY PLOTS
    # ----------------------
    for param in ["alpha", "beta", "feed", "kill"]:
        if param in df.columns:
            plt.figure(figsize=(6,4))
            df.groupby(param)["coherence"].mean().plot(marker='o')
            plt.title(f"Mean Coherence vs {param}")
            plt.xlabel(param)
            plt.ylabel("Coherence")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"coherence_vs_{param}.png"))
            plt.close()

    print("Plots saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
