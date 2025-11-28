# Analysis Readme — Dynamic Tau Sweep

This document explains how to aggregate and visualise results from the sweep.

## 1. Master index
Open `outputs/dynamic_tau_sweep_index.json` which lists all runs and their `outdir`.

## 2. Per-run metrics
Each run directory contains `metrics.csv` with time series:
- time, coherence, energy, entropy, autocat

And `meta.json` with parameters.

## 3. Aggregation script (outline)
Use Python/pandas to:
- Read the master index.
- For each run, read meta.json and metrics.csv and extract key summary stats:
  - final_coherence = last coherence value
  - max_coherence, mean_autocat, persistence (time at which autocat < threshold)
- Build a dataframe with columns: alpha, beta, feed, kill, seed, final_coherence, max_coherence, persistence, outdir.

## 4. Phase diagram
Pivot the dataframe to compute the probability of pocket formation (e.g., final_coherence > threshold) for each (alpha,beta) pair (averaging over ensemble & feed/kill ranges or fixing them). Plot as heatmap.

## 5. Example plotting
- Time-series overlays for representative runs.
- Montage of snapshots (B images) for each regime.
- Attractor plots: mean(B) vs mean(tau) using the metrics.csv series.

## 6. Statistical tests
- Bootstrap to estimate confidence intervals on pocket formation probability.
- Permutation test comparing alpha=0 control vs alpha>0 families.
