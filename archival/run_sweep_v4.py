#!/usr/bin/env python3

"""
run_sweep_v4.py

Parameter sweep for dynamic τ models (v3 and v4),
with optional post-sweep analysis modules.

Includes v4-specific parameters:
    - kappa_tau: curvature-coupling strength in τ
    - tau_noise: environmental τ-noise amplitude

Sweep modes:
    --mode global  : wide v3-style scan (alpha, beta, gamma, feed, kill, kappa_tau, tau_noise)
    --mode local   : fine-grained exploration around a Q-ridge anchor
                     (fixed beta, gamma, feed, kill; small grid in alpha, kappa_tau, tau_noise)

Analysis modules (use via --analyze):
    - summary       -> runs_summary.csv
    - phase         -> phase_alpha_beta.png
    - oscillon      -> best_run_snaps/, figure3_oscillon_candidate.png, cross_section_B.png
    - maintenance   -> maintenance_summary.csv, maintenance_iou_hist.png
    - metrics       -> runs_metrics.csv (trends + proto_life_score)
    - tau_structure -> tau_structure_summary.csv (+ tau_var_hist.png)

Usage examples:
    # global v4 sweep (with v4 params)
    python run_sweep_v4.py --mode global --workers 4

    # global v4 sweep + proto-life analysis
    python run_sweep_v4.py --mode global --workers 4 --analyze summary oscillon maintenance metrics tau_structure

    # local Q-ridge exploration
    python run_sweep_v4.py --mode local --tag qridge --workers 4 --analyze summary maintenance metrics tau_structure oscillon

    # v3 sweep for comparison (v4 keys ignored by v3)
    python run_sweep_v4.py --mode global --model v3 --tag v3test --workers 2
"""

import os
import json
import glob
import hashlib
import argparse
from multiprocessing import Pool

import dynamic_tau_v4 as model_v4
import dynamic_tau_v3 as model_v3  # assumes file is available

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ------------------------------------------------------------
# Config builder
# ------------------------------------------------------------

def build_configs(mode: str = "global"):
    """
    Build a list of cfg dicts for the chosen sweep mode.

    mode = "global":
        Wide sweep over alpha, beta, gamma, feed, kill,
        and small v4-specific grids in kappa_tau, tau_noise.

    mode = "local":
        Fine-grained exploration around a Q-ridge anchor
        derived from previous sweeps:
            alpha ≈ 0, beta = 0.005, gamma = 0.005,
            feed = 0.035, kill = 0.065,
            kappa_tau ≈ 0, tau_noise ≈ 0.

        Local grid:
            alpha ∈ {0.0, 0.005, 0.01}
            kappa_tau ∈ {0.0, 0.01, 0.02}
            tau_noise ∈ {0.0, 0.005, 0.01}
    """
    if mode == "global":
        # ---- global v4 sweep (384 runs) ----
        alpha_vals    = [0.0, 0.01, 0.02, 0.04]
        beta_vals     = [0.001, 0.005, 0.01]
        gamma_vals    = [0.0, 0.005]
        feed_vals     = [0.03, 0.035]
        kill_vals     = [0.055, 0.065]

        # v4-specific sweeps (keep modest)
        kappa_tau_vals = [0.0, 0.02]   # 0: v3-like, 0.02: mild curvature
        tau_noise_vals = [0.0, 0.01]   # 0: no noise, 0.01: small noise

        configs = []
        for a in alpha_vals:
            for b in beta_vals:
                for g in gamma_vals:
                    for f in feed_vals:
                        for k in kill_vals:
                            for kap in kappa_tau_vals:
                                for tn in tau_noise_vals:
                                    cfg = {
                                        "nx": 150, "ny": 150,
                                        "dx": 1.1, "dy": 1.0,
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
                                        "noise": 0.02,
                                        # v4-specific knobs
                                        "kappa_tau": kap,
                                        "tau_noise": tn,
                                    }
                                    configs.append(cfg)
        return configs

    elif mode == "local":
        # ---- local exploration around Q-ridge anchor (27 runs) ----
        # Anchor chosen from best proto_life_score run in previous sweep:
        #   alpha = 0.0, beta = 0.005, gamma = 0.005,
        #   feed = 0.035, kill = 0.065, kappa_tau = 0.0, tau_noise = 0.0
        # We keep beta, gamma, feed, kill fixed, and explore:
        #   alpha ∈ {0.0, 0.005, 0.01}
        #   kappa_tau ∈ {0.0, 0.01, 0.02}
        #   tau_noise ∈ {0.0, 0.005, 0.01}
        alpha_vals_local    = [0.0, 0.005, 0.01]
        kappa_tau_vals_local = [0.0, 0.01, 0.02]
        tau_noise_vals_local = [0.0, 0.005, 0.01]

        beta_anchor   = 0.005
        gamma_anchor  = 0.005
        feed_anchor   = 0.035
        kill_anchor   = 0.065

        configs = []
        for a in alpha_vals_local:
            for kap in kappa_tau_vals_local:
                for tn in tau_noise_vals_local:
                    cfg = {
                        "nx": 150, "ny": 150,
                        "dx": 1.1, "dy": 1.0,
                        "dt": 0.01,
                        "steps": 4000,
                        "snap_every": 200,
                        "log_every": 20,
                        "Da": 0.16, "Db": 0.08,
                        "feed": feed_anchor,
                        "kill": kill_anchor,
                        "alpha": a,
                        "beta": beta_anchor,
                        "gamma": gamma_anchor,
                        "tau0": 1.0, "tau_min": 0.2, "tau_max": 3.0,
                        "N0": 1.0,
                        "nutrient_use": 0.01,
                        "memory_decay": 0.01,
                        "seed_radius": 10,
                        "noise": 0.02,
                        "kappa_tau": kap,
                        "tau_noise": tn,
                    }
                    configs.append(cfg)
        return configs

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------------------------------------------
# Worker utils
# ------------------------------------------------------------

def make_outdir(base, cfg):
    """
    Build a unique directory name for a given config.
    """
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(base, h)


def worker(job):
    """
    Top-level worker function for multiprocessing.

    job: (model_name, base_outdir, cfg_dict)
    """
    model_name, base_outdir, cfg = job

    # choose model based on name (v3 vs v4)
    if model_name == "v4":
        model = model_v4
    else:
        model = model_v3

    outdir = make_outdir(base_outdir, cfg)
    os.makedirs(base_outdir, exist_ok=True)
    print(f"→ Running {outdir}")
    model.run_simulation(cfg, outdir)
    print(f"✓ Done   {outdir}")
    return outdir


# ------------------------------------------------------------
# Shared helper: build run summary dataframe
# ------------------------------------------------------------

def build_runs_summary_df(base_outdir):
    """
    Scan all runs under base_outdir, read metrics + summary, and
    build a DataFrame with basic per-run stats.

    Columns:
        run_dir, alpha, beta, gamma, feed, kill,
        kappa_tau, tau_noise,
        mean_coherence, max_coherence, mean_entropy, mean_autocat
    """
    runs = []
    for d in sorted(os.listdir(base_outdir)):
        run_dir = os.path.join(base_outdir, d)
        metrics_path = os.path.join(run_dir, "metrics.csv")
        meta_path = os.path.join(run_dir, "summary.json")
        if not (os.path.exists(metrics_path) and os.path.exists(meta_path)):
            continue

        try:
            m = pd.read_csv(metrics_path)
            with open(meta_path, "r") as f:
                meta = json.load(f)
            cfg = meta.get("cfg", meta)

            mean_coh = m["coherence"].mean()
            max_coh  = m["coherence"].max()
            mean_ent = m["entropy"].mean()
            mean_auto = m["autocat"].mean()

            runs.append({
                "run_dir": run_dir,
                "alpha": cfg.get("alpha"),
                "beta": cfg.get("beta"),
                "gamma": cfg.get("gamma"),
                "feed": cfg.get("feed"),
                "kill": cfg.get("kill"),
                "kappa_tau": cfg.get("kappa_tau", 0.0),
                "tau_noise": cfg.get("tau_noise", 0.0),
                "mean_coherence": mean_coh,
                "max_coherence": max_coh,
                "mean_entropy": mean_ent,
                "mean_autocat": mean_auto,
            })
        except Exception as e:
            print("Error reading run", run_dir, ":", e)

    if not runs:
        return None

    df = pd.DataFrame(runs)
    df = df.sort_values("mean_coherence", ascending=False).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Analysis modules
# ------------------------------------------------------------

def run_summary_analysis(base_outdir, plots_dir):
    df = build_runs_summary_df(base_outdir)
    if df is None:
        print("No runs to summarize in", base_outdir)
        return

    os.makedirs(plots_dir, exist_ok=True)
    summary_path = os.path.join(plots_dir, "runs_summary.csv")
    df.to_csv(summary_path, index=False)
    print("Wrote", summary_path, "with", len(df), "runs.")


def run_phase_map_analysis(base_outdir, plots_dir):
    rows = []
    for d in sorted(os.listdir(base_outdir)):
        run_dir = os.path.join(base_outdir, d)
        metrics_path = os.path.join(run_dir, "metrics.csv")
        meta_path = os.path.join(run_dir, "summary.json")
        if not (os.path.exists(metrics_path) and os.path.exists(meta_path)):
            continue

        try:
            m = pd.read_csv(metrics_path)
            with open(meta_path, "r") as f:
                meta = json.load(f)
            cfg = meta.get("cfg", meta)

            m["alpha"] = cfg.get("alpha")
            m["beta"] = cfg.get("beta")
            rows.append(m[["time", "coherence", "alpha", "beta"]])
        except Exception as e:
            print("Error loading for phase map:", run_dir, ":", e)

    if not rows:
        print("No runs for phase map in", base_outdir)
        return

    df = pd.concat(rows, ignore_index=True)
    phase = df.groupby(["alpha", "beta"])["coherence"].mean().unstack()

    os.makedirs(plots_dir, exist_ok=True)
    outpath = os.path.join(plots_dir, "phase_alpha_beta.png")

    plt.figure(figsize=(6, 5))
    plt.imshow(phase, origin="lower", cmap="viridis", aspect="auto")
    plt.xticks(range(len(phase.columns)), phase.columns)
    plt.yticks(range(len(phase.index)), phase.index)
    plt.xlabel("beta")
    plt.ylabel("alpha")
    plt.title("Mean Coherence (alpha × beta)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved phase map to", outpath)


def run_oscillon_analysis(base_outdir, plots_dir):
    df = build_runs_summary_df(base_outdir)
    if df is None:
        print("No runs for oscillon analysis in", base_outdir)
        return

    os.makedirs(plots_dir, exist_ok=True)

    best = df.iloc[0]
    best_dir = best["run_dir"]
    print("Oscillon analysis: best run:", best_dir)

    snaps_dir = os.path.join(plots_dir, "best_run_snaps")
    os.makedirs(snaps_dir, exist_ok=True)

    def pick_and_copy(pattern, dest_prefix):
        snaps = sorted(glob.glob(os.path.join(best_dir, pattern)))
        if not snaps:
            return []
        n = len(snaps)
        idxs = [0] if n == 1 else [0, n // 2, n - 1]
        copied = []
        for i in idxs:
            src = snaps[i]
            dst = os.path.join(
                snaps_dir,
                f"{dest_prefix}_{i:02d}{os.path.splitext(src)[1]}"
            )
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
            copied.append(dst)
        return copied

    b_copied = pick_and_copy("B_snapshot_*.png", "B")
    tau_copied = pick_and_copy("tau_snapshot_*.png", "tau")

    print(
        f"Copied snapshots for oscillon candidate: "
        f"{len(b_copied)} B, {len(tau_copied)} tau."
    )

    if b_copied:
        imgs = [Image.open(p).convert("RGB") for p in b_copied]
        w, h = imgs[0].size
        canvas = Image.new("RGB", (w * len(imgs), h))
        for i, im in enumerate(imgs):
            canvas.paste(im, (i * w, 0))
        panel_path = os.path.join(plots_dir, "figure3_oscillon_candidate.png")
        canvas.save(panel_path)
        print("Saved oscillon candidate panel to", panel_path)

    b_snaps = sorted(glob.glob(os.path.join(best_dir, "B_snapshot_*.png")))
    if b_snaps:
        final_b_path = b_snaps[-1]
        arr = np.array(Image.open(final_b_path).convert("L"), dtype=float)
        center = arr[arr.shape[0] // 2, :]
        plt.figure(figsize=(10, 3))
        plt.plot(center, lw=1.2)
        plt.title("Centerline cross-section of final B snapshot (oscillon run)")
        plt.xlabel("x")
        plt.ylabel("B intensity")
        plt.tight_layout()
        cs_path = os.path.join(plots_dir, "cross_section_B.png")
        plt.savefig(cs_path)
        plt.close()
        print("Saved cross section to", cs_path)


def run_maintenance_analysis(base_outdir, plots_dir):
    rows = []

    for d in sorted(os.listdir(base_outdir)):
        run_dir = os.path.join(base_outdir, d)
        meta_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)
        cfg = meta.get("cfg", meta)

        b_snaps = sorted(glob.glob(os.path.join(run_dir, "B_snapshot_*.png")))
        if len(b_snaps) < 2:
            continue

        mid_path = b_snaps[len(b_snaps) // 3]
        final_path = b_snaps[-1]

        try:
            img_mid = np.array(Image.open(mid_path).convert("L"), dtype=float)
            img_fin = np.array(Image.open(final_path).convert("L"), dtype=float)

            if img_mid.max() <= 0 or img_fin.max() <= 0:
                iou = np.nan
            else:
                t1 = 0.5 * img_mid.max()
                t2 = 0.5 * img_fin.max()
                m1 = img_mid > t1
                m2 = img_fin > t2

                inter = np.logical_and(m1, m2).sum()
                union = np.logical_or(m1, m2).sum()
                iou = np.nan if union == 0 else inter / union
        except Exception as e:
            print("Error computing maintenance IoU for", run_dir, ":", e)
            iou = np.nan

        rows.append({
            "run_dir": run_dir,
            "alpha": cfg.get("alpha"),
            "beta": cfg.get("beta"),
            "gamma": cfg.get("gamma"),
            "feed": cfg.get("feed"),
            "kill": cfg.get("kill"),
            "kappa_tau": cfg.get("kappa_tau", 0.0),
            "tau_noise": cfg.get("tau_noise", 0.0),
            "maintenance_iou": iou,
        })

    if not rows:
        print("No runs for maintenance analysis in", base_outdir)
        return

    df = pd.DataFrame(rows)
    os.makedirs(plots_dir, exist_ok=True)
    out_csv = os.path.join(plots_dir, "maintenance_summary.csv")
    df.to_csv(out_csv, index=False)
    print("Wrote maintenance summary to", out_csv)

    valid = df["maintenance_iou"].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(valid.values, bins=20, alpha=0.8)
        plt.xlabel("Maintenance IoU (mid vs final B)")
        plt.ylabel("Count")
        plt.title("Distribution of maintenance scores across runs")
        plt.tight_layout()
        hist_path = os.path.join(plots_dir, "maintenance_iou_hist.png")
        plt.savefig(hist_path)
        plt.close()
        print("Saved maintenance IoU histogram to", hist_path)


def run_metrics_analysis(base_outdir, plots_dir):
    rows = []

    for d in sorted(os.listdir(base_outdir)):
        run_dir = os.path.join(base_outdir, d)
        metrics_path = os.path.join(run_dir, "metrics.csv")
        meta_path = os.path.join(run_dir, "summary.json")
        if not (os.path.exists(metrics_path) and os.path.exists(meta_path)):
            continue

        try:
            m = pd.read_csv(metrics_path)
            with open(meta_path, "r") as f:
                meta = json.load(f)
            cfg = meta.get("cfg", meta)

            if not {"time", "coherence", "entropy", "autocat"}.issubset(m.columns):
                print("Skipping run (missing columns):", run_dir)
                continue

            t = m["time"].to_numpy(dtype=float)
            c = m["coherence"].to_numpy(dtype=float)
            e = m["entropy"].to_numpy(dtype=float)
            a = m["autocat"].to_numpy(dtype=float)

            if len(t) > 1:
                coh_slope = np.polyfit(t, c, 1)[0]
                ent_slope = np.polyfit(t, e, 1)[0]
                rho_ce = np.corrcoef(c, e)[0, 1]
                rho_ca = np.corrcoef(c, a)[0, 1]
            else:
                coh_slope = np.nan
                ent_slope = np.nan
                rho_ce = np.nan
                rho_ca = np.nan

            mean_coh = np.nanmean(c)
            max_coh  = np.nanmax(c)
            mean_ent = np.nanmean(e)
            mean_auto = np.nanmean(a)

            rows.append({
                "run_dir": run_dir,
                "alpha": cfg.get("alpha"),
                "beta": cfg.get("beta"),
                "gamma": cfg.get("gamma"),
                "feed": cfg.get("feed"),
                "kill": cfg.get("kill"),
                "kappa_tau": cfg.get("kappa_tau", 0.0),
                "tau_noise": cfg.get("tau_noise", 0.0),
                "mean_coherence": mean_coh,
                "max_coherence": max_coh,
                "mean_entropy": mean_ent,
                "mean_autocat": mean_auto,
                "coh_slope": coh_slope,
                "ent_slope": ent_slope,
                "rho_coh_entropy": rho_ce,
                "rho_coh_autocat": rho_ca,
            })
        except Exception as e:
            print("Error in metrics analysis for run", run_dir, ":", e)

    if not rows:
        print("No runs for metrics analysis in", base_outdir)
        return

    df = pd.DataFrame(rows)

    def zscore_series(s: pd.Series) -> pd.Series:
        arr = s.to_numpy(dtype=float)
        mu = np.nanmean(arr)
        sigma = np.nanstd(arr)
        if not np.isfinite(sigma) or sigma == 0:
            return pd.Series(np.zeros_like(arr), index=s.index)
        return pd.Series((arr - mu) / sigma, index=s.index)

    for col in [
        "mean_coherence",
        "mean_entropy",
        "coh_slope",
        "ent_slope",
        "rho_coh_entropy",
        "rho_coh_autocat",
    ]:
        if col in df.columns:
            df[f"z_{col}"] = zscore_series(df[col])

    df["proto_life_score"] = (
        df.get("z_mean_coherence", 0.0)
        - df.get("z_mean_entropy", 0.0)
        + df.get("z_coh_slope", 0.0)
        - df.get("z_ent_slope", 0.0)
        - df.get("z_rho_coh_entropy", 0.0)
        + df.get("z_rho_coh_autocat", 0.0)
    )

    df = df.sort_values("proto_life_score", ascending=False).reset_index(drop=True)

    os.makedirs(plots_dir, exist_ok=True)
    out_csv = os.path.join(plots_dir, "runs_metrics.csv")
    df.to_csv(out_csv, index=False)
    print("Wrote metrics summary (with proto_life_score) to", out_csv)


def run_tau_structure_analysis(base_outdir, plots_dir):
    rows = []

    for d in sorted(os.listdir(base_outdir)):
        run_dir = os.path.join(base_outdir, d)
        meta_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)
        cfg = meta.get("cfg", meta)

        tau_snaps = sorted(glob.glob(os.path.join(run_dir, "tau_snapshot_*.png")))
        if not tau_snaps:
            continue

        final_tau_path = tau_snaps[-1]

        try:
            arr = np.array(Image.open(final_tau_path).convert("L"), dtype=float)
            if arr.max() > 0:
                arr_norm = arr / arr.max()
            else:
                arr_norm = arr.copy()

            tau_var = float(arr_norm.var())

            dx = arr_norm[:, 1:] - arr_norm[:, :-1]
            dy = arr_norm[1:, :] - arr_norm[:-1, :]
            grad2 = float(np.mean(dx ** 2) + np.mean(dy ** 2))

            rows.append({
                "run_dir": run_dir,
                "alpha": cfg.get("alpha"),
                "beta": cfg.get("beta"),
                "gamma": cfg.get("gamma"),
                "feed": cfg.get("feed"),
                "kill": cfg.get("kill"),
                "kappa_tau": cfg.get("kappa_tau", 0.0),
                "tau_noise": cfg.get("tau_noise", 0.0),
                "tau_var_final": tau_var,
                "tau_grad2_final": grad2,
            })
        except Exception as e:
            print("Error in tau structure analysis for", run_dir, ":", e)

    if not rows:
        print("No runs for tau_structure analysis in", base_outdir)
        return

    df = pd.DataFrame(rows)
    os.makedirs(plots_dir, exist_ok=True)
    out_csv = os.path.join(plots_dir, "tau_structure_summary.csv")
    df.to_csv(out_csv, index=False)
    print("Wrote tau structure summary to", out_csv)

    valid = df["tau_var_final"].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(valid.values, bins=20, alpha=0.8)
        plt.xlabel("tau_var_final")
        plt.ylabel("Count")
        plt.title("Distribution of τ variance (final snapshot)")
        plt.tight_layout()
        hist_path = os.path.join(plots_dir, "tau_var_hist.png")
        plt.savefig(hist_path)
        plt.close()
        print("Saved τ variance histogram to", hist_path)


ANALYSIS_REGISTRY = {
    "summary":       run_summary_analysis,
    "phase":         run_phase_map_analysis,
    "oscillon":      run_oscillon_analysis,
    "maintenance":   run_maintenance_analysis,
    "metrics":       run_metrics_analysis,
    "tau_structure": run_tau_structure_analysis,
}


def run_selected_analyses(base_outdir, tag, modules):
    plots_dir = os.path.join("plots", f"proto_life_{tag}")
    os.makedirs(plots_dir, exist_ok=True)

    if not modules:
        print("No analysis modules selected.")
        return

    for name in modules:
        fn = ANALYSIS_REGISTRY.get(name)
        if fn is None:
            print(f"[warn] Unknown analysis module: {name}")
            continue
        print(f"\n[analysis] Running module '{name}'...")
        fn(base_outdir, plots_dir)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers.")
    parser.add_argument("--model", choices=["v3", "v4"], default="v4",
                        help="Which simulator to use.")
    parser.add_argument("--mode", choices=["global", "local"], default="global",
                        help="Sweep mode: 'global' for wide scan, 'local' for Q-ridge exploration.")
    parser.add_argument("--tag", type=str, default="v4",
                        help="Label for this sweep (affects output dirs and plot prefix).")
    parser.add_argument("--analyze", nargs="*",
                        help=("Optional analysis modules to run after sweep "
                              "(e.g. 'summary phase oscillon maintenance metrics tau_structure')."))
    args = parser.parse_args()

    # Choose base output directory; keep global layout stable, add mode suffix for local
    if args.model == "v4":
        base_name = f"dynamic_tau_v4_{args.tag}"
    else:
        base_name = f"dynamic_tau_v3_{args.tag}"

    if args.mode == "local":
        base_name += "_local"

    base_outdir = os.path.join("outputs", base_name)

    cfgs = build_configs(mode=args.mode)
    print(f"Total simulations ({args.mode} mode): {len(cfgs)}")
    os.makedirs(base_outdir, exist_ok=True)

    jobs = [(args.model, base_outdir, cfg) for cfg in cfgs]

    if args.workers > 1:
        with Pool(args.workers) as P:
            P.map(worker, jobs)
    else:
        for job in jobs:
            worker(job)

    print("\nSweep complete.")

    if args.analyze is not None:
        print("\nStarting post-sweep analysis...")
        run_selected_analyses(base_outdir, args.tag, args.analyze)
        print("Analysis complete.")


if __name__ == "__main__":
    main()
