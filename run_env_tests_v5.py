#!/usr/bin/env python3
"""
run_env_tests_v5.py

Environmental robustness tests for selected dynamic_tau_v5_qridge runs.

For each selected candidate run (identified by its hash folder under
outputs/dynamic_tau_v5_qridge/), this script:

  1. Loads the original configuration from meta.json.
  2. Constructs several environment variants by tweaking feed and kill:
       - baseline       (original cfg)
       - feed_low       (feed * 0.8)
       - feed_high      (feed * 1.2)
       - kill_low       (kill * 0.8)
       - kill_high      (kill * 1.2)
  3. Runs dynamic_tau_v5.run_simulation(cfg, outdir) for each variant,
     writing results under outputs/dynamic_tau_v5_env/<hash>/<variant>/.
  4. After simulations, compares the final B snapshot of each variant
     to the baseline final B for the same candidate using:
       - IoU of the active region (simple threshold on B)
       - Pixelwise correlation inside the union mask

Outputs:
  - Simulation directories:
        outputs/dynamic_tau_v5_env/<hash>/<variant>/
  - Summary CSV:
        plots/proto_life_v5/env_tests_summary.csv

This does NOT modify dynamic_tau_v5.py; it simply reuses run_simulation
with slightly altered parameters, so it’s safe to drop in.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from simulations.dynamic_tau_v5 import run_simulation  # adjust import if your layout differs


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_cfg_from_meta(run_dir: Path):
    """
    Load cfg from meta.json or summary.json in a run directory.
    Assumes the file contains { "cfg": {...}, ... } or is itself the cfg dict.
    """
    for name in ("meta.json", "summary.json"):
        p = run_dir / name
        if p.exists():
            with open(p, "r") as fh:
                meta = json.load(fh)
            if isinstance(meta, dict) and "cfg" in meta:
                return meta["cfg"]
            elif isinstance(meta, dict):
                return meta
    raise FileNotFoundError(f"No meta.json or summary.json with cfg in {run_dir}")


def make_env_variants(base_cfg: dict):
    """
    Build a dict of named environment variant cfgs from a base cfg.
    We tweak only feed and kill here, but you can extend this later.
    """
    variants = {}

    feed = base_cfg.get("feed", base_cfg.get("f", None))
    kill = base_cfg.get("kill", base_cfg.get("k", None))

    if feed is None or kill is None:
        raise ValueError("Base cfg must contain 'feed' and 'kill' (or 'f' and 'k').")

    # Helper to copy cfg and tweak one field
    def clone_with(name, feed_factor=1.0, kill_factor=1.0):
        cfg = dict(base_cfg)  # shallow copy is enough for simple types
        new_feed = max(feed * feed_factor, 0.0)
        new_kill = max(kill * kill_factor, 0.0)
        if "feed" in cfg:
            cfg["feed"] = new_feed
        if "f" in cfg:
            cfg["f"] = new_feed
        if "kill" in cfg:
            cfg["kill"] = new_kill
        if "k" in cfg:
            cfg["k"] = new_kill
        cfg["env_variant"] = name
        return name, cfg

    # Baseline: unchanged
    variants["baseline"] = dict(base_cfg, env_variant="baseline")

    # Slight perturbations (20% up/down)
    for name, cfg in [
        clone_with("feed_low",  feed_factor=0.8, kill_factor=1.0),
        clone_with("feed_high", feed_factor=1.2, kill_factor=1.0),
        clone_with("kill_low",  feed_factor=1.0, kill_factor=0.8),
        clone_with("kill_high", feed_factor=1.0, kill_factor=1.2),
    ]:
        variants[name] = cfg

    return variants


def load_png_gray(path: Path):
    arr = np.array(Image.open(path).convert("L"), dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def get_last_B_snapshot(run_dir: Path):
    """
    Return path and array for the last B snapshot in a run directory.
    Supports B_snapshot_*.png or legacy snapshot_*.png.
    """
    snaps = sorted(run_dir.glob("B_snapshot_*.png"))
    if not snaps:
        snaps = sorted(run_dir.glob("snapshot_*.png"))
    if not snaps:
        raise FileNotFoundError(f"No B snapshots found in {run_dir}")
    last = snaps[-1]
    arr = load_png_gray(last)
    return last, arr


def compute_similarity(ref_B: np.ndarray, cand_B: np.ndarray, thresh=0.3):
    """
    Compute IoU and intensity correlation between ref_B and cand_B.

    - IoU: threshold B fields at 'thresh' and compute intersection / union.
    - corr: Pearson correlation of intensities inside union mask.
    """
    if ref_B.shape != cand_B.shape:
        raise ValueError("Shape mismatch between reference and candidate B fields.")

    mask_ref = ref_B > thresh
    mask_cand = cand_B > thresh
    union_mask = mask_ref | mask_cand
    inter_mask = mask_ref & mask_cand

    union = union_mask.sum()
    inter = inter_mask.sum()
    iou = 0.0 if union == 0 else float(inter / union)

    if union == 0:
        corr = np.nan
    else:
        v_ref = ref_B[union_mask].flatten()
        v_cand = cand_B[union_mask].flatten()
        v_ref_c = v_ref - v_ref.mean()
        v_cand_c = v_cand - v_cand.mean()
        denom = np.sqrt((v_ref_c**2).sum() * (v_cand_c**2).sum())
        if denom <= 0:
            corr = np.nan
        else:
            corr = float((v_ref_c * v_cand_c).sum() / denom)

    return iou, corr


# ------------------------------------------------------------
# Main environment test pipeline
# ------------------------------------------------------------

def run_env_suite_for_candidate(
    base_run_dir: Path,
    env_out_root: Path,
    skip_existing: bool = True,
):
    """
    For a single candidate run (under outputs/dynamic_tau_v5_qridge),
    run all environment variants and compute similarity to baseline.

    Returns:
        candidate_hash, list of result dicts for each variant.
    """
    candidate_hash = base_run_dir.name
    print(f"\n=== Candidate {candidate_hash} ===")
    cfg_base = load_cfg_from_meta(base_run_dir)
    variants = make_env_variants(cfg_base)

    # Where to write environment runs for this candidate
    cand_out_root = env_out_root / candidate_hash
    cand_out_root.mkdir(parents=True, exist_ok=True)

    # 1. Run all variants
    variant_outdirs = {}
    for name, cfg in variants.items():
        outdir = cand_out_root / name
        if outdir.exists() and any(outdir.glob("B_snapshot_*.png")) and skip_existing:
            print(f"  [skip] {name} already has snapshots in {outdir}")
            variant_outdirs[name] = outdir
            continue

        print(f"  → Running variant '{name}' into {outdir}")
        outdir.mkdir(parents=True, exist_ok=True)
        run_simulation(cfg, str(outdir))
        variant_outdirs[name] = outdir
        print(f"  ✓ Done '{name}'")

    # 2. Load baseline final B
    baseline_outdir = variant_outdirs["baseline"]
    _, B_base = get_last_B_snapshot(baseline_outdir)

    # 3. Compare each variant to baseline
    results = []
    for name, outdir in variant_outdirs.items():
        try:
            _, B_var = get_last_B_snapshot(outdir)
            iou, corr = compute_similarity(B_base, B_var, thresh=0.3)
        except Exception as e:
            print(f"  ! Error computing similarity for variant '{name}': {e}")
            iou, corr = np.nan, np.nan

        res = {
            "candidate_hash": candidate_hash,
            "env_variant": name,
            "baseline_dir": str(baseline_outdir),
            "variant_dir": str(outdir),
            "iou_vs_baseline": iou,
            "corr_vs_baseline": corr,
            "feed": variants[name].get("feed", variants[name].get("f", np.nan)),
            "kill": variants[name].get("kill", variants[name].get("k", np.nan)),
        }
        results.append(res)
        print(f"  [variant {name}] IoU={iou:.3f}  corr={corr:.3f}")

    return candidate_hash, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        nargs="*",
        default=["40cdedd754", "94392f5ff1", "6fc841af45"],
        help=(
            "List of candidate run hashes under outputs/dynamic_tau_v5_qridge/. "
            "Default: three example organisms (breathing, crystallising, melting)."
        ),
    )
    ap.add_argument(
        "--qroot",
        type=str,
        default="outputs/dynamic_tau_v5_qridge",
        help="Root directory for q-ridge runs (original candidates).",
    )
    ap.add_argument(
        "--env-root",
        type=str,
        default="outputs/dynamic_tau_v5_env",
        help="Root directory for environment test runs.",
    )
    ap.add_argument(
        "--summary-out",
        type=str,
        default="plots/proto_life_v5/env_tests_summary.csv",
        help="Path to write the summary CSV of environment tests.",
    )
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run variants even if outputs already exist.",
    )
    args = ap.parse_args()

    qroot = Path(args.qroot)
    env_root = Path(args.env_root)
    env_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for h in args.candidates:
        base_run_dir = qroot / h
        if not base_run_dir.exists():
            print(f"Warning: candidate directory not found: {base_run_dir}, skipping.")
            continue

        _, results = run_env_suite_for_candidate(
            base_run_dir,
            env_root,
            skip_existing=not args.no_skip_existing,
        )
        all_results.extend(results)

    if not all_results:
        print("No results generated; nothing to write.")
        return

    # Write summary CSV
    import csv

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "candidate_hash",
        "env_variant",
        "baseline_dir",
        "variant_dir",
        "feed",
        "kill",
        "iou_vs_baseline",
        "corr_vs_baseline",
    ]

    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print("\nSummary written to:", summary_path)


if __name__ == "__main__":
    main()
