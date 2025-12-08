#!/usr/bin/env python3
"""
analyze_internal_v5.py

Characterise subtle, internal morphogenesis in dynamic_tau_v5 runs.

Reads:
    - a runs_summary CSV (e.g. runs_summary_v5_qridge.csv)
    - per-run snapshots: B_snapshot_*.png, tau_snapshot_*.png
    - per-run metrics.csv (time, coherence, entropy, energy, autocat)

Adds per-run metrics:
    - internal_reorg_index  (0 = frozen interior, 1 = highly rearranged)
    - com_shift_B           (center-of-mass movement inside the cell)
    - coherence_osc_index   (variance of detrended coherence time-series)

Writes:
    - <input_stem>_with_internal.csv  (augmented summary)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ----------------- helpers -----------------

def load_png_gray(path: Path):
    arr = np.array(Image.open(path).convert("L"), dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def choose_indices(n, k=3):
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    return [0, n // 2, n - 1]


def center_of_mass(field, mask=None):
    """
    Center of mass of 'field' (2D array). If mask is given, weight is
    field * mask and we ignore masked-out areas.
    Returns (cy, cx) in pixel coordinates.
    """
    f = np.array(field, dtype=float)
    if mask is not None:
        f = f * mask
    total = f.sum()
    if total <= 0:
        return np.nan, np.nan
    ny, nx = f.shape
    ys = np.arange(ny).reshape(-1, 1)
    xs = np.arange(nx).reshape(1, -1)
    cy = (f * ys).sum() / total
    cx = (f * xs).sum() / total
    return float(cy), float(cx)


def compute_internal_reorg(run_dir: Path, thresh=0.3):
    """
    For a single run directory, compute:
      - internal_reorg_index
      - com_shift_B

    Procedure:
      1) Load B_snapshot_*.png
      2) Pick mid and last snapshots
      3) Create a cell mask = (B_mid > thresh) OR (B_last > thresh)
      4) Compute correlation of B_mid vs B_last inside mask
      5) Compute center-of-mass of B_mid and B_last inside mask
    """
    snaps = sorted(run_dir.glob("B_snapshot_*.png"))
    if not snaps:
        # fallback to legacy snapshot name if needed
        snaps = sorted(run_dir.glob("snapshot_*.png"))
    if len(snaps) < 2:
        return np.nan, np.nan

    n = len(snaps)
    mid = snaps[n // 2]
    last = snaps[-1]

    B_mid = load_png_gray(mid)
    B_last = load_png_gray(last)

    # build cell mask (union of mid and last)
    mask = (B_mid > thresh) | (B_last > thresh)
    if mask.sum() == 0:
        return np.nan, np.nan

    v_mid = B_mid[mask].flatten()
    v_last = B_last[mask].flatten()

    # correlation inside mask
    if v_mid.size < 2:
        corr = np.nan
    else:
        v_mid_c = v_mid - v_mid.mean()
        v_last_c = v_last - v_last.mean()
        denom = np.sqrt((v_mid_c**2).sum() * (v_last_c**2).sum())
        if denom <= 0:
            corr = np.nan
        else:
            corr = float((v_mid_c * v_last_c).sum() / denom)

    # internal reorganisation index
    if np.isnan(corr):
        internal_reorg_index = np.nan
    else:
        internal_reorg_index = float(1.0 - corr)  # 0 = identical, 1 = very different

    # center-of-mass shift inside mask
    cy_mid, cx_mid = center_of_mass(B_mid, mask=mask)
    cy_last, cx_last = center_of_mass(B_last, mask=mask)
    if np.isnan(cy_mid) or np.isnan(cy_last):
        com_shift = np.nan
    else:
        com_shift = float(np.sqrt((cy_last - cy_mid)**2 + (cx_last - cx_mid)**2))

    return internal_reorg_index, com_shift


def compute_coherence_osc_index(run_dir: Path):
    """
    Load metrics.csv and compute a simple oscillation index for coherence:
      - detrend coherence by subtracting a linear fit
      - return variance of the residuals
    """
    mpath = run_dir / "metrics.csv"
    if not mpath.exists():
        return np.nan

    df = pd.read_csv(mpath)
    if "time" not in df.columns or "coherence" not in df.columns:
        return np.nan
    t = df["time"].values
    c = df["coherence"].values
    if len(t) < 3:
        return np.nan

    # linear detrend: c ≈ a*t + b
    A = np.vstack([t, np.ones_like(t)]).T
    try:
        a, b = np.linalg.lstsq(A, c, rcond=None)[0]
        trend = a * t + b
        resid = c - trend
        osc_index = float(np.var(resid))
    except Exception:
        osc_index = np.nan

    return osc_index


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to runs_summary CSV (e.g. plots/proto_life_v5/runs_summary_v5_qridge.csv)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Default: <input_stem>_with_internal.csv in same dir.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print("Loaded", len(df), "rows from", csv_path)

    internal_reorg = []
    com_shifts = []
    osc_indices = []

    for i, row in df.iterrows():
        rd = Path(row["run_dir"])
        if not rd.exists():
            print("Warning: run_dir does not exist, skipping:", rd)
            internal_reorg.append(np.nan)
            com_shifts.append(np.nan)
            osc_indices.append(np.nan)
            continue

        iri, cms = compute_internal_reorg(rd)
        coi = compute_coherence_osc_index(rd)

        internal_reorg.append(iri)
        com_shifts.append(cms)
        osc_indices.append(coi)

    df["internal_reorg_index"] = internal_reorg
    df["com_shift_B"] = com_shifts
    df["coherence_osc_index"] = osc_indices

    if args.out is None:
        out_path = csv_path.parent / f"{csv_path.stem}_with_internal.csv"
    else:
        out_path = Path(args.out)

    df.to_csv(out_path, index=False)
    print("Wrote augmented CSV with internal metrics to:", out_path)


if __name__ == "__main__":
    main()
