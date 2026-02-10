#!/usr/bin/env python3
"""run_search_v7_minimal_w.py

Minimal *pure-maths* QD (MAP-Elites grid) search driver for dynamic_tau_v7.

Core idea
- The simulator defines an organism indicator field w(x,t) and logs w_mean(t).
- The runner steers search using ONLY w_mean(t) via universal signal functionals.

Signals from w_mean(t)
- alive_frac: fraction of samples where w_mean > w_gate
- growth_slope: slope of log(w_mean+eps) vs time (positive-only in score)
- osc_log_peak: log(peak/median) of PSD within a frequency band ("soft lock-in")

Archive
- 2D MAP-Elites grid over descriptors:
    desc_0 = sigmoid(growth_slope / growth_scale)
    desc_1 = sigmoid(osc_log_peak / osc_scale)

Outputs (under --out_root)
- boqd_log.csv
- qd_elites.json
- qd_map.png
- best_so_far.json

Run example
  python simulations/run_search_v7_minimal_w.py \
    --out_root outputs/dynamic_tau_v7_minimal_w \
    --workers 8 --budget 400 --init_random 80 \
    --bins 24 \
    --steps 3000 --nx 150 --ny 150 \
    --w_enabled 1 --w_tau_gain_max 0.15 \
    --w_gate 0.01 \
    --osc_fmin 0.002 --osc_fmax 0.03

Notes
- Frequencies are in cycles per unit of the 'time' column in metrics.csv.
  If your metrics.csv uses step-index time (e.g. 0,20,40...), the effective dt
  is larger, so tune osc_fmin/osc_fmax accordingly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure we can import sibling sim modules when executed from repo root.
import sys
sys.path.insert(0, os.path.dirname(__file__))

# --- simulator import (prefer v7, fail loudly if user asks to require v7) ---
MODEL_NAME = None
try:
    import dynamic_tau_v7_seeddiv as model  # type: ignore
    MODEL_NAME = "dynamic_tau_v7"
except Exception:
    import dynamic_tau_v6 as model  # type: ignore
    MODEL_NAME = "dynamic_tau_v6"


# -------------------------
# JSON helper
# -------------------------

def _json_sanitize(o: Any):
    import numpy as _np

    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    return str(o)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_sanitize)


def stable_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, default=_json_sanitize)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


# -------------------------
# Parameter space
# -------------------------

# Keep the exploration set small on purpose.
# Add more once w-feedback is clearly producing diverse persistent pockets.
PARAM_SPACE: List[Tuple[str, float, float]] = [
    ("feed", 0.020, 0.050),
    ("kill", 0.045, 0.085),
    ("alpha", 0.000, 0.080),
    ("beta", 0.001, 0.020),
    ("gamma", 0.000, 0.800),
    ("kappa_tau", 0.000, 0.100),
    ("tau_noise", 0.000, 0.010),
    # w -> tau feedback strength (critical for "self-maintaining" pockets)
    ("w_tau_gain", 0.0, 0.15),
]


def sample_params(rng: np.random.Generator) -> Dict[str, float]:
    p: Dict[str, float] = {}
    for name, lo, hi in PARAM_SPACE:
        p[name] = float(lo + (hi - lo) * rng.random())
    return p


def mutate_params(rng: np.random.Generator, base: Dict[str, float], sigma: float) -> Dict[str, float]:
    out = dict(base)
    for name, lo, hi in PARAM_SPACE:
        span = (hi - lo)
        out[name] = float(np.clip(out[name] + rng.normal(0.0, sigma) * span, lo, hi))
    return out


# -------------------------
# Simulator cfg
# -------------------------

def base_sim_cfg(nx: int, ny: int, steps: int, dt: float, log_every: int) -> Dict[str, Any]:
    # Minimal set; your dynamic_tau_v7 can ignore unknown keys.
    return {
        "nx": nx,
        "ny": ny,
        "dx": 1.0,
        "dy": 1.0,
        "dt": float(dt),
        "steps": steps,
        "log_every": int(log_every),
        "save_metrics": True,
        "save_states": True,
        "mid_state_step": steps // 2,
        # snapshots off by default in search
        "save_snapshots": False,
        "snap_every": 200,
        # Gray-Scott defaults
        "Da": 0.16,
        "Db": 0.08,
        # tau defaults
        "tau0": 1.0,
        "tau_min": 0.2,
        "tau_max": 5.0,
        "kappa_tau": 0.02,
        "tau_noise": 0.001,
        "alpha": 0.03,
        "beta": 0.006,
        "gamma": 0.3,
        # nutrient defaults (safe)
        "N0": 1.0,
        "nutrient_use": 0.01,
        "nutrient_replenish": 0.001,
        "use_diffusive_nutrient": True,
        "D_N": 0.02,
        "eta_N": 0.1,
        "rho_N": 0.0005,
        # --- w indicator system ---
        "w_enabled": True,
        "w_mode": "varratio",
        "w_radius": 1,
        "w_amp_thr": 0.05,
        "w_amp_width": 0.02,
        "w_tau_gain": 0.0,
        "w_tau_bias": 0.0,
        "save_w": False,
        # init
        "seed_radius": 10,
        "noise": 0.02,
    }


def make_cfg(params: Dict[str, float], args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    cfg = base_sim_cfg(args.nx, args.ny, args.steps, dt=args.dt, log_every=args.log_every)

    # Optional: write state_XXXXXX snapshots for later visualization.
    cfg["save_snapshots"] = bool(int(getattr(args, "save_snapshots", 0)))
    cfg["snap_every"] = int(getattr(args, "snap_every", 200))
    cfg["snapshot_format"] = str(getattr(args, "snapshot_format", "npz"))

    cfg["feed"] = float(params["feed"])
    cfg["kill"] = float(params["kill"])
    cfg["alpha"] = float(params["alpha"])
    cfg["beta"] = float(params["beta"])
    cfg["gamma"] = float(params["gamma"])
    cfg["kappa_tau"] = float(params["kappa_tau"])
    cfg["tau_noise"] = float(params["tau_noise"])

    cfg["w_enabled"] = bool(args.w_enabled)
    cfg["w_mode"] = str(args.w_mode)
    cfg["w_radius"] = int(args.w_radius)
    cfg["w_amp_thr"] = float(args.w_amp_thr)
    cfg["w_amp_width"] = float(args.w_amp_width)
    cfg["w_tau_gain"] = float(params.get("w_tau_gain", 0.0))
    cfg["w_tau_bias"] = float(args.w_tau_bias)
    cfg["save_w"] = bool(args.save_w)

    
    # --- seed diversity (initial condition) ---
    cfg["init_mode"] = str(getattr(args, "init_mode", "square"))
    cfg["seed_pos_mode"] = str(getattr(args, "seed_pos_mode", "center"))
    cfg["seed_count"] = int(getattr(args, "seed_count", 1))
    cfg["seed_sigma"] = float(getattr(args, "seed_sigma", 4.0))
    cfg["seed_ring_width"] = float(getattr(args, "seed_ring_width", 2.5))
    cfg["seed_stripe_period"] = int(getattr(args, "seed_stripe_period", 16))
    cfg["seed_margin"] = int(getattr(args, "seed_margin", 12))
    cfg["seed_radius"] = int(getattr(args, "seed_radius", cfg.get("seed_radius", 10)))
    cfg["seed"] = int(seed)
    return cfg


# -------------------------
# Universal functionals on w_mean(t)
# -------------------------

def _weighted_slope(t: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Weighted least-squares slope of y ~ a*t + b."""
    if w is None:
        w = np.ones_like(t)
    w = np.asarray(w, dtype=float)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # guard
    if len(t) < 3 or not np.isfinite(t).all() or not np.isfinite(y).all():
        return float("nan")

    sw = w.sum()
    if sw <= 0:
        return float("nan")

    tbar = (w * t).sum() / sw
    ybar = (w * y).sum() / sw
    num = (w * (t - tbar) * (y - ybar)).sum()
    den = (w * (t - tbar) ** 2).sum()
    if den <= 1e-18:
        return float("nan")
    return float(num / den)


def _welch_psd(x: np.ndarray, dt: float, nperseg: int, noverlap_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal Welch PSD (no scipy dependency).
    Returns (freqs, psd) where freqs are in 1/time units of dt.

    Notes:
      - Uses Hann window
      - Averages periodograms across segments
      - Normalizes so that psd integrates to ~variance (roughly; good enough for peak-finding)
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n < 8:
        return np.array([], dtype=float), np.array([], dtype=float)

    nper = int(max(8, min(int(nperseg), n)))
    nover = int(max(0, min(int(round(nper * float(noverlap_frac))), nper - 1)))
    step = nper - nover
    if step <= 0:
        step = nper

    # Segment starts
    starts = list(range(0, n - nper + 1, step))
    if not starts:
        starts = [0]

    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nper) / max(nper - 1, 1))
    win_norm = float(np.sum(win * win))

    acc = None
    for s in starts:
        seg = x[s:s + nper].astype(float, copy=True)
        seg = seg - float(np.mean(seg))
        # Linear detrend (cheap)
        t = np.arange(seg.size, dtype=float)
        a, b = np.polyfit(t, seg, 1)
        seg = seg - (a * t + b)

        seg = seg * win
        Y = np.fft.rfft(seg)
        P = (np.abs(Y) ** 2) / (win_norm + 1e-12)
        acc = P if acc is None else (acc + P)

    psd = acc / max(1, len(starts))
    freqs = np.fft.rfftfreq(nper, d=float(dt))
    return freqs, psd


def _osc_diagnostics(x: np.ndarray, dt: float, fmin: float, fmax: float, args) -> Dict[str, float]:
    """
    Compute oscillation diagnostics on a scalar time series x(t).

    Key outputs
    - osc_log:         log(peak_ratio) where peak_ratio = peak / (robust_floor + eps)
    - osc_peak:        max PSD value in-band
    - osc_floor:       robust floor estimate from out-of-band PSD (excludes DC and a small neighborhood of the peak)
    - osc_peak_ratio:  osc_peak / (osc_floor + eps)
    - osc_ratio:       band_power / total_power (after detrend)

    Frequency estimates (all in cycles per unit time)
    - osc_f_peak_bin:      argmax bin frequency in-band (can lock to band edge if spectrum is monotone)
    - osc_f_peak_refined:  parabolic (sub-bin) refinement around the argmax (when neighbors exist)
    - osc_f_centroid:      band-limited centroid frequency (moves even when argmax is at the edge)

    For convenience when mapping "stable regions in osc space", we set:
      osc_f_peak = osc_f_centroid
    and keep the bin-argmax as osc_f_peak_bin.
    """
    x = np.asarray(x, dtype=float)

    out_nan = {
        "osc_log": float("nan"),
        "osc_peak": float("nan"),
        "osc_floor": float("nan"),
        "osc_f_peak": float("nan"),
        "osc_f_peak_bin": float("nan"),
        "osc_f_peak_refined": float("nan"),
        "osc_f_centroid": float("nan"),
        "osc_peak_ratio": float("nan"),
        "osc_ratio": float("nan"),
        "osc_band_empty": 1.0,
    }

    if x.size < 16 or not np.isfinite(x).any() or not (dt > 0):
        return out_nan

    # --- detrend (removes DC + slow drift dominance) ---
    x = x - float(np.nanmean(x))
    t = np.arange(x.size, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() >= 4:
        a, b = np.polyfit(t[ok], x[ok], 1)
        x = x - (a * t + b)

    # --- PSD estimator ---
    method = str(getattr(args, "osc_method", "fft")).lower()
    if method == "welch":
        nper = int(getattr(args, "osc_welch_nperseg", 256))
        nover = float(getattr(args, "osc_welch_noverlap", 0.5))
        freqs, psd = _welch_psd(x, dt=dt, nperseg=nper, noverlap_frac=nover)
    else:
        freqs = np.fft.rfftfreq(x.size, d=float(dt))
        Y = np.fft.rfft(x)
        psd = (np.abs(Y) ** 2) / float(x.size)

    if freqs.size < 2 or psd.size < 2 or not np.isfinite(psd).any():
        return out_nan

    # Drop DC (index 0) from all downstream computations.
    freqs2 = freqs[1:]
    psd2 = psd[1:]

    # Band mask
    fmin = float(fmin)
    fmax = float(fmax)
    in_band = (freqs2 >= fmin) & (freqs2 <= fmax)
    if not np.any(in_band):
        return out_nan

    # Extract in-band arrays (they are contiguous for an interval band)
    band_inds = np.where(in_band)[0]
    band_psd = psd2[band_inds]
    band_freqs = freqs2[band_inds]

    # Peak in-band (bin argmax)
    peak_local = int(np.nanargmax(band_psd))
    peak_global = int(band_inds[peak_local])
    osc_peak = float(band_psd[peak_local])
    osc_f_peak_bin = float(freqs2[peak_global])

    # --- Parabolic (sub-bin) refinement around the peak ---
    osc_f_peak_refined = float(osc_f_peak_bin)
    if 0 < peak_global < (freqs2.size - 1):
        # Use log-PSD for stability across orders of magnitude
        med_all = float(np.nanmedian(psd2[np.isfinite(psd2)])) if np.isfinite(psd2).any() else 0.0
        eps_small = max(1e-30, 1e-12 * max(med_all, 1e-30))
        y_m1 = float(np.log(psd2[peak_global - 1] + eps_small))
        y_0  = float(np.log(psd2[peak_global]     + eps_small))
        y_p1 = float(np.log(psd2[peak_global + 1] + eps_small))
        denom = (y_m1 - 2.0 * y_0 + y_p1)
        if abs(denom) > 1e-12 and np.isfinite(denom):
            delta = 0.5 * (y_m1 - y_p1) / denom  # peak offset in bins
            # Clamp to a sane sub-bin range
            delta = float(np.clip(delta, -1.0, 1.0))
            df_bin = float(freqs2[peak_global + 1] - freqs2[peak_global])
            osc_f_peak_refined = float(osc_f_peak_bin + delta * df_bin)

    # --- Band-limited centroid frequency (moves even when argmax is at band edge) ---
    # Use trapezoidal integrals for consistent units
    band_power = float(np.trapz(band_psd, band_freqs)) if band_psd.size >= 2 else float(band_psd.sum())
    if band_power > 0 and np.isfinite(band_power):
        num = float(np.trapz(band_freqs * band_psd, band_freqs)) if band_psd.size >= 2 else float((band_freqs * band_psd).sum())
        osc_f_centroid = float(num / (band_power + 1e-30))
    else:
        osc_f_centroid = float("nan")

    # --- Robust floor: out-of-band median excluding a small neighborhood around the peak ---
    k = 2  # exclude +/- k bins around the peak to avoid leakage
    exclude = np.zeros_like(psd2, dtype=bool)
    lo = max(0, peak_global - k)
    hi = min(psd2.size, peak_global + k + 1)
    exclude[lo:hi] = True

    floor_mask = (~in_band) & (~exclude)
    floor_psd = psd2[floor_mask]
    if floor_psd.size == 0:
        floor_psd = psd2[~in_band]
    if floor_psd.size == 0:
        floor_psd = psd2

    med_all = float(np.nanmedian(psd2[np.isfinite(psd2)])) if np.isfinite(psd2).any() else 0.0
    eps = max(1e-15, 1e-12 * max(med_all, 0.0))

    osc_floor = float(np.nanmedian(floor_psd[np.isfinite(floor_psd)])) if np.isfinite(floor_psd).any() else float(med_all)

    peak_ratio = float(osc_peak / (osc_floor + eps))
    osc_log = float(np.log(max(peak_ratio, 1e-300)))

    # Band / total power ratio
    total_power = float(np.trapz(psd2, freqs2)) if psd2.size >= 2 else float(psd2.sum())
    osc_ratio = float(band_power / (total_power + eps))

    return {
        "osc_log": osc_log,
        "osc_peak": osc_peak,
        "osc_floor": osc_floor,
        # NOTE: map-friendly frequency summary
        "osc_f_peak": float(osc_f_centroid),
        # plus more detailed frequency diagnostics
        "osc_f_peak_bin": osc_f_peak_bin,
        "osc_f_peak_refined": osc_f_peak_refined,
        "osc_f_centroid": float(osc_f_centroid),
        "osc_peak_ratio": peak_ratio,
        "osc_ratio": osc_ratio,
        "osc_band_empty": 0.0,
    }


def _acf_diagnostics(y: np.ndarray, dt_sample: float, fmin: float, fmax: float, args=None) -> Dict[str, float]:
    """Autocorrelation-based oscillation diagnostics with sub-sample peak refinement.

    This version fixes "lag quantization" by refining the ACF peak location using a
    3-point parabolic interpolation around the discrete peak lag. The returned
    lag is (k_peak + delta) * dt_sample, where delta in [-1, 1].

    Returns:
      osc_acf_peak: peak normalized ACF value at the (refined) peak (still reported as the
                    discrete ACF value at k_peak; refinement affects lag/f_est)
      osc_acf_lag_s: refined lag (seconds) at which the peak occurs
      osc_f_est: 1 / osc_acf_lag_s
      osc_acf_band_empty: 1 if the requested band has no valid lags, else 0
      osc_acf_cycles: estimated cycles across sampled window
      osc_acf_too_few_cycles: 1 if min-cycles gate fails, else 0
      osc_acf_k_peak: discrete peak lag index (samples)
      osc_acf_delta: sub-sample peak offset (bins)
    """
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    out_nan = {
        "osc_acf_peak": float("nan"),
        "osc_acf_lag_s": float("nan"),
        "osc_f_est": float("nan"),
        "osc_acf_band_empty": 1.0,
        "osc_acf_cycles": float("nan"),
        "osc_acf_too_few_cycles": 0.0,
        "osc_acf_k_peak": float("nan"),
        "osc_acf_delta": float("nan"),
        "osc_acf_no_local_peak": 0.0,
    }

    if n < 8 or not np.isfinite(dt_sample) or dt_sample <= 0:
        return out_nan

    y0 = y - np.nanmean(y)
    # Replace NaNs with 0 after centering (keeps length consistent)
    y0 = np.where(np.isfinite(y0), y0, 0.0)

    var = float(np.mean(y0 * y0))
    if not np.isfinite(var) or var <= 1e-18:
        return out_nan

    # Normalized autocorrelation for lags 0..n-1 (biased estimator)
    acf_full = np.correlate(y0, y0, mode="full")
    acf = acf_full[n - 1 :] / (var * n)  # acf[0] ~ 1

    # Band in periods (seconds): [1/fmax, 1/fmin]
    Tmin = 1.0 / max(float(fmax), 1e-12)
    Tmax = 1.0 / max(float(fmin), 1e-12)
    kmin = int(np.ceil(Tmin / dt_sample))
    kmax = int(np.floor(Tmax / dt_sample))

    # Clamp to available lags (exclude lag 0)
    kmin = max(kmin, 1)
    kmax = min(kmax, n - 1)

    if kmax < kmin:
        return out_nan

    band = acf[kmin : kmax + 1]

    # Prefer true local maxima to avoid monotone-decay artifacts.
    # If no local maxima exist, fall back to the global max in-band.
    local = []
    for kk in range(kmin + 1, kmax):
        a = float(acf[kk - 1])
        b = float(acf[kk])
        c = float(acf[kk + 1])
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (b > a) and (b > c):
            local.append(kk)

    if local:
        # Pick the highest local maximum
        k_peak = int(max(local, key=lambda kk: float(acf[kk])))
        osc_acf_no_local_peak = 0.0
    else:
        # No strict local maximum in-band -> likely monotone relaxation.
        return {
            "osc_acf_peak": float("nan"),
            "osc_acf_lag_s": float("nan"),
            "osc_f_est": float("nan"),
            "osc_acf_band_empty": 0.0,
            "osc_acf_cycles": float("nan"),
            "osc_acf_too_few_cycles": 0.0,
            "osc_acf_k_peak": float("nan"),
            "osc_acf_delta": float("nan"),
            "osc_acf_no_local_peak": 1.0,
        }

    peak = float(acf[k_peak])

    # --- Sub-sample parabolic refinement around k_peak ---
    delta = 0.0
    if 1 <= k_peak <= (n - 2):
        y_m1 = float(acf[k_peak - 1])
        y_0 = float(acf[k_peak])
        y_p1 = float(acf[k_peak + 1])
        denom = (y_m1 - 2.0 * y_0 + y_p1)
        if np.isfinite(denom) and abs(denom) > 1e-12:
            delta = 0.5 * (y_m1 - y_p1) / denom
            # Clamp to prevent crazy extrapolation when the parabola is shallow
            delta = float(np.clip(delta, -1.0, 1.0))

    lag_s = float((k_peak + delta) * dt_sample)
    f_est = float(1.0 / lag_s) if (np.isfinite(lag_s) and lag_s > 0) else float("nan")

    # Optional minimum-cycles gate: require at least args.osc_min_cycles cycles across the sampled window.
    total_T = float((n - 1) * dt_sample)
    osc_acf_cycles = float(total_T * f_est) if np.isfinite(f_est) else float("nan")
    min_cycles = float(getattr(args, "osc_min_cycles", 0.0) or 0.0) if args is not None else 0.0
    if min_cycles > 0.0 and np.isfinite(osc_acf_cycles) and osc_acf_cycles < min_cycles:
        return {
            "osc_acf_peak": float("nan"),
            "osc_acf_lag_s": float("nan"),
            "osc_f_est": float("nan"),
            "osc_acf_band_empty": 0.0,
            "osc_acf_cycles": osc_acf_cycles,
            "osc_acf_too_few_cycles": 1.0,
            "osc_acf_k_peak": float(k_peak),
            "osc_acf_delta": float(delta),
            "osc_acf_no_local_peak": 0.0,
        }

    return {
        "osc_acf_peak": peak,
        "osc_acf_lag_s": lag_s,
        "osc_f_est": f_est,
        "osc_acf_band_empty": 0.0,
        "osc_acf_cycles": osc_acf_cycles,
        "osc_acf_too_few_cycles": 0.0,
        "osc_acf_k_peak": float(k_peak),
        "osc_acf_delta": float(delta),
        "osc_acf_no_local_peak": float(osc_acf_no_local_peak),
    }


def compute_universal_from_metrics(metrics_csv: str, args: argparse.Namespace) -> Dict[str, float]:
    df = pd.read_csv(metrics_csv)
    if df.empty:
        raise RuntimeError("metrics.csv empty")

    # Candidate oscillation signals (in priority order).
    # We fall back to w_mean so this runner still works with older simulator logs.
    cand_cols = []
    for c in ["w_p95", "w_std", "w_max", "w_mean"]:
        if c in df.columns:
            cand_cols.append(c)

    if not cand_cols:
        raise RuntimeError("metrics.csv missing any of: w_p95, w_std, w_max, w_mean")

    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

    # NOTE: dt_sample for oscillation analysis must come from the simulation config,
    # never inferred from df["time"] (which is often logged in *steps*).
    dt0 = float(getattr(args, "dt", args.log_dt_fallback))
    log_every = int(getattr(args, "log_every", 1))
    dt_sample = dt0 * float(log_every)

    # We still build a physical-time axis for optional windowing/plots.
    # Heuristic only affects t_phys; it does NOT affect dt_sample.
    dt_col = float(np.median(np.diff(t))) if len(t) >= 2 else float("nan")
    time_is_steps = (np.isfinite(dt_col) and dt_col >= 1.0 and dt0 <= 0.5)
    t_phys = t * dt0 if time_is_steps else t

    w_gate = float(args.w_gate)

    # choose a window (ignore early transient)
    n = len(t_phys)
    i0 = int(np.floor(n * float(args.window_start_frac)))
    i0 = max(0, min(i0, n - 3))
    tt = t_phys[i0:]

    # Evaluate each candidate column for oscillatory recurrence; choose the "best"
    best = {
        "name": None,
        "series": None,
        "acf_peak": -np.inf,
        "acf_no_local_peak": 1.0,
        "acf_lag_s": float("nan"),
        "f_est": float("nan"),
        "band_empty": 1.0,
        "osc_log": float("nan"),
        "osc_peak": float("nan"),
        "osc_floor": float("nan"),
        "osc_peak_ratio": float("nan"),
        "osc_ratio": float("nan"),
        "osc_band_empty": float("nan"),
        "osc_f_peak": float("nan"),
        "osc_f_peak_bin": float("nan"),
        "osc_f_peak_refined": float("nan"),
        "osc_f_centroid": float("nan"),
    }

    for col in cand_cols:
        wcol = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        ww = wcol[i0:]
        # skip pathological columns (all NaN or constant)
        finite = ww[np.isfinite(ww)]
        if finite.size < 8 or float(np.nanstd(ww)) <= 1e-12:
            continue

        osc = _osc_diagnostics(ww, dt=dt_sample, fmin=float(args.osc_fmin), fmax=float(args.osc_fmax), args=args)
        ad = _acf_diagnostics(ww, dt_sample=dt_sample, fmin=float(args.osc_fmin), fmax=float(args.osc_fmax), args=args)
        acf_peak = float(ad.get("osc_acf_peak", float("nan")))
        if not np.isfinite(acf_peak):
            acf_peak = -np.inf

        # Primary selection: ACF local peak strength in-band.
        if acf_peak > float(best["acf_peak"]):
            best.update({
                "name": col,
                "series": wcol,
                "acf_peak": acf_peak,
                "acf_no_local_peak": float(ad.get("osc_acf_no_local_peak", 0.0)),
                "acf_lag_s": float(ad.get("osc_acf_lag_s", float("nan"))),
                "f_est": float(ad.get("osc_f_est", float("nan"))),
                "band_empty": float(ad.get("osc_acf_band_empty", 1.0)),
                "osc_log": float(osc.get("osc_log", float("nan"))),
                "osc_peak": float(osc.get("osc_peak", float("nan"))),
                "osc_floor": float(osc.get("osc_floor", float("nan"))),
                "osc_peak_ratio": float(osc.get("osc_peak_ratio", float("nan"))),
                "osc_ratio": float(osc.get("osc_ratio", float("nan"))),
                "osc_band_empty": float(osc.get("osc_band_empty", float("nan"))),
                "osc_f_peak": float(osc.get("osc_f_peak", float("nan"))),
                "osc_f_peak_bin": float(osc.get("osc_f_peak_bin", float("nan"))),
                "osc_f_peak_refined": float(osc.get("osc_f_peak_refined", float("nan"))),
                "osc_f_centroid": float(osc.get("osc_f_centroid", float("nan"))),
            })

    if best["series"] is None:
        # If no candidate was usable, fall back to w_mean explicitly.
        if "w_mean" not in df.columns:
            raise RuntimeError("No usable oscillation signal found (and w_mean missing)")
        best["name"] = "w_mean"
        best["series"] = pd.to_numeric(df["w_mean"], errors="coerce").to_numpy(dtype=float)

    # Ensure the stored diagnostics correspond to the chosen series.
    # (If we fell back to w_mean, the loop above never populated best["osc_*"] / best["acf_*"].)
    # best["series"] is already a numpy array (from pd.to_numeric(...).to_numpy above).
    # pd.to_numeric(ndarray) returns an ndarray, which does not have .to_numpy().
    _w_chosen = np.asarray(best["series"], dtype=float)
    _ww = _w_chosen[i0:] if _w_chosen.size else np.asarray([], dtype=float)
    if _ww.size >= 8 and float(np.nanstd(_ww)) > 1e-12:
        _osc = _osc_diagnostics(_ww, dt=dt_sample, fmin=float(args.osc_fmin), fmax=float(args.osc_fmax), args=args)
        _ad = _acf_diagnostics(_ww, dt_sample=dt_sample, fmin=float(args.osc_fmin), fmax=float(args.osc_fmax), args=args)
        best["osc_log"] = float(_osc.get("osc_log", float("nan")))
        best["osc_peak"] = float(_osc.get("osc_peak", float("nan")))
        best["osc_floor"] = float(_osc.get("osc_floor", float("nan")))
        best["osc_peak_ratio"] = float(_osc.get("osc_peak_ratio", float("nan")))
        best["osc_ratio"] = float(_osc.get("osc_ratio", float("nan")))
        best["osc_band_empty"] = float(_osc.get("osc_band_empty", float("nan")))
        best["osc_f_peak"] = float(_osc.get("osc_f_peak", float("nan")))
        best["osc_f_peak_bin"] = float(_osc.get("osc_f_peak_bin", float("nan")))
        best["osc_f_peak_refined"] = float(_osc.get("osc_f_peak_refined", float("nan")))
        best["osc_f_centroid"] = float(_osc.get("osc_f_centroid", float("nan")))

        best["acf_peak"] = float(_ad.get("osc_acf_peak", float("nan")))
        best["acf_lag_s"] = float(_ad.get("osc_acf_lag_s", float("nan")))
        best["f_est"] = float(_ad.get("osc_f_est", float("nan")))
        best["band_empty"] = float(_ad.get("osc_acf_band_empty", 1.0))
        best["acf_no_local_peak"] = float(_ad.get("osc_acf_no_local_peak", 0.0))

    # Chosen signal time series (FULL length), plus windowed copy for growth/weights.
    w = best["series"].astype(float)
    ww = w[i0:]

    alive_frac = float(np.mean(ww > w_gate))

    # If the simulator logs applied feedback, surface it here so the runner log can verify engagement.
    w_tau_gain_measured = float("nan")
    w_tau_gain_missing = 1
    if "w_tau_gain_applied" in df.columns:
        s = pd.to_numeric(df["w_tau_gain_applied"], errors="coerce").to_numpy(dtype=float)
        finite = s[np.isfinite(s)]
        if finite.size > 0:
            w_tau_gain_measured = float(finite[-1])
            w_tau_gain_missing = 0
    elif "w_tau_gain_used" in df.columns:
        s = pd.to_numeric(df["w_tau_gain_used"], errors="coerce").to_numpy(dtype=float)
        finite = s[np.isfinite(s)]
        if finite.size > 0:
            w_tau_gain_measured = float(finite[-1])
            w_tau_gain_missing = 0

    # viable weights: only care when alive
    wt = 1.0 / (1.0 + np.exp(-(ww - w_gate) / max(float(args.w_gate_width), 1e-9)))

    eps = 1e-12
    logw = np.log(ww + eps)
    growth_slope = _weighted_slope(tt, logw, wt)

    return {
        "osc_signal": str(best["name"]),
        "alive_frac": alive_frac,
        "w_tau_gain_measured": w_tau_gain_measured,
        "w_tau_gain_missing": w_tau_gain_missing,
        "growth_slope": float(growth_slope),

        # PSD-derived osc metrics (computed on chosen signal)
        "osc_log": float(best["osc_log"]),
        "osc_peak": float(best["osc_peak"]),
        "osc_floor": float(best["osc_floor"]),
        "osc_f_peak": float(best["osc_f_peak"]),
        "osc_f_peak_bin": float(best["osc_f_peak_bin"]),
        "osc_f_peak_refined": float(best["osc_f_peak_refined"]),
        "osc_f_centroid": float(best["osc_f_centroid"]),
        "osc_band_empty": float(best["osc_band_empty"]),
        "osc_peak_ratio": float(best["osc_peak_ratio"]),
        "osc_ratio": float(best["osc_ratio"]),

        # ACF-derived osc metrics (computed on chosen signal)
        "osc_acf_peak": float(best["acf_peak"]) if np.isfinite(best["acf_peak"]) else float("nan"),
        "osc_acf_lag_s": float(best["acf_lag_s"]),
        "osc_f_est": float(best["f_est"]),
        "osc_acf_band_empty": float(best["band_empty"]),
        "osc_acf_no_local_peak": float(best.get("acf_no_local_peak", 0.0)),

        # simple summaries of chosen signal
        "w_sig_mean": float(np.nanmean(w)),
        "w_sig_final": float(w[-1]) if len(w) else float("nan"),

        "dt_metric": float(dt_sample),
        "dt_raw_metric": float(dt_col) if np.isfinite(dt_col) else float("nan"),
        "time_is_steps": int(time_is_steps),
    }

def sigmoid01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp01(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.clip(x, 0.0, 1.0))


def _extract_w_from_state_npz(npz_path: str) -> Optional[np.ndarray]:
    """Load w-like field from a saved state_*.npz file.

    We prefer keys in this order:
      - 'w' (explicit)
      - 'N' (organism indicator field in dynamic_tau_v7)
      - 'n' (fallback)
    """
    try:
        z = np.load(npz_path)
        for k in ("w", "N", "n"):
            if k in z.files:
                arr = z[k]
                return np.asarray(arr, dtype=float)
    except Exception:
        return None
    return None


def morph_descriptors_from_w(
    w: np.ndarray,
    q: float = 0.10,
    low_is_structure: bool = True,
    max_ncomp: int = 10,
) -> Dict[str, float]:
    """Compute simple morphology descriptors from a 2D field.

    Returns:
      morph_area_frac: fraction of pixels in the structure mask
      morph_ncomp: number of connected components in mask (4-neighbor), clipped to max_ncomp
      morph_compactness: compactness of the largest component (perimeter^2 / (4*pi*area)), NaN if undefined
    """
    w2 = np.asarray(w, dtype=float)
    if w2.ndim != 2:
        return {"morph_area_frac": float("nan"), "morph_ncomp": float("nan"), "morph_compactness": float("nan")}

    # Threshold by quantile so it's scale-robust across runs.
    thr = float(np.quantile(w2[np.isfinite(w2)], np.clip(q, 1e-6, 1 - 1e-6)))
    mask = (w2 <= thr) if low_is_structure else (w2 >= thr)

    area = int(mask.sum())
    area_frac = float(area / mask.size)

    # Connected components (4-neighbor)
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    def neighbors(r, c):
        if r > 0: yield (r - 1, c)
        if r + 1 < H: yield (r + 1, c)
        if c > 0: yield (r, c - 1)
        if c + 1 < W: yield (r, c + 1)

    ncomp = 0
    largest_area = 0
    largest_perim = 0

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue
            # BFS
            ncomp += 1
            stack = [(r, c)]
            visited[r, c] = True
            comp_area = 0
            comp_perim = 0
            while stack:
                rr, cc = stack.pop()
                comp_area += 1
                # perimeter: count edges to non-mask or boundary
                if rr == 0 or not mask[rr - 1, cc]: comp_perim += 1
                if rr == H - 1 or not mask[rr + 1, cc]: comp_perim += 1
                if cc == 0 or not mask[rr, cc - 1]: comp_perim += 1
                if cc == W - 1 or not mask[rr, cc + 1]: comp_perim += 1

                for nr, nc in neighbors(rr, cc):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            if comp_area > largest_area:
                largest_area = comp_area
                largest_perim = comp_perim

    ncomp_clipped = int(min(max_ncomp, ncomp))

    # Compactness of the largest component (circle ~ 1; higher => less compact / more jagged)
    if largest_area > 0:
        compactness = float((largest_perim ** 2) / (4.0 * math.pi * float(largest_area)))
    else:
        compactness = float("nan")

    return {
        "morph_area_frac": float(area_frac),
        "morph_ncomp": float(ncomp_clipped),
        "morph_compactness": float(compactness),
    }


def classify_morphology(morph: Dict[str, float]) -> str:
    """Coarse morphology class label for captions."""
    n = morph.get("morph_ncomp", float("nan"))
    c = morph.get("morph_compactness", float("nan"))
    a = morph.get("morph_area_frac", float("nan"))
    if not np.isfinite(n) or not np.isfinite(a):
        return "unknown"
    n = int(n)
    if n >= 5:
        return "multi-compartment"
    if n >= 2:
        return "few-compartment"
    # single component
    if np.isfinite(c) and c > 3.0:
        return "labyrinth/jagged"
    if a < 0.02:
        return "tiny compartment"
    return "single compartment"
def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_desc_calib(path: str) -> Dict[str, float]:
    """Load descriptor calibration (if present)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {}
        # ensure numeric
        out = {}
        for k, v in d.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    except Exception:
        return {}

def apply_desc_calib(
    raw_f: float,
    raw_acf_peak: float,
    calib: Dict[str, float],
    args: argparse.Namespace,
) -> Tuple[float, float]:
    """Map raw descriptors into [0,1] for MAP-Elites.

    Descriptor 0: normalized frequency estimate in [osc_fmin, osc_fmax].
    Descriptor 1: linearly normalized ACF peak within calibrated [acf_peak_lo, acf_peak_hi].
    """
    # d0: frequency in-band normalization
    fmin = float(args.osc_fmin)
    fmax = float(args.osc_fmax)
    if not np.isfinite(raw_f):
        d0 = float("nan")
    else:
        d0 = (float(raw_f) - fmin) / (max(fmax - fmin, 1e-12))
        d0 = float(np.clip(d0, 0.0, 1.0))

    # d1: ACF peak normalization (calibrated)
    lo = float(calib.get("acf_peak_lo", getattr(args, "acf_peak_lo_default", 0.0)))
    hi = float(calib.get("acf_peak_hi", getattr(args, "acf_peak_hi_default", 1.0)))
    if not np.isfinite(raw_acf_peak):
        # Bootstrap-safe: treat missing/invalid ACF peak as "no recurrence".
        d1 = 0.0
    else:
        d1 = (float(raw_acf_peak) - lo) / (max(hi - lo, 1e-12))
        d1 = float(np.clip(d1, 0.0, 1.0))

    return d0, d1


@dataclass
class Job:
    """One simulation evaluation job (multiprocessing-friendly)."""

    params: Dict[str, Any]
    cfg: Dict[str, Any]
    outdir: str
    method: str

def run_one(job: Job, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    try:
        ensure_dir(job.outdir)
        model.run_simulation(job.cfg, job.outdir)

        mpath = os.path.join(job.outdir, "metrics.csv")
        um = compute_universal_from_metrics(mpath, args)
        # Raw descriptors (calibrated later in the main process). (calibrated later in the main process).
        # We compute both oscillation diagnostics (legacy) and morphology diagnostics (recommended),
        # then choose which ones populate MAP-Elites via --desc_mode.

        # --- Oscillation raw descriptors (legacy) ---
        _f_acf = float(um.get("osc_f_est", float("nan")))
        _f_fft = float(um.get("osc_f_peak_refined", um.get("osc_f_peak", um.get("osc_f_centroid", float("nan")))))
        raw_osc_f = _f_acf if np.isfinite(_f_acf) else _f_fft
        _acf = float(um.get("osc_acf_peak", float("nan")))
        raw_osc_acf = _acf if np.isfinite(_acf) else 0.0

        # --- Morphology raw descriptors (area fraction + component count) ---
        morph = {"morph_area_frac": float("nan"), "morph_ncomp": float("nan"), "morph_compactness": float("nan")}
        w_for_morph = None
        for nm in ("state_mid.npz", "state_final.npz"):
            pth = os.path.join(job.outdir, nm)
            if os.path.exists(pth):
                w_for_morph = _extract_w_from_state_npz(pth)
                if w_for_morph is not None:
                    break
        if w_for_morph is not None:
            morph = morph_descriptors_from_w(
                w_for_morph,
                q=float(args.morph_q),
                low_is_structure=bool(int(args.morph_low)),
                max_ncomp=int(args.morph_max_ncomp),
            )

        # Choose descriptors for the archive
        if str(getattr(args, "desc_mode", "morph")).lower() == "osc":
            raw_d0 = float(raw_osc_f)
            raw_d1 = float(raw_osc_acf)
        else:
            # raw morphology descriptors are already in [0,1] (area fraction) and [0,max_ncomp] (component count)
            raw_d0 = float(morph.get("morph_area_frac", float("nan")))
            ncomp = float(morph.get("morph_ncomp", float("nan")))
            raw_d1 = float(ncomp / max(float(args.morph_max_ncomp), 1.0)) if np.isfinite(ncomp) else float("nan")

        # Placeholder; calibrated later in main (or identity for morph)
        d0 = float("nan")
        d1 = float("nan")


        # score: alive gate * (positive growth + osc)
        alive_gate = sigmoid01((float(um["alive_frac"]) - float(args.alive_gate)) / max(float(args.alive_gate_width), 1e-9))
        growth_pos = max(float(um["growth_slope"]), 0.0)
        score = alive_gate * (float(args.w_growth) * growth_pos + float(args.w_osc) * max(float(um["osc_log"]), 0.0))

        row: Dict[str, Any] = {
            "run_dir": job.outdir,
            "method": job.method,
            "model": MODEL_NAME,
            "elapsed_s": float(time.time() - t0),
            "score": float(score),
            "desc_0": float(d0),
            "desc_1": float(d1),
            "raw_desc_0": float(raw_d0),
            "raw_desc_1": float(raw_d1),
            "raw_osc_f": float(raw_osc_f),
            "raw_osc_acf": float(raw_osc_acf),
            "morph_class": classify_morphology(morph),
            **morph,
            **{k: float(v) for k, v in job.params.items()},
            **{k: job.cfg.get(k) for k in ["seed", "nx", "ny", "steps"]},
            **um,
        }

        save_json(os.path.join(job.outdir, "meta.json"), {
            "params": job.params,
            "cfg": job.cfg,
            "method": job.method,
            "model": MODEL_NAME,
            "universal": um,
            "desc_0": d0,
            "desc_1": d1,
            "raw_desc_0": raw_d0,
            "raw_desc_1": raw_d1,
            "raw_osc_f": raw_osc_f,
            "raw_osc_acf": raw_osc_acf,
            "morph": morph,
            "morph_class": classify_morphology(morph),
            "score": score,
        })

        return row

    except Exception as e:
        return {
            "run_dir": job.outdir,
            "method": job.method,
            "model": MODEL_NAME,
            "elapsed_s": float(time.time() - t0),
            "error": str(e),
            **{k: float(v) for k, v in job.params.items()},
        }


# -------------------------
# CLI / main
# -------------------------


# -----------------------------
# MAP-Elites style grid archive
# -----------------------------
class GridArchive:
    """Simple 2D grid archive storing best (score) elite per bin.

    Each elite is a dict with at least:
      - d0, d1: descriptor coords in [0,1]
      - score: float
      - run_dir: str
      - params: dict (genome / config params)
    """

    def __init__(self, bins: int = 20):
        self.bins = int(bins)
        self.best_score = -np.inf * np.ones((self.bins, self.bins), dtype=np.float64)
        self.grid_elites = [[None for _ in range(self.bins)] for _ in range(self.bins)]

    @property
    def elites(self):
        out = []
        for i in range(self.bins):
            for j in range(self.bins):
                e = self.grid_elites[i][j]
                if e is not None:
                    out.append(e)
        return out

    def _bin(self, d0: float, d1: float):
        # Robust to NaNs: caller should drop, but guard anyway.
        if not np.isfinite(d0) or not np.isfinite(d1):
            return None
        i = int(np.clip(math.floor(float(d0) * self.bins), 0, self.bins - 1))
        j = int(np.clip(math.floor(float(d1) * self.bins), 0, self.bins - 1))
        return i, j

    def occupancy(self) -> int:
        return int(np.isfinite(self.best_score).sum())

    def update(self, row: dict) -> bool:
        """Consider a completed run row. Returns True if archive improved."""
        d0 = float(row.get("desc_0", np.nan))
        d1 = float(row.get("desc_1", np.nan))
        score = float(row.get("score", np.nan))
        if not np.isfinite(score):
            return False
        ij = self._bin(d0, d1)
        if ij is None:
            return False
        i, j = ij

        if score > float(self.best_score[i, j]):
            self.best_score[i, j] = score
            # store a small elite payload; keep params so we can mutate later
            elite = {
                "i": int(i),
                "j": int(j),
                "d0": d0,
                "d1": d1,
                "score": score,
                "run_dir": str(row.get("run_dir", "")),
                "params": dict(row.get("params", row.get("genome", {}))) if isinstance(row.get("params", row.get("genome", {})), dict) else {},
            }
            # If params aren't present in row, try reconstructing from columns that look like sampled params
            if not elite["params"]:
                # Best-effort: pull any non-metric scalar columns that were used as params.
                for k, v in row.items():
                    if k.startswith("osc_") or k.startswith("w_") or k in ("feed","kill","alpha","beta","gamma","kappa_tau","tau_noise","seed","w_enabled"):
                        # these are either params or diagnostics; include params anyway (runner uses w_* and RD params)
                        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                            elite["params"][k] = float(v)
                        elif isinstance(v, (bool, np.bool_)):
                            elite["params"][k] = bool(v)
            self.grid_elites[i][j] = elite
            return True
        return False

    def best_elite_global(self):
        best = None
        best_s = -np.inf
        for e in self.elites:
            s = float(e.get("score", -np.inf))
            if s > best_s:
                best_s = s
                best = e
        return best

    # Back-compat: earlier runners called archive.best_elite()
    def best_elite(self):
        return self.best_elite_global()


    def random_elite(self, rng: np.random.Generator):
        es = self.elites
        if not es:
            return None
        return es[int(rng.integers(0, len(es)))]

    def to_json(self, path: Optional[str] = None):
        """Return a JSON-serializable dict for the archive, and optionally write it to *path*.

        - If path is None: return the dict (caller can decide how to persist).
        - If path is provided: write the JSON file and return the path.
        """
        data = {
            "bins": int(self.bins),
            "elites": self.elites,
        }
        if path is None:
            return data
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_sanitize)
        return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible search (0 = random)")
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--init_random", type=int, default=40)

    p.add_argument("--bins", type=int, default=24, help="MAP-Elites bins per axis")

    # Descriptor mode: choose what fills the MAP-Elites grid.
    # - osc: frequency/ACF peak (legacy)
    # - morph: morphology descriptors from the w/N field (recommended for phenotype diversity)
    p.add_argument("--desc_mode", type=str, default="morph", choices=["morph", "osc"])

    # Morphology descriptor settings (used when desc_mode=morph)
    p.add_argument("--morph_low", type=int, default=1, help="1: treat low values as structure (mask = w <= quantile); 0: high values as structure")
    p.add_argument("--morph_q", type=float, default=0.10, help="Quantile for thresholding structure mask (e.g. 0.10)")
    p.add_argument("--morph_max_ncomp", type=int, default=10, help="Clip connected-components count at this value for descriptor scaling")

    p.add_argument("--nx", type=int, default=150)
    p.add_argument("--ny", type=int, default=150)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--dt", type=float, default=0.01, help="simulation dt (used for PSD sample dt)")
    p.add_argument("--log_every", type=int, default=20, help="metrics logging cadence in steps (used for PSD sample dt)")

    # --- seed diversity (initial condition) ---
    p.add_argument("--init_mode", type=str, default="square",
                   choices=["square","circle","gaussian","ring","stripes"],
                   help="Initial seeding pattern for B (shape diversity lever).")
    p.add_argument("--seed_pos_mode", type=str, default="center",
                   choices=["center","random"],
                   help="Seed placement: center or random positions.")
    p.add_argument("--seed_count", type=int, default=1,
                   help="Number of seed loci for multi-seed modes.")
    p.add_argument("--seed_sigma", type=float, default=4.0,
                   help="Gaussian sigma for init_mode=gaussian.")
    p.add_argument("--seed_ring_width", type=float, default=2.5,
                   help="Ring width (sigma) for init_mode=ring.")
    p.add_argument("--seed_stripe_period", type=int, default=16,
                   help="Stripe period for init_mode=stripes.")
    p.add_argument("--seed_margin", type=int, default=12,
                   help="Margin from edges for random seed placement.")
    p.add_argument("--seed_radius", type=int, default=10,
                   help="Seed radius for init_mode=square/circle, and ring radius for init_mode=ring.")


    p.add_argument("--save_snapshots", type=int, default=0, help="Save per-step snapshots for visualization (0/1).")
    p.add_argument("--snap_every", type=int, default=200, help="Snapshot cadence in steps when --save_snapshots=1.")
    p.add_argument("--snapshot_format", type=str, default="npz", choices=["npz", "npy"], help="Snapshot format: npz (recommended) or npy legacy.")
    p.add_argument("--osc_method", type=str, default="fft", choices=["fft","welch"], help="PSD estimator for osc metrics")
    p.add_argument("--osc_welch_nperseg", type=int, default=256, help="Welch segment length (samples)")
    p.add_argument("--osc_welch_noverlap", type=float, default=0.5, help="Welch overlap fraction in [0,1)")

    # steering weights
    p.add_argument("--w_growth", type=float, default=1.0)
    p.add_argument("--w_osc", type=float, default=1.0)

    # descriptor scaling
    p.add_argument("--growth_scale", type=float, default=0.002, help="scale for sigmoid(growth_slope/scale)")
    p.add_argument("--osc_scale", type=float, default=1.0, help="scale for sigmoid(osc_log/scale)")

    # viability gate from alive_frac
    p.add_argument("--alive_gate", type=float, default=0.4, help="alive_frac midpoint")
    p.add_argument("--alive_gate_width", type=float, default=0.1)

    # time windowing
    p.add_argument("--window_start_frac", type=float, default=0.2, help="ignore first fraction of samples")
    p.add_argument("--w_gate", type=float, default=0.01, help="alive threshold on w_mean")
    p.add_argument("--w_gate_width", type=float, default=0.01)
    p.add_argument("--log_dt_fallback", type=float, default=1.0)

    # oscillation band (units: cycles per time unit in metrics.csv)
    p.add_argument("--osc_fmin", type=float, default=0.002)
    p.add_argument("--osc_fmax", type=float, default=0.03)
    p.add_argument("--osc_min_cycles", type=float, default=2.0, help="Minimum cycles required in sampled window for ACF freq estimate; set 0 to disable")

    # descriptor calibration (writes <out_root>/desc_calib.json)
    p.add_argument("--calib_min_rows", type=int, default=80, help="min finished rows before writing desc_calib.json")
    p.add_argument("--calib_every", type=int, default=40, help="update desc_calib.json every N finished rows")
    p.add_argument("--calib_q_lo", type=float, default=0.05)
    p.add_argument("--calib_q_hi", type=float, default=0.95)
    p.add_argument("--osc_log_lo_default", type=float, default=-10.0)
    p.add_argument("--osc_log_hi_default", type=float, default=-7.0)


    # QD proposal
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--flip_prob", type=float, default=0.35)

    # allow tuning w system from runner
    p.add_argument("--w_enabled", type=int, default=1)
    p.add_argument("--w_mode", type=str, default="varratio")
    p.add_argument("--w_radius", type=int, default=1)
    p.add_argument("--w_amp_thr", type=float, default=0.05)
    p.add_argument("--w_amp_width", type=float, default=0.02)
    p.add_argument("--w_tau_bias", type=float, default=0.0)
    p.add_argument("--save_w", type=int, default=0)

    # allow tightening the search range of w_tau_gain without editing PARAM_SPACE
    p.add_argument("--w_tau_gain_max", type=float, default=0.15)

    return p.parse_args()


def append_csv(path: str, row: Dict[str, Any]) -> None:
    """Append one row to a CSV with a *stable* column order.

    The v16 runner used pandas-to_csv on single-row DataFrames; when dict key order
    differs between rows, pandas may write values in a different column order than
    the header, corrupting the file. This function locks column order to the
    existing header. If new keys appear, it expands the header and rewrites the CSV
    once (cheap at these run sizes).
    """
    import csv
    import os

    def _stringify(v: Any) -> str:
        # Keep CSV scalar-friendly; nested objects become JSON.
        if v is None:
            return ""
        if isinstance(v, (bool, np.bool_)):
            return "1" if bool(v) else "0"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            # preserve NaN as blank for pandas friendliness
            fv = float(v)
            return "" if (not np.isfinite(fv)) else repr(fv)
        if isinstance(v, (str,)):
            return v
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, default=_json_sanitize)
        return str(v)

    if not os.path.exists(path):
        # First write: choose a deterministic header order (preferred keys first).
        preferred = [
            "run_dir","method","model","elapsed_s","score","desc_0","desc_1","raw_desc_0","raw_desc_1","error",
        ]
        header = []
        for k in preferred:
            if k in row and k not in header:
                header.append(k)
        # then all remaining keys in sorted order for stability across runs
        for k in sorted(row.keys()):
            if k not in header:
                header.append(k)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow([_stringify(row.get(k, "")) for k in header])
        return

    # Read existing header
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    row_keys = set(row.keys())
    header_set = set(header)
    extras = [k for k in row_keys if k not in header_set]

    if extras:
        # Expand header and rewrite file once to avoid column drift.
        new_header = list(header) + sorted(extras)
        # Read existing rows as dicts, then rewrite with new header.
        with open(path, "r", newline="", encoding="utf-8") as f:
            dr = csv.DictReader(f)
            rows = list(dr)

        with open(path, "w", newline="", encoding="utf-8") as f:
            dw = csv.DictWriter(f, fieldnames=new_header, extrasaction="ignore")
            dw.writeheader()
            for r in rows:
                # DictReader gives strings; keep as-is.
                dw.writerow({k: r.get(k, "") for k in new_header})

        header = new_header

    # Append aligned row
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([_stringify(row.get(k, "")) for k in header])


def main() -> None:
    args = parse_args()

    # patch PARAM_SPACE upper bound for w_tau_gain
    for i, (name, lo, hi) in enumerate(PARAM_SPACE):
        if name == "w_tau_gain":
            PARAM_SPACE[i] = (name, lo, float(args.w_tau_gain_max))

    print(f"[runner] Using simulator: {MODEL_NAME} ({getattr(model, '__file__', 'unknown')})")

    ensure_dir(args.out_root)
    log_path = os.path.join(args.out_root, "boqd_log.csv")
    best_path = os.path.join(args.out_root, "best_so_far.json")
    elites_path = os.path.join(args.out_root, "qd_elites.json")

    # Descriptor calibration state (shared in main process)
    calib_path = os.path.join(args.out_root, "desc_calib.json")
    calib = load_desc_calib(calib_path)
    if float(calib.get("updated_rows", 0)) <= 0:
        calib["osc_log_lo"] = float(args.osc_log_lo_default)
        calib["osc_log_hi"] = float(args.osc_log_hi_default)
    osc_log_samples: List[float] = []


    rng = np.random.default_rng(None if int(args.seed)==0 else int(args.seed))
    archive = GridArchive(bins=args.bins)

    # multiproc
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=int(args.workers))

    pending: List[mp.pool.ApplyResult] = []
    run_idx = 0

    def make_outdir(method: str, params: Dict[str, float], seed: int) -> str:
        h = stable_hash({"method": method, "seed": seed, **params, "nx": args.nx, "ny": args.ny, "steps": args.steps})
        return os.path.join(args.out_root, method, f"run_{run_idx:05d}_{h}")

    def submit(method: str, params: Dict[str, float]) -> None:
        nonlocal run_idx
        seed = int(rng.integers(0, 2**31 - 1))
        cfg = make_cfg(params, args, seed)
        outdir = make_outdir(method, params, seed)
        run_idx += 1
        job = Job(params=params, cfg=cfg, outdir=outdir, method=method)
        pending.append(pool.apply_async(run_one, (job, args)))

    # init random
    total = int(args.budget)
    init_n = int(min(args.init_random, total))
    remaining = total

    for _ in range(init_n):
        submit("init", sample_params(rng))
        remaining -= 1

    best_row: Optional[Dict[str, Any]] = None

    def handle(row: Dict[str, Any]) -> None:
        nonlocal best_row

        if "error" not in row:
            # Calibrated descriptors (computed in main so calibration can update over time)
            raw_f = float(row.get("raw_desc_0", row.get("osc_f_est", row.get("osc_f_peak", float("nan")))))
            raw_acf = float(row.get("raw_desc_1", row.get("osc_acf_peak", float("nan"))))
            raw_log = float(row.get("osc_log", (row.get("universal", {}) or {}).get("osc_log", float("nan"))))

            if str(getattr(args, "desc_mode", "morph")).lower() == "morph":
                # Identity mapping: morphology descriptors are already normalized (or clipped) into [0,1].
                d0, d1 = clamp01(raw_f), clamp01(raw_acf)
            else:
                d0, d1 = apply_desc_calib(raw_f, raw_acf, calib, args)

            row["desc_0"] = d0
            row["desc_1"] = d1
            score = float(row["score"])
            if np.isfinite(d0) and np.isfinite(d1) and np.isfinite(score):
                updated = archive.update(row)
            else:
                updated = False

            # Update calibration periodically from observed osc_log values
            if np.isfinite(raw_log):
                osc_log_samples.append(raw_log)
            n_ok = len(osc_log_samples)
            if (n_ok >= int(args.calib_min_rows)) and (int(args.calib_every) > 0) and (n_ok % int(args.calib_every) == 0):
                qlo = float(args.calib_q_lo); qhi = float(args.calib_q_hi)
                lo = float(np.quantile(osc_log_samples, qlo))
                hi = float(np.quantile(osc_log_samples, qhi))
                # guard against degenerate ranges
                if hi <= lo + 1e-9:
                    lo = float(args.osc_log_lo_default)
                    hi = float(args.osc_log_hi_default)
                calib["osc_log_lo"] = lo
                calib["osc_log_hi"] = hi
                calib["updated_rows"] = float(n_ok)
                with open(os.path.join(args.out_root, "desc_calib.json"), "w", encoding="utf-8") as f:
                    json.dump(calib, f, indent=2, default=_json_sanitize)

            if (best_row is None) or (score > float(best_row.get("score", -1e18))):
                best_row = dict(row)
                save_json(best_path, best_row)

            if updated:
                save_json(elites_path, archive.to_json())

        # IMPORTANT: write to boqd_log *after* desc_0/desc_1 are attached.
        append_csv(log_path, row)

    # main loop
    while True:
        while (remaining > 0) and (len(pending) < int(args.workers)):
            # propose from elites when available
            if archive.elites and rng.random() > 0.1:
                parent = archive.random_elite(rng) if (rng.random() < float(args.flip_prob)) else archive.best_elite()
                if parent is None:
                    params = sample_params(rng)
                else:
                    params = mutate_params(rng, parent['params'] if isinstance(parent, dict) else parent.params, sigma=float(args.sigma))
                submit("qd", params)
            else:
                submit("rand", sample_params(rng))
            remaining -= 1

        if (remaining <= 0) and (len(pending) == 0):
            break

        still: List[mp.pool.ApplyResult] = []
        for r in pending:
            if r.ready():
                handle(r.get())
            else:
                still.append(r)
        pending = still

        time.sleep(0.05)

    pool.close()
    pool.join()

    save_json(elites_path, archive.to_json())

    print("\n[done]")
    print("log:", log_path)
    print("elites:", elites_path)
    if best_row:
        print("best score:", best_row.get("score"))
        print("best run:", best_row.get("run_dir"))
    print("occupancy:", archive.occupancy())


if __name__ == "__main__":

    # --- seed diversity (initial condition) ---
    main()
