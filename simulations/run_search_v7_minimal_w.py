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

# --- simulator import (prefer v7) ---
MODEL_NAME = None
try:
    import dynamic_tau_v7 as model  # type: ignore
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

# Guardrail: enforce a minimum feedback gain when debugging (set from CLI).
W_TAU_GAIN_MIN: float = 0.0


def _clamp_w_tau_gain(p: Dict[str, float]) -> Dict[str, float]:
    """Enforce the CLI guardrail on the genome value."""
    if "w_tau_gain" in p:
        p["w_tau_gain"] = float(max(p["w_tau_gain"], W_TAU_GAIN_MIN))
    return p


def sample_params(rng: np.random.Generator) -> Dict[str, float]:
    p: Dict[str, float] = {}
    for name, lo, hi in PARAM_SPACE:
        p[name] = float(lo + (hi - lo) * rng.random())
    return _clamp_w_tau_gain(p)


def mutate_params(rng: np.random.Generator, base: Dict[str, float], sigma: float) -> Dict[str, float]:
    out = dict(base)
    for name, lo, hi in PARAM_SPACE:
        span = (hi - lo)
        out[name] = float(np.clip(out[name] + rng.normal(0.0, sigma) * span, lo, hi))
    return _clamp_w_tau_gain(out)


# -------------------------
# Simulator cfg
# -------------------------

def base_sim_cfg(nx: int, ny: int, steps: int) -> Dict[str, Any]:
    # Minimal set; your dynamic_tau_v7 can ignore unknown keys.
    return {
        "nx": nx,
        "ny": ny,
        "dx": 1.0,
        "dy": 1.0,
        "dt": 0.01,
        "steps": steps,
        "log_every": 20,
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
    cfg = base_sim_cfg(args.nx, args.ny, args.steps)

    # continuous params
    cfg["feed"] = float(params["feed"])
    cfg["kill"] = float(params["kill"])
    cfg["alpha"] = float(params["alpha"])
    cfg["beta"] = float(params["beta"])
    cfg["gamma"] = float(params["gamma"])
    cfg["kappa_tau"] = float(params["kappa_tau"])
    cfg["tau_noise"] = float(params["tau_noise"])

    # w-system toggles/shape
    cfg["w_enabled"] = bool(args.w_enabled)
    cfg["w_mode"] = str(args.w_mode)
    cfg["w_radius"] = int(args.w_radius)
    cfg["w_amp_thr"] = float(args.w_amp_thr)
    cfg["w_amp_width"] = float(args.w_amp_width)

    # feedback gain from genome (guardrail already applied)
    cfg["w_tau_gain"] = float(params.get("w_tau_gain", 0.0))
    cfg["w_tau_bias"] = float(args.w_tau_bias)

    # some sims use *_max naming; harmless if ignored
    cfg["w_tau_gain_max"] = float(args.w_tau_gain_max)

    cfg["save_w"] = bool(args.save_w)
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


def _psd_peak_log_ratio(x: np.ndarray, dt: float, fmin: float, fmax: float) -> Tuple[float, float, float]:
    """Return (osc_log_peak, peak, floor) using rFFT PSD, excluding DC."""
    x = np.asarray(x, dtype=float)
    if len(x) < 8 or not np.isfinite(x).all():
        return float("nan"), float("nan"), float("nan")

    x = x - np.mean(x)
    # Hann window
    w = np.hanning(len(x))
    xw = x * w

    X = np.fft.rfft(xw)
    psd = (np.abs(X) ** 2)
    freqs = np.fft.rfftfreq(len(xw), d=dt)

    # exclude DC
    psd = psd.copy()
    psd[0] = 0.0

    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return float("nan"), float("nan"), float("nan")

    band_psd = psd[band]
    peak = float(np.max(band_psd))
    floor = float(np.median(band_psd))
    eps = 1e-18
    osc_log = float(math.log((peak + eps) / (floor + eps)))
    return osc_log, peak, floor


def compute_universal_from_metrics(metrics_csv: str, args: argparse.Namespace) -> Dict[str, float]:
    df = pd.read_csv(metrics_csv)
    if df.empty or "w_mean" not in df.columns:
        raise RuntimeError("metrics.csv missing w_mean")

    t = df["time"].to_numpy(dtype=float)
    w = df["w_mean"].to_numpy(dtype=float)

    # infer dt from time column
    if len(t) >= 2:
        dt = float(np.median(np.diff(t)))
    else:
        dt = float(args.log_dt_fallback)

    w_gate = float(args.w_gate)
    alive_frac = float(np.mean(w > w_gate))

    # choose a window (ignore early transient)
    n = len(w)
    i0 = int(np.floor(n * float(args.window_start_frac)))
    i0 = max(0, min(i0, n - 3))

    tt = t[i0:]
    ww = w[i0:]

    # viable weights: only care when alive
    wt = 1.0 / (1.0 + np.exp(-(ww - w_gate) / max(float(args.w_gate_width), 1e-9)))

    eps = 1e-12
    logw = np.log(ww + eps)
    growth_slope = _weighted_slope(tt, logw, wt)

    osc_log, osc_peak, osc_floor = _psd_peak_log_ratio(
        x=ww,
        dt=dt,
        fmin=float(args.osc_fmin),
        fmax=float(args.osc_fmax),
    )

    # ---- NEW: measured feedback gain (from metrics.csv if present) ----
    if "w_tau_gain" in df.columns:
        xg = pd.to_numeric(df["w_tau_gain"], errors="coerce").to_numpy(dtype=float)
        # prefer final value (could also use mean)
        w_tau_gain_measured = float(xg[-1]) if len(xg) else float("nan")
        w_tau_gain_missing = 0.0
    else:
        w_tau_gain_measured = float("nan")
        w_tau_gain_missing = 1.0

    return {
        "alive_frac": alive_frac,
        "growth_slope": float(growth_slope),
        "osc_log": float(osc_log),
        "osc_peak": float(osc_peak),
        "osc_floor": float(osc_floor),
        "w_mean_mean": float(np.nanmean(w)),
        "w_mean_final": float(w[-1]),
        "dt_metric": dt,
        "w_tau_gain_measured": w_tau_gain_measured,
        "w_tau_gain_missing": w_tau_gain_missing,
    }


def sigmoid01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# -------------------------
# MAP-Elites grid
# -------------------------

@dataclass
class Elite:
    score: float
    run_dir: str
    desc0: float
    desc1: float
    params: Dict[str, float]


class GridArchive:
    def __init__(self, bins: int):
        self.bins = int(bins)
        self.elites: Dict[Tuple[int, int], Elite] = {}

    def key(self, d0: float, d1: float) -> Tuple[int, int]:
        i = int(np.clip(math.floor(d0 * self.bins), 0, self.bins - 1))
        j = int(np.clip(math.floor(d1 * self.bins), 0, self.bins - 1))
        return (i, j)

    def update(self, d0: float, d1: float, score: float, run_dir: str, params: Dict[str, float]) -> bool:
        k = self.key(d0, d1)
        cur = self.elites.get(k)
        if (cur is None) or (score > cur.score):
            self.elites[k] = Elite(score=score, run_dir=run_dir, desc0=d0, desc1=d1, params=params)
            return True
        return False

    def random_elite(self, rng: np.random.Generator) -> Optional[Elite]:
        if not self.elites:
            return None
        ks = list(self.elites.keys())
        k = ks[int(rng.integers(0, len(ks)))]
        return self.elites[k]

    def best_elite(self) -> Optional[Elite]:
        if not self.elites:
            return None
        return max(self.elites.values(), key=lambda e: e.score)

    def occupancy(self) -> float:
        return float(len(self.elites) / max(1, self.bins * self.bins))

    def to_json(self) -> Dict[str, Any]:
        out = {
            "bins": self.bins,
            "n_filled": int(len(self.elites)),
            "occupancy": self.occupancy(),
            "elites": [],
        }
        for (i, j), e in self.elites.items():
            out["elites"].append({
                "i": int(i),
                "j": int(j),
                "score": float(e.score),
                "run_dir": str(e.run_dir),
                "desc_0": float(e.desc0),
                "desc_1": float(e.desc1),
                "params": e.params,
            })
        return out


# -------------------------
# Jobs
# -------------------------

@dataclass
class Job:
    params: Dict[str, float]
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

        # descriptors
        d0 = sigmoid01(float(um["growth_slope"]) / max(float(args.growth_scale), 1e-18))
        d1 = sigmoid01(float(um["osc_log"]) / max(float(args.osc_scale), 1e-18))

        # score: alive gate * (positive growth + osc)
        alive_gate = sigmoid01(
            (float(um["alive_frac"]) - float(args.alive_gate)) / max(float(args.alive_gate_width), 1e-9)
        )
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
            **{k: float(v) for k, v in job.params.items()},  # includes w_tau_gain (PARAM)
            **{k: job.cfg.get(k) for k in ["seed", "nx", "ny", "steps"]},
            **um,  # includes w_tau_gain_measured + w_tau_gain_missing
        }

        save_json(os.path.join(job.outdir, "meta.json"), {
            "params": job.params,
            "cfg": job.cfg,
            "method": job.method,
            "model": MODEL_NAME,
            "universal": um,
            "desc_0": d0,
            "desc_1": d1,
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--init_random", type=int, default=40)

    p.add_argument("--bins", type=int, default=24, help="MAP-Elites bins per axis")

    p.add_argument("--nx", type=int, default=150)
    p.add_argument("--ny", type=int, default=150)
    p.add_argument("--steps", type=int, default=3000)

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

    # oscillation band
    p.add_argument("--osc_fmin", type=float, default=0.002)
    p.add_argument("--osc_fmax", type=float, default=0.03)

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

    # ---- NEW: guardrail + debug printing ----
    p.add_argument("--w_tau_gain_min", type=float, default=0.0,
                   help="Guardrail: clamp sampled/mutated w_tau_gain to at least this value.")
    p.add_argument("--debug_print_first", type=int, default=0,
                   help="Print the first N submitted genomes for sanity checks.")
    p.add_argument("--debug_print_every", type=int, default=0,
                   help="Print every K-th submitted genome (0 disables).")

    return p.parse_args()


def append_csv(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not exists, index=False)


def main() -> None:
    args = parse_args()

    # patch PARAM_SPACE upper bound for w_tau_gain
    for i, (name, lo, hi) in enumerate(PARAM_SPACE):
        if name == "w_tau_gain":
            PARAM_SPACE[i] = (name, lo, float(args.w_tau_gain_max))

    # Guardrails / sanity prints for feedback activation.
    global W_TAU_GAIN_MIN
    W_TAU_GAIN_MIN = float(args.w_tau_gain_min)

    # Validate the patched bounds.
    w_lo, w_hi = next((lo, hi) for (n, lo, hi) in PARAM_SPACE if n == "w_tau_gain")
    print(f"[runner] w_tau_gain bounds: lo={w_lo} hi={w_hi} (min clamp={W_TAU_GAIN_MIN})")
    if w_hi <= w_lo:
        raise ValueError(
            f"Invalid w_tau_gain bounds after patch: lo={w_lo} hi={w_hi}. "
            f"Did you pass --w_tau_gain_max 0?"
        )

    print(f"[runner] Using simulator: {MODEL_NAME} ({getattr(model, '__file__', 'unknown')})")

    ensure_dir(args.out_root)
    log_path = os.path.join(args.out_root, "boqd_log.csv")
    best_path = os.path.join(args.out_root, "best_so_far.json")
    elites_path = os.path.join(args.out_root, "qd_elites.json")

    rng = np.random.default_rng()
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

        # ---- FIX #1: safe formatting in debug prints ----
        def _fmt6g(v):
            if v is None:
                return "NA"
            try:
                return f"{float(v):.6g}"
            except Exception:
                return "NA"

        # Optional: print a sample of submitted genomes so we can verify w_tau_gain is not collapsing to 0.
        if args.debug_print_first and run_idx <= int(args.debug_print_first):
            print(
                f"[submit {run_idx:05d}] method={method} "
                f"w_tau_gain={_fmt6g(params.get('w_tau_gain'))} "
                f"w_enabled={int(cfg.get('w_enabled', 0) or 0)} "
                f"w_tau_bias={_fmt6g(cfg.get('w_tau_bias'))} "
                f"w_gate={_fmt6g(args.w_gate)} "
                f"outdir={outdir}"
            )

        if args.debug_print_every and (run_idx % int(args.debug_print_every) == 0):
            print(
                f"[submit {run_idx:05d}] method={method} "
                f"w_tau_gain={_fmt6g(params.get('w_tau_gain'))} "
                f"w_enabled={int(cfg.get('w_enabled', 0) or 0)} "
                f"outdir={outdir}"
            )

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
        append_csv(log_path, row)

        if "error" not in row:
            d0 = float(row["desc_0"])
            d1 = float(row["desc_1"])
            score = float(row["score"])
            updated = archive.update(
                d0, d1, score, str(row["run_dir"]),
                {k: float(row[k]) for k, _, _ in PARAM_SPACE}
            )

            if (best_row is None) or (score > float(best_row.get("score", -1e18))):
                best_row = dict(row)
                save_json(best_path, best_row)

            if updated:
                save_json(elites_path, archive.to_json())

    # main loop
    while True:
        while (remaining > 0) and (len(pending) < int(args.workers)):
            # propose from elites when available
            if archive.elites and rng.random() > 0.1:
                parent = archive.random_elite(rng) if (rng.random() < float(args.flip_prob)) else archive.best_elite()
                if parent is None:
                    params = sample_params(rng)
                else:
                    params = mutate_params(rng, parent.params, sigma=float(args.sigma))
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
    main()
