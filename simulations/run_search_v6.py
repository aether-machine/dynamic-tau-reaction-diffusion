#!/usr/bin/env python3
"""
run_search_v6.py

Hybrid search driver for dynamic_tau_v6:

- Bayesian Optimization (simple GP + Expected Improvement) for "best proto-life score"
- Quality-Diversity (MAP-Elites) to intentionally harvest diverse regimes

Outputs (under --out_root):
  boqd_log.csv             # one row per run (params + metrics + score + descriptors)
  qd_map.png               # heatmap of elite scores in behavior space
  qd_elites.json           # which run_dir is elite for each cell
  best_so_far.json         # current best run (by score)

Run example:
  python simulations/run_search_v6.py \
    --out_root outputs/dynamic_tau_v6_search \
    --workers 8 --budget 400 --init_random 40 \
    --mode hybrid --p_qd 0.5 \
    --steps 3000 --nx 150 --ny 150 \
    --save_snapshots_elites 1

This does NOT "force" dynamics; it only chooses (alpha,beta,feed,kill,...) intelligently.
"""

import os, json, math, time, hashlib

# -------------------------
# JSON helper (numpy-safe)
# -------------------------
def _json_sanitize(o):
    """Make common numpy / pathlib types JSON-serializable."""
    import numpy as _np
    from pathlib import Path as _Path
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    if isinstance(o, _Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


# -------------------------
# Search / QD / Score configuration (set by CLI in main())
# -------------------------
QD_CFG = {
    "bins": 20,
    "maint_max": 0.35,
    "reorg_scale": 0.05,
    "osc_scale": 1e-5,
    "mix_osc": 0.30,
    "sigma": 0.14,
    "flip_prob": 0.35,
}

SCORE_CFG = {
    "version": "v1",        # "v1" or "v2"
    "iou_gate": 0.08,
    "iou_width": 0.02,
    "iou_sat_scale": 0.10,
}

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dynamic_tau_v6 as model


# -----------------------------
# Parameter space (edit freely)
# -----------------------------
# We keep this intentionally modest-dimensional for BO stability.
PARAM_SPACE = [
    ("alpha",      "float", 0.000, 0.080),
    ("beta",       "float", 0.001, 0.020),
    ("gamma",      "float", 0.000, 0.800),
    ("feed",       "float", 0.020, 0.050),
    ("kill",       "float", 0.045, 0.085),
    ("kappa_tau",  "float", 0.000, 0.100),
    ("tau_noise",  "float", 0.000, 0.010),
    # optional: enable if you want BO to explore these too
    # ("memory_decay", "float", 0.001, 0.050),
]

# Discrete flags explored more naturally by QD mutations (can be randomized too)
FLAG_SPACE = {
    "use_multiscale_memory": [False, True],
    "use_diffusive_nutrient": [False, True],
    "num_tau_species": [1, 2],  # keep small at first
}

# -----------------------------
# Defaults for sim cfg (V6)
# -----------------------------
def base_sim_cfg(nx: int, ny: int, steps: int) -> Dict[str, Any]:
    return {
        "nx": nx, "ny": ny,
        "dx": 1.0, "dy": 1.0,
        "dt": 0.01,
        "steps": steps,
        "log_every": 20,

        # snapshots off by default for search
        "save_snapshots": False,
        "save_montage": False,
        "snap_every": 200,

        # save NPZ mid/final for fast metrics
        "save_states": True,
        "mid_state_step": steps // 2,

        # Gray-Scott
        "Da": 0.16, "Db": 0.08,

        # τ / nutrient defaults (overridden by params)
        "tau0": 1.0, "tau_min": 0.2, "tau_max": 5.0,
        "alpha": 0.03, "beta": 0.006, "gamma": 0.3,
        "kappa_tau": 0.02,
        "tau_noise": 0.001,

        # nutrient
        "N0": 1.0,
        "nutrient_use": 0.01,
        "nutrient_replenish": 0.001,
        "use_diffusive_nutrient": True,
        "D_N": 0.02,
        "eta_N": 0.1,
        "rho_N": 0.0005,

        # memory
        "use_multiscale_memory": True,
        "memory_decay": 0.01,
        "mem_decay_fast": 0.02,
        "mem_decay_slow": 0.002,
        "mem_w_fast": 0.7,
        "mem_w_slow": 0.3,

        # multi-τ
        "num_tau_species": 1,

        # init
        "seed_radius": 10,
        "noise": 0.02,
    }


# -----------------------------
# Small GP for BO (no sklearn)
# -----------------------------
def rbf_kernel(X: np.ndarray, Y: np.ndarray, lengthscale: float = 0.20) -> np.ndarray:
    # X: (n,d), Y: (m,d)
    # returns (n,m)
    X2 = np.sum(X*X, axis=1, keepdims=True)
    Y2 = np.sum(Y*Y, axis=1, keepdims=True).T
    dist2 = X2 + Y2 - 2.0 * (X @ Y.T)
    return np.exp(-0.5 * dist2 / (lengthscale**2))

def std_norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5*z*z) / math.sqrt(2.0*math.pi)

def std_norm_cdf(z: np.ndarray) -> np.ndarray:
    # 0.5 * (1 + erf(z/sqrt(2)))
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

class SimpleGP:
    def __init__(self, lengthscale: float = 0.20, noise: float = 1e-6):
        self.lengthscale = float(lengthscale)
        self.noise = float(noise)
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X.copy()
        self.y = y.copy()
        K = rbf_kernel(self.X, self.X, self.lengthscale)
        K = K + (self.noise + 1e-12) * np.eye(len(self.X))
        # Cholesky
        self.L = np.linalg.cholesky(K)
        # alpha = K^-1 y via solves
        tmp = np.linalg.solve(self.L, self.y)
        self.alpha = np.linalg.solve(self.L.T, tmp)

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ks = rbf_kernel(Xs, self.X, self.lengthscale)
        mu = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        # variance
        kss = np.ones(len(Xs), dtype=np.float64)  # rbf(x,x)=1
        var = kss - np.sum(v*v, axis=0)
        var = np.maximum(var, 1e-12)
        return mu, var


def expected_improvement(mu: np.ndarray, var: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.sqrt(var)
    imp = mu - best - xi
    z = imp / (sigma + 1e-12)
    ei = imp * std_norm_cdf(z) + sigma * std_norm_pdf(z)
    ei[sigma < 1e-12] = 0.0
    return ei


# -----------------------------
# Encoding / decoding params
# -----------------------------
def encode_params(p: Dict[str, Any]) -> np.ndarray:
    xs = []
    for (name, typ, lo, hi) in PARAM_SPACE:
        v = float(p[name])
        u = (v - lo) / (hi - lo + 1e-12)
        xs.append(float(np.clip(u, 0.0, 1.0)))
    return np.array(xs, dtype=np.float64)

def decode_params(x: np.ndarray) -> Dict[str, Any]:
    out = {}
    for i, (name, typ, lo, hi) in enumerate(PARAM_SPACE):
        u = float(np.clip(x[i], 0.0, 1.0))
        out[name] = lo + u * (hi - lo)
    return out


def stable_hash(d: Dict[str, Any]) -> str:
    # hash only a canonical subset (params + important toggles) so it’s reproducible
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# -----------------------------
# Metrics computation
# -----------------------------
def load_npz_state(run_dir: str, which: str) -> Optional[Dict[str, np.ndarray]]:
    p = Path(run_dir) / which
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}

def maintenance_iou(B_mid: np.ndarray, B_final: np.ndarray) -> float:
    # threshold on combined distribution to stabilize
    combo = np.concatenate([B_mid.ravel(), B_final.ravel()])
    th = float(np.percentile(combo, 90))
    m1 = B_mid > th
    m2 = B_final > th
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(inter / union)

def internal_reorg_index(B_mid: np.ndarray, B_final: np.ndarray) -> float:
    # correlation inside the FINAL mask (body persists, interior changes)
    th = float(np.percentile(B_final, 90))
    mask = B_final > th
    if mask.sum() < 10:
        return 0.0
    x = B_mid[mask].ravel().astype(np.float64)
    y = B_final[mask].ravel().astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    vx = float(np.sum(x*x))
    vy = float(np.sum(y*y))
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    corr = float(np.sum(x*y) / math.sqrt(vx*vy))
    corr = max(-1.0, min(1.0, corr))
    return float(1.0 - corr)  # in [0,2]

def coherence_osc_index(metrics_csv: str) -> float:
    df = pd.read_csv(metrics_csv)
    if df.empty or "time" not in df.columns or "coherence" not in df.columns:
        return float("nan")
    t = df["time"].to_numpy(dtype=np.float64)
    c = df["coherence"].to_numpy(dtype=np.float64)
    if len(t) < 10:
        return float("nan")
    # detrend linear
    a, b = np.polyfit(t, c, 1)
    resid = c - (a*t + b)
    return float(np.var(resid))

def tau_structure(tau_final: np.ndarray) -> Tuple[float, float]:
    tau_var = float(np.var(tau_final))
    gy, gx = np.gradient(tau_final)
    grad2 = gx*gx + gy*gy
    return tau_var, float(np.mean(grad2))

def summarize_timeseries(metrics_csv: str) -> Dict[str, float]:
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return {
            "mean_coherence": np.nan, "std_coherence": np.nan, "max_coherence": np.nan,
            "mean_entropy": np.nan, "mean_autocat": np.nan,
            "coherence_slope": np.nan,
        }
    mean_coh = float(df["coherence"].mean()) if "coherence" in df.columns else np.nan
    std_coh  = float(df["coherence"].std())  if "coherence" in df.columns else np.nan
    max_coh  = float(df["coherence"].max())  if "coherence" in df.columns else np.nan
    mean_ent = float(df["entropy"].mean())   if "entropy" in df.columns else np.nan
    mean_aut = float(df["autocat"].mean())   if "autocat" in df.columns else np.nan

    # slope
    if "time" in df.columns and "coherence" in df.columns and len(df) >= 5:
        t = df["time"].to_numpy(dtype=np.float64)
        c = df["coherence"].to_numpy(dtype=np.float64)
        a, b = np.polyfit(t, c, 1)
        slope = float(a)
    else:
        slope = np.nan

    return {
        "mean_coherence": mean_coh,
        "std_coherence": std_coh,
        "max_coherence": max_coh,
        "mean_entropy": mean_ent,
        "mean_autocat": mean_aut,
        "coherence_slope": slope,
    }


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_score(m: Dict[str, float]) -> float:
    """Objective score optimized by BO/QD."""
    iou = float(m.get("maintenance_iou", 0.0))
    reorg = float(m.get("internal_reorg_index", 0.0))
    osc = float(m.get("coherence_osc_index", 0.0))
    coh = float(m.get("coherence_index", 0.0))
    ent = float(m.get("entropy_proxy", 0.0))

    # base weights (keep your intent; tune later if needed)
    w_iou = 2.0
    w_reorg = 1.0
    w_osc = 0.8
    w_coh = 0.8
    w_ent = 0.2

    version = str(SCORE_CFG.get("version", "v1")).lower()
    if version == "v1":
        return float(w_iou * iou + w_reorg * reorg + w_osc * osc + w_coh * coh - w_ent * ent)

    if version != "v2":
        raise ValueError(f"Unknown SCORE_CFG['version']={version!r}")

    gate = float(SCORE_CFG.get("iou_gate", 0.08))
    width = float(SCORE_CFG.get("iou_width", 0.02))
    sat = float(SCORE_CFG.get("iou_sat_scale", 0.10))

    # soft gate in [0,1]
    g = _sigmoid((iou - gate) / max(width, 1e-9))

    # saturating iou term in [0,1)
    iou_excess = max(iou - gate, 0.0)
    iou_term = 1.0 - math.exp(-iou_excess / max(sat, 1e-9))

    # dynamics terms count primarily when viable
    score = (
        w_iou * (g * iou_term)
        + g * (w_reorg * reorg + w_osc * osc + w_coh * coh)
        - w_ent * ent
    )
    return float(score)


def _norm_exp(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return 1.0 - math.exp(-max(x, 0.0) / scale)


def descriptor_pair(m: Dict[str, float]) -> Tuple[float, float]:
    """Mixed descriptor mapping: d1=maintenance, d2=blend(reorg,osc)."""
    maint = float(m.get("maintenance_iou", 0.0))
    reorg = float(m.get("internal_reorg_index", 0.0))
    osc = float(m.get("coherence_osc_index", 0.0))

    maint_max = float(QD_CFG.get("maint_max", 0.35))
    reorg_scale = float(QD_CFG.get("reorg_scale", 0.05))
    osc_scale = float(QD_CFG.get("osc_scale", 1e-5))
    mix = float(QD_CFG.get("mix_osc", 0.30))

    d1 = max(0.0, min(1.0, maint / max(maint_max, 1e-12)))
    reorg_n = _norm_exp(reorg, reorg_scale)
    osc_n = _norm_exp(osc, osc_scale)
    d2 = (1.0 - mix) * reorg_n + mix * osc_n
    d2 = max(0.0, min(1.0, d2))
    return float(d1), float(d2)

class QDMap:
    def __init__(self, bins_x: int = 20, bins_y: int = 20):
        self.bx = int(bins_x)
        self.by = int(bins_y)
        self.best_score = -np.inf * np.ones((self.bx, self.by), dtype=np.float64)
        self.best_run = [[None for _ in range(self.by)] for _ in range(self.bx)]

    def _bin(self, d1: float, d2: float) -> Tuple[int, int]:
        ix = int(np.clip(math.floor(d1 * self.bx), 0, self.bx - 1))
        iy = int(np.clip(math.floor(d2 * self.by), 0, self.by - 1))
        return ix, iy

    def consider(self, d1: float, d2: float, score: float, run_dir: str):
        ix, iy = self._bin(d1, d2)
        if score > self.best_score[ix, iy]:
            self.best_score[ix, iy] = score
            self.best_run[ix][iy] = run_dir


    def best_elite(self) -> Optional[str]:
        """Return the run_dir of the single best elite in the map (or None if empty)."""
        best_dir: Optional[str] = None
        best_s: float = -np.inf
        for ix in range(self.bx):
            for iy in range(self.by):
                rd = self.best_run[ix][iy]
                if rd is None:
                    continue
                s = float(self.best_score[ix, iy])
                if s > best_s:
                    best_s = s
                    best_dir = rd
        return best_dir

    def random_elite(self, rng: np.random.Generator) -> Optional[str]:
        coords = [(i, j) for i in range(self.bx) for j in range(self.by) if self.best_run[i][j] is not None]
        if not coords:
            return None
        i, j = coords[int(rng.integers(0, len(coords)))]
        return self.best_run[i][j]

    def save(self, out_root: str, fname: str = "qd_elites.json") -> str:
        """Persist current MAP-Elites archive to JSON under out_root.

        The archive is small (bins_x * bins_y). Each filled cell stores:
          - ix, iy: integer cell indices
          - score: best score seen for that cell
          - run_dir: path to the run directory for that elite
        """
        Path(out_root).mkdir(parents=True, exist_ok=True)

        elites = []
        for ix in range(self.bx):
            for iy in range(self.by):
                rd = self.best_run[ix][iy]
                if rd is None:
                    continue
                elites.append({
                    "ix": int(ix),
                    "iy": int(iy),
                    "score": float(self.best_score[ix, iy]),
                    "run_dir": str(rd),
                })

        out_path = os.path.join(out_root, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "bins_x": int(self.bx),
                "bins_y": int(self.by),
                "elites": elites,
            }, f, indent=2, default=_json_sanitize)
        return out_path





@dataclass
class Job:
    cfg: Dict[str, Any]
    outdir: str
    method: str
    xvec: Optional[List[float]] = None

def run_and_eval(job: Job) -> Dict[str, Any]:
    """
    Worker-safe function (top-level) so multiprocessing can pickle it.
    """
    t0 = time.time()
    outdir = job.outdir
    try:
        model.run_simulation(job.cfg, outdir)
        metrics_csv = os.path.join(outdir, "metrics.csv")

        mid = load_npz_state(outdir, "state_mid.npz")
        fin = load_npz_state(outdir, "state_final.npz")
        if mid is None or fin is None:
            raise RuntimeError("Missing state_mid.npz/state_final.npz; set save_states=True in cfg.")

        B_mid = mid["B"]
        B_fin = fin["B"]
        tau_fin = fin["tau"]

        m = summarize_timeseries(metrics_csv)
        m["maintenance_iou"] = maintenance_iou(B_mid, B_fin)
        m["internal_reorg_index"] = internal_reorg_index(B_mid, B_fin)
        m["coherence_osc_index"] = coherence_osc_index(metrics_csv)

        tau_var, tau_g2 = tau_structure(tau_fin)
        m["tau_var_final"] = tau_var
        m["tau_grad2_final"] = tau_g2

        score = compute_score(m)
        d1, d2 = descriptor_pair(m)

        elapsed = time.time() - t0

        row = {
            "run_dir": outdir,
            "method": job.method,
            "elapsed_s": elapsed,
            "score": score,
            "desc_1": d1,
            "desc_2": d2,
            **{k: job.cfg.get(k) for k,_,_,_ in PARAM_SPACE},
            **{k: job.cfg.get(k) for k in FLAG_SPACE.keys()},
            "seed": job.cfg.get("seed", None),
            **m,
        }
        return row

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "run_dir": outdir,
            "method": job.method,
            "elapsed_s": elapsed,
            "error": str(e),
        }


# -----------------------------
# Proposal logic
# -----------------------------
def sample_random_params(rng: np.random.Generator) -> Dict[str, Any]:
    p = {}
    for (name, typ, lo, hi) in PARAM_SPACE:
        p[name] = float(lo + (hi - lo) * rng.random())
    # randomize flags occasionally
    for k, choices in FLAG_SPACE.items():
        p[k] = choices[int(rng.integers(0, len(choices)))]
    return p

def mutate_params(rng: np.random.Generator, x: np.ndarray, sigma: float = 0.08) -> np.ndarray:
    y = x + rng.normal(0.0, sigma, size=x.shape)
    return np.clip(y, 0.0, 1.0)

def propose_bo_batch(rng: np.random.Generator, df: pd.DataFrame, k: int, xi: float = 0.01) -> List[np.ndarray]:
    # Need enough data
    if df is None or len(df) < 15 or "score" not in df.columns:
        return []

    df2 = df[np.isfinite(df["score"].to_numpy(dtype=float))].copy()
    if len(df2) < 15:
        return []

    # Build X,y
    X = []
    y = []
    for _, r in df2.iterrows():
        p = {name: float(r[name]) for (name,_,_,_) in PARAM_SPACE}
        X.append(encode_params(p))
        y.append(float(r["score"]))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Standardize y for GP stability
    y_mean = float(np.mean(y))
    y_std = float(np.std(y) + 1e-12)
    ys = (y - y_mean) / y_std

    gp = SimpleGP(lengthscale=0.22, noise=1e-6)
    gp.fit(X, ys)

    best = float(np.max(ys))

    # random candidate pool
    Ncand = 6000
    cand = rng.random((Ncand, X.shape[1]))  # uniform in [0,1]^d
    mu, var = gp.predict(cand)
    ei = expected_improvement(mu, var, best, xi=xi)

    # pick top-k unique
    idx = np.argsort(-ei)
    out = []
    for j in idx:
        out.append(cand[j])
        if len(out) >= k:
            break
    return out

def make_cfg_from_params(params: Dict[str, Any], nx: int, ny: int, steps: int, seed: int) -> Dict[str, Any]:
    cfg = base_sim_cfg(nx, ny, steps)

    # apply continuous params
    for (name, _, _, _) in PARAM_SPACE:
        cfg[name] = float(params[name])

    # apply flags
    for k in FLAG_SPACE.keys():
        if k in params:
            cfg[k] = params[k]

    # If num_tau_species > 1, create deterministic per-species params based on base alpha/beta/gamma
    ns = int(cfg.get("num_tau_species", 1))
    if ns > 1:
        # deterministic pseudo-random offsets based on hash of params (not seed)
        h = int(hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)
        rr = np.random.default_rng(h)
        sp = []
        for i in range(ns):
            sp.append({
                "alpha": cfg["alpha"] * (0.85 + 0.30 * rr.random()),
                "beta":  cfg["beta"]  * (0.85 + 0.30 * rr.random()),
                "gamma": cfg["gamma"] * (0.85 + 0.30 * rr.random()),
            })
        cfg["tau_species_params"] = sp

    cfg["seed"] = int(seed)
    return cfg


# -----------------------------
# Main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, default="outputs/dynamic_tau_v6_search")
    p.add_argument("--budget", type=int, default=200, help="total evaluations")
    p.add_argument("--init_random", type=int, default=40, help="initial random evals")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--mode", type=str, default="hybrid", choices=["bo", "qd", "hybrid"])
    p.add_argument("--p_qd", type=float, default=0.5, help="hybrid: probability of QD-proposed candidate")
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--save_snapshots_elites", type=int, default=0, help="if >0, save snapshots for elites every N updates")
    p.add_argument("--resume", action="store_true")

    # QD / MAP-Elites configuration
    p.add_argument("--qd_bins", type=int, default=20, help="bins per descriptor axis")
    p.add_argument("--qd_maint_max", type=float, default=0.35, help="maintenance_iou normalization max for descriptor d1")
    p.add_argument("--qd_reorg_scale", type=float, default=0.05, help="exp squash scale for internal_reorg_index")
    p.add_argument("--qd_osc_scale", type=float, default=1e-5, help="exp squash scale for coherence_osc_index")
    p.add_argument("--qd_mix_osc", type=float, default=0.30, help="mix weight for osc in descriptor d2")
    p.add_argument("--qd_sigma", type=float, default=0.14, help="mutation sigma used for QD proposals")
    p.add_argument("--qd_flip_prob", type=float, default=0.35, help="probability to mutate a random elite (vs best elite)")

    # Score configuration
    p.add_argument("--score_version", type=str, default="v1", choices=["v1", "v2"])
    p.add_argument("--score_iou_gate", type=float, default=0.08)
    p.add_argument("--score_iou_width", type=float, default=0.02)
    p.add_argument("--score_iou_sat_scale", type=float, default=0.10)

    return p.parse_args()

def main():
    args = parse_args()

    # Apply CLI knobs into global configs used by descriptor_pair() and compute_score()
    QD_CFG.update({
        "bins": int(args.qd_bins),
        "maint_max": float(args.qd_maint_max),
        "reorg_scale": float(args.qd_reorg_scale),
        "osc_scale": float(args.qd_osc_scale),
        "mix_osc": float(args.qd_mix_osc),
        "sigma": float(args.qd_sigma),
        "flip_prob": float(args.qd_flip_prob),
    })
    SCORE_CFG.update({
        "version": str(args.score_version),
        "iou_gate": float(args.score_iou_gate),
        "iou_width": float(args.score_iou_width),
        "iou_sat_scale": float(args.score_iou_sat_scale),
    })

    out_root = args.out_root
    Path(out_root).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(out_root, "boqd_log.csv")

    rng = np.random.default_rng()

    # Load existing log if resuming
    if args.resume and os.path.exists(log_path):
        df = pd.read_csv(log_path)
        print(f"[V6] Resuming from {log_path} with {len(df)} rows")
    else:
        df = pd.DataFrame([])
        if os.path.exists(log_path):
            # don't overwrite silently
            ts = int(time.time())
            os.rename(log_path, os.path.join(out_root, f"boqd_log_backup_{ts}.csv"))
        print("[V6] Starting fresh search")

    qd = QDMap(bins_x=int(args.qd_bins), bins_y=int(args.qd_bins))

    # Replay existing rows into QD map
    if len(df) > 0 and "desc_1" in df.columns and "desc_2" in df.columns and "score" in df.columns:
        for _, r in df.iterrows():
            if np.isfinite(r.get("score", np.nan)):
                qd.consider(float(r["desc_1"]), float(r["desc_2"]), float(r["score"]), str(r["run_dir"]))

    # Helper: record one row
    def append_row(row: Dict[str, Any]):
        nonlocal df
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(log_path, index=False)

    # Search loop
    done = 0 if len(df) == 0 else len(df)
    print(f"[V6] Target budget={args.budget}, already_have={done}")

    while done < args.budget:
        batch = min(args.workers, args.budget - done)

        # decide how many BO vs QD
        if args.mode == "bo":
            n_bo, n_qd = batch, 0
        elif args.mode == "qd":
            n_bo, n_qd = 0, batch
        else:
            n_qd = int(round(batch * args.p_qd))
            n_bo = batch - n_qd

        proposals: List[Tuple[str, Dict[str, Any]]] = []

        # initial random phase
        if len(df) < args.init_random:
            for _ in range(batch):
                proposals.append(("random", sample_random_params(rng)))
        else:
            # BO proposals
            bo_x = propose_bo_batch(rng, df, k=n_bo, xi=0.01) if n_bo > 0 else []
            for x in bo_x:
                proposals.append(("bo", {**decode_params(x)}))

            # QD proposals: mutate elites or random
            for _ in range(n_qd):
                # choose elite parent: mostly best elite, sometimes random elite (flip_prob)
                elite_dir = None
                if rng.random() < float(args.qd_flip_prob):
                    elite_dir = qd.random_elite(rng)
                else:
                    elite_dir = qd.best_elite() or qd.random_elite(rng)
                if elite_dir is None:
                    proposals.append(("qd", sample_random_params(rng)))
                else:
                    # find elite params in df
                    sub = df[df["run_dir"] == elite_dir]
                    if sub.empty:
                        proposals.append(("qd", sample_random_params(rng)))
                    else:
                        r = sub.iloc[0]
                        p0 = {name: float(r[name]) for (name,_,_,_) in PARAM_SPACE}
                        x0 = encode_params(p0)
                        x1 = mutate_params(rng, x0, sigma=float(args.qd_sigma))
                        p1 = decode_params(x1)
                        # randomly flip some flags
                        for fk, choices in FLAG_SPACE.items():
                            if rng.random() < 0.25:
                                p1[fk] = choices[int(rng.integers(0, len(choices)))]
                            else:
                                # inherit if present in row
                                if fk in r and str(r[fk]) != "nan":
                                    p1[fk] = r[fk]
                        proposals.append(("qd", p1))

            # if BO returned fewer than expected, fill
            while len(proposals) < batch:
                proposals.append(("fill", sample_random_params(rng)))

        # build jobs
        jobs: List[Job] = []
        for (method, p) in proposals[:batch]:
            seed = int(rng.integers(0, 2**31 - 1))
            cfg = make_cfg_from_params(p, args.nx, args.ny, args.steps, seed=seed)

            # output folder
            # hash includes params + flags + seed to avoid collisions
            h = stable_hash({
                "method": method,
                "seed": seed,
                **{k: cfg.get(k) for k,_,_,_ in PARAM_SPACE},
                **{k: cfg.get(k) for k in FLAG_SPACE.keys()},
            })
            outdir = os.path.join(out_root, method, h)
            cfg["outdir"] = outdir

            print(f"→ {method:6s} | {h} | alpha={cfg['alpha']:.4f} beta={cfg['beta']:.4f} f={cfg['feed']:.4f} k={cfg['kill']:.4f} "
                  f"kappa={cfg['kappa_tau']:.4f} noise={cfg['tau_noise']:.4f} "
                  f"flags(ms={cfg['use_multiscale_memory']}, diffN={cfg['use_diffusive_nutrient']}, tauN={cfg.get('num_tau_species',1)})")

            jobs.append(Job(cfg=cfg, outdir=outdir, method=method))

        # run batch
        if args.workers > 1:
            import multiprocessing as mp
            with mp.Pool(processes=args.workers) as pool:
                results = pool.map(run_and_eval, jobs)
        else:
            results = [run_and_eval(j) for j in jobs]

        # ingest results
        for row in results:
            if "error" in row:
                print("✗ ERROR:", row["run_dir"], row["error"])
                append_row(row)
                done += 1
                continue

            score = float(row.get("score", np.nan))
            d1 = float(row.get("desc_1", 0.0))
            d2 = float(row.get("desc_2", 0.0))
            run_dir = str(row["run_dir"])

            print(f"✓ {row['method']:6s} score={score:.4f} desc=({d1:.2f},{d2:.2f}) "
                  f"iou={row.get('maintenance_iou', np.nan):.3f} reorg={row.get('internal_reorg_index', np.nan):.3f} osc={row.get('coherence_osc_index', np.nan):.4g}")

            qd.consider(d1, d2, score, run_dir)
            append_row(row)
            done += 1

        # Save QD artifacts periodically
        qd.save(out_root)

        # Track best so far
        if len(df) > 0 and "score" in df.columns:
            dff = df[np.isfinite(df["score"].to_numpy(dtype=float))]
            if len(dff) > 0:
                best = dff.sort_values("score", ascending=False).iloc[0].to_dict()
                with open(os.path.join(out_root, "best_so_far.json"), "w") as fh:
                    json.dump(best, fh, indent=2, default=_json_sanitize)

        print(f"[V6] Progress: {done}/{args.budget}\n")

    print("[V6] Search complete.")
    print("Log:", log_path)
    print("QD map:", os.path.join(out_root, "qd_map.png"))


if __name__ == "__main__":
    main()
