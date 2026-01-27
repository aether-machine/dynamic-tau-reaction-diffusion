#!/usr/bin/env python3
"""dynamic_tau_v6.py (v6 + organism-indicator feedback)

This file is a drop-in compatible variant of your original `dynamic_tau_v6.py`.

What it adds
------------
1) A single spatial "organism indicator" field `w(x,t) in [0,1]`, computed from a
   *local coherence proxy* of the chemical fields.

2) Optional feedback: `w` can be fed back into **one place** in the dynamics
   (default: **tau modulation**) to test whether coherent regions can become
   *self-maintaining*.

Why this fits your "universal metrics" direction
------------------------------------------------
- You get one canonical indicator `w(x,t)` that you can:
  - log as a scalar `w_mean(t)` ("organism mass")
  - use for growth metrics (log-slope of `w_mean(t)`)
  - use to condition other metrics (weighted/ROI measures)
  - feed back into dynamics (closing the loop)

New cfg keys (all optional)
---------------------------
- w_enabled: bool (default False)
- w_mode: str in {"varratio", "energy"} (default "varratio")
  - "energy": w is a gated local activity (smoothed A^2+B^2)
  - "varratio": w is gated *and* prefers locally smooth / coherent activity
- w_radius: int neighborhood radius (default 1 -> 3x3 box)
- w_amp_thr: float threshold on local activity mean for gating (default 0.05)
- w_amp_width: float width for gating sigmoid (default 0.02)

Feedback (pick one place; start with tau):
- w_tau_gain: float (default 0.0)  # add + w_tau_gain*(w - w_tau_bias) to tau_dot
- w_tau_bias: float (default 0.0)

I/O:
- save_w: bool (default False)  # include w in state_mid.npz/state_final.npz

Outputs
-------
- metrics.csv gets an extra column `w_mean` when w_enabled.
- state_mid.npz/state_final.npz can optionally include `w`.

Notes
-----
- Defaults preserve original behavior: w_enabled=False and w_tau_gain=0.0.
"""

import os
import json
from typing import Dict, Any, Optional, List

import numpy as np


# ----------
# Utilities
# ----------

def laplacian(X: np.ndarray) -> np.ndarray:
    """2D periodic Laplacian."""
    return (
        np.roll(X, 1, axis=0)
        + np.roll(X, -1, axis=0)
        + np.roll(X, 1, axis=1)
        + np.roll(X, -1, axis=1)
        - 4.0 * X
    )


def montage(field: np.ndarray) -> np.ndarray:
    """Return an RGB montage for quick visualization."""
    f = field
    f = (f - f.min()) / (f.max() - f.min() + 1e-12)
    rgb = np.stack([f, f, f], axis=-1)
    return (255.0 * rgb).astype(np.uint8)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid for arrays
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def box_filter(X: np.ndarray, radius: int) -> np.ndarray:
    """Periodic (2r+1)x(2r+1) box filter using rolls."""
    r = int(radius)
    if r <= 0:
        return X
    acc = np.zeros_like(X, dtype=np.float64)
    n = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            acc += np.roll(np.roll(X, dy, axis=0), dx, axis=1)
            n += 1
    return (acc / float(n)).astype(np.float64)


# -------------------------------------
# Local coherence proxy -> organism w(x,t)
# -------------------------------------

def local_coherence_proxy(
    A: np.ndarray,
    B: np.ndarray,
    *,
    mode: str = "varratio",
    radius: int = 1,
    amp_thr: float = 0.05,
    amp_width: float = 0.02,
) -> np.ndarray:
    """Compute organism indicator w(x,t) in [0,1].

    We build w from local activity in the *B field*:

      S(x,t) := B(x,t)

    Rationale: in Gray-Scott style systems, A is often near-uniform while B carries
    the patterned "organism" structure. Using B avoids the failure mode where A
    makes S large everywhere, causing w to be ~1 everywhere.

    mode="energy":
      w = sigmoid((mean(S)-thr)/width)

    mode="varratio" (recommended starting point):
      mean = blur(S)
      var  = blur(S^2) - mean^2
      coh  = mean/(mean+var+eps)           # high when activity is locally smooth
      gate = sigmoid((mean-thr)/width)     # suppress background
      w    = gate * coh

    This is intentionally "pure math": local averages + ratios + a smooth gate.
    """
    # IMPORTANT: use B (not A^2+B^2) so the indicator can localize to patterned
    # regions instead of turning on everywhere due to A being ~1.
    S = B.astype(np.float64)
    m = box_filter(S, radius)

    gate = _sigmoid((m - float(amp_thr)) / max(float(amp_width), 1e-12))

    mode = str(mode).lower()
    if mode == "energy":
        w = gate
        return np.clip(w, 0.0, 1.0)

    if mode != "varratio":
        raise ValueError(f"Unknown w_mode={mode!r} (expected 'varratio' or 'energy')")

    m2 = box_filter(S * S, radius)
    var = np.maximum(m2 - m * m, 0.0)
    coh = m / (m + var + 1e-12)

    w = gate * coh
    return np.clip(w, 0.0, 1.0)


# --------------------------
# Metrics (global, legacy)
# --------------------------

def entropy(B: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(B.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    return float(-np.sum(hist * np.log(hist)))


def coherence(A: np.ndarray, B: np.ndarray) -> float:
    """Legacy global "coherence" proxy: mean(|A+iB|^2)."""
    M = A + 1j * B
    return float(np.mean(np.abs(M) ** 2))


def autocatalysis(B: np.ndarray, alpha: float = 2.0) -> float:
    """Simple autocatalysis proxy."""
    return float(np.mean(B ** alpha))


# --------------------------
# I/O
# --------------------------

def save_state_npz(
    outdir: str,
    name: str,
    A: np.ndarray,
    B: np.ndarray,
    tau: np.ndarray,
    N: np.ndarray,
    w: Optional[np.ndarray] = None,
):
    path = os.path.join(outdir, name)
    if w is None:
        np.savez_compressed(path, A=A, B=B, tau=tau, N=N)
    else:
        np.savez_compressed(path, A=A, B=B, tau=tau, N=N, w=w)


# --------------------------
# Main simulation
# --------------------------

def run_simulation(cfg: Dict[str, Any], outdir: str):
    os.makedirs(outdir, exist_ok=True)

    nx = int(cfg.get("nx", 128))
    ny = int(cfg.get("ny", 128))
    dt = float(cfg.get("dt", 0.01))
    steps = int(cfg.get("steps", 2000))
    log_every = int(cfg.get("log_every", 20))

    # Gray-Scott
    Da = float(cfg.get("Da", 0.16))
    Db = float(cfg.get("Db", 0.08))
    feed = float(cfg.get("feed", 0.035))
    kill = float(cfg.get("kill", 0.060))

    # tau field
    tau0 = float(cfg.get("tau0", 1.0))
    tau_min = float(cfg.get("tau_min", 0.2))
    tau_max = float(cfg.get("tau_max", 5.0))
    alpha = float(cfg.get("alpha", 0.03))
    beta = float(cfg.get("beta", 0.006))
    gamma = float(cfg.get("gamma", 0.3))
    kappa_tau = float(cfg.get("kappa_tau", 0.02))
    tau_noise = float(cfg.get("tau_noise", 0.001))

    # nutrient
    N0 = float(cfg.get("N0", 1.0))
    nutrient_use = float(cfg.get("nutrient_use", 0.01))
    nutrient_replenish = float(cfg.get("nutrient_replenish", 0.001))
    use_diffusive_nutrient = bool(cfg.get("use_diffusive_nutrient", True))
    D_N = float(cfg.get("D_N", 0.02))

    # memory
    use_multiscale_memory = bool(cfg.get("use_multiscale_memory", True))
    memory_decay = float(cfg.get("memory_decay", 0.01))
    mem_decay_fast = float(cfg.get("mem_decay_fast", 0.02))
    mem_decay_slow = float(cfg.get("mem_decay_slow", 0.002))
    mem_w_fast = float(cfg.get("mem_w_fast", 0.7))
    mem_w_slow = float(cfg.get("mem_w_slow", 0.3))

    # init
    seed_radius = int(cfg.get("seed_radius", 10))
    noise = float(cfg.get("noise", 0.02))
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    # organism indicator w
    w_enabled = bool(cfg.get("w_enabled", False))
    w_mode = str(cfg.get("w_mode", "varratio"))
    w_radius = int(cfg.get("w_radius", 1))
    w_amp_thr = float(cfg.get("w_amp_thr", 0.05))
    w_amp_width = float(cfg.get("w_amp_width", 0.02))
    save_w = bool(cfg.get("save_w", False))

    # feedback: tau modulation
    w_tau_gain = float(cfg.get("w_tau_gain", 0.0))
    w_tau_bias = float(cfg.get("w_tau_bias", 0.0))

    # Save snapshots / states
    save_snapshots = bool(cfg.get("save_snapshots", False))
    save_montage = bool(cfg.get("save_montage", False))
    snap_every = int(cfg.get("snap_every", 200))
    save_states = bool(cfg.get("save_states", True))
    mid_state_step = int(cfg.get("mid_state_step", steps // 2))

    # Initialize fields (all arrays are shaped (ny, nx))
    A = np.ones((ny, nx), dtype=np.float64)
    B = np.zeros((ny, nx), dtype=np.float64)
    tau = tau0 * np.ones((ny, nx), dtype=np.float64)
    N = N0 * np.ones((ny, nx), dtype=np.float64)

    # Seed patch in center
    cx, cy = nx // 2, ny // 2
    rr = seed_radius
    A[cy - rr : cy + rr, cx - rr : cx + rr] = 0.50
    B[cy - rr : cy + rr, cx - rr : cx + rr] = 0.25

    # Add noise
    A += noise * rng.standard_normal(size=A.shape)
    B += noise * rng.standard_normal(size=B.shape)
    A = np.clip(A, 0.0, 1.0)
    B = np.clip(B, 0.0, 1.0)

    # Memory fields
    mem = np.zeros_like(B)
    mem_fast = np.zeros_like(B)
    mem_slow = np.zeros_like(B)

    metrics_path = os.path.join(outdir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        cols = ["time", "coherence", "entropy", "autocat"]
        if w_enabled:
            cols += ["w_mean"]
        f.write(",".join(cols) + "\n")

    # Run
    w_field: Optional[np.ndarray] = None

    for t in range(steps):
        # Effective diffusion scaling by tau
        tau_eff = np.clip(tau, tau_min, tau_max)
        Da_eff = Da / tau_eff
        Db_eff = Db / tau_eff

        # Reaction
        reaction = A * B * B

        # Update A, B
        A += dt * (Da_eff * laplacian(A) - reaction + feed * (1.0 - A))
        B += dt * (Db_eff * laplacian(B) + reaction - (kill + feed) * B)
        A = np.clip(A, 0.0, 1.0)
        B = np.clip(B, 0.0, 1.0)

        # Nutrient update
        if use_diffusive_nutrient:
            N += dt * (D_N * laplacian(N) - nutrient_use * B + nutrient_replenish * (N0 - N))
        else:
            N += dt * (-nutrient_use * B + nutrient_replenish * (N0 - N))
        N = np.clip(N, 0.0, N0)

        # Memory update
        if use_multiscale_memory:
            mem_fast = (1.0 - dt * mem_decay_fast) * mem_fast + dt * mem_decay_fast * B
            mem_slow = (1.0 - dt * mem_decay_slow) * mem_slow + dt * mem_decay_slow * B
            mem_eff = mem_w_fast * mem_fast + mem_w_slow * mem_slow
        else:
            mem = (1.0 - dt * memory_decay) * mem + dt * memory_decay * B
            mem_eff = mem

        # Organism indicator (computed on current A,B)
        if w_enabled:
            w_field = local_coherence_proxy(
                A,
                B,
                mode=w_mode,
                radius=w_radius,
                amp_thr=w_amp_thr,
                amp_width=w_amp_width,
            )
        else:
            w_field = None

        # Tau update (optionally includes w feedback)
        tau_dot = alpha * mem_eff - beta * (tau - tau0) + gamma * N + kappa_tau * laplacian(tau)
        if w_enabled and abs(w_tau_gain) > 0.0:
            # feedback term: increase tau where w is high (or relative to bias)
            tau_dot = tau_dot + w_tau_gain * (w_field - w_tau_bias)  # type: ignore

        tau += dt * tau_dot
        tau += tau_noise * rng.standard_normal(size=tau.shape)
        tau = np.clip(tau, tau_min, tau_max)

        # Logging
        if t % log_every == 0:
            coh = coherence(A, B)
            ent = entropy(B)
            aut = autocatalysis(B)
            row = [t, coh, ent, aut]
            if w_enabled:
                row.append(float(np.mean(w_field)))  # type: ignore
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(",".join(map(str, row)) + "\n")

        # Snapshots / states
        if save_snapshots and (t % snap_every == 0):
            np.save(os.path.join(outdir, f"A_{t:06d}.npy"), A)
            np.save(os.path.join(outdir, f"B_{t:06d}.npy"), B)
            np.save(os.path.join(outdir, f"tau_{t:06d}.npy"), tau)
            if w_enabled and save_w:
                np.save(os.path.join(outdir, f"w_{t:06d}.npy"), w_field)

        if save_montage and (t % snap_every == 0):
            from imageio import imwrite

            imwrite(os.path.join(outdir, f"B_{t:06d}.png"), montage(B))

        if save_states and (t == mid_state_step):
            save_state_npz(outdir, "state_mid.npz", A, B, tau, N, w_field if (w_enabled and save_w) else None)

    if save_states:
        save_state_npz(outdir, "state_final.npz", A, B, tau, N, w_field if (w_enabled and save_w) else None)

    # Save meta
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)

    return outdir


if __name__ == "__main__":
    # Minimal smoke test
    cfg = {
        "nx": 128,
        "ny": 128,
        "steps": 2000,
        "seed": 0,
        "w_enabled": True,
        "w_mode": "varratio",
        "w_tau_gain": 0.05,
        "save_states": True,
        "save_w": True,
    }
    run_simulation(cfg, outdir="outputs/_smoke_dynamic_tau_v6_w")
    print("done")
