#!/usr/bin/env python3
"""
dynamic_tau_v4.py

Modular dynamic-τ reaction–diffusion simulation.

This is a refactor/extension of dynamic_tau_v3.py with:
  - same base dynamics as v3
  - modular structure for future extensions
  - OPTIONAL v4 terms:
      * kappa_tau: curvature-coupled τ (via ∇²τ)
      * tau_noise: environmental τ-noise

External API:
    run_simulation(cfg: dict, outdir: str) -> str

This is intentionally compatible with the v3 sweep configs:
extra keys like kappa_tau and tau_noise are optional and
default to 0.0 if not provided.
"""

import os
import json
import csv
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Utility functions
# ============================================================

def laplacian(Z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    5-point discrete Laplacian with periodic boundary conditions.
    """
    return (
        -4 * Z
        + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
    ) / (dx * dy)


def montage(images: List[str], outpath: str, cols: int = 3) -> None:
    """
    Simple montage: up to 9 images in a grid.
    images: list of image paths.
    """
    if not images:
        return

    imgs = [Image.open(im) for im in images[:9]]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h))

    for idx, im in enumerate(imgs):
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(im, (x, y))

    canvas.save(outpath)


def entropy(field: np.ndarray) -> float:
    """
    Global Shannon-like entropy of a scalar field.

    We shift to non-negative values, normalize to get a probability
    distribution, then compute -sum p log p.
    """
    x = field.flatten()
    x = x - x.min()
    s = x.sum()
    if s <= 0:
        return 0.0
    p = x / (s + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


# ============================================================
# Simulation state container
# ============================================================

class SimState:
    """
    Container for all dynamic fields.

    Currently mirrors v3:
    - A, B: Gray–Scott concentrations
    - tau: time-density field
    - mem: exponential memory of |reaction|
    - N: nutrient field

    v4 extensions (kappa_tau, tau_noise) act on tau but do not
    add new fields yet.
    """

    def __init__(self, A, B, tau, mem, N):
        self.A = A
        self.B = B
        self.tau = tau
        self.mem = mem
        self.N = N


# ============================================================
# Initialization
# ============================================================

def init_fields(cfg: Dict[str, Any]) -> SimState:
    """
    Initialize A, B, tau, mem, N using the same scheme as v3.
    """
    nx, ny = cfg["nx"], cfg["ny"]

    # Base fields
    A = np.ones((ny, nx))
    B = np.zeros((ny, nx))

    tau0 = cfg["tau0"]
    tau = tau0 * np.ones((ny, nx))

    mem = np.zeros((ny, nx))

    # Nutrient
    N0 = cfg["N0"]
    N = N0 * np.ones((ny, nx))

    # Initial perturbation in centre
    r = cfg["seed_radius"]
    cx, cy = nx // 2, ny // 2
    A[cy - r:cy + r, cx - r:cx + r] = 0.5
    B[cy - r:cy + r, cx - r:cx + r] = 0.25

    # Add noise
    noise = cfg["noise"]
    if noise > 0:
        A += noise * np.random.rand(ny, nx)
        B += noise * np.random.rand(ny, nx)

    return SimState(A=A, B=B, tau=tau, mem=mem, N=N)


# ============================================================
# Core update steps
# ============================================================

def compute_reaction(state: SimState, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Gray–Scott reaction term R = A * B^2

    Future extension point:
      - nutrient-weighted reactions
      - alternative reaction kinetics
    """
    return state.A * (state.B ** 2)


def update_memory(state: SimState, R: np.ndarray, cfg: Dict[str, Any]) -> None:
    """
    Exponential memory kernel of |R| (v3 behaviour).

    Future extension point:
      - multiple timescales (mem_fast, mem_slow)
      - Hebbian-like correlations
    """
    decay = cfg["memory_decay"]
    state.mem = state.mem * (1.0 - decay) + np.abs(R)


def update_tau(state: SimState, cfg: Dict[str, Any], dt: float,
               R: np.ndarray, dx: float, dy: float) -> None:
    """
    Update τ according to v3 + optional v4 terms.

    Base v3 term:
        tau += dt * (alpha * mem - beta * (tau - tau0) + gamma * N)

    Optional v4 curvature term:
        + dt * kappa_tau * laplacian(tau)

    Optional v4 noise term:
        + sqrt(dt) * tau_noise * ξ(x,t)

    kappa_tau and tau_noise default to 0.0 if not provided.
    """
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    gamma = cfg["gamma"]
    tau0 = cfg["tau0"]
    tau_min, tau_max = cfg["tau_min"], cfg["tau_max"]

    kappa_tau = cfg.get("kappa_tau", 0.0)
    tau_noise = cfg.get("tau_noise", 0.0)

    # Base deterministic term (v3)
    base = alpha * state.mem - beta * (state.tau - tau0) + gamma * state.N

    # Curvature / geometry term (optional)
    curv = 0.0
    if kappa_tau != 0.0:
        curv = kappa_tau * laplacian(state.tau, dx, dy)

    # Stochastic τ-noise (optional)
    noise = 0.0
    if tau_noise != 0.0:
        noise = tau_noise * np.sqrt(dt) * np.random.randn(*state.tau.shape)

    state.tau += dt * (base + curv) + noise

    # Clip
    np.clip(state.tau, tau_min, tau_max, out=state.tau)


def update_nutrient(state: SimState, R: np.ndarray, cfg: Dict[str, Any],
                    dt: float, dx: float, dy: float) -> None:
    """
    Nutrient update.

    v3 behaviour is a simple depletion:

        N -= dt * nutrient_use * |R|

    Future extension point:
      - add diffusion
      - add spatially varying supply
      - feedback from tau or geometry
    """
    use_rate = cfg["nutrient_use"]
    state.N -= dt * use_rate * np.abs(R)
    np.clip(state.N, 0.0, cfg["N0"], out=state.N)


def update_fields(state: SimState, cfg: Dict[str, Any],
                  dt: float, dx: float, dy: float) -> None:
    """
    Update A and B using τ-modified diffusion and Gray–Scott kinetics.

    This reproduces v3 behaviour when kappa_tau = tau_noise = 0,
    since the τ-field then follows the original dynamics.
    """
    Da = cfg["Da"]
    Db = cfg["Db"]
    feed = cfg["feed"]
    kill = cfg["kill"]

    A, B, tau = state.A, state.B, state.tau

    # Laplacians
    lapA = laplacian(A, dx, dy)
    lapB = laplacian(B, dx, dy)

    # Reaction
    R = A * (B ** 2)

    # τ-modified effective diffusion (geometric focusing)
    Da_eff = Da / tau
    Db_eff = Db / tau

    dA = Da_eff * lapA - R + feed * (1 - A)
    dB = Db_eff * lapB + R - (kill + feed) * B

    # Euler step
    A += dt * dA
    B += dt * dB

    np.clip(A, 0.0, 3.0, out=A)
    np.clip(B, 0.0, 3.0, out=B)

    state.A, state.B = A, B


# ============================================================
# Logging and snapshots
# ============================================================

def log_metrics(state: SimState, R: np.ndarray, t: float,
                logs: Dict[str, list]) -> None:
    """
    Compute global metrics and append to logs dict.

    Metrics:
      - coherence = mean(|A + iB|^2)
      - entropy   = global entropy of B
      - autocat   = mean(R)
    """
    A, B = state.A, state.B

    M = A + 1j * B
    coherence = np.mean(np.abs(M) ** 2)
    ent = entropy(B)
    autocat = np.mean(R)

    logs["time"].append(t)
    logs["coherence"].append(coherence)
    logs["entropy"].append(ent)
    logs["autocat"].append(autocat)


def take_snapshots(state: SimState, step: int, cfg: Dict[str, Any],
                   outdir: str) -> None:
    """
    Save B, tau, N snapshots as PNG images every snap_every steps.
    """
    snap_every = cfg["snap_every"]
    if step % snap_every != 0:
        return

    idx = step // snap_every

    B = state.B
    tau = state.tau
    N = state.N

    plt.imsave(os.path.join(outdir, f"B_snapshot_{idx:04d}.png"),
               B, cmap="magma")
    plt.imsave(os.path.join(outdir, f"tau_snapshot_{idx:04d}.png"),
               tau, cmap="viridis")
    plt.imsave(os.path.join(outdir, f"N_snapshot_{idx:04d}.png"),
               N, cmap="plasma")


def save_metrics(logs: Dict[str, list], outdir: str) -> None:
    """
    Write metrics.csv in the same format as v3.
    """
    path = os.path.join(outdir, "metrics.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "coherence", "entropy", "autocat"])
        for t, c, e, a in zip(
            logs["time"],
            logs["coherence"],
            logs["entropy"],
            logs["autocat"],
        ):
            w.writerow([t, c, e, a])


def save_run_summaries(cfg: Dict[str, Any], outdir: str) -> None:
    """
    Save meta.json and summary.json with a simple pointer to metrics
    and list of B snapshots (same pattern as v3).
    """
    # meta.json: just the cfg
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # summary.json: cfg + metrics filename + B snapshots
    snaps = sorted(
        [f for f in os.listdir(outdir) if f.startswith("B_snapshot_")]
    )
    summary = {
        "cfg": cfg,
        "metrics": "metrics.csv",
        "snapshots": snaps,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


def save_montages(outdir: str) -> None:
    """
    Build simple 3x3 montages for B, tau, N snapshots.
    """
    snapsB = sorted(
        os.path.join(outdir, s)
        for s in os.listdir(outdir)
        if s.startswith("B_snapshot_")
    )
    snapsT = sorted(
        os.path.join(outdir, s)
        for s in os.listdir(outdir)
        if s.startswith("tau_snapshot_")
    )
    snapsN = sorted(
        os.path.join(outdir, s)
        for s in os.listdir(outdir)
        if s.startswith("N_snapshot_")
    )

    montage(snapsB, os.path.join(outdir, "montage_B.png"))
    montage(snapsT, os.path.join(outdir, "montage_tau.png"))
    montage(snapsN, os.path.join(outdir, "montage_N.png"))


# ============================================================
# Main simulation driver
# ============================================================

def run_simulation(cfg: Dict[str, Any], outdir: str) -> str:
    """
    Main entry point. Mirrors dynamic_tau_v3.run_simulation, but
    internally uses the modular update steps above.

    Returns the output directory path.
    """
    os.makedirs(outdir, exist_ok=True)

    # Save cfg immediately for reproducibility
    with open(os.path.join(outdir, "cfg.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # Grid / time params
    nx, ny = cfg["nx"], cfg["ny"]
    dx, dy = cfg["dx"], cfg["dy"]
    dt = cfg["dt"]
    steps = cfg["steps"]
    log_every = cfg["log_every"]

    # Initialize fields
    state = init_fields(cfg)

    # Metric logs
    logs = {
        "time": [],
        "coherence": [],
        "entropy": [],
        "autocat": [],
    }

    # Main time loop
    for step in range(steps):

        # Reaction term
        R = compute_reaction(state, cfg)

        # Memory update
        update_memory(state, R, cfg)

        # τ update (v3 + optional v4 terms)
        update_tau(state, cfg, dt, R, dx, dy)

        # Nutrient update
        update_nutrient(state, R, cfg, dt, dx, dy)

        # A, B update
        update_fields(state, cfg, dt, dx, dy)

        # Logging (coarse)
        if step % log_every == 0:
            t = step * dt
            log_metrics(state, R, t, logs)

        # Snapshots
        take_snapshots(state, step, cfg, outdir)

    # After time loop: persist metrics & summaries
    save_metrics(logs, outdir)
    save_run_summaries(cfg, outdir)
    save_montages(outdir)

    return outdir


if __name__ == "__main__":
    # Example minimal cfg for standalone testing
    cfg_example = {
        "nx": 150, "ny": 150,
        "dx": 1.0, "dy": 1.0,
        "dt": 0.01,
        "steps": 1000,
        "snap_every": 100,
        "log_every": 10,
        "Da": 0.16, "Db": 0.08,
        "feed": 0.035, "kill": 0.065,
        "alpha": 0.02, "beta": 0.005, "gamma": 0.0,
        "tau0": 1.0, "tau_min": 0.2, "tau_max": 3.0,
        "N0": 1.0,
        "nutrient_use": 0.01,
        "memory_decay": 0.01,
        "seed_radius": 10,
        "noise": 0.02,

        # v4 extensions (optional)
        "kappa_tau": 0.0,
        "tau_noise": 0.0,
    }

    out = "outputs/dynamic_tau_v4_example"
    print("Running example simulation to:", out)
    run_simulation(cfg_example, out)
    print("Done.")
