#!/usr/bin/env python3
"""
dynamic_tau_v7.py

Dynamic time-density (tau) reaction–diffusion substrate with optional
"organism indicator" feedback channel w(x,t) that can modulate tau.

This version logs applied feedback scalars per step into metrics.csv:
  - w_mean
  - w_enabled_applied
  - w_tau_gain_applied
  - w_tau_bias_applied
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# -----------------------------
# Helpers / numerics
# -----------------------------

def laplacian(X: np.ndarray) -> np.ndarray:
    return (
        -4.0 * X
        + np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0)
        + np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)
    )


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def sigmoid(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    # stable sigmoid
    z = k * x
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))

# -----------------------------
# Initialization / seeding
# -----------------------------

def seed_initial_condition(
    A: np.ndarray,
    B: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: str = "square",
    pos_mode: str = "center",
    seed_radius: int = 10,
    seed_count: int = 1,
    seed_sigma: float = 4.0,
    ring_width: float = 2.5,
    stripe_period: int = 16,
    margin: int = 12,
    value: float = 1.0,
) -> None:
    """
    Seed B with a configurable pattern to encourage morphology diversity.

    mode:
      - square: one or many filled squares
      - circle: one or many filled disks
      - gaussian: one or many gaussian blobs (additive)
      - ring: one or many rings
      - stripes: vertical stripes (ignores seed_count/pos_mode)

    pos_mode:
      - center: place the (first) seed at the center (additional seeds jittered around)
      - random: place seeds uniformly random (with margin)

    Notes:
      - For multi-seed modes, we *add* contributions for gaussian/ring, and clip to [0,1].
      - For square/circle we set B to `value` on the mask.
    """
    ny, nx = B.shape
    rr = int(max(1, seed_radius))

    def pick_center(k: int):
        # small jitter around center for multi-seed
        cy0, cx0 = ny // 2, nx // 2
        if k == 0:
            return cy0, cx0
        jy = rng.integers(-2 * rr, 2 * rr + 1)
        jx = rng.integers(-2 * rr, 2 * rr + 1)
        cy = int(np.clip(cy0 + jy, margin, ny - margin - 1))
        cx = int(np.clip(cx0 + jx, margin, nx - margin - 1))
        return cy, cx

    def pick_random():
        cy = int(rng.integers(margin, ny - margin))
        cx = int(rng.integers(margin, nx - margin))
        return cy, cx

    if mode == "stripes":
        # deterministic vertical stripes; phase randomized by seed
        phase = int(rng.integers(0, stripe_period))
        x = (np.arange(nx)[None, :] + phase) % stripe_period
        mask = (x < (stripe_period // 2))
        B[mask.repeat(ny, axis=0)] = value
        return

    yy, xx = np.mgrid[0:ny, 0:nx]

    for k in range(int(max(1, seed_count))):
        if pos_mode == "random":
            cy, cx = pick_random()
        else:
            cy, cx = pick_center(k)

        if mode == "square":
            B[max(0, cy - rr):min(ny, cy + rr), max(0, cx - rr):min(nx, cx + rr)] = value

        elif mode == "circle":
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= (rr ** 2)
            B[mask] = value

        elif mode == "gaussian":
            sig = float(max(1e-3, seed_sigma))
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sig * sig))
            B[:] = np.clip(B + value * blob, 0.0, 1.0)

        elif mode == "ring":
            sig = float(max(1e-3, ring_width))
            r0 = float(rr)
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            ring = np.exp(-((r - r0) ** 2) / (2.0 * sig * sig))
            B[:] = np.clip(B + value * ring, 0.0, 1.0)

        else:
            # fallback
            B[max(0, cy - rr):min(ny, cy + rr), max(0, cx - rr):min(nx, cx + rr)] = value


# -----------------------------
# Simple per-step metrics
# -----------------------------

def coherence(A: np.ndarray, B: np.ndarray) -> float:
    # Example coherence proxy (keep consistent with your prior v6/v7)
    # Here: inverse of mean absolute gradient magnitude (higher = smoother/structured)
    gx = np.diff(B, axis=1)
    gy = np.diff(B, axis=0)
    g = np.mean(np.abs(gx)) + np.mean(np.abs(gy))
    return float(1.0 / (1e-9 + g))


def entropy(B: np.ndarray, nbins: int = 64) -> float:
    x = B.ravel()
    x = np.clip(x, 0.0, 1.0)
    hist, _ = np.histogram(x, bins=nbins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    p = p / (np.sum(p) + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def autocatalysis(B: np.ndarray) -> float:
    # Example autocat proxy: mean of B^2 (emphasize hotspots)
    return float(np.mean(B * B))


# -----------------------------
# Organism indicator w(x,t)
# -----------------------------

def local_coherence_proxy(
    A: np.ndarray,
    B: np.ndarray,
    mode: str = "grad",
    radius: int = 3,
    amp_thr: float = 0.05,
    amp_width: float = 0.02,
) -> np.ndarray:
    """
    Produce a soft indicator field w(x,t) in [0,1], intended to mark "organism-like"
    foreground regions. This is a light-weight, fully local proxy.

    mode:
      - "grad": high where |∇B| is structured (low gradient noise suppressed by amp gate)
      - "amp": purely amplitude-gated on B
    """
    Bc = np.clip(B, 0.0, 1.0)

    # amplitude gate (soft)
    w_amp = sigmoid((Bc - amp_thr) / max(amp_width, 1e-9), k=1.0)

    if mode == "amp":
        return w_amp

    # gradient coherence proxy: low gradient => high coherence, but gated by amplitude
    gy, gx = np.gradient(Bc)
    gmag = np.sqrt(gx * gx + gy * gy)
    # squash
    w_grad = 1.0 / (1.0 + 10.0 * gmag)
    w = w_amp * w_grad
    return np.clip(w, 0.0, 1.0)


# -----------------------------
# Main simulation
# -----------------------------

def run_simulation(cfg: Dict[str, Any], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    nx = int(cfg.get("nx", 128))
    ny = int(cfg.get("ny", 128))
    steps = int(cfg.get("steps", 2000))
    dt = float(cfg.get("dt", 0.01))
    log_every = int(cfg.get("log_every", 20))

    save_states = bool(cfg.get("save_states", True))
    mid_state_step = int(cfg.get("mid_state_step", steps // 2))

    save_snapshots = bool(cfg.get("save_snapshots", False))
    snap_every = int(cfg.get("snap_every", 200))

    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    # Gray-Scott diffusion
    Da = float(cfg.get("Da", 0.16))
    Db = float(cfg.get("Db", 0.08))

    # feed/kill
    feed = float(cfg.get("feed", 0.0367))
    kill = float(cfg.get("kill", 0.0649))

    # tau dynamics
    tau0 = float(cfg.get("tau0", 1.0))
    tau_min = float(cfg.get("tau_min", 0.2))
    tau_max = float(cfg.get("tau_max", 5.0))

    alpha = float(cfg.get("alpha", 0.03))
    beta = float(cfg.get("beta", 0.006))
    gamma = float(cfg.get("gamma", 0.3))
    kappa_tau = float(cfg.get("kappa_tau", 0.02))
    tau_noise = float(cfg.get("tau_noise", 0.001))

    # nutrient (optional)
    N0 = float(cfg.get("N0", 1.0))
    nutrient_use = float(cfg.get("nutrient_use", 0.01))
    nutrient_replenish = float(cfg.get("nutrient_replenish", 0.001))
    use_diffusive_nutrient = bool(cfg.get("use_diffusive_nutrient", True))
    D_N = float(cfg.get("D_N", 0.02))
    eta_N = float(cfg.get("eta_N", 0.1))
    rho_N = float(cfg.get("rho_N", 0.0005))

    # memory
    use_multiscale_memory = bool(cfg.get("use_multiscale_memory", True))
    memory_decay = float(cfg.get("memory_decay", 0.01))
    mem_decay_fast = float(cfg.get("mem_decay_fast", 0.02))
    mem_decay_slow = float(cfg.get("mem_decay_slow", 0.002))
    mem_w_fast = float(cfg.get("mem_w_fast", 0.7))
    mem_w_slow = float(cfg.get("mem_w_slow", 0.3))

    # w feedback toggles
    w_enabled = bool(cfg.get("w_enabled", False))
    w_mode = str(cfg.get("w_mode", "grad"))
    w_radius = int(cfg.get("w_radius", 3))
    w_amp_thr = float(cfg.get("w_amp_thr", 0.05))
    w_amp_width = float(cfg.get("w_amp_width", 0.02))
    w_tau_gain = float(cfg.get("w_tau_gain", 0.0))
    w_tau_bias = float(cfg.get("w_tau_bias", 0.0))

    # init
    noise = float(cfg.get("noise", 0.02))
    seed_radius = int(cfg.get("seed_radius", 10))

    # Fields
    A = np.ones((ny, nx), dtype=np.float64)
    B = np.zeros((ny, nx), dtype=np.float64)
    tau = np.full((ny, nx), tau0, dtype=np.float64)
    N = np.full((ny, nx), N0, dtype=np.float64)

    # Seed B using configurable patterns (seed diversity lever)
    init_mode = str(cfg.get("init_mode", "square"))
    seed_pos_mode = str(cfg.get("seed_pos_mode", "center"))
    seed_count = int(cfg.get("seed_count", 1))
    seed_sigma = float(cfg.get("seed_sigma", 4.0))
    seed_ring_width = float(cfg.get("seed_ring_width", 2.5))
    seed_stripe_period = int(cfg.get("seed_stripe_period", 16))
    seed_margin = int(cfg.get("seed_margin", 12))
    seed_initial_condition(
        A, B, rng,
        mode=init_mode,
        pos_mode=seed_pos_mode,
        seed_radius=seed_radius,
        seed_count=seed_count,
        seed_sigma=seed_sigma,
        ring_width=seed_ring_width,
        stripe_period=seed_stripe_period,
        margin=seed_margin,
        value=1.0,
    )
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
            cols += ["w_mean", "w_enabled_applied", "w_tau_gain_applied", "w_tau_bias_applied"]
        f.write(",".join(cols) + "\n")

    # Run
    w_field: Optional[np.ndarray] = None

    for t in range(steps):
        # Effective diffusion scaling by tau
        tau_eff = np.clip(tau, tau_min, tau_max)
        Da_eff = Da / tau_eff
        Db_eff = Db / tau_eff

        # Gray-Scott reaction terms
        AB2 = A * (B * B)
        A += dt * (Da_eff * laplacian(A) - AB2 + feed * (1.0 - A))
        B += dt * (Db_eff * laplacian(B) + AB2 - (kill + feed) * B)
        A = np.clip(A, 0.0, 1.0)
        B = np.clip(B, 0.0, 1.0)

        # Nutrient update
        if use_diffusive_nutrient:
            N += dt * (D_N * laplacian(N) - eta_N * B * N + rho_N * (N0 - N))
        else:
            N += dt * (-nutrient_use * B * N + nutrient_replenish * (N0 - N))
        N = np.clip(N, 0.0, 2.0 * N0)

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
                row.append(int(w_enabled))
                row.append(float(w_tau_gain))
                row.append(float(w_tau_bias))

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(",".join(map(str, row)) + "\n")

        # Snapshots / states
        if save_snapshots and (t % snap_every == 0):
            np.save(os.path.join(outdir, f"A_{t:06d}.npy"), A)
            np.save(os.path.join(outdir, f"B_{t:06d}.npy"), B)
            np.save(os.path.join(outdir, f"tau_{t:06d}.npy"), tau)

        if save_states and (t == mid_state_step):
            np.savez(os.path.join(outdir, "state_mid.npz"), A=A, B=B, tau=tau, N=N)

    if save_states:
        np.savez(os.path.join(outdir, "state_final.npz"), A=A, B=B, tau=tau, N=N)
