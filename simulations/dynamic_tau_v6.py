#!/usr/bin/env python3
"""
dynamic_tau_v6.py

Dynamic-τ Gray–Scott with:
- nutrient field N (optional diffusion)
- τ feedback + relaxation + nutrient coupling
- τ diffusion (kappa_tau)
- τ noise (tau_noise)
- single or multiscale memory
- optional multi-τ species

V6 additions (optimization-friendly):
- save_states: save mid & final states as NPZ (B,tau,N) for fast metrics
- save_snapshots: allow disabling PNG snapshots for large searches
- mid_state_step: choose which step is "mid" (default steps//2)

Compatible with V5-style configs.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def laplacian(Z, dx, dy):
    # 5-point Laplacian with correct dx/dy handling
    return (
        (np.roll(Z, -1, axis=0) - 2*Z + np.roll(Z, 1, axis=0)) / (dy*dy)
        + (np.roll(Z, -1, axis=1) - 2*Z + np.roll(Z, 1, axis=1)) / (dx*dx)
    )

def montage(images, outpath):
    if len(images) == 0:
        return
    imgs = [Image.open(im) for im in images[:9]]
    w, h = imgs[0].size
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols*w, rows*h))
    for idx, im in enumerate(imgs):
        canvas.paste(im, ((idx % cols)*w, (idx // cols)*h))
    canvas.save(outpath)

def save_state_npz(outdir, name, A, B, tau, N):
    path = os.path.join(outdir, name)
    np.savez_compressed(path, A=A, B=B, tau=tau, N=N)
    return path

# ------------------------------------------------------------
# Main simulation
# ------------------------------------------------------------
def run_simulation(cfg, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Save meta
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # Grid
    nx, ny = int(cfg["nx"]), int(cfg["ny"])
    dx, dy = float(cfg["dx"]), float(cfg["dy"])
    dt = float(cfg["dt"])
    steps = int(cfg["steps"])

    # Snapshot controls
    save_snapshots = bool(cfg.get("save_snapshots", True))
    snap_every = int(cfg.get("snap_every", 200))
    save_montage = bool(cfg.get("save_montage", True))

    # State saving (V6)
    save_states = bool(cfg.get("save_states", False))
    mid_state_step = int(cfg.get("mid_state_step", steps // 2))

    # Base Gray–Scott parameters
    Da = float(cfg["Da"])
    Db = float(cfg["Db"])
    feed = float(cfg["feed"])
    kill = float(cfg["kill"])

    # τ dynamics parameters
    tau0 = float(cfg["tau0"])
    tau_min, tau_max = float(cfg["tau_min"]), float(cfg["tau_max"])

    # global τ parameters (used if num_tau_species==1)
    alpha = float(cfg["alpha"])
    beta  = float(cfg["beta"])
    gamma = float(cfg["gamma"])

    # τ diffusion & noise
    kappa_tau = float(cfg.get("kappa_tau", 0.0))
    tau_noise = float(cfg.get("tau_noise", 0.0))

    # nutrient
    N0 = float(cfg.get("N0", 1.0))
    nutrient_use = float(cfg.get("nutrient_use", 0.01))
    nutrient_replenish = float(cfg.get("nutrient_replenish", 0.001))

    use_diffusive_nutrient = bool(cfg.get("use_diffusive_nutrient", False))
    D_N = float(cfg.get("D_N", 0.02))
    eta_N = float(cfg.get("eta_N", 0.1))
    rho_N = float(cfg.get("rho_N", 0.0005))

    # memory
    use_multiscale_memory = bool(cfg.get("use_multiscale_memory", False))
    mem_decay = float(cfg.get("memory_decay", 0.01))

    mem_decay_fast = float(cfg.get("mem_decay_fast", 0.02))
    mem_decay_slow = float(cfg.get("mem_decay_slow", 0.002))
    mem_w_fast = float(cfg.get("mem_w_fast", 0.7))
    mem_w_slow = float(cfg.get("mem_w_slow", 0.3))

    # multi-τ species
    num_species = int(cfg.get("num_tau_species", 1))
    species_params = cfg.get("tau_species_params", None)

    # RNG
    seed = cfg.get("seed", None)
    rng = np.random.default_rng(seed)

    # Fields
    A = np.ones((ny, nx), dtype=np.float64)
    B = np.zeros((ny, nx), dtype=np.float64)

    if num_species > 1:
        tau = np.zeros((num_species, ny, nx), dtype=np.float64)
        tau[:] = tau0
    else:
        tau = tau0 * np.ones((ny, nx), dtype=np.float64)

    N = N0 * np.ones((ny, nx), dtype=np.float64)

    # memory field(s)
    if use_multiscale_memory:
        mem_fast = np.zeros((ny, nx), dtype=np.float64)
        mem_slow = np.zeros((ny, nx), dtype=np.float64)
    else:
        mem = np.zeros((ny, nx), dtype=np.float64)

    # initial perturbation
    r = int(cfg.get("seed_radius", 10))
    cx, cy = nx // 2, ny // 2
    A[cy-r:cy+r, cx-r:cx+r] = 0.5
    B[cy-r:cy+r, cx-r:cx+r] = 0.25

    noise = float(cfg.get("noise", 0.02))
    if noise > 0:
        A += noise * rng.random((ny, nx))
        B += noise * rng.random((ny, nx))

    # Logging arrays
    times = []
    coherences = []
    entropies = []
    autocats = []

    def entropy(field):
        x = field.flatten()
        x = x - x.min()
        s = x.sum()
        if s <= 0:
            return 0.0
        p = x / (s + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))

    # --------------------------------------------------------
    # Time loop
    # --------------------------------------------------------
    for step in range(steps + 1):

        # reaction
        R = A * (B**2)

        # diffusion for A,B uses τ as geometric focusing: D_eff = D / τ
        if num_species > 1:
            tau_eff = np.mean(tau, axis=0)
        else:
            tau_eff = tau

        tau_eff = np.clip(tau_eff, tau_min, tau_max)

        lapA = laplacian(A, dx, dy)
        lapB = laplacian(B, dx, dy)

        Da_eff = Da / tau_eff
        Db_eff = Db / tau_eff

        dA = Da_eff * lapA - R + feed * (1.0 - A)
        dB = Db_eff * lapB + R - (kill + feed) * B

        A += dt * dA
        B += dt * dB
        np.clip(A, 0.0, 3.0, out=A)
        np.clip(B, 0.0, 3.0, out=B)

        # Memory update
        if use_multiscale_memory:
            mem_fast = mem_fast * (1.0 - mem_decay_fast) + np.abs(R)
            mem_slow = mem_slow * (1.0 - mem_decay_slow) + np.abs(R)
            mem_eff = mem_w_fast * mem_fast + mem_w_slow * mem_slow
        else:
            mem = mem * (1.0 - mem_decay) + np.abs(R)
            mem_eff = mem

        # Nutrient update
        if use_diffusive_nutrient:
            lapN = laplacian(N, dx, dy)
            dN = D_N * lapN - eta_N * N * B + rho_N
            N += dt * dN
        else:
            N -= dt * nutrient_use * np.abs(R)
            N += dt * nutrient_replenish

        np.clip(N, 0.0, N0, out=N)

        # τ update (global or species)
        if num_species > 1:
            if species_params is None:
                # fallback: small random per species
                species_params = []
                for _ in range(num_species):
                    species_params.append({
                        "alpha": alpha * (0.8 + 0.4*rng.random()),
                        "beta":  beta  * (0.8 + 0.4*rng.random()),
                        "gamma": gamma * (0.8 + 0.4*rng.random()),
                    })

            for i in range(num_species):
                ai = float(species_params[i].get("alpha", alpha))
                bi = float(species_params[i].get("beta",  beta))
                gi = float(species_params[i].get("gamma", gamma))

                # tau diffusion term
                lapTau = laplacian(tau[i], dx, dy) if kappa_tau != 0.0 else 0.0
                xi = (tau_noise * rng.standard_normal((ny, nx)) * np.sqrt(dt)) if tau_noise != 0.0 else 0.0

                tau[i] += dt * (ai * mem_eff - bi * (tau[i] - tau0) + gi * N + kappa_tau * lapTau) + xi
                np.clip(tau[i], tau_min, tau_max, out=tau[i])
        else:
            lapTau = laplacian(tau, dx, dy) if kappa_tau != 0.0 else 0.0
            xi = (tau_noise * rng.standard_normal((ny, nx)) * np.sqrt(dt)) if tau_noise != 0.0 else 0.0

            tau += dt * (alpha * mem_eff - beta * (tau - tau0) + gamma * N + kappa_tau * lapTau) + xi
            np.clip(tau, tau_min, tau_max, out=tau)

        # Logging
        log_every = int(cfg.get("log_every", 20))
        if (step % log_every) == 0:
            M = A + 1j*B
            coherence = float(np.mean(np.abs(M)**2))
            coherences.append(coherence)
            entropies.append(entropy(B))
            autocats.append(float(np.mean(R)))
            times.append(step * dt)

        # Save mid/final states (NPZ)
        if save_states and step == mid_state_step:
            # save "mid" state
            if num_species > 1:
                tau_mid = np.mean(tau, axis=0)
            else:
                tau_mid = tau
            save_state_npz(outdir, "state_mid.npz", A, B, tau_mid, N)

        # Snapshots
        if save_snapshots and (step % snap_every == 0):
            idx = step // snap_every
            plt.imsave(os.path.join(outdir, f"B_snapshot_{idx:04d}.png"), B, cmap="magma")
            if num_species > 1:
                tau_vis = np.mean(tau, axis=0)
            else:
                tau_vis = tau
            plt.imsave(os.path.join(outdir, f"tau_snapshot_{idx:04d}.png"), tau_vis, cmap="viridis")
            plt.imsave(os.path.join(outdir, f"N_snapshot_{idx:04d}.png"), N, cmap="plasma")

    # Save final state
    if save_states:
        if num_species > 1:
            tau_final = np.mean(tau, axis=0)
        else:
            tau_final = tau
        save_state_npz(outdir, "state_final.npz", A, B, tau_final, N)

    # Save metrics.csv
    import csv
    with open(os.path.join(outdir, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time","coherence","entropy","autocat"])
        for t, c, e, a in zip(times, coherences, entropies, autocats):
            w.writerow([t,c,e,a])

    # Summary
    summary = {
        "cfg": cfg,
        "metrics": "metrics.csv",
        "save_snapshots": save_snapshots,
        "save_states": save_states,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Montage (only if snapshots exist)
    if save_snapshots and save_montage:
        snapsB = sorted([os.path.join(outdir, s) for s in os.listdir(outdir) if s.startswith("B_snapshot_")])
        snapsT = sorted([os.path.join(outdir, s) for s in os.listdir(outdir) if s.startswith("tau_snapshot_")])
        snapsN = sorted([os.path.join(outdir, s) for s in os.listdir(outdir) if s.startswith("N_snapshot_")])

        montage(snapsB, os.path.join(outdir, "montage_B.png"))
        montage(snapsT, os.path.join(outdir, "montage_tau.png"))
        montage(snapsN, os.path.join(outdir, "montage_N.png"))

    return outdir
