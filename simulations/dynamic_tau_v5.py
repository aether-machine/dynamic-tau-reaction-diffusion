#!/usr/bin/env python3
"""
dynamic_tau_v5.py

Dynamic time-density (τ) reaction–diffusion simulation with optional modules:

Core:
 - Gray–Scott A/B dynamics
 - Time-density field τ coupled to activity (mem) and nutrient N
 - Snapshots of B, τ, N
 - Metrics over time (coherence, entropy, energy, autocat)
 - Simple montages of snapshots

New optional modules:
 1) Multiscale memory kernel
    - mem_fast, mem_slow with different decay rates and weights
    - collapses to single mem if disabled

 2) Diffusive nutrient field
    - N diffuses, is depleted by reaction, replenished toward N0
    - collapses to simple local depletion if disabled

 3) Multi-τ species
    - num_tau_species >= 1
    - multiple τ_i fields with separate parameters
    - combined into τ_eff for diffusion via tau_mix_weights
    - collapses to single τ if num_tau_species == 1

Used by run_sweep_v5.py (not included here).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import csv


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def laplacian(Z, dx, dy):
    """
    5-point stencil Laplacian with periodic boundary conditions.
    """
    return (
        -4.0 * Z
        + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
    ) / (dx * dy)


def montage(images, outpath, max_images=9, cols=3):
    """
    Make a simple montage from a list of image paths.
    """
    if not images:
        return
    images = images[:max_images]
    imgs = [Image.open(im) for im in images]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h))
    for idx, im in enumerate(imgs):
        canvas.paste(im, ((idx % cols) * w, (idx // cols) * h))
    canvas.save(outpath)


def entropy(field):
    """
    Shannon entropy over a nonnegative field (up to an additive constant).
    """
    x = field.flatten()
    x = x - x.min()
    s = x.sum()
    if s <= 0:
        return 0.0
    p = x / (s + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


# ------------------------------------------------------------
# Main simulation
# ------------------------------------------------------------
def run_simulation(cfg, outdir):
    """
    Run a single dynamic τ simulation with the given configuration.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary specifying all parameters. At minimum:
        - nx, ny, dx, dy, dt, steps
        - snap_every, log_every
        - Da, Db, feed, kill
        - tau0, tau_min, tau_max
        - alpha, beta, gamma
        - N0, nutrient_use, memory_decay
        - seed_radius, noise

        Optional:
        - use_multiscale_memory (bool)
        - memory_fast_decay, memory_slow_decay
        - memory_fast_weight, memory_slow_weight
        - use_diffusive_nutrient (bool)
        - Dn, nutrient_feed_rate
        - num_tau_species (int >= 1)
        - tau0_list, alpha_list, beta_list, gamma_list,
          kappa_tau_list, tau_min_list, tau_max_list,
          tau_mix_weights
        - kappa_tau, tau_noise, save_tau_species (bool)
    outdir : str
        Output directory. Will be created if it does not exist.

    Returns
    -------
    outdir : str
        The output directory path.
    """

    os.makedirs(outdir, exist_ok=True)

    # Save config
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # Grid / time
    nx, ny = int(cfg["nx"]), int(cfg["ny"])
    dx, dy = float(cfg["dx"]), float(cfg["dy"])
    dt = float(cfg["dt"])
    steps = int(cfg["steps"])
    snap_every = int(cfg["snap_every"])
    log_every = int(cfg["log_every"])

    # Base reaction–diffusion parameters
    Da = float(cfg["Da"])
    Db = float(cfg["Db"])
    feed = float(cfg["feed"])
    kill = float(cfg["kill"])

    # Time-density / memory parameters (single-τ defaults)
    tau0 = float(cfg.get("tau0", 1.0))
    tau_min = float(cfg.get("tau_min", 0.2))
    tau_max = float(cfg.get("tau_max", 3.0))
    alpha = float(cfg.get("alpha", 0.0))
    beta = float(cfg.get("beta", 0.01))
    gamma = float(cfg.get("gamma", 0.0))
    kappa_tau = float(cfg.get("kappa_tau", 0.0))
    tau_noise = float(cfg.get("tau_noise", 0.0))

    # Nutrient
    N0 = float(cfg.get("N0", 1.0))
    nutrient_use = float(cfg.get("nutrient_use", 0.01))

    # Memory options
    use_multiscale_memory = bool(cfg.get("use_multiscale_memory", False))
    memory_decay = float(cfg.get("memory_decay", 0.01))
    memory_fast_decay = float(cfg.get("memory_fast_decay", 0.1))
    memory_slow_decay = float(cfg.get("memory_slow_decay", 0.005))
    memory_fast_weight = float(cfg.get("memory_fast_weight", 0.7))
    memory_slow_weight = float(cfg.get("memory_slow_weight", 0.3))

    # Nutrient module options
    use_diffusive_nutrient = bool(cfg.get("use_diffusive_nutrient", False))
    Dn = float(cfg.get("Dn", 0.05))
    nutrient_feed_rate = float(cfg.get("nutrient_feed_rate", 0.01))

    # Multi-τ options
    num_tau_species = int(cfg.get("num_tau_species", 1))
    save_tau_species = bool(cfg.get("save_tau_species", False))

    # Fields
    A = np.ones((ny, nx), dtype=float)
    B = np.zeros((ny, nx), dtype=float)
    N = N0 * np.ones((ny, nx), dtype=float)

    # Memory fields
    if use_multiscale_memory:
        mem_fast = np.zeros((ny, nx), dtype=float)
        mem_slow = np.zeros((ny, nx), dtype=float)
        mem = None
    else:
        mem = np.zeros((ny, nx), dtype=float)
        mem_fast = mem_slow = None

    # Time-density fields
    if num_tau_species <= 1:
        tau = tau0 * np.ones((ny, nx), dtype=float)
        tau_list = None
        tau0_list = None
        alpha_list = beta_list = gamma_list = None
        kappa_tau_list = tau_min_list = tau_max_list = None
        tau_mix_weights = None
    else:
        # Expect lists in cfg; if missing, derive trivial defaults
        tau0_list = cfg.get("tau0_list", [tau0] * num_tau_species)
        alpha_list = cfg.get("alpha_list", [alpha] * num_tau_species)
        beta_list = cfg.get("beta_list", [beta] * num_tau_species)
        gamma_list = cfg.get("gamma_list", [gamma] * num_tau_species)
        kappa_tau_list = cfg.get("kappa_tau_list", [kappa_tau] * num_tau_species)
        tau_min_list = cfg.get("tau_min_list", [tau_min] * num_tau_species)
        tau_max_list = cfg.get("tau_max_list", [tau_max] * num_tau_species)
        tau_mix_weights = np.array(cfg.get("tau_mix_weights", [1.0] * num_tau_species), dtype=float)

        if len(tau0_list) != num_tau_species:
            raise ValueError("tau0_list length must match num_tau_species")
        if len(alpha_list) != num_tau_species:
            raise ValueError("alpha_list length must match num_tau_species")
        if len(beta_list) != num_tau_species:
            raise ValueError("beta_list length must match num_tau_species")
        if len(gamma_list) != num_tau_species:
            raise ValueError("gamma_list length must match num_tau_species")
        if len(kappa_tau_list) != num_tau_species:
            raise ValueError("kappa_tau_list length must match num_tau_species")
        if len(tau_min_list) != num_tau_species:
            raise ValueError("tau_min_list length must match num_tau_species")
        if len(tau_max_list) != num_tau_species:
            raise ValueError("tau_max_list length must match num_tau_species")
        if tau_mix_weights.shape[0] != num_tau_species:
            raise ValueError("tau_mix_weights length must match num_tau_species")

        # normalise mixing weights
        if tau_mix_weights.sum() == 0:
            tau_mix_weights = np.ones_like(tau_mix_weights) / num_tau_species
        else:
            tau_mix_weights = tau_mix_weights / tau_mix_weights.sum()

        tau = None
        tau_list = [
            tau0_list[i] * np.ones((ny, nx), dtype=float)
            for i in range(num_tau_species)
        ]

    # Initial perturbation
    r = int(cfg.get("seed_radius", 10))
    cx, cy = nx // 2, ny // 2
    A[cy - r:cy + r, cx - r:cx + r] = 0.5
    B[cy - r:cy + r, cx - r:cx + r] = 0.25

    noise = float(cfg.get("noise", 0.02))
    if noise > 0.0:
        A += noise * np.random.rand(ny, nx)
        B += noise * np.random.rand(ny, nx)

    # Clip to safe bounds
    np.clip(A, 0.0, 3.0, out=A)
    np.clip(B, 0.0, 3.0, out=B)

    # --------------------------------------------------------
    # Metrics logging
    # --------------------------------------------------------
    times = []
    coherences = []
    entropies = []
    energies = []
    autocats = []

    # --------------------------------------------------------
    # Time loop
    # --------------------------------------------------------
    for step in range(steps):

        # Laplacians for A, B
        lapA = laplacian(A, dx, dy)
        lapB = laplacian(B, dx, dy)

        # Gray–Scott reaction
        R = A * (B ** 2)

        # Update memory
        if use_multiscale_memory:
            mem_fast = mem_fast * (1.0 - memory_fast_decay) + np.abs(R)
            mem_slow = mem_slow * (1.0 - memory_slow_decay) + np.abs(R)
            mem_eff = memory_fast_weight * mem_fast + memory_slow_weight * mem_slow
        else:
            mem = mem * (1.0 - memory_decay) + np.abs(R)
            mem_eff = mem

        # Update nutrient
        if use_diffusive_nutrient:
            lapN = laplacian(N, dx, dy)
            dN = (
                Dn * lapN
                - nutrient_use * np.abs(R)
                + nutrient_feed_rate * (N0 - N)
            )
            N += dt * dN
            np.clip(N, 0.0, N0, out=N)
        else:
            N -= dt * nutrient_use * np.abs(R)
            np.clip(N, 0.0, N0, out=N)

        # Update τ (single or multi-species) and compute effective τ for diffusion
        if num_tau_species <= 1:
            lap_tau = laplacian(tau, dx, dy) if kappa_tau != 0.0 else 0.0
            tau += dt * (
                alpha * mem_eff
                - beta * (tau - tau0)
                + gamma * N
                + (kappa_tau * lap_tau if kappa_tau != 0.0 else 0.0)
                + tau_noise * np.random.randn(ny, nx)
            )
            np.clip(tau, tau_min, tau_max, out=tau)
            tau_eff = tau
        else:
            # multi-species τ
            for i in range(num_tau_species):
                kappa_i = float(kappa_tau_list[i])
                lap_ti = laplacian(tau_list[i], dx, dy) if kappa_i != 0.0 else 0.0
                tau_list[i] += dt * (
                    float(alpha_list[i]) * mem_eff
                    - float(beta_list[i]) * (tau_list[i] - float(tau0_list[i]))
                    + float(gamma_list[i]) * N
                    + (kappa_i * lap_ti if kappa_i != 0.0 else 0.0)
                    + tau_noise * np.random.randn(ny, nx)
                )
                np.clip(tau_list[i], float(tau_min_list[i]), float(tau_max_list[i]), out=tau_list[i])

            # combine into τ_eff
            tau_eff = tau_mix_weights[0] * tau_list[0]
            for i in range(1, num_tau_species):
                tau_eff = tau_eff + tau_mix_weights[i] * tau_list[i]

        # τ clamps to avoid division problems
        tau_eff_safe = np.clip(tau_eff, 1e-3, None)

        # τ-modified diffusion
        Da_eff = Da / tau_eff_safe
        Db_eff = Db / tau_eff_safe

        dA = Da_eff * lapA - R + feed * (1.0 - A)
        dB = Db_eff * lapB + R - (kill + feed) * B

        # Update A, B
        A += dt * dA
        B += dt * dB
        np.clip(A, 0.0, 3.0, out=A)
        np.clip(B, 0.0, 3.0, out=B)

        # Logging
        if step % log_every == 0:
            t = step * dt
            M = A + 1j * B
            coherence = float(np.mean(np.abs(M) ** 2))
            ent = entropy(B)
            energy = float(0.5 * np.mean(A ** 2 + B ** 2))
            autocat = float(np.mean(R))

            times.append(t)
            coherences.append(coherence)
            entropies.append(ent)
            energies.append(energy)
            autocats.append(autocat)

        # Snapshots
        if step % snap_every == 0:
            idx = step // snap_every
            # B field
            plt.imsave(
                os.path.join(outdir, f"B_snapshot_{idx:04d}.png"),
                B,
                cmap="magma",
            )
            # τ effective
            plt.imsave(
                os.path.join(outdir, f"tau_snapshot_{idx:04d}.png"),
                tau_eff,
                cmap="viridis",
            )
            # optional τ species snapshots
            if num_tau_species > 1 and save_tau_species:
                for i in range(num_tau_species):
                    plt.imsave(
                        os.path.join(outdir, f"tau{i+1}_snapshot_{idx:04d}.png"),
                        tau_list[i],
                        cmap="viridis",
                    )
            # Nutrient
            plt.imsave(
                os.path.join(outdir, f"N_snapshot_{idx:04d}.png"),
                N,
                cmap="plasma",
            )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------
    metrics_path = os.path.join(outdir, "metrics.csv")
    with open(metrics_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "coherence", "entropy", "energy", "autocat"])
        for t, c, e, en, a in zip(times, coherences, entropies, energies, autocats):
            w.writerow([t, c, e, en, a])

    # Summary JSON
    snapshots_B = sorted(
        f for f in os.listdir(outdir) if f.startswith("B_snapshot_")
    )
    snapshots_tau = sorted(
        f for f in os.listdir(outdir) if f.startswith("tau_snapshot_")
    )
    snapshots_N = sorted(
        f for f in os.listdir(outdir) if f.startswith("N_snapshot_")
    )

    summary = {
        "cfg": cfg,
        "metrics": os.path.basename(metrics_path),
        "snapshots_B": snapshots_B,
        "snapshots_tau": snapshots_tau,
        "snapshots_N": snapshots_N,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # --------------------------------------------------------
    # Montages
    # --------------------------------------------------------
    snapsB_full = [os.path.join(outdir, s) for s in snapshots_B]
    snapsT_full = [os.path.join(outdir, s) for s in snapshots_tau]
    snapsN_full = [os.path.join(outdir, s) for s in snapshots_N]

    montage(snapsB_full, os.path.join(outdir, "montage_B.png"))
    montage(snapsT_full, os.path.join(outdir, "montage_tau.png"))
    montage(snapsN_full, os.path.join(outdir, "montage_N.png"))

    return outdir


if __name__ == "__main__":
    # Minimal CLI usage example (not a sweep):
    import argparse
    parser = argparse.ArgumentParser(description="Run a single dynamic_tau_v5 simulation.")
    parser.add_argument("--outdir", type=str, default="outputs/dynamic_tau_v5/test_run")
    args = parser.parse_args()

    example_cfg = {
        "nx": 150, "ny": 150,
        "dx": 1.0, "dy": 1.0,
        "dt": 0.01,
        "steps": 4000,
        "snap_every": 200,
        "log_every": 20,
        "Da": 0.16, "Db": 0.08,
        "feed": 0.035,
        "kill": 0.065,
        "alpha": 0.0,
        "beta": 0.005,
        "gamma": 0.005,
        "tau0": 1.0,
        "tau_min": 0.2,
        "tau_max": 3.0,
        "N0": 1.0,
        "nutrient_use": 0.01,
        "memory_decay": 0.01,
        "seed_radius": 10,
        "noise": 0.02,
        # optional modules (can be omitted for v4-like behaviour):
        "use_multiscale_memory": False,
        "use_diffusive_nutrient": False,
        "num_tau_species": 1,
        "kappa_tau": 0.0,
        "tau_noise": 0.0,
    }

    run_simulation(example_cfg, args.outdir)
