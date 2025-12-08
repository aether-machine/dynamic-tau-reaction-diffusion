#!/usr/bin/env python3
"""
tau_reaction_diffusion_v2.py
Dynamic tau RD with resource N, D_eff = D0/(tau+eps), and tau-noise.
Simple, single-file simulation for exploration.
"""

import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- Config ----------
cfg = {
    "nx": 128, "ny": 128,
    "dx": 1.0, "dy": 1.0,
    "dt": 0.5, "steps": 4000, "snap_every": 400,
    # Gray-Scott base
    "D_A0": 0.16, "D_B0": 0.08,
    "f": 0.03, "k": 0.065,
    # tau dynamics
    "tau0": 1.0, "alpha": 0.01, "beta": 0.01, "gamma": 0.5,
    "tau_min": 0.2, "tau_max": 5.0, "eps": 1e-3,
    # resource N
    "D_N": 0.02, "eta": 0.1, "rho": 0.0005,
    # noise
    "noise_tau_amp": 0.001,
    "seed": 42,
    "outdir": "outputs/dynamic_tau_v2"
}

os.makedirs(cfg["outdir"], exist_ok=True)

# ---------- helpers ----------
rng = np.random.RandomState(cfg["seed"])

def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
    ) / (cfg["dx"]*cfg["dy"])

def shannon_entropy(field):
    flat = field.flatten()
    p = flat - flat.min()
    s = p.sum()
    if s <= 0: return 0.0
    p = p/(s+1e-12)
    p = np.maximum(p, 1e-12)
    return -np.sum(p*np.log(p))

# ---------- initialize fields ----------
nx, ny = cfg["nx"], cfg["ny"]
A = np.ones((ny,nx))
B = np.zeros((ny,nx))
N = np.ones((ny,nx))*0.5   # resource background
tau = np.ones((ny,nx))*cfg["tau0"]

# seed central perturbation in B and N hotspot
r = 8
cy, cx = ny//2, nx//2
A[cy-r:cy+r, cx-r:cx+r] = 0.5
B[cy-r:cy+r, cx-r:cx+r] = 0.25
N[cy-2*r:cy+2*r, cx-2*r:cx+2*r] = 1.0

# small noise
A += 0.02*rng.rand(ny,nx)
B += 0.02*rng.rand(ny,nx)

# metrics
times, coherence_ts, entropy_ts, energy_ts = [], [], [], []

# ---------- main loop ----------
for t in range(cfg["steps"]):
    # compute local D_eff
    D_A = cfg["D_A0"] / (tau + cfg["eps"])
    D_B = cfg["D_B0"] / (tau + cfg["eps"])

    lapA = laplacian(A)
    lapB = laplacian(B)
    lapN = laplacian(N)

    reaction = A * (B**2)

    # PDE updates with local diffusion coefficient approximation:
    dA = D_A * lapA - reaction + cfg["f"]*(1 - A)
    dB = D_B * lapB + reaction - (cfg["k"]) * B

    # resource dynamics
    dN = cfg["D_N"] * lapN - cfg["eta"] * N * B + cfg["rho"]

    # integrate (explicit)
    A += cfg["dt"] * dA
    B += cfg["dt"] * dB
    N += cfg["dt"] * dN

    # tau update: activity S = reaction + 0.5*|grad B|
    gradBx = (np.roll(B, -1, axis=1) - np.roll(B, 1, axis=1)) / (2*cfg["dx"])
    gradBy = (np.roll(B, -1, axis=0) - np.roll(B, 1, axis=0)) / (2*cfg["dy"])
    S = np.abs(reaction) + 0.5*np.sqrt(gradBx**2 + gradBy**2)

    # noise term
    xi = cfg["noise_tau_amp"] * rng.randn(ny, nx) * math.sqrt(cfg["dt"])

    tau += cfg["dt"] * (cfg["alpha"]*S - cfg["beta"]*(tau - cfg["tau0"]) + cfg["gamma"]*N) + xi
    np.clip(tau, cfg["tau_min"], cfg["tau_max"], out=tau)

    # clamp concentrations
    np.clip(A, 0.0, 2.0, out=A)
    np.clip(B, 0.0, 2.0, out=B)
    np.clip(N, 0.0, 2.0, out=N)

    # logging
    if (t % 10) == 0:
        M = A + 1j*B
        coherence = np.mean(np.abs(M)**2)
        energy = 0.5*np.sum(A**2 + B**2)/(nx*ny)
        entropy = shannon_entropy(B)
        times.append(t*cfg["dt"])
        coherence_ts.append(coherence)
        energy_ts.append(energy)
        entropy_ts.append(entropy)

    # snapshots
    if (t % cfg["snap_every"]) == 0:
        idx = t // cfg["snap_every"]
        # plot B, N, tau
        fig, axes = plt.subplots(1,3, figsize=(12,4))
        axes[0].imshow(B, cmap='magma', origin='lower'); axes[0].set_title(f'B t={t*cfg["dt"]:.1f}')
        axes[1].imshow(N, cmap='viridis', origin='lower'); axes[1].set_title('N')
        axes[2].imshow(tau, cmap='inferno', origin='lower'); axes[2].set_title('tau')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        fig.savefig(os.path.join(cfg["outdir"], f"snapshot_{idx:04d}.png"))
        plt.close(fig)

# ---------- save metrics ----------
import csv
meta = {"created": datetime.utcnow().isoformat()+"Z", "cfg": cfg}
with open(os.path.join(cfg["outdir"], "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2)

with open(os.path.join(cfg["outdir"], "metrics.csv"), "w", newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(["time","coherence","energy","entropy"])
    for i in range(len(times)):
        writer.writerow([times[i], coherence_ts[i], energy_ts[i], entropy_ts[i]])

print("Simulation complete. Outputs in:", cfg["outdir"])
