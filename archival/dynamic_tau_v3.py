#!/usr/bin/env python3
"""
dynamic_tau_v3.py

Full dynamic τ simulation with:
 - τ feedback (α)
 - τ relaxation (β)
 - nutrient coupling (γ)
 - geometric focusing (D_eff = D / τ)
 - memory kernel (exponential decay)
 - snapshot output for B, τ, N
 - montage output

Used by run_sweep_v3.py
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
    return (
        -4*Z
        + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
    ) / (dx * dy)

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

# ------------------------------------------------------------
# Main simulation
# ------------------------------------------------------------
def run_simulation(cfg, outdir):

    os.makedirs(outdir, exist_ok=True)

    # Save meta
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # Grid
    nx, ny = cfg["nx"], cfg["ny"]
    dx, dy = cfg["dx"], cfg["dy"]
    dt = cfg["dt"]
    steps = cfg["steps"]
    snap_every = cfg["snap_every"]

    # Parameters
    Da = cfg["Da"]
    Db = cfg["Db"]
    feed = cfg["feed"]
    kill = cfg["kill"]

    # τ-dynamics parameters
    tau0 = cfg["tau0"]
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    gamma = cfg["gamma"]
    tau_min, tau_max = cfg["tau_min"], cfg["tau_max"]

    # Memory kernel weight
    mem_decay = cfg["memory_decay"]

    # Fields
    A = np.ones((ny, nx))
    B = np.zeros((ny, nx))
    tau = tau0 * np.ones((ny, nx))
    mem = np.zeros((ny, nx))   # accumulates |reaction|

    # Optional nutrient field
    N = cfg["N0"] * np.ones((ny, nx))

    # initial perturbation
    r = cfg["seed_radius"]
    cx, cy = nx//2, ny//2
    A[cy-r:cy+r, cx-r:cx+r] = 0.5
    B[cy-r:cy+r, cx-r:cx+r] = 0.25

    # noise
    A += cfg["noise"] * np.random.rand(ny, nx)
    B += cfg["noise"] * np.random.rand(ny, nx)

    # Logging arrays
    times = []
    coherences = []
    entropies = []
    autocats = []

    def entropy(field):
        x = field.flatten()
        x = x - x.min()
        s = x.sum()
        if s <= 0: return 0
        p = x / (s + 1e-12)
        return -np.sum(p * np.log(p + 1e-12))

    # --------------------------------------------------------
    # Time loop
    # --------------------------------------------------------
    for step in range(steps):

        # Laplacians
        lapA = laplacian(A, dx, dy)
        lapB = laplacian(B, dx, dy)

        # Reaction term (Gray–Scott)
        R = A * (B**2)

        # τ-modified diffusion
        Da_eff = Da / tau
        Db_eff = Db / tau
        dA = Da_eff * lapA - R + feed * (1 - A)
        dB = Db_eff * lapB + R - (kill + feed) * B

        # Update A,B
        A += dt * dA
        B += dt * dB
        np.clip(A, 0.0, 3.0, out=A)
        np.clip(B, 0.0, 3.0, out=B)

        # Update memory (exponential kernel)
        mem = mem * (1 - mem_decay) + np.abs(R)

        # Update τ
        tau += dt * (alpha * mem - beta * (tau - tau0) + gamma * N)
        np.clip(tau, tau_min, tau_max, out=tau)

        # Simple nutrient depletion
        N -= dt * cfg["nutrient_use"] * np.abs(R)
        np.clip(N, 0.0, cfg["N0"], out=N)

        # Logging
        if step % cfg["log_every"] == 0:
            M = A + 1j*B
            coherence = np.mean(np.abs(M)**2)
            coherences.append(coherence)
            entropies.append(entropy(B))
            autocats.append(np.mean(R))
            times.append(step * dt)

        # Snapshots
        if step % snap_every == 0:
            idx = step // snap_every
            plt.imsave(os.path.join(outdir, f"B_snapshot_{idx:04d}.png"), B, cmap="magma")
            plt.imsave(os.path.join(outdir, f"tau_snapshot_{idx:04d}.png"), tau, cmap="viridis")
            plt.imsave(os.path.join(outdir, f"N_snapshot_{idx:04d}.png"), N, cmap="plasma")

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------
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
        "snapshots": sorted([f for f in os.listdir(outdir) if f.startswith("B_snapshot_")])
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # --------------------------------------------------------
    # Montage
    # --------------------------------------------------------
    snapsB = sorted([os.path.join(outdir, s) for s in os.listdir(outdir) if s.startswith("B_snapshot_")])
    snapsT = sorted([os.path.join(outdir, s) for s in os.listdir(outdir) if s.startswith("tau_snapshot_")])
    snapsN = sorted([os.path.join(outdir, s) for s in os.listdir(outdir) if s.startswith("N_snapshot_")])

    montage(snapsB, os.path.join(outdir, "montage_B.png"))
    montage(snapsT, os.path.join(outdir, "montage_tau.png"))
    montage(snapsN, os.path.join(outdir, "montage_N.png"))

    return outdir
