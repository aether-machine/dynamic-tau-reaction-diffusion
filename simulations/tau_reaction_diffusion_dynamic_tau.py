"""
Simulation 4 — Dynamic τ with feedback from chemical activity
Author: [Your Name], 2025
License: MIT

Drop into: simulations/tau_reaction_diffusion_dynamic_tau.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math

# --- Output folder
outdir = "outputs/dynamic_tau"
os.makedirs(outdir, exist_ok=True)

# --- Grid and time parameters
nx, ny = 160, 160
dx = dy = 1.0
dt = 0.01
steps = 4000        # total time steps
snap_every = 200    # save snapshot every N steps

# --- Reaction-diffusion parameters (Gray-Scott style)
Da, Db = 0.16, 0.08
feed, kill = 0.035, 0.065

# --- τ dynamics parameters
# tau evolves slowly: tau_t = alpha * (S_activity) - beta * (tau - tau0)
tau0 = 1.0                # baseline tau
alpha = 0.02              # coupling strength from activity -> tau increase
beta = 0.005              # relaxation rate back to baseline
tau_min, tau_max = 0.2, 3.0

# --- initialize fields
A = np.ones((ny, nx))
B = np.zeros((ny, nx))

# small perturbation in center
r = 12
A[ny//2 - r:ny//2 + r, nx//2 - r:nx//2 + r] = 0.50
B[ny//2 - r:ny//2 + r, nx//2 - r:nx//2 + r] = 0.25

# small random noise
np.random.seed(42)
A += 0.02 * np.random.rand(ny, nx)
B += 0.02 * np.random.rand(ny, nx)

# time-density field τ initial
tau = tau0 * np.ones((ny, nx))
# optionally seed a localized high-tau region
cx, cy = nx//3, ny//3
tau[cy-6:cy+6, cx-6:cx+6] = tau0 + 0.8

# laplacian helper
def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
    ) / (dx*dy)

# metrics time series
times = []
coherence_ts = []   # mean |M|^2
energy_ts = []      # integral 0.5*(A^2 + B^2)
entropy_ts = []     # Shannon over B distribution
autocat_ts = []     # mean(A*B^2)

# small helper: Shannon entropy of normalized B
def shannon_entropy(field):
    flat = field.flatten()
    # add small epsilon to avoid log(0)
    eps = 1e-12
    p = flat - flat.min()
    if p.sum() == 0:
        return 0.0
    p /= (p.sum() + eps)
    p = np.maximum(p, eps)
    return -np.sum(p * np.log(p))

# --- main loop
for t in range(steps):
    # compute laplacians
    lapA = laplacian(A)
    lapB = laplacian(B)

    # reaction term
    reaction = A * (B**2)
    dA = Da * lapA - reaction + feed * (1 - A)
    dB = Db * lapB + reaction - (kill + feed) * B

    # update concentrations with tau modulation (slower/faster local dynamics)
    A += (dA * dt * tau)
    B += (dB * dt * tau)

    # ensure non-negativity (numerical stability)
    A = np.clip(A, 0.0, 2.0)
    B = np.clip(B, 0.0, 2.0)

    # --- compute local activity S (we'll use |reaction| + gradient magnitude of B)
    gradBx = (np.roll(B, -1, axis=1) - np.roll(B, 1, axis=1)) / (2*dx)
    gradBy = (np.roll(B, -1, axis=0) - np.roll(B, 1, axis=0)) / (2*dy)
    gradB_mag = np.sqrt(gradBx**2 + gradBy**2)
    S_activity = np.abs(reaction) + 0.5 * gradB_mag

    # --- update tau slowly (local feedback)
    # simple ODE: tau_t = alpha * S_activity - beta * (tau - tau0)
    tau += dt * (alpha * S_activity - beta * (tau - tau0))

    # clamp tau to physical bounds
    tau = np.clip(tau, tau_min, tau_max)

    # --- metrics logging every step or periodically
    if t % 10 == 0:
        M = A + 1j * B
        coherence = np.mean(np.abs(M)**2)               # mean |M|^2
        energy = 0.5 * np.sum(A**2 + B**2) / (nx*ny)
        entropy = shannon_entropy(B)
        autocat = np.mean(reaction)
        times.append(t * dt)
        coherence_ts.append(coherence)
        energy_ts.append(energy)
        entropy_ts.append(entropy)
        autocat_ts.append(autocat)

    # --- snapshots
    if t % snap_every == 0:
        idx = t // snap_every
        # save B snapshot image
        plt.figure(figsize=(4,4))
        plt.imshow(B, cmap='magma', origin='lower')
        plt.title(f"B(t={t*dt:.2f})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{outdir}/B_snapshot_{idx:03d}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(4,4))
        plt.imshow(tau, cmap='viridis', origin='lower')
        plt.title(f"tau(t={t*dt:.2f})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{outdir}/tau_snapshot_{idx:03d}.png", dpi=150)
        plt.close()

# --- end main loop

# --- Save time-series diagnostics plots
plt.figure(figsize=(6,4))
plt.plot(times, coherence_ts, label="mean |M|^2")
plt.plot(times, energy_ts, label="energy-like")
plt.xlabel("time")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/diagnostics_coherence_energy.png", dpi=150)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(times, entropy_ts, label="entropy(B)")
plt.plot(times, autocat_ts, label="autocat mean")
plt.xlabel("time")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/diagnostics_entropy_autocat.png", dpi=150)
plt.close()

# --- final snapshots
plt.figure(figsize=(4,4))
plt.imshow(B, cmap='magma', origin='lower')
plt.title("B final")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{outdir}/B_final.png", dpi=200)
plt.close()

plt.figure(figsize=(4,4))
plt.imshow(tau, cmap='viridis', origin='lower')
plt.title("tau final")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{outdir}/tau_final.png", dpi=200)
plt.close()

# Optionally write a small CSV with metrics
import csv
with open(f"{outdir}/metrics.csv", "w", newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(["time", "coherence", "energy", "entropy", "autocat"])
    for i in range(len(times)):
        writer.writerow([times[i], coherence_ts[i], energy_ts[i], entropy_ts[i], autocat_ts[i]])

print("Simulation complete. Outputs saved to:", outdir)
# --- Animation builder (optional)
import imageio
import glob

# Collect all saved B snapshots (change pattern if you want tau animation instead)
frames = sorted(glob.glob(f"{outdir}/B_snapshot_*.png"))
if not frames:
    print("No snapshot frames found. Skipping animation.")
else:
    # Build MP4
    mp4_path = f"{outdir}/B_evolution.mp4"
    with imageio.get_writer(mp4_path, fps=15, codec='libx264', quality=8) as writer:
        for fn in frames:
            writer.append_data(imageio.imread(fn))
    print("MP4 animation saved to:", mp4_path)

    # Build GIF
    gif_path = f"{outdir}/B_evolution.gif"
    with imageio.get_writer(gif_path, mode='I', duration=0.07) as writer:
        for fn in frames:
            writer.append_data(imageio.imread(fn))
    print("GIF animation saved to:", gif_path)
