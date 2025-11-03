"""
Coupled τ-Diffusion + Reaction System
-------------------------------------
Demonstrates how variable local time density (τ) modulates
reaction–diffusion pattern formation.

Author: [Your Name], 2025
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx, ny = 150, 150
dx = dy = 1.0
dt = 0.01
steps = 4000

# Diffusion coefficients
Da, Db = 0.1, 0.05

# Reaction coefficients (Gray–Scott–like)
feed, kill = 0.036, 0.065

# ----------------------------------------------------------------
# Define 2-D time-density field τ(x, y)
def tau_field(nx, ny, mode="wave"):
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    X, Y = np.meshgrid(x, y)
    if mode == "wave":
        return 1.0 + 0.3*np.sin(X)*np.cos(Y)
    elif mode == "vortex":
        return 1.0 + 0.6*np.exp(-((X-np.pi)**2 + (Y-np.pi)**2)/(0.4**2))
    else:
        return np.ones((ny, nx))

tau = tau_field(nx, ny, mode="wave")

# ----------------------------------------------------------------
# Initialize fields A (substrate) and B (reactant)
A = np.ones((ny, nx))
B = np.zeros((ny, nx))

# small perturbation
r = 20
A[ny//2 - r:ny//2 + r, nx//2 - r:nx//2 + r] = 0.50
B[ny//2 - r:ny//2 + r, nx//2 - r:nx//2 + r] = 0.25

# Laplacian operator (finite differences)
def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
    ) / (dx*dy)

# ----------------------------------------------------------------
# Time evolution
snapshots = []
for t in range(steps):
    lapA = laplacian(A)
    lapB = laplacian(B)

    # Reaction terms (Gray–Scott model)
    reaction = A * B**2
    dA = Da * lapA - reaction + feed*(1 - A)
    dB = Db * lapB + reaction - (kill + feed)*B

    # τ-modulated update (local time dilation)
    A += (dA * dt * tau)
    B += (dB * dt * tau)

    if t % 500 == 0:
        snapshots.append(B.copy())

# ----------------------------------------------------------------
# Plotting
fig, axes = plt.subplots(1, len(snapshots), figsize=(15, 3))
for ax, img, i in zip(axes, snapshots, range(len(snapshots))):
    ax.imshow(img, cmap='plasma', origin='lower')
    ax.set_title(f"t = {i*500*dt:.1f}")
    ax.axis('off')

plt.suptitle("τ-Dependent Reaction–Diffusion: Emergent Coherent Pockets")
plt.tight_layout()
plt.show()

# Visualize τ field itself
plt.figure(figsize=(4,4))
plt.imshow(tau, cmap='gray', origin='lower')
plt.title("Time-Density Field τ(x, y)")
plt.axis('off')
plt.tight_layout()
plt.show()
