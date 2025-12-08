# Temporal Diffusion Model
# ------------------------------------------------------------
# Concept: diffusion under a variable "time density" field τ(x)
# Author: [Your Name], 2025
# License: MIT

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Model parameters

nx = 200           # spatial points
nt = 800           # time steps
L = 1.0            # domain length (arbitrary units)
D0 = 1e-3          # baseline diffusion coefficient
dx = L / nx
dt = 0.25 * dx**2 / D0  # stability condition

x = np.linspace(0, L, nx)

# ------------------------------------------------------------
# 2. Define the time-density field τ(x)
# You can experiment with different spatial profiles

def tau_field(x, mode="gaussian"):
    if mode == "gaussian":
        return 1.0 + 0.8 * np.exp(-((x - 0.5 * L)**2) / (0.05**2))
    elif mode == "linear":
        return 1.0 + 0.5 * (x / L)
    elif mode == "sinusoidal":
        return 1.0 + 0.5 * np.sin(4 * np.pi * x / L)
    else:
        return np.ones_like(x)

tau = tau_field(x, mode="gaussian")

# Effective diffusion coefficient D(x) = D0 * τ(x)
D = D0 * tau

# ------------------------------------------------------------
# 3. Initial condition (concentration profile)
C = np.zeros(nx)
C[nx // 2] = 1.0  # delta-like initial pulse

# For recording evolution
frames = [C.copy()]

# ------------------------------------------------------------
# 4. Time evolution
for t in range(nt):
    # second derivative (diffusion term)
    laplacian = np.zeros_like(C)
    laplacian[1:-1] = D[1:-1] * (C[2:] - 2 * C[1:-1] + C[:-2]) / dx**2
    
    # add gradient of D(x) * ∂C/∂x term (since D varies in space)
    gradC = np.zeros_like(C)
    gradC[1:-1] = (C[2:] - C[:-2]) / (2 * dx)
    gradD = np.zeros_like(D)
    gradD[1:-1] = (D[2:] - D[:-2]) / (2 * dx)
    
    flux_term = gradD * gradC
    C[1:-1] += dt * (laplacian + flux_term[1:-1])
    
    # boundary conditions: zero flux
    C[0] = C[1]
    C[-1] = C[-2]
    
    if t % 40 == 0:
        frames.append(C.copy())

# ------------------------------------------------------------
# 5. Visualization

plt.figure(figsize=(8, 5))
for i, f in enumerate(frames):
    plt.plot(x, f, label=f"t={i * 40 * dt:.3f}")
plt.title("Temporal Diffusion under Variable Time Density τ(x)")
plt.xlabel("x")
plt.ylabel("Concentration C(x, t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot τ(x)
plt.figure(figsize=(8, 2))
plt.plot(x, tau, 'k--')
plt.title("Time-Density Field τ(x)")
plt.xlabel("x")
plt.ylabel("τ(x)")
plt.grid(True)
plt.tight_layout()
plt.show()
