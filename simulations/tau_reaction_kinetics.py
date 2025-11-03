# τ-Dependent Reaction Kinetics
# ------------------------------------------------------------
# Concept: A → B reaction where the local reaction rate k(x)
# scales with the "time density" field τ(x).
# Author: [Your Name], 2025
# License: MIT

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Parameters
nx = 200
nt = 600
L = 1.0
dx = L / nx
dt = 0.002

x = np.linspace(0, L, nx)

# ------------------------------------------------------------
# 2. Define the time-density field τ(x)
def tau_field(x, mode="sinusoidal"):
    if mode == "sinusoidal":
        return 1.0 + 0.4 * np.sin(2 * np.pi * x / L)
    elif mode == "gaussian":
        return 1.0 + 0.8 * np.exp(-((x - 0.5 * L)**2) / (0.06**2))
    else:
        return np.ones_like(x)

tau = tau_field(x, mode="sinusoidal")

# Reaction constant baseline
k0 = 1.0
k = k0 * tau         # local reaction rate

# ------------------------------------------------------------
# 3. Initial concentrations
A = np.ones(nx) * 1.0   # initial reactant
B = np.zeros(nx)        # product starts at zero

frames_A = [A.copy()]
frames_B = [B.copy()]

# ------------------------------------------------------------
# 4. Reaction kinetics
# dA/dt = -k(x) * A
# dB/dt =  k(x) * A
for t in range(nt):
    A -= k * A * dt
    B += k * A * dt

    if t % 30 == 0:
        frames_A.append(A.copy())
        frames_B.append(B.copy())

# ------------------------------------------------------------
# 5. Visualization

plt.figure(figsize=(8,5))
for i,f in enumerate(frames_A):
    plt.plot(x, f, label=f"A, t={i*30*dt:.2f}")
plt.title("τ-Dependent Reaction Kinetics: A(x,t)")
plt.xlabel("x")
plt.ylabel("Concentration A")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
for i,f in enumerate(frames_B):
    plt.plot(x, f, label=f"B, t={i*30*dt:.2f}")
plt.title("τ-Dependent Reaction Kinetics: B(x,t)")
plt.xlabel("x")
plt.ylabel("Concentration B")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,2))
plt.plot(x, tau, "k--")
plt.title("Time-Density Field τ(x)")
plt.xlabel("x")
plt.ylabel("τ(x)")
plt.tight_layout()
plt.show()
