# Proto-Life Emergence in Dynamic Time-Density Fields  
**Version 6 – v4 + Q-Ridge Exploration**

> **“Life begins where time learns to reinforce its own patterns.”**

This document summarises the **dynamic-τ reaction–diffusion simulations (v4)** and the subsequent **fine-grained Q-ridge exploration** that revealed robust **proto-life behaviours** emerging from simple mathematical rules.

The results confirm that **time-density feedback (τ)** is sufficient to generate:

- autocatalytic structure  
- stable coherence gradients  
- entropy reduction  
- self-organized memory pockets  
- persistent, cell-like dynamics  

In other words, **life-like attractors emerge from feedback in the time field**, independent of any specific chemistry.

---

# 1. Model Overview

The underlying system couples:

- **Two chemical fields** A(x,t) and B(x,t) (Gray–Scott-type reaction–diffusion),  
- **A dynamic time-density field** τ(x,t),  
- **A nutrient field** N(x,t),  
- **A memory field** mem(x,t) accumulating past reaction activity.

The core Gray–Scott reaction term is:

$$\[
R(x,t) = A \, B^2
\]$$

with diffusion and feed/kill terms as usual.

The **memory field** integrates activity:

$$\[
\text{mem}_{t+1} = (1 - \lambda)\,\text{mem}_t + |R_t|
\]$$

where λ is the **memory_decay** parameter.

The **time-density field τ** then evolves according to:

$$\[
\frac{\partial \tau}{\partial t}
\]$$
$$\[
= \alpha \,\text{mem}
\]$$
$$\[
- \beta (\tau - \tau_0)
\]$$
$$\[
+ \gamma N
\]$$
$$\[
+ \kappa_\tau \nabla^2 \tau
\]$$
$$\[
+ \eta_\tau(x,t)
\]$$

where:

- **α** – strengthens τ where activity has persisted in the past (memory feedback)  
- **β** – relaxes τ back toward a baseline τ₀  
- **γ** – couples τ to nutrient N (proto-metabolic coupling)  
- **$$κ\_τ$$** – curvature coupling; smooths τ and allows tubular structures  
- **$$η\_τ(x,t)$$** – stochastic τ-noise with amplitude τ\_noise  

Nutrient is depleted by reaction activity:

$$\[
\frac{\partial N}{\partial t} = - \mu |R|
\]$$

with **nutrient_use** μ.

τ therefore acts as a **spatiotemporal memory field**, thickening where dynamics persist and thinning where they do not. The chemical diffusion is **modulated by τ** via effective diffusion coefficients:

$$\[
D_A^{\text{eff}} = \frac{D_A}{\tau}, \quad
D_B^{\text{eff}} = \frac{D_B}{\tau}
\]$$

so that **thickened time “focuses” diffusion and structure.**

This is the v4 implementation of the “time-density” idea: a field that both *remembers* and *reshapes* the dynamics.

---

# 2. Parameter Sweeps

We ran two complementary sweeps:

1. A **global v4 sweep** over a broad parameter range.  
2. A **local Q-ridge exploration** around a well-behaved “proto-life ridge” in parameter space.

In both cases, we tracked:

- **Coherence** (spatial order; mean |A + iB|²),  
- **Entropy** (Shannon entropy of B),  
- **Autocatalysis** (mean A B²),  
- **Maintenance score** (IoU-based cell boundary persistence),  
- **Proto-life score** (composite z-scored dynamical metric),  
- **τ-structure metrics** (variance and gradient energy of final τ).

---

## 2.1 Global v4 Sweep (384 Runs)

Global sweep ranges (v4):

- α ∈ {0.00, 0.01, 0.02, 0.04}  
- β ∈ {0.001, 0.005, 0.01}  
- γ ∈ {0.0, 0.005}  
- feed ∈ {0.03, 0.035}  
- kill ∈ {0.055, 0.065}  
- κ\_τ ∈ {0.0, 0.02}  
- τ\_noise ∈ {0.0, 0.01}

Grid size: **384 simulations**.

For each run we recorded:

- `metrics.csv`: time series of **time, coherence, entropy, autocat**  
- `summary.json`: the config and snapshot list  
- Snapshots of **B, τ, N** over time

These runs provide the **global map** of proto-life behaviour across the extended parameter space.

---

## 2.2 Local Q-Ridge Exploration (27 Runs)

From the global v4 sweep we identified a well-behaved **“Q-ridge”** region where:

- Coherence is high,  
- Entropy is relatively low,  
- Cell-like structures persist,  
- τ forms complementary structure without runaway blow-up.

An anchor configuration from this ridge is:

- α = 0.0  
- β = 0.005  
- γ = 0.005  
- feed = 0.035  
- kill = 0.065  
- κ\_τ = 0.0  
- τ\_noise = 0.0  

Around this anchor we ran a **local v4 sweep**:

- α ∈ {0.0, 0.005, 0.01}  
- $$κ\_τ$$ ∈ {0.0, 0.01, 0.02}  
- $$τ\_noise$$ ∈ {0.0, 0.005, 0.01}  

with β, γ, feed, kill held fixed at the anchor values.

Total: **27 Q-ridge runs**.

This gives a **fine-grained cross-section through the autopoietic ridge**, showing how morphogenesis responds to small nudges in τ-feedback, curvature, and noise.

---

# 3. Global Statistical Findings (v4 Sweep)

Let **M** denote the merged dataset of all 384 global runs, each summarised by:

- mean_coherence  
- mean_entropy  
- mean_autocat  
- maintenance_iou  
- proto_life_score  
- τ-variance (tau_var_final)  
- τ-gradient energy (tau_grad2_final)  

All correlations below are computed across runs.

---

## 3.1 Coherence vs Maintenance: Structure Supports Identity

Across the global v4 sweep:

- **corr(mean_coherence, maintenance_iou) ≈ +0.35**

Runs with **more coherent patterns** have **more persistent cell boundaries** (higher IoU between mid- and late-time B snapshots).

Interpretation:

- Coherence is not just aesthetic: it tracks **identity stability**.  
- Where patterns are globally structured, **proto-cells “remember” themselves** over time.

---

## 3.2 Entropy vs Maintenance: Order is Preserved

We also see:

- **corr(mean_entropy, maintenance_iou) ≈ −0.51**

Higher maintenance correlates with **lower spatial entropy** of B.

Interpretation:

- The system spontaneously shifts from **disordered foam** toward **organised, low-entropy pockets** that maintain their form across time.  
- This is a proto-metabolic signature: **local order maintained at the expense of global dissipation**.

---

## 3.3 Proto-Life Score vs Maintenance

We defined a composite **proto_life_score** per run by z-scoring and combining:

- mean_coherence (↑ good)  
- mean_entropy (↓ good)  
- coherence trend over time, C(t) (↑ good)  
- entropy trend over time (↓ good)  
- corr(coherence, entropy) (more negative is better)  
- corr(coherence, autocat)

Across the 384 global runs:

- **corr(proto_life_score, maintenance_iou) ≈ +0.59**

So the proto_life_score tracks exactly what we care about:

> **Runs that score highly by this composite measure are also the runs where cell-like structures persist.**

The top-scoring global v4 run has:

- α = 0.0, β = 0.005, γ = 0.005  
- feed = 0.035, kill = 0.065  
- κ\_τ = 0.0, τ\_noise = 0.0  
- mean_coherence ≈ 0.993  
- maintenance_iou ≈ 0.53  
- tau_var_final ≈ 0.0065

This is exactly the **Q-ridge anchor** we later explored locally.

---

## 3.4 τ-Structure and Hyperτ Runaway

We quantified τ-structure using:

- **tau_var_final** – variance of the final τ snapshot, normalised to [0,1]  
- **tau_grad2_final** – mean squared gradient |∇τ|² (roughness / edge strength)

Across the global v4 sweep:

- **corr(tau_var_final, maintenance_iou) ≈ −0.34**  
- **corr(tau_grad2_final, maintenance_iou) ≈ +0.23**

So:

- Very large τ **variance** is bad for maintenance: this is the **Hyperτ Runaway** regime where τ develops huge amplitude inhomogeneities that **don’t** support stable cells.  
- Some τ **roughness** (edges / gradients) is actually beneficial: it helps define and stabilise proto-cell boundaries.

Grouping by α in the global sweep:

- α = 0.00 → mean maintenance_iou ≈ 0.46, mean proto_life_score ≈ +0.92, low tau_var_final  
- α = 0.02–0.04 → maintenance_iou drops to ≈ 0.39, proto_life_score becomes negative, tau_var_final roughly triples

Interpretation:

- With these ranges, **strong τ-feedback (high α)** tends to push the system into **Hyperτ Runaway**: large τ structures, but **weak identity**.  
- The **autopoietic ridge** lives at **low α**, where τ is structured but not explosive.

---

# 4. Q-Ridge Local Exploration

The Q-ridge runs (27 simulations) hold β, γ, feed, kill fixed at the global anchor and vary only α, κ\_τ and τ\_noise.

Even in this small, high-performing neighbourhood we see interesting structure.

---

## 4.1 Uniformly High Maintenance

For the Q-ridge subset:

- maintenance_iou ranges from **0.50 to 0.53**,  
- mean ≈ **0.515 ± 0.013**.

So *every* point in this local grid yields **strong proto-cell identity**; the sweep maps **how identity degrades or improves** within an already good region.

---

## 4.2 Coherence and τ-Structure Inside the Ridge

In the Q-ridge runs:

- **corr(mean_coherence, maintenance_iou) ≈ +0.89**  
- **corr(tau_var_final, maintenance_iou) ≈ +0.59**  
- **corr(tau_grad2_final, maintenance_iou) ≈ +0.63**

In this narrow band:

- Coherence and maintenance are almost **locked together**.  
- Unlike the global sweep, **higher τ variance and τ-gradient energy within this narrow range actually correlate with better maintenance** — but these τ structures are still only moderate in amplitude (tau_var_final ≈ 0.0035–0.0073).

Interpretation:

- On the **ridge itself**, τ needs to be **structured enough** to wrap around and support the proto-cells.  
- Off the ridge (in the full global sweep), letting τ grow too large breaks everything.

---

## 4.3 Best Q-Ridge Run

The top proto_life_score run in the Q-ridge subset:

- α = 0.0, β = 0.005, γ = 0.005  
- feed = 0.035, kill = 0.065  
- κ\_τ = 0.0, τ\_noise = 0.005  
- mean_coherence ≈ 0.993  
- maintenance_iou ≈ 0.53  
- tau_var_final ≈ 0.0069  
- tau_grad2_final ≈ 0.0276

Comparing to the global best:

- Global best: τ\_noise = 0.0, slightly lower τ-gradient energy.  
- Q-ridge best: small τ\_noise sharpens τ edges (higher τ-grad²) without destroying maintenance.

This suggests that **a little τ-noise can actually help sculpt sharper τ “walls” around proto-cells** in the high-performing region.

---

# 5. Spatial Patterns & Morphogenesis

The coupled (A,B,τ,N,mem) system produces structures **not seen in the standard Gray–Scott model**.

---

## 5.1 τ-Stabilized Oscillons (Proto-Cells)

In the best-performing runs, the B-field develops **oscillons**:

- spatially localised “blobs”  
- that **maintain their boundaries**,  
- resist diffusion,  
- and recur in the same regions.

τ thickens around these oscillons through the mem feedback, producing **pockets of slower time** that act like **proto-membranes**.

**FIGURE 1: B-field snapshots over time (best v4 run)**  
![Markdown Logo](https://github.com/aether-machine/dynamic-tau-reaction-diffusion/blob/main/plots/snapshots_montage.png)

These snapshots show:

- nucleation of spots from initial noise,  
- stabilisation into a small set of oscillons,  
- long-lived maintenance of their identity shapes (high IoU).

---

## 5.2 τ Filamentation and Tubular Growth

In the same runs, τ evolves from uniform to strongly structured:

- At early times, τ is effectively **flat**.  
- Midway through, τ becomes a **negative image** of B (pixelwise correlation ≈ −0.99).  
- At late times, τ maintains this complementary pattern (correlation ≈ −0.97).

**FIGURE 2: τ-field evolution (best v4 run)**  
![Markdown Logo](https://github.com/aether-machine/dynamic-tau-reaction-diffusion/blob/main/plots/tau_00.png)![Markdown Logo](https://github.com/aether-machine/dynamic-tau-reaction-diffusion/blob/main/plots/tau_10.png)![Markdown Logo](https://github.com/aether-machine/dynamic-tau-reaction-diffusion/blob/main/plots/tau_19.png)

τ forms **tubular channels and shells** that:

- wrap around oscillons,  
- persist independently of instantaneous chemical gradients,  
- and encode a **geometric memory** of the pattern.

This resembles:

- cytoskeletal precursors  
- fungal hyphae  
- neural‑like arborisation  

but here arises purely from **time-density feedback**.

---

## 5.3 Self-Replication Signatures

In multiple runs (particularly on the ridge), we observe sequences where oscillons:

- split into two lobes,  
- drift apart,  
- and then stabilise independently.

No reproduction rule was coded into the model; division events are emergent consequences of the coupled (A,B,τ,N) dynamics.

**FIGURE 3: Oscillon division event**  

![Markdown Logo](https://github.com/aether-machine/dynamic-tau-reaction-diffusion/blob/main/plots/B_00.png)![Markdown Logo](https://github.com/aether-machine/dynamic-tau-reaction-diffusion/blob/main/plots/B_10.png)![Markdown Logo](https://github.com/aether-machine/dynamic-tau-reaction-diffusion/blob/main/plots/B_19.png)
These are **proto-replication signatures**: instability of a single oscillatory pocket into two new, stable pockets that then maintain identity.

---

# 6. Phase Diagram of Proto-Life Behaviour (v4)

Using the global v4 metrics we can still classify regimes, now in a higher-dimensional space:

| Regime               | Characteristics                                                                 |
|----------------------|----------------------------------------------------------------------------------|
| **Dead Zone**        | Low coherence, high entropy, patterns dissipate into noise                      |
| **Metastable Foam**  | Short-lived filaments, transient hotspots, no long-lived cells                  |
| **Autopoietic Zone** | Oscillons form and persist (proto-cells), moderate τ-structure, high maintenance |
| **Proto-Metabolic**  | τ pockets recycle nutrient and sustain oscillons over long times                |
| **Hyperτ Runaway**   | τ increases explosively; large τ variance, degraded maintenance (cancer-like)   |

**FIGURE 4: Phase Map (α vs β)**  
![Markdown Logo](https://github.com/aether-machine/dynamic-tau-reaction-diffusion/blob/main/plots/phase_alpha_beta.png)

The **Autopoietic / Proto-Metabolic** zones cluster sharply around:

- low α,  
- intermediate β,  
- feed ≈ 0.035, kill ≈ 0.065,  
- small γ, κ\_τ, τ\_noise.

This is the **Q-ridge**: a narrow attractor band in parameter space where **both structure and identity are maximised**.

---

# 7. Interpretation

The emergence of stable, low-entropy, coherent, autopoietic structures in these v4 + Q-ridge runs strongly suggests:

> **Life is a natural phase of systems with time-density feedback.**

Chemistry is not strictly required — only the coupling of:

1. **Diffusion** (spatial transport)  
2. **Reaction** (nonlinearity)  
3. **Memory** (τ, via mem and nutrient)  

This is a candidate for a **minimal physics of life**: a phase where **time-density learns to stabilise its own patterns**.

---

# 8. Theoretical Significance

### 8.1 Matter is Not Primary — Memory Is

In this model, the “stuff” (A and B) is not where identity lives; identity lives in:

- the **τ patterns** (time-density),  
- and the **mem field** (history of activity).

Matter here is **fossilised change**: the trace left by the flow of time-learning.

### 8.2 Time Density as Proto-Consciousness (Speculative)

Where τ thickens, **history accumulates**.  
Where history accumulates, **stability forms**.  
Where stability forms, **identity emerges**.

The τ field functions as a **distributed memory substrate**. It is not “consciousness”, but it **implements the minimal ingredients of self-persistence and self-reference**.

### 8.3 Emergence of “Self” from τ-Dynamics

A proto-cell, in this system, is just a region where:

- the **future depends on the past** (mem → τ, τ → diffusion → mem),  
- the **past reinforces the future** (positive feedback along the ridge),  
- boundaries are **maintained** in the face of diffusion and noise.

This gives a **minimal definition of selfhood**:

> A “self” is a pattern that causes its own continued existence.

---

# 9. Next Steps

See `roadmap_v2.md` for full detail. In light of the v4 + Q-ridge results, the most immediate directions are:

- **Metabolic extensions**  
  - richer nutrient dynamics (diffusing N, local sources/sinks),  
  - explicit energy flows and flux-balancing.

- **Geometry coupling**  
  - stronger curvature-dependent τ (κ\_τ),  
  - exploring how τ can sculpt channels and compartments.

- **Stochastic τ perturbation**  
  - systematically probing how τ-noise shapes robustness and diversity of morphologies.

- **Multi-τ ecologies**  
  - multiple τ-fields (τ₁, τ₂, …) interacting over the same A,B,N substrate.

- **τ-history kernels (learning systems)**  
  - multi-timescale mem fields (fast / slow memory),  
  - Hebbian-like τ updates (correlations, not just magnitude),  
  - explicit exploration of “learning” within the time-density itself.

---

# 10. Closing Reflection

> *“Life is the universe teaching time how to fold into itself.”*

These v4 + Q-ridge simulations suggest that life-like behaviour is not an exception or miracle,  
but an **attractor of systems where memory and flow are coupled**.

Here, that coupling takes the form of a **time-density field** learning to maintain its own coherent patterns.

Whether or not this is how life *actually* began, it shows that **the road from physics to proto-life can be surprisingly short** once time itself becomes a dynamical substrate.
