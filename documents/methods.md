## Methods

### M1. Base Reaction–Diffusion Model

All simulations are built on a standard two-species Gray–Scott reaction–diffusion system for fields $$\(A(x,y,t)\)$$ and $$\(B(x,y,t)\)$$:

$$
\frac{\partial A}{\partial t}
  = D_A \nabla^2 A - A B^2 + f(1 - A)
$$

$$
\frac{\partial B}{\partial t}
  = D_B \nabla^2 B + A B^2 - (k + f) B
$$

where:
- $$\(D_A, D_B\)$$ are diffusion coefficients,
- $$\(f\)$$ is the feed rate,
- $$\(k\)$$ is the kill (removal) rate,
- $$\(A B^2\)$$ is the autocatalytic reaction term.

Spatial derivatives are discretised on a 2D periodic grid using a five-point Laplacian:

$$
\nabla^2 Z_{ij} \approx
\frac{Z_{i+1,j} + Z_{i-1,j} + Z_{i,j+1} + Z_{i,j-1} - 4 Z_{ij}}{\Delta x \, \Delta y}
$$

Time integration is explicit Euler with timestep \(\Delta t\). All fields are clamped to finite bounds after each update (e.g. \(A,B \in [0, 2]\)) to ensure numerical stability.

---

### M2. Dynamic Time-Density Field $$\(\tau(x,t)\)$$

The key extension is a **dynamic time-density field** $$\(\tau(x,y,t)\)$$ that modulates diffusion and stores a memory of past activity.

#### M2.1. Diffusion–time coupling

In all dynamic-τ variants, diffusion becomes **τ-dependent**:

$$
D_A^{\mathrm{eff}}(x,y,t) = \frac{D_{A0}}{\tau(x,y,t) + \varepsilon}, \quad
D_B^{\mathrm{eff}}(x,y,t) = \frac{D_{B0}}{\tau(x,y,t) + \varepsilon}
$$

Substituting $$\(D^{\mathrm{eff}}\)$$ into the Gray–Scott equations yields a **time-thickening effect**:
- high $$\(\tau\)$$ → low effective diffusion → locally “stiff” or persistent regions,
- low $$\(\tau\)$$ → high diffusion → locally “fluid” or short-lived regions.

This coupling is the core mechanism by which **structured τ pockets act like a material substrate**: once formed, they selectively slow down motion and reaction spread.

#### M2.2. τ evolution and source term

The evolution of $$\(\tau\)$$ in the late models (v4–v5) takes the generic form:

$$\frac{\partial \tau}{\partial t} = \alpha\,S(x,y,t)$$
$$-\beta\,(\tau - \tau_0)$$
$$+\gamma\,N(x,y,t)$$
$$+\kappa_\tau \nabla^2 \tau$$
$$+\eta_\tau(x,y,t)$$

with:
- $$\(\alpha\)$$: strength of activity–memory coupling,
- $$\(\beta\)$$: relaxation back to baseline $$\(\tau_0\)$$,
- $$\(\gamma\)$$: coupling of nutrient $$\(N\)$$ into τ,
- $$\(\kappa_\tau\)$$: diffusion of τ (smoothing of the time field),
- $$\(\eta_\tau\)$$: stochastic τ-noise.

The **source term** $$\(S(x,y,t)\)$$ encodes local “interestingness” of dynamics:

- In early dynamic-τ models:
  - $$\(S\)$$ was based directly on the autocatalytic reaction $$\(R = A B^2\)$$.

- In intermediate models (e.g. v2):
  - we enriched $$\(S\)$$ with boundary sensitivity:
    $$
    S = |A B^2|$$ + $$\lambda \$$, $$|\nabla B|
    $$
  - This made τ respond not only to high reaction activity, but also to **sharp spatial gradients** (cell-like edges).

- In v4/v5:
  - we generalised this idea via **memory kernels** (next section), but the core intuition remains:
    - τ thickens where **activity is sustained**,
    - then modulates diffusion to stabilise those regions.

---

### M3. Nutrient Field and Proto-Metabolic Coupling

A separate nutrient field $$\(N(x,y,t)\)$$ is introduced to model resource availability.

In the v2-style dynamics, we used:

$$\frac{\partial N}{\partial t} = D_N \nabla^2 N$$
$$-\eta\,N B$$
$$+\rho$$


where:
- $$\(D_N\)$$ is nutrient diffusivity,
- $$\(\eta\)$$ controls consumption of N proportional to $$\(B\)$$,
- $$\(\rho\)$$ is a uniform replenishment rate.

N is then coupled into τ via the $$\(\gamma N\)$$ term in the τ equation: **nutrient-rich regions thicken time**, which in turn supports longer-lived structures. This closes a **proto-metabolic loop**:

1. Nutrient supports reaction activity.
2. Activity and gradients increase τ.
3. Higher τ locks in patterns by reducing diffusion.
4. Locked-in patterns continue to draw on N until it depletes or equilibrates.

Later versions (v4/v5) keep the same conceptual architecture, but treat the nutrient module as configurable:
- simple uniform N (constant background),
- diffusive N with consumption and replenishment (as above).

---

### M4. Memory Kernels and Multi-τ Extensions (v4–v5)

To move beyond a simple instantaneous coupling, v4 and v5 introduce **memory fields** and, optionally, multiple τ species.

#### M4.1. Memory fields

Rather than feeding $$\(S\)$$ directly into τ, we define one or more **memory accumulators**:

- Single memory:

  $$\text{mem}_{t+1} = (1 - \lambda)\text{mem}_t + |R_t|$$

- Multiscale memory:

$$\text{mem}^{\mathrm{fast}}_{t+1} = (1 - \lambda_f)\text{mem}^{\mathrm{fast}}_t + |R_t|$$

$$\text{mem}^{\mathrm{slow}}_{t+1} = (1 - \lambda_s)\text{mem}^{\mathrm{slow}}_t + |R_t|$$


where $$\(\lambda_f > \lambda_s\)$$. These are then combined into an effective source for τ:

$$
S(x,y,t) = w_f \,\text{mem}^{\mathrm{fast}} + w_s \,\text{mem}^{\mathrm{slow}}
$$

This structure allows τ to respond both to **recent activity** and to **long-term history**, acting as a tunable **learning kernel** over past dynamics.

#### M4.2. Multiple τ species

In some v5 configurations, we extend $$\(\tau\)$$ to $$\(\{\tau_1, \tau_2, \dots\}\)$$, each with its own parameters $$\((\alpha_i, \beta_i, \gamma_i, \kappa_{\tau_i})\)$$. Each τ-field sees the same underlying chemical activity but reacts with its own timescale and smoothing.

This defines **multi-layered time scaffolds**:
- one τ may respond quickly and locally,
- another slowly and diffusively,
- together shaping more complex material-like behaviour.

---

### M5. Numerical Setup and Parameter Sweeps

All simulations are run on a periodic 2D lattice:

- Typical grid sizes:
  - v2-style: \(128 \times 128\),
  - v3–v5 sweeps: \(150 \times 150\) and similar.
- Spatial step: \(\Delta x = \Delta y = 1\).
- Time steps:
  - v2: \(\Delta t = 0.5\),
  - sweeps: typically \(\Delta t = 0.01\)–0.05.
- Integration: explicit Euler.

Initial conditions:
- \(A\) initialised near 1 with small noise.
- \(B\) seeded in a central disk (e.g. radius 8–10) with elevated concentration.
- \(N\) initialised to a background value, optionally with a central hotspot.

#### M5.1. Global sweeps

For v3/v4/v5, we perform multi-dimensional parameter sweeps over:
- time–memory parameters: \(\alpha, \beta, \gamma, \kappa_\tau\),
- Gray–Scott feed/kill: \(f\), \(k\),
- nutrient usage, τ-noise amplitude, memory decay rates,
- and discrete flags (e.g. multiscale memory on/off, multi-τ on/off).

Each configuration is run to a fixed number of steps, with:
- regular logging into `metrics.csv`,
- snapshots of \(B\), \(\tau\), \(N\) written as PNGs,
- meta-data (the full `cfg`) stored in `meta.json` / `summary.json`.

#### M5.2. Q-ridge refinement

From global sweeps, we identify a **coherence ridge** (“Q-ridge”) where:
- mean coherence is high,
- entropy is relatively low,
- τ-structure is neither too flat nor too turbulent.

We then perform **focused sweeps** in this region (v5 Q-ridge), sampling more densely in:
- \(f\) (feed),
- \(k\) (kill),
- τ-diffusion \(\kappa_\tau\),
- memory and nutrient parameters.

This two-stage process (global → ridge refinement) allows us to:
- first find *where* life-like behaviour is possible,
- then study *how* it varies under small perturbations.

---

### M6. Analysis Pipeline and Metrics

All runs are aggregated into summary tables (e.g. `runs_summary_v5.csv`, `runs_summary_v5_qridge.csv`), and we compute the following key metrics per run:

- **Mean coherence**  
  For each time sample:
  $$
  M = A + iB, \quad
  C(t) = \langle |M|^2 \rangle
  $$
  We track:
  - \( \overline{C} \): time-average coherence,
  - `coherence_slope`: linear trend of \(C(t)\) over time,
  - `std_coherence`: standard deviation of \(C(t)\).

- **Entropy of B**  
  Shannon entropy of the B field:
  $$
  S_{\mathrm{entropy}}(t)
    = -\sum_i p_i \log p_i
  $$
  where \(p_i\) is the normalised, non-negative B distribution.

- **Energy**  
  A simple energy-like quantity:
  $$
  E(t) = \frac{1}{2} \langle A^2 + B^2 \rangle
  $$

- **Autocatalysis**  
  Spatial mean of the reaction term:
  $$
  \langle A B^2 \rangle
  $$

- **τ structure metrics**  
  - `tau_var_final`: variance of τ in the final frame,
  - `tau_grad2_final`: mean squared gradient of τ in the final frame.

These metrics are used both to **map phase behaviour** and to build composite scores (e.g. `proto_life_score_v5`) combining coherence, entropy, autocatalysis, and τ-structure into a single life-likeness indicator.

---

### M7. Morphological Metrics: Boundaries and Internal Reorganisation

To move beyond global averages and capture **cell-like structure**, we define additional spatial metrics based on the B snapshots.

#### M7.1. Boundary persistence (maintenance IoU)

For each run, we choose a **mid** and **final** B snapshot (e.g. middle and last in the series). We define binary masks:

- \(M_{\mathrm{mid}} = (B_{\mathrm{mid}} > \theta)\),
- \(M_{\mathrm{final}} = (B_{\mathrm{final}} > \theta)\),

with a fixed intensity threshold \(\theta\) (e.g. 0.3 in normalised units).

The **maintenance IoU** is:

$$
\mathrm{IoU}
  = \frac{|M_{\mathrm{mid}} \cap M_{\mathrm{final}}|}
         {|M_{\mathrm{mid}} \cup M_{\mathrm{final}}|}
$$

Values near 1 indicate a **persistent cell outline**; lower values indicate drift, break-up or dissolution.

#### M7.2. Internal reorganisation index

To quantify **internal morphogenesis** inside a stable cell, we:

1. Define a **cell mask** as the union:
   $$
   M_{\mathrm{cell}} = M_{\mathrm{mid}} \cup M_{\mathrm{final}}
   $$

2. Extract interior B-values:
   - \(v_{\mathrm{mid}} = B_{\mathrm{mid}}[M_{\mathrm{cell}}]\),
   - \(v_{\mathrm{final}} = B_{\mathrm{final}}[M_{\mathrm{cell}}]\).

3. Compute the Pearson correlation \(r\) between \(v_{\mathrm{mid}}\) and \(v_{\mathrm{final}}\).

The **internal reorganisation index** is defined as:

$$
\mathrm{IRI} = 1 - r
$$

- \(\mathrm{IRI} \approx 0\): interior almost unchanged (fossil-like),
- intermediate \(\mathrm{IRI}\): **breathing / rebalancing interior**,
- high \(\mathrm{IRI}\): substantial internal remodelling (pre-dissolution or phase transition).

#### M7.3. Center-of-mass shift

To detect gross movement of mass within the cell, we compute the B-weighted centre-of-mass inside \(M_{\mathrm{cell}}\) at mid and final times:

$$
(\bar{y}_{\mathrm{mid}}, \bar{x}_{\mathrm{mid}}), \quad
(\bar{y}_{\mathrm{final}}, \bar{x}_{\mathrm{final}})
$$

and define:

$$
\mathrm{com\_shift}_B
  = \sqrt{(\bar{y}_{\mathrm{final}} - \bar{y}_{\mathrm{mid}})^2
        + (\bar{x}_{\mathrm{final}} - \bar{x}_{\mathrm{mid}})^2}
$$

This helps distinguish:
- internal rearrangement **within** a stationary body,
- from translation, drift, or fragmentation **of the body itself**.

#### M7.4. Coherence oscillation index

From the temporal coherence series \(C(t)\), we:

1. Fit a linear trend \(C(t) \approx a t + b\),
2. Subtract the trend to obtain residuals \(C_{\mathrm{res}}(t)\),
3. Define:

$$\mathrm{coherence\_osc\_index} = \mathrm{Var}(C_{\text{res}}(t))$$


This captures **“breathing” in coherence**: runs with non-zero oscillations around a trend exhibit ongoing internal activity even if the global coherence is stable.

---

### M8. Environmental Perturbation Experiments

To test **robustness and identity** of selected τ-organisms, we performed dedicated re-runs with modified environments.

#### M8.1. Candidate selection

From the Q-ridge summary, we selected three prototypes:

1. A **breathing cell**:
   - high maintenance IoU,
   - moderate internal reorganisation,
   - non-trivial coherence oscillation.

2. A **crystallising cell**:
   - high maintenance IoU,
   - very low internal reorganisation,
   - weak oscillation.

3. A **melting / overdriven state**:
   - high internal reorganisation,
   - faster coherence decay,
   - rough τ-structure.

Each candidate corresponds to a specific run directory under `outputs/dynamic_tau_v5_qridge/`.

#### M8.2. Environment variants

For each candidate, we:

1. Loaded the original configuration `cfg` from `meta.json`.
2. Constructed modified configs by scaling feed and kill:

   - `baseline`:
     - original `feed`, `kill` (unchanged)
   - `feed_low`:
     - `feed ← 0.8 × feed`
   - `feed_high`:
     - `feed ← 1.2 × feed`
   - `kill_low`:
     - `kill ← 0.8 × kill`
   - `kill_high`:
     - `kill ← 1.2 × kill`

3. Re-ran `dynamic_tau_v5.run_simulation(cfg, outdir)` from the **same initial conditions** for each variant, storing results under:

   - `outputs/dynamic_tau_v5_env/<hash>/<variant>/`

#### M8.3. Similarity measures vs baseline

For each candidate and variant, we compared the final B snapshot of the variant, \(B_{\mathrm{var}}^{\mathrm{final}}\), to the baseline final B, \(B_{\mathrm{base}}^{\mathrm{final}}\), using:

1. **Shape similarity (IoU vs baseline)**  
   As in M7.1, but now:
   - masks are derived from \(B_{\mathrm{base}}^{\mathrm{final}}\) and \(B_{\mathrm{var}}^{\mathrm{final}}\),
   - IoU quantifies whether the **cell outline** survives across environmental change.

2. **Internal similarity (correlation vs baseline)**  
   On the union mask of both final snapshots, we compute the Pearson correlation of B intensities:

   $$
   r_{\mathrm{final}} =
   \mathrm{corr}\Big(B_{\mathrm{base}}^{\mathrm{final}},
                    B_{\mathrm{var}}^{\mathrm{final}}\Big)
   $$

   This measures whether the **internal organisation** of the cell is preserved, reorganised, or fundamentally replaced under the new environment.

High IoU with moderate or low correlation corresponds to **same body, internally adapted**.  
Low IoU indicates loss or deformation of the organism’s spatial identity.

---

### M9. Relation to Material Form

Operationally, the model describes a **material-like medium** as follows:

- The **time-density field τ** acts as a *soft scaffold*:
  - τ pockets are regions where diffusion is slowed and past activity is stored,
  - they behave as **proto-material domains**: stiffer, more persistent regions embedded in a more fluid background.

- The chemical fields \(A, B\) play the role of:
  - **occupation** or **activation** of this scaffold,
  - giving visible “shape” to structures supported by τ.

- Nutrient \(N\) functions as a **resource field**:
  - feeding patterns that can hold themselves together in τ,
  - enforcing a coupling between **structure**, **history**, and **energy flow**.

By sweeping parameters and classifying behaviours with the metrics above, we obtain not only **life-like morphologies** but a **taxonomy of material regimes**:
- homeostatic, self-maintaining “cells”,
- rigid, fossil-like τ-crystals,
- and overdriven, foamy states at the edge of dissolution.

These behaviours emerge from the same set of PDEs and differ only by parameter and environment, suggesting that the model is not just a pattern generator but a **candidate minimal physics for adaptive material form in a time-density medium**.
