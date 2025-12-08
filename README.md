# Time-Density Physics: From Reaction–Diffusion to Proto-Life

> **“Matter is the memory of change — life is memory learning to organize itself.”**

This repository develops and explores a new physical principle:

> **dynamic time density** $$\( \tau(x,y,t) \)$$ as a fundamental field  
> that couples to reaction–diffusion systems, induces coherence,  
> and produces **proto-life structures**.

What began as a narrow question —

> *Can temporal density fields influence chemical and structural transformations?*

— has evolved into a general framework for understanding:

- self-organization  
- autocatalytic dynamics  
- emergent coherence  
- proto-metabolic structures  
- memory-driven morphogenesis  
- environmental robustness of “proto-organisms”

all arising from **τ-modulated feedback loops** in a continuous medium.

This repository contains the simulations, analysis pipeline, and results behind these experiments.

---

## 1. Conceptual Overview

### 1.1 Time Density (τ)

We model a **time-density field** $$\( \tau(x,y,t) \)$$ that represents local “thickness of time” or **memory density**:

- where $$\( \tau \)$$ is high, processes slow and history accumulates;
- where $$\( \tau \)$$ is low, processes are more labile and quickly forgotten.

In this picture:

- **Matter** ≈ fossilised change in a time-density field  
- **Chemistry** ≈ structured manipulation of that field  
- **Life** ≈ regions where the time field learns to reinforce its own patterns

Mathematically, τ evolves according to a feedback equation of the form:

$$\frac{\partial \tau}{\partial t}
  = \alpha\,S(x,y,t)
  - \beta\,(\tau - \tau_0)
  + \gamma\,N(x,y,t)
  + \kappa_\tau \nabla^2 \tau
  + \eta_\tau(x,y,t)
$$

where:

- $$\( S(x,y,t) \)$$ is a **local activity / memory source** (reaction + gradients),
- $$\( N(x,y,t) \)$$ is a **resource field** (“nutrient”),
- $$\( \alpha, \beta, \gamma, \kappa_\tau \)$$ control feedback, relaxation, resource coupling, and τ-smoothing,
- $$\( \eta_\tau \)$$ is τ-noise.

The **key idea** is that τ both:

1. **remembers** where interesting dynamics have happened, and  
2. **feeds back** to those dynamics by modulating diffusion and stability.

---

## 2. Simulation Progression

The project has gone through several generations of models, converging on a robust **dynamic τ + Gray–Scott** framework (v5) where proto-life behaviour is most clearly expressed.

### 2.1 Early prototypes (v1–v3)

These are kept as **archival / exploratory** code and notebooks:

- τ-modulated diffusion (temporal diffusion)
- τ-modulated reaction rates (temporal catalysis)
- τ-modulated “phase” behaviour (τ shifting effective criticality)

They established that:

- τ-gradients distort diffusion and reaction fronts,
- τ can act as a hidden “medium” that focuses or disperses activity,
- simple feedback is enough to induce complex spatial structure.

### 2.2 Dynamic τ + Reaction–Diffusion (v2–v4)

The first fully coupled PDE experiments combined:

- a Gray–Scott reaction–diffusion system for $$\(A(x,y,t), B(x,y,t)\)$$,
- a dynamic τ-field that responds to **reaction activity** and **gradients**,
- an optional resource field $$\(N(x,y,t)\)$$.

The v2 models already produced:

- τ pockets that **stabilised oscillons** (proto-cell structures),
- tubular τ filaments,
- visually striking “walling”, “eating” and “tunnelling” behaviours.

The v3–v4 runs generalised this into sweeps over τ-feedback parameters (α, β, γ) and Gray–Scott parameters (feed, kill), revealing regions in parameter space where **coherent, cell-like morphologies** are common.

---

## 3. Dynamic τ v5: Proto-Life and Q-Ridge

The current flagship model is:

- `simulations/dynamic_tau_v5.py`  
- sweeps and analyses in `simulations/run_sweep_v5.py` and `analyze_*_v5.py`.

### 3.1 Core model

The chemical subsystem is a 2-species Gray–Scott system:

- \( A(x,y,t), B(x,y,t) \) with diffusion and autocatalytic reaction \( R = A B^2 \),
- **effective diffusion** modulated by τ:

  $$D_A^{\text{eff}} = \frac{D_{A0}}{\tau + \varepsilon}, \quad
    D_B^{\text{eff}} = \frac{D_{B0}}{\tau + \varepsilon}$$

The τ equation uses an **activity + memory + resource** source:

- activity \( S(x,y,t) \) built from \( |A B^2| \) and \( |\nabla B| \),
- optional **memory kernels** (`mem_fast`, `mem_slow`) that integrate activity over time,
- an optional **resource field** $$\(N(x,y,t)\)$$ that diffuses, is consumed, and replenished:

  $$\frac{\partial N}{\partial t}
    = D_N \nabla^2 N
    - \eta N B
    + \rho$$

The result is a **time-density medium** where:

- τ **remembers** sustained dynamics,
- τ **sculpts** the landscape by slowing diffusion where memory accumulates,
- N acts as a **resource abstraction layer** controlling how easily τ can thicken.

### 3.2 Parameter sweeps and Q-ridge

Using `run_sweep_v5.py`, we perform large sweeps over:

- Gray–Scott parameters: `feed`, `kill`
- τ parameters: `alpha`, `beta`, `gamma`, `kappa_tau`, `tau_noise`
- memory options: single vs multiscale memory
- nutrient options: with/without resource coupling

Each configuration `cfg` is run via:

```python
dynamic_tau_v5.run_simulation(cfg, outdir)
```
Outputs are stored under:

- `outputs/dynamic_tau_v5/` (global sweeps)
- `outputs/dynamic_tau_v5_qridge/` (refined sweeps along coherence “ridges”)

For every run we track:

- **Coherence**  
  – mean \( \langle |A + iB|^2 \rangle \)

- **Entropy**  
  – Shannon entropy of \(B\)

- **Energy-like quantity**  
  – \( \tfrac{1}{2} \langle A^2 + B^2 \rangle \)

- **Autocatalysis**  
  – mean \( \langle A B^2 \rangle \)

- **τ structure**  
  – variance and gradient energy of τ in the final frame

From these, `analyze_proto_life_v5.py` produces summary CSVs (e.g. `runs_summary_v5_qridge.csv`), and identifies:

- high-coherence, low-entropy, τ-structured runs,

forming a **“Q-ridge”** in parameter space where proto-life behaviours concentrate.

---

### 4. From Patterns to Proto-Organisms

To move beyond “pretty patterns”, we add morphological and dynamical metrics (via `analyze_internal_v5.py`):

- `maintenance_iou`  
  – IoU of mid vs final B-activity masks  
  – *Do we keep the same body outline over time?*

- `internal_reorg_index`  
  – \( 1 - \mathrm{corr}(B_{\text{mid}}, B_{\text{final}}) \) inside the cell mask  
  – *How much does the interior reorganise while the body persists?*

- `com_shift_B`  
  – centre-of-mass shift of B  
  – *Does the mass move, drift, or split?*

- `coherence_osc_index`  
  – variance of detrended coherence time-series  
  – *Does the organism “breathe” in coherence around a trend?*

Together with τ-structure metrics, these reveal three robust regimes on the Q-ridge:

#### Breathing cells

- high boundary persistence (IoU ≈ 1)  
- moderate internal reorganisation  
- clear coherence oscillations  

→ **homeostatic proto-organisms**.

#### Crystallising cells

- extremely stable outlines and interiors  
- weak oscillation  

→ **τ-fossils**: beautiful but dynamically stiff structures.

#### Melting foam

- fragile shapes  
- large internal reorganisation  
- faster coherence decay  

→ **overdriven states** near dissolution or phase transition.

These classes are described in detail in `docs/proto_life_results.md`.

---

### 5. Environmental Perturbation Experiments

We test whether these τ-cells are just delicate patterns or genuine **attractors with identity**.

Using `run_env_tests_v5.py`, we:

- select representative Q-ridge runs (breathing, crystallising, melting)

- for each, construct environment variants:
  - `baseline` (original `feed`, `kill`)
  - `feed_low`, `feed_high`
  - `kill_low`, `kill_high`

- re-run from the same initial condition with modified parameters into:
  - `outputs/dynamic_tau_v5_env/<hash>/<variant>/`

- compare final B fields to the baseline final B:
  - `iou_vs_baseline` (shape similarity)
  - `corr_vs_baseline` (internal pattern similarity)

**Findings (informal summary):**

- **Breathing cells**  
  preserve their body outline across a band of environments,  
  while reorganising their interior under stress → proto-homeostasis.

- **Crystallising cells**  
  preserve both shape and interior almost rigidly → memory without plasticity.

- **Melting regimes**  
  often keep a rough outline but completely rewrite their interior (or dissolve) → edge-of-failure states.

These results support the interpretation that the **dynamic τ medium** supports stable, environment-sensitive proto-organisms, not just static patterns.

Details and figures are in `docs/proto_life_results.md` and `plots/proto_life_v5/`.

---

### 6. Repository Structure
```
time-density/
│
├── README.md                     ← overview and entry point
│
├── docs/
│   ├── theory_overview.md        ← conceptual background (time-density, τ)
│   ├── proto_life_results.md     ← main v5 proto-life analysis & figures
│   └── roadmap_v2.md             ← research roadmap and future directions
│
├── simulations/
│   ├── dynamic_tau_v5.py         ← core v5 time-density + Gray–Scott model
│   ├── run_sweep_v5.py           ← parameter sweeps (global + Q-ridge)
│   ├── tau_reaction_diffusion_v2.py   ← archival v2 exploratory model
│   └── (older notebooks / scripts for v1–v4, kept for reference)
│
├── analysis/
│   ├── analyze_proto_life_v5.py  ← aggregate metrics & scoring
│   ├── analyze_internal_v5.py    ← boundary / internal dynamics metrics
│   └── run_env_tests_v5.py       ← environmental perturbation experiments
│
├── outputs/
│   ├── dynamic_tau_v5/           ← raw global sweeps
│   ├── dynamic_tau_v5_qridge/    ← refined Q-ridge sweeps
│   └── dynamic_tau_v5_env/       ← environment-variant runs
│
├── plots/
│   └── proto_life_v5/
│       ├── runs_summary_v5.csv
│       ├── runs_summary_v5_qridge.csv
│       ├── runs_summary_v5_qridge_with_internal.csv
│       ├── env_tests_summary.csv
│       ├── phase maps, scatterplots
│       └── best-run montages & cross-sections
│
├── LICENSE
└── zenodo.json
```
## 7. How to Run
### 7.1 Setup
```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy matplotlib pandas pillow tqdm
```
### 7.2 Run a sweep
From repository root:
```bash
# Global sweep (example)
python simulations/run_sweep_v5.py --mode global --workers 4

# Q-ridge refinement sweep (example)
python simulations/run_sweep_v5.py --mode qridge --workers 4
```
Outputs will appear under outputs/dynamic_tau_v5/ and outputs/dynamic_tau_v5_qridge/.
### 7.3 Analyse runs
```bash
# Aggregate metrics
python analysis/analyze_proto_life_v5.py \
    --root outputs/dynamic_tau_v5_qridge \
    --out plots/proto_life_v5/runs_summary_v5_qridge.csv

# Add internal dynamics metrics
python analysis/analyze_internal_v5.py \
    --csv plots/proto_life_v5/runs_summary_v5_qridge.csv

# Environmental tests for selected candidates
python analysis/run_env_tests_v5.py \
    --candidates 40cdedd754 94392f5ff1 6fc841af45
```
Then inspect the CSVs and figures in plots/proto_life_v5/.

8. Citation

This work will be archived in Zenodo upon Version 1.0 release.

Example citation:

[Author Name], Time-Density Physics: Proto-Life from Temporal Memory Fields.
GitHub (2025), Zenodo DOI: 10.5281/zenodo.XXXXXXX (to be updated)

9. Roadmap

The next stages of this project focus on:

richer nutrient and resource dynamics (multi-resource ecologies)

geometry-coupled τ (curvature-dependent time, spatial self-sculpting)

systematic τ-noise and perturbation studies (resilience and evolution)

multiple interacting τ-species (proto-ecologies)

more elaborate memory kernels (learning systems, adaptation timescales)

possible links to experimental active media and field-theoretic models

For details, see docs/roadmap_v2.md.

“When time thickens, matter forms.
When matter remembers, life begins.”

