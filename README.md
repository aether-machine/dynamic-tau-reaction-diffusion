# Time-Density Physics: From Reaction–Diffusion to Proto-Life

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17856677.svg)](https://doi.org/10.5281/zenodo.17856677)

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


$$\frac{\partial \tau}{\partial t} = \alpha\,S(x,y,t)$$
$$-\beta\,(\tau - \tau_0)$$
$$+\gamma\,N(x,y,t)$$
$$+\kappa_\tau \nabla^2 \tau$$
$$+\eta_\tau(x,y,t)$$

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

- activity $$\( S(x,y,t) \)$$ built from $$\( |A B^2| \)$$ and $$\( |\nabla B| \)$$,
- optional **memory kernels** (`mem_fast`, `mem_slow`) that integrate activity over time,
- an optional **resource field** $$\(N(x,y,t)\)$$ that diffuses, is consumed, and replenished:

$$\frac{\partial N}{\partial t} = D_N \nabla^2 N$$
$$-\eta\,N B$$
$$+\rho$$

The result is a **time-density medium** where:

- τ **remembers** sustained dynamics,
- τ **sculpts** the landscape by slowing diffusion where memory accumulates,
- N acts as a **resource abstraction layer** controlling how easily τ can thicken.

---

## What’s new in v6

v6 adds a search layer on top of the simulator:

- **BO** (Bayesian optimization) to push score upward.
- **QD / MAP‑Elites** to intentionally harvest *diverse* regimes.
- A **hybrid BO+QD loop** (probabilistically picks BO vs QD per run).
- A sharper **“proto‑life score v2”** designed to avoid “IoU dominates everything”, while still requiring viability.

Key scripts:

- `simulations/dynamic_tau_v6.py` — the simulator (reaction–diffusion + dynamic τ).
- `simulations/run_search_v6.py` — BO / QD / hybrid search driver.
- `simulations/analyze_search_v6.py` — aggregates results and produces plots + tables.

---

## Repository layout (high level)

```
simulations/
  dynamic_tau_v5.py
  dynamic_tau_v6.py
  run_sweep_v5.py
  run_sweep_v5_qridge.py
  run_search_v6.py
  analyze_search_v6.py

outputs/
  ... (generated; safe to delete / regenerate)
```

---

## Setup

Minimal dependencies used by the v6 pipeline:

- Python 3.9+
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow` (used by some sim rendering paths)

A typical environment setup:

```bash
python -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib pillow
```

---

## Metrics glossary (the “proto‑life” signals)

The v6 search uses the same core metrics you’ve been tracking:

### `maintenance_iou`
A **viability / persistence** proxy: overlap between a mid‑simulation structure and the final structure.

- Higher → the macroscopic pattern persists.
- Lower → the system collapses, diffuses away, or explodes into noise.

### `internal_reorg_index`
An **internal reorganization** proxy: how much internal rearrangement happens while the outer form may remain.

- Higher → richer internal change.
- Too high can mean violent rearrangement that destroys viability.

### `coherence_osc_index`
A **coherent oscillation** proxy: is there sustained rhythmic / coherent activity (not just noise)?

- Higher → more sustained oscillatory dynamics.

### `mean_entropy` and `mean_coherence`
Optional “texture” metrics used for diagnostics / ranking / plotting.

- Entropy helps distinguish structured dynamics vs noise.
- Coherence helps distinguish coordinated oscillation vs uncorrelated flicker.

---

## v6 Search driver (BO / QD / hybrid)

### 1) Basic run

```bash
python simulations/run_search_v6.py \
  --out_root outputs/dynamic_tau_v6_search \
  --workers 8 \
  --budget 200 \
  --init_random 40 \
  --mode hybrid \
  --p_qd 0.7 \
  --nx 150 --ny 150 \
  --steps 3000
```

### 2) Recommended: QD “knobs” exposed

These tune the **behavior space mapping** (descriptors → bins) and how hard QD pushes for novelty.

Example (matches the tuned run you posted):

```bash
python simulations/run_search_v6.py \
  --out_root outputs/dynamic_tau_v6_search_tuned \
  --workers 8 --budget 200 --init_random 40 \
  --mode hybrid --p_qd 0.7 \
  --qd_bins 12 \
  --qd_maint_max 0.35 \
  --qd_reorg_scale 0.05 \
  --qd_osc_scale 1e-5 \
  --qd_mix_osc 0.3 \
  --qd_sigma 0.14 \
  --qd_flip_prob 0.35 \
  --steps 3000 --nx 150 --ny 150
```

What these mean (conceptually):

- `--qd_bins`: grid resolution in behavior space (MAP‑Elites grid is `bins × bins`).
- `--qd_maint_max`: sets how maintenance IoU is normalized into descriptor space.
- `--qd_reorg_scale`, `--qd_osc_scale`: control how reorg/osc are squashed into `[0,1]` (think “soft normalization”).
- `--qd_mix_osc`: mixes reorg and oscillation into the second descriptor:
  - `0.0` → descriptor 2 is “pure reorg”
  - `1.0` → descriptor 2 is “pure oscillation”
- `--qd_sigma`: adds Gaussian noise in descriptor space to prevent collapse into a tiny region.
- `--qd_flip_prob`: occasionally swaps descriptor axes (another anti‑collapse trick).

### 3) Score version

The search driver can rank runs using a score function.

- `v1` is a simple weighted blend.
- `v2` adds **a soft viability gate** + **saturation** so IoU can’t dominate everything.

Example:

```bash
python simulations/run_search_v6.py \
  --out_root outputs/dynamic_tau_v6_search_scorev2 \
  --workers 8 --budget 200 --init_random 40 \
  --mode hybrid --p_qd 0.7 \
  --qd_bins 12 --qd_maint_max 0.35 --qd_reorg_scale 0.05 --qd_osc_scale 1e-5 --qd_mix_osc 0.3 \
  --qd_sigma 0.14 --qd_flip_prob 0.35 \
  --score_version v2 \
  --score_iou_gate 0.08 \
  --score_iou_width 0.02 \
  --score_iou_sat_scale 0.10 \
  --steps 3000 --nx 150 --ny 150
```

Interpretation of the main v2 knobs:

- `--score_iou_gate`: below this IoU the score is strongly suppressed.
- `--score_iou_width`: how soft / sharp the gate is.
- `--score_iou_sat_scale`: scale for the saturating IoU term (smaller = saturates earlier).

If you want a *hard* “must be alive” filter, push `score_iou_gate` upward.
If you want exploration of barely‑viable borderline regimes, lower it slightly.

### 4) Search modes

- `--mode bo`: always use Bayesian optimization (best for maximizing score).
- `--mode qd`: always use QD/MAP‑Elites sampling (best for diversity).
- `--mode hybrid`: each run chooses BO vs QD; `--p_qd` controls the fraction of QD runs.

### 5) Resume

If the process is interrupted:

```bash
python simulations/run_search_v6.py --out_root outputs/dynamic_tau_v6_search_tuned --resume
```

---

## What `run_search_v6.py` writes

Inside `--out_root`, you’ll typically see:

- `boqd_log.csv` — one row per run (method, params, metrics, score, descriptors, run_dir)
- `best_so_far.json` — rolling best run (score + params)
- `qd_elites.json` — MAP‑Elites archive: best run_dir per behavior cell

And per run directory (e.g. `outputs/.../qd/<id>/`):

- `cfg.json` — parameters + flags
- `metrics.csv` — scalar metrics
- `state_mid.npz`, `state_final.npz` — saved state (includes at least `B` and `tau`)
- `summary.json` — compact record used by the analyzer

---

## Analyzing v6 runs

After a search completes (or during, if you want live snapshots), run:

```bash
python simulations/analyze_search_v6.py \
  --run_dir outputs/dynamic_tau_v6_search_tuned \
  --bins 12
```

The analyzer produces:

- `descriptor_scatter.png` — points in descriptor space, colored by score
- `qd_elite_heatmap.png` — best score per MAP‑Elites cell
- `coverage_over_time.png` — how quickly the search fills behavior space
- `score_vs_metrics.png` — score vs each metric (overlay)
- `top25_by_score.csv` — fastest way to find “what to look at next”
- `analysis_quantiles.json` — quick distribution summary for sanity checks

---

## Regime characterization (a practical taxonomy)

Once you have `descriptor_scatter.png` and `qd_elite_heatmap.png`, you can label clusters into regimes.

Here’s a simple, *actionable* taxonomy that maps directly to the metrics:

### 1) Stable / static
**High `maintenance_iou`**, **low `internal_reorg_index`**, **low `coherence_osc_index`**.

- “Alive” only in the trivial sense: the pattern persists but does nothing.
- Useful as a control group / baseline.

### 2) Breathing maintenance
**High `maintenance_iou`**, **moderate reorg**, **moderate-to-high oscillation**.

- This is often the most “proto‑life‑ish” region: persistent boundary + active interior.
- Expect higher `mean_coherence` than noise‑dominated cases.

### 3) Metastable reorganization
**Moderate IoU**, **high reorg**, **mixed oscillation**.

- Pattern may hold for long periods but undergoes punctuated internal change.
- These are great for “fractal resonance” / “internal negotiation” hypotheses because transitions can be sharp.

### 4) Chaotic / noisy
**Low IoU**, often **high entropy**, often **low coherence**.

- Usually not helpful unless you’re explicitly studying noise‑driven exploration.

### How to use QD to harvest regimes intentionally

- To push toward “breathing maintenance”: increase `qd_mix_osc` (bias descriptor 2 toward oscillation) and use score v2 with a firm IoU gate.
- To harvest reorganizers: decrease `qd_mix_osc`, lower `qd_reorg_scale` (more sensitivity), and slightly relax `score_iou_gate`.
- If coverage collapses into a tiny region: increase `qd_sigma` and/or `qd_flip_prob`, and consider lowering `--p_qd` temporarily so BO can “jump” you to a new viable area.

---

## Legacy: v5 sweeps (baseline + Q‑ridge)

The older v5 workflow is still useful for controlled sweeps and for reproducing earlier results.

### Uniform / grid sweep

```bash
python simulations/run_sweep_v5.py \
  --out_root outputs/dynamic_tau_v5_sweep \
  --steps 3000 --nx 150 --ny 150
```

### Q‑ridge sweep

```bash
python simulations/run_sweep_v5_qridge.py \
  --out_root outputs/dynamic_tau_v5_qridge \
  --steps 3000 --nx 150 --ny 150
```

---

## Notes / tips

- **Budget sizing:** if behavior coverage plateaus quickly, increase `--budget` and/or increase `--qd_sigma`.
- **Low‑frequency / slow dynamics:** increase `--steps` (and consider saving snapshots more frequently).
- **Reproducibility:** if you need determinism, add a fixed seed to the search driver and to the simulator RNG paths.
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
GitHub (2025), 


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

