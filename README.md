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
---

## 4. Setup

### 4.1 Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 Install dependencies
If you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

If not, you’ll typically need:
```bash
pip install numpy scipy pandas matplotlib imageio tqdm
```

Some systems require an extra codec package for GIF/MP4, but `imageio` generally works out-of-the-box.

---

## 4.2) Core directory layout

Typical important paths:

- `simulations/`
  - `dynamic_tau_*.py` simulators (PDE stepper + state saving)
  - `run_search_*.py` runners (search driver: init/rand/qd, scoring, logging)
- `outputs/<run_name>/`
  - `boqd_log.csv` / `metrics.csv` (summary rows per run)
  - `qd_elites.json` / `best_so_far.json` (best candidates)
  - `init/`, `rand/`, `qd/` (per-run folders with `meta.json` + state snapshots)
- `outputs/<run_name>/replay/`
  - replayed top candidates with dense snapshots and rendered GIFs

---

## 4.3) Running a search

### 4.1 Basic “v7” search command (example)
Run from repo root:

```bash
python simulations/run_search_v7_minimal_w_acf_v25.py   --out_root outputs/v25_example   --workers 8 --budget 400 --init_random 80   --bins 24   --steps 3000 --nx 150 --ny 150   --dt 0.01 --log_every 20   --w_enabled 1 --w_gate 0.01 --w_tau_gain_max 0.15   --osc_fmin 0.05 --osc_fmax 1.5 --osc_min_cycles 2.0   --seed 1
```

### 4.2 Switching initial condition modes (examples)

**Rings**
```bash
python simulations/run_search_v7_minimal_w_acf_v25.py   --out_root outputs/v25_rings   --init_mode ring   --seed_count 3 --seed_pos_mode random   --seed_radius 22 --seed_ring_width 4   --workers 8 --budget 400 --init_random 80   --bins 24 --steps 3000 --nx 150 --ny 150 --dt 0.01 --log_every 20   --w_enabled 1 --w_gate 0.01 --w_tau_gain_max 0.15   --osc_fmin 0.05 --osc_fmax 1.5 --osc_min_cycles 2.0   --seed 2
```

**Stripes**
```bash
python simulations/run_search_v7_minimal_w_acf_v25.py   --out_root outputs/v25_stripes   --init_mode stripes   --workers 8 --budget 400 --init_random 80   --bins 24 --steps 3000 --nx 150 --ny 150 --dt 0.01 --log_every 20   --w_enabled 1 --w_gate 0.01 --w_tau_gain_max 0.15   --osc_fmin 0.05 --osc_fmax 1.5 --osc_min_cycles 2.0   --seed 3
```

**Gaussian blobs / packets (seed diversity)**
```bash
python simulations/run_search_v7_minimal_w_acf_v25.py   --out_root outputs/v25_gaussian_packets   --init_mode gaussian   --seed_count 6 --seed_pos_mode random   --seed_sigma 6   --workers 8 --budget 600 --init_random 120   --bins 32 --steps 4000 --nx 180 --ny 180 --dt 0.01 --log_every 20   --w_enabled 1 --w_gate 0.01 --w_tau_gain_max 0.15   --osc_fmin 0.05 --osc_fmax 1.5 --osc_min_cycles 2.0   --seed 4
```

> Use `python simulations/<runner>.py --help` to see which init flags are supported in your current runner.

---

## 4.4) Understanding outputs

### 4.1 Top-level summary files
In `outputs/<run_name>/` you’ll usually see:
- `boqd_log.csv` and/or `metrics.csv`: one row per attempted run
- `qd_elites.json`: elite archive (best per bin)
- `best_so_far.json`: global best candidate

### 4.2 Per-run folders
In `outputs/<run_name>/{init,rand,qd}/run_*/` you’ll typically get:
- `meta.json`: parameters, descriptors, score, and diagnostics
- `state_mid.npz` / `state_final.npz`: mid/final snapshots

If `save_snapshots` is enabled during the search, you may also see dense sequences, but for large searches it’s common to only save mid/final and replay later.

---

## 5) Replaying top runs with dense snapshots

The search phase is optimized for *coverage*. Replay is where you generate *high-quality time series* for visualization.

### 5.1 Replay top-K
Make sure `PYTHONPATH` can find the simulator modules:

```bash
PYTHONPATH=simulations python replay_runs_with_snapshots_patched_v2.py   --out_root outputs/v25_example   --top_k 50   --include_best 1   --snap_every 25   --snapshot_format npz   --clean 1
```

This creates:
- `outputs/v25_example/replay/<run_name>/...` with dense snapshots

### 5.2 Snapshot formats
Depending on the simulator version, replay may save:
- combined `state_000025.npz` files **or**
- per-field files like `A_000025.npy`, `B_000025.npy`, `tau_000025.npy`

Both are supported by the latest visualizer (see below).

---

## 6) Rendering montages and GIFs

### 6.1 Render GIFs from replay outputs
```bash
python make_viz_from_runs_v8.py   --out_root outputs/v25_example/replay   --top_k 50   --field B   --make_gifs 1   --per_run_scale 0
```

Recommended fields:
- `B` often shows boundaries/structure crisply
- `tau` shows the memory scaffolding
- `A` can be less visually salient depending on parameters

### 6.2 If you only get 2-frame GIFs
Two-frame GIFs usually mean the visualizer is only finding:
- `state_mid.npz` and `state_final.npz`

Check whether dense snapshots exist:
```bash
RUN="outputs/v25_example/replay/<some_run_dir>"
ls -1 "$RUN" | head
ls -1 "$RUN"/state_*.npz 2>/dev/null | wc -l
ls -1 "$RUN"/A_*.npy 2>/dev/null | wc -l
ls -1 "$RUN"/B_*.npy 2>/dev/null | wc -l
```

If you see `A_*.npy`/`B_*.npy` but no `state_*.npz`, render using `--field B` (or `A`/`tau`) and ensure you’re using `make_viz_from_runs_v8.py` or later.

---

## 7) Common troubleshooting

### 7.1 `ModuleNotFoundError: No module named 'dynamic_tau_v7'`
This almost always means `PYTHONPATH` is not set to include `simulations/`.

Fix:
```bash
PYTHONPATH=simulations python replay_runs_with_snapshots_patched_v2.py ...
```

Or export once per shell:
```bash
export PYTHONPATH=simulations
```

### 7.2 “QD folder didn’t appear”
If the run never reaches QD phase, check `boqd_log.csv` / `metrics.csv` for errors.

Common cause: runner crashed inside diagnostics. Look for an `error` column and count non-empty values.

### 7.3 “Unrecognized arguments” errors
Runner interfaces change across versions. Always confirm flags with:
```bash
python simulations/run_search_v7_minimal_w_acf_v25.py --help
```

### 7.4 Replay writes `.npy` even when `snapshot_format=npz`
Some simulator versions implement replay snapshots as per-field `.npy` (A/B/tau), regardless of the `snapshot_format` request. This is OK as long as the visualizer supports it (v8+).

---

## 8) Suggested workflow (repeatable)

1. Run a search:
   - `outputs/<name>/` is created
2. Inspect top-level logs:
   - ensure `error rows: 0`
3. Replay top runs:
   - generates `outputs/<name>/replay/`
4. Render GIFs:
   - run `make_viz_from_runs_v8.py`
5. Curate:
   - pick representative runs per attractor class for writeups

---

## 9) Reproducibility tips

- Use `--seed` for the search driver.
- Keep `meta.json` from interesting runs; it contains the configuration needed to replay.
- If you change runner versions, keep output roots separate (`outputs/v25_*`, `outputs/v26_*`, etc.).

---

## 10) Quick “smoke test” commands

### Verify search output has no errors
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/v25_example/boqd_log.csv")
err = df["error"].notna().sum() if "error" in df.columns else 0
print("rows:", len(df), "error_rows:", err)
print("methods:", df["method"].value_counts().to_dict() if "method" in df.columns else None)
PY
```

### Verify replay has dense snapshots
```bash
RUN="outputs/v25_example/replay"
find "$RUN" -maxdepth 2 -type f -name "B_*.npy" | head
find "$RUN" -maxdepth 2 -type f -name "state_*.npz" | head
```

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
## V7 “Measurement Calculus” Update: Universal `w`-conditioned metrics + crosstalk control

V7 shifts the search/analysis stack from a “zoo of scalar metrics” to a **single shared measurement calculus** computed from one organism indicator. The goal is to **handle crosstalk *before* computation**, so descriptors and scores become *different projections of the same conditioned process* rather than competing objectives.

> **Pipeline:** Field → weighted signals → shared viability window → orthogonalized functionals

---

### 1) One shared organism measure inside the sim

We define a soft organism indicator field `w(x,t)` (or equivalently use the logged `w_mean(t)` as its global summary). The key idea is that *all* downstream signals are computed as **conditional expectations under `w`**, rather than over the whole dish/grid.

A typical soft indicator:
$$\[
w(x,t)=\sigma\!\left(\frac{B(x,t)-b_0}{b_w}\right)
\]$$
where $$\(\sigma\)$$ is a sigmoid-like soft threshold.

This induces “body-conditioned” statistics for any field $$\(f(x,t)\)$$:

$$\[
E_w[f](t)=\frac{\sum_x w(x,t)f(x,t)}{\sum_x w(x,t)+\varepsilon},\quad
\mathrm{Var}_w[f](t)=E_w[f^2](t)-E_w[f](t)^2
\]$$

**Why this kills crosstalk:**  
Metrics like entropy/coherence computed on the full grid are dominated by background. Conditioning on `w` makes them “about the organism,” not the dish—so growth doesn’t trivially improve every other metric.

---

### 2) A shared viability window *before* any metric is computed

Rather than compute metrics on all timesteps equally, we define a soft time-weight \(W(t)\) from mass-like signal \(M(t)=\langle w\rangle\) (or directly from `w_mean(t)`):

$$\[
W(t)=\sigma\!\left(\frac{M(t)-M^\*}{s}\right)
\]$$

Optionally add a survival gate relative to early mass \(M_\text{ref}\):
$$\[
S(t)=\sigma\!\left(\frac{M(t)-\alpha M_\text{ref}}{s}\right)
\]$$

Then the effective analysis window is:
$$\[
\widetilde{W}(t)=W(t)\cdot S(t)\cdot \text{taper}(t)
\]$$

**Everything uses $$\(\widetilde{W}(t)\)$$.** This is the “handle crosstalk before computation” step.

---

### 3) A minimal basis of conditioned signals (small, interpretable)

We aim for a small set of channels spanning “life-like” behaviors:

- **Growth channel:** scale-free log mass  
  $$\[
  g(t)=\log(M(t)+\varepsilon)
  \]$$
- **Oscillation (“breathing”) channel:** pick a scalar that should oscillate when internal dynamics stabilize. In minimal V7 we use time series derived from `w_mean(t)`; richer variants can use conditioned tau-structure like $$\(E_w[\|\nabla\tau\|^2]\)$$.
- **Reorganization channel:** snapshot distance computed in the same `w`-measure (optional but recommended).
- **Complexity channel:** conditioned entropy / spectral entropy (optional regularizer).

---

### 4) Universal metric operators (the only “allowed” functionals)

Instead of inventing bespoke metrics, V7 uses a small library of operators—each applied consistently under \(\widetilde{W}(t)\):

**Operator A — weighted slope (growth)**
$$\[
\text{slope}_{\widetilde{W}}(g)=\arg\min_a\sum_t\widetilde{W}(t)\big(g(t)-(at+b)\big)^2
\]$$

**Operator B — oscillation lock-in via PSD peakiness (preferred)**
Compute PSD $$\(P(f)\)$$ on a windowed/detrended signal $$\(s(t)\)$$. Then:

- **Peak ratio (“lock-in”)**
  $$\[
  \text{peak\_ratio}(s)=\frac{\max_{f\in B}P(f)}{\mathrm{median}_{f\in B}P(f)+\varepsilon}
  \]$$
  Often reported as:
  $$\[
  \log\_{{\rm peak}}=\log(\text{peak\_ratio})
  \]$$
- **Peak frequency**
  $$\[
  f_{\rm peak}=\arg\max_{f\in B}P(f)
  \]$$

**Operator C — conditional distance (reorg/maintenance)**
Weighted correlation distance between snapshots $$\(B_1, B_2\)$$:
$$\[
D(B_1,B_2)=1-\mathrm{corr}_w(B_1,B_2)
\]$$

**Operator D — conditioned entropy / spectral entropy (optional)**
Entropy of a weighted histogram or spectral entropy of a conditioned PSD.

---

### 5) Explicit decomposition: remove shared variance (“orthogonalize”)

The step most pipelines skip: **orthogonalize secondary channels against mass** so they don’t just measure “big organisms do more of everything.”

Example: oscillation often correlates with mass. We regress $$\(s(t)\)$$ on $$\(g(t)\)$$ within the viable window and use the residual:
$$\[
s_\perp(t)=s(t)-\hat{a}g(t)-\hat{b}
\]$$
Then compute oscillation metrics on $$\(s_\perp(t)\)$$, not $$\(s(t)\)$$.

This is still pure math (weighted least squares) and it makes metrics *stop fighting*.

---

### 6) Practical V7 stack (growth + oscillation, minimal steering)

This is the minimal set that stays “fluid / analog” (supports subtle incremental changes):

- **Shared:** `w_mean(t)` (proxy for $$\(M(t)\))$$, viability window $$\(\widetilde{W}(t)\)$$
- **Score:** growth functional (weighted slope of `log(w_mean + eps)`)
- **Osc diagnostics:** `f_peak`, `peak_ratio`, `log_peak_ratio`, plus a continuity `osc_ratio` if desired
- **Descriptors:** typically `(growth_sigmoid, osc_sigmoid)` where `osc_sigmoid` is based on `log_peak_ratio` (peakiness), not tiny band-energy fractions
- **Feedback parameter:** `w_tau_gain` (and related `w_*` toggles) are logged and treated as first-class controls for “self-maintaining” region formation

---

### 7) What’s new in V7 (summary)

- **Universal life signal:** use `w` / `w_mean(t)` as the shared organism measure.
- **Shared viability window:** gate *all* metrics consistently in time.
- **Oscillation “lock-in”:** add PSD peakiness + `f_peak` (more sensitive than band/total energy).
- **Crosstalk control:** optional residualization against log-mass to make oscillation/complexity “pure axes.”
- **Minimal steering:** focus search pressure on stable regimes without needing ad-hoc metric piles.

---

### 8) Immediate sanity probes (recommended)

To verify the pipeline is behaving (and not a resolution/bug artifact):

- Print a 10-row sample table per sweep:
  `f_peak, peak_ratio, log_peak_ratio, osc_ratio, w_tau_gain`
- Ensure `f_peak` is *not* constant across all runs (otherwise frequency resolution/banding is dominating).
- Ensure `w_tau_gain` is nonzero/variable during evaluation and matches the CSV (otherwise activation/logging mismatch).




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

