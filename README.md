# Dynamic-τ Reaction–Diffusion (Time-Density Physics)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17856677.svg)](https://doi.org/10.5281/zenodo.17856677)

> **“Matter is the memory of change — life is memory learning to organize itself.”**

This repository explores a hypothesis: that adding a **memory / time‑density field** `τ(x,y,t)` to a standard reaction–diffusion system can reshape the attractor landscape and produce **persistent, recurrent, bounded structures** (rings, packets, multi‑compartments, stripes, etc.).

It includes:
- **Simulators**: Gray–Scott-style reaction–diffusion + dynamic `τ` (and optional resource `N`)
- **Search drivers**: BO/QD (MAP‑Elites) / hybrid loops for discovering interesting regimes
- **Diagnostics**: oscillation/recurrence (v7 uses ACF), stability gates, morphology descriptors
- **Replay + visualization**: replay top runs with dense snapshots and render GIFs/montages

> ⚠️ **Research sandbox.** “Proto‑life” here refers to *dynamical signatures* (boundedness, persistence, recurrence, internal reorg), not biological claims.

---

## Quick start

For a step‑by‑step guide (install → run → replay → GIFs), see:

- **`RUNNING_GUIDE.md`**

### 1) Create an environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies (either your `requirements.txt` or the minimal set):
```bash
pip install -r requirements.txt
# or
pip install numpy scipy pandas matplotlib imageio tqdm
```

### 2) Run a search (v7-style, example)

> Script names evolve (e.g. `run_search_v7_minimal_w_acf_v25.py`). Use the latest `run_search_v7_*.py` in `simulations/`.

```bash
python simulations/run_search_v7_minimal_w_acf_v25.py   --out_root outputs/v25_example   --workers 8 --budget 400 --init_random 80   --bins 24   --steps 3000 --nx 150 --ny 150   --dt 0.01 --log_every 20   --w_enabled 1 --w_gate 0.01 --w_tau_gain_max 0.15   --osc_fmin 0.05 --osc_fmax 1.5 --osc_min_cycles 2.0   --seed 1
```

### 3) Replay the top runs with dense snapshots

Replay creates `outputs/<run>/replay/...` with a snapshot sequence every `snap_every` steps.

```bash
PYTHONPATH=simulations python replay_runs_with_snapshots_patched_v2.py   --out_root outputs/v25_example   --top_k 50   --include_best 1   --snap_every 25   --snapshot_format npz   --clean 1
```

### 4) Render GIFs / montages

```bash
python make_viz_from_runs_v8.py   --out_root outputs/v25_example/replay   --top_k 50   --field B   --make_gifs 1   --per_run_scale 0
```

**Field tips**
- `B` often shows “membrane/boundary” structure most clearly.
- `tau` shows the memory scaffolding.

**Two‑frame GIF?** That usually means the visualizer only found `state_mid` + `state_final`. Check whether dense snapshots exist (e.g. `B_000025.npy`), and ensure you’re using `make_viz_from_runs_v8.py` or later.

---

## Conceptual overview

### Time density / memory field `τ(x,y,t)`

We treat `τ` as a local “time thickness” / memory density:

- where `τ` is high, processes slow and history accumulates
- where `τ` is low, dynamics are faster and more labile

A generic `τ` evolution used across model versions:

```math
\frac{\partial \tau}{\partial t}
= \alpha\,S(x,y,t)
- \beta\,(\tau-\tau_0)
+ \gamma\,N(x,y,t)
+ \kappa_{\tau}\nabla^2\tau
+ \eta_{\tau}(x,y,t)
```

Where:
- `S(x,y,t)` is an activity / boundary source (reaction + gradients)
- `N(x,y,t)` is an optional resource field (diffuse/consume/replenish)
- `α,β,γ,κτ` control feedback, relaxation, resource coupling, and smoothing
- `ητ` is stochasticity/noise in the memory field

The defining feature is **feedback**:
1) `τ` increases where dynamics are sustained,
2) and then feeds back (typically through diffusion/reaction modulation) to stabilize or reshape those dynamics.

---

## Model versions (high level)

This project has evolved through multiple “versions” as the simulator and measurement stack matured.

- **v1–v4**: exploratory couplings (τ‑modulated diffusion, reaction, kernels). Kept for reference.
- **v5**: stable dynamic‑τ + Gray–Scott + resource `N` sweeps (“Q‑ridge” style analysis).
- **v6**: adds a **search layer** (BO/QD/hybrid) to discover diverse regimes automatically.
- **v7**: measurement + tooling upgrades:
  - ACF‑based oscillation/recurrence diagnostics (replacing FFT window locking issues)
  - seed‑diversity init modes (rings, gaussian packets, stripes, multi‑seed)
  - replay + dense snapshots + visualization pipeline
  - morphology descriptors to begin classifying attractor families

---

## Outputs: what gets written

A search `--out_root outputs/<name>` typically contains:
- `boqd_log.csv` / `metrics.csv`: one row per attempted run (params, score, descriptors, errors)
- `qd_elites.json`: MAP‑Elites archive (best run per bin)
- `best_so_far.json`: global best candidate

Per-run directories:
- `outputs/<name>/{init,rand,qd}/run_*/meta.json`
- `state_mid.npz`, `state_final.npz`

Replay directories:
- `outputs/<name>/replay/<run_name>/...` with dense snapshots.

Depending on simulator settings, dense snapshots may be saved as:
- combined `state_000025.npz`, **or**
- per‑field sequences like `A_000025.npy`, `B_000025.npy`, `tau_000025.npy`

The visualizer (v8+) supports both layouts.

---

## Repository map (recommended)

Your local repo may differ, but a typical structure looks like:

```text
.
├── simulations/
│   ├── dynamic_tau_v5.py
│   ├── dynamic_tau_v6.py
│   ├── dynamic_tau_v7*.py
│   ├── run_sweep_v5*.py
│   └── run_search_v6.py / run_search_v7*.py
├── analysis/
│   └── (optional sweep/search analyzers)
├── replay_runs_with_snapshots_patched_v2.py
├── make_viz_from_runs_v8.py
├── RUNNING_GUIDE.md
├── README.md
└── outputs/   (generated)
```

---

## Citation

If you use this work in academic writing or derivative software, please cite the Zenodo archive:

- DOI: https://doi.org/10.5281/zenodo.17856677

(See Zenodo for the recommended citation string.)

---

## Roadmap

Near-term directions:
- richer morphology descriptors (ringness, stripe anisotropy, filament measures)
- better “behavior space” descriptors for QD (avoid collapse into one basin)
- robustness tests (gentle parameter shifts / perturbations)
- interacting seeds / “ecologies” (multi-compartment persistence, drift, splitting)

Longer-term ideas:
- multi-resource fields and structured environments
- multiple interacting memory fields
- links to active matter / excitable media / field-theoretic analogies

---

## License

See `LICENSE`.
