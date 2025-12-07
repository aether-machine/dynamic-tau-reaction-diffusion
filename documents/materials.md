### M10. Materials and Code

All simulations and analyses were performed using a small, self-contained Python codebase organised under the `time-density/` repository.

#### M10.1. Software environment

- **Language**: Python 3.10+ (tested also on 3.12)
- **Core dependencies**:
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `Pillow` (PIL)
- Optional but recommended:
  - `tqdm` for progress bars in long sweeps
  - `jupyter` / `ipykernel` for interactive inspection

A minimal environment can be set up with:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

pip install numpy matplotlib pandas pillow tqdm
```
All paths below assume the repository root is time-density/.

M10.2. Core simulation code

The main dynamic time-density model used for the v4/v5 experiments is implemented in:

simulations/dynamic_tau_v5.py

This module defines a single entry point:

```bash
run_simulation(cfg: dict, outdir: str) -> str
```
where:

cfg is a Python dictionary containing:

grid and numerical parameters (nx, ny, dx, dy, dt, steps)

Gray–Scott parameters (feed, kill, Da, Db)

τ parameters (tau0, alpha, beta, gamma, tau_min, tau_max, kappa_tau, tau_noise, etc.)

nutrient parameters (use_nutrient, D_N, nutrient_use, nutrient_replenish, …)

memory options (use_multiscale_memory, decay rates, weights)

multi-τ options (num_tau_species, per-species parameters)

logging / snapshot settings (snap_every, log_every)

outdir is a filesystem path where:

meta.json (or summary.json) stores the configuration,

metrics.csv stores time-series,

PNG snapshots (B_snapshot_*.png, tau_snapshot_*.png, N_snapshot_*.png) are written.

Early exploratory models (e.g. tau_reaction_diffusion_v2.py) are kept in the repo as archival code illustrating the evolution from a simple single-τ model to the full v5 architecture, but all quantitative results in this document refer to dynamic_tau_v5.py and its sweeps.

M10.3. Parameter sweeps

Parameter sweeps are driven by standalone scripts in simulations/:

simulations/run_sweep_v5.py

Constructs a list of configuration dictionaries (cfg objects) sampling:

feed, kill

τ parameters (alpha, beta, gamma, kappa_tau, tau_noise)

nutrient and memory flags

Optionally restricts sampling to a Q-ridge region around previously identified high-coherence / low-entropy configurations.

Supports parallel execution via the Python multiprocessing module.

For each cfg, calls:

```bash
dynamic_tau_v5.run_simulation(cfg, outdir)
```
Outputs are stored under:
```bash
outputs/dynamic_tau_v5/...
outputs/dynamic_tau_v5_qridge/...
```
Typical invocation (from repository root):
```bash
source .venv/bin/activate

# Global sweep
python simulations/run_sweep_v5.py --mode global --workers 4

# Q-ridge refinement sweep
python simulations/run_sweep_v5.py --mode qridge --workers 4

```
The exact CLI options (--mode, --workers, etc.) are documented in the script header and argument parser.

M10.4. Aggregate analysis and scoring

High-level aggregation and scoring of runs are handled by:

analyze_proto_life_v5.py

This script:

Scans a chosen output root, e.g.:
