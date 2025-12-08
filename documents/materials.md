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

`cfg` is a Python dictionary containing:

- **Grid and numerical parameters**  
  - `nx`, `ny`, `dx`, `dy`, `dt`, `steps`

- **Gray–Scott parameters**  
  - `feed`, `kill`, `Da`, `Db`

- **τ parameters**  
  - `tau0`, `alpha`, `beta`, `gamma`, `tau_min`, `tau_max`, `kappa_tau`, `tau_noise`, etc.

- **Nutrient parameters**  
  - `use_nutrient`, `D_N`, `nutrient_use`, `nutrient_replenish`, …

- **Memory options**  
  - `use_multiscale_memory`, decay rates, weights

- **Multi-τ options**  
  - `num_tau_species`, per-species parameters

- **Logging / snapshot settings**  
  - `snap_every`, `log_every`


`outdir` is a filesystem path where:

- `meta.json` (or `summary.json`) stores the configuration
- `metrics.csv` stores time-series
- PNG snapshots are written:
  - `B_snapshot_*.png`
  - `tau_snapshot_*.png`
  - `N_snapshot_*.png`


Early exploratory models (e.g. `tau_reaction_diffusion_v2.py`) are kept in the repo as archival code illustrating the evolution from a simple single-τ model to the full v5 architecture, but all quantitative results in this document refer to `dynamic_tau_v5.py` and its sweeps.

---

### M10.3. Parameter sweeps

Parameter sweeps are driven by standalone scripts in `simulations/`:

- `simulations/run_sweep_v5.py`

This script:

- constructs a list of configuration dictionaries (`cfg` objects) sampling:
  - `feed`, `kill`
  - τ parameters (`alpha`, `beta`, `gamma`, `kappa_tau`, `tau_noise`)
  - nutrient and memory flags
- optionally restricts sampling to a **Q-ridge** region around previously identified high-coherence / low-entropy configurations
- supports parallel execution via the Python `multiprocessing` module
- for each `cfg`, calls:

```python
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

### M10.4. Aggregate analysis and scoring

High-level aggregation and scoring of runs are handled by:

- `analyze_proto_life_v5.py`

This script:

- scans a chosen output root, e.g.:

  - `outputs/dynamic_tau_v5/`
  - `outputs/dynamic_tau_v5_qridge/`

- for each run directory that contains:

  - `metrics.csv`
  - `meta.json` or `summary.json`

  it extracts:

  - configuration parameters (e.g. `feed`, `kill`, τ-parameters, flags)
  - time-averaged metrics (`mean_coherence`, `mean_entropy`, `mean_autocat`,
    `std_coherence`, `coherence_slope`, etc.)
  - τ-structure metrics (`tau_var_final`, `tau_grad2_final`)

- computes composite scores (e.g. `proto_life_score_v5`) based on weighted combinations of:

  - coherence / entropy
  - autocatalysis
  - τ-structure features

- writes summary CSVs to:

  - `plots/proto_life_v5/runs_summary_v5.csv`
  - `plots/proto_life_v5/runs_summary_v5_qridge.csv`

- optionally generates:

  - phase maps (e.g. coherence vs. α–β)
  - coherence/entropy scatterplots
  - best-run montages and other diagnostic plots

Example usage:

```bash
python analyze_proto_life_v5.py \
    --root outputs/dynamic_tau_v5 \
    --out plots/proto_life_v5/runs_summary_v5.csv

python analyze_proto_life_v5.py \
    --root outputs/dynamic_tau_v5_qridge \
    --out plots/proto_life_v5/runs_summary_v5_qridge.csv
```
### M10.5. Internal dynamics metrics

To quantify boundary persistence, internal reorganisation, and coherence oscillation, we use:

- `analyze_internal_v5.py`

This script takes an existing summary CSV (e.g. `runs_summary_v5_qridge.csv`), re-visits each run directory, and computes:

- `maintenance_iou`  
  – IoU between mid and final B-masks (shape persistence)

- `internal_reorg_index`  
  – 1 minus the correlation between mid and final B inside the cell mask  
    (internal pattern change without losing the body)

- `com_shift_B`  
  – centre-of-mass shift of B inside the cell (net movement of mass)

- `coherence_osc_index`  
  – variance of the detrended coherence time-series (coherence “breathing”)

It writes an augmented CSV alongside the original:

```bash
python analyze_internal_v5.py --csv plots/proto_life_v5/runs_summary_v5_qridge.csv

# Produces:
# plots/proto_life_v5/runs_summary_v5_qridge_with_internal.csv
```
These augmented tables are the basis for the classification into “breathing cells”, “crystallising cells”, and “melting foam” regimes.

M10.6. Environmental perturbation experiments

To probe robustness and proto-homeostasis, we use:

run_env_tests_v5.py

This script:

takes a list of candidate run hashes under outputs/dynamic_tau_v5_qridge/
(e.g. 40cdedd754, 94392f5ff1, 6fc841af45)

for each candidate:

loads the original config from meta.json

constructs environment variants by scaling feed and kill:

baseline

feed_low

feed_high

kill_low

kill_high

runs dynamic_tau_v5.run_simulation for each variant into
outputs/dynamic_tau_v5_env/<hash>/<variant>/

for each variant, compares the final B snapshot to the baseline final B, computing:

iou_vs_baseline (shape similarity)

corr_vs_baseline (internal organisation similarity)

writes a summary CSV:

plots/proto_life_v5/env_tests_summary.csv

Example invocation:

python run_env_tests_v5.py --candidates 40cdedd754 94392f5ff1 6fc841af45

This provides the empirical basis for the statements about homeostatic cells, fossil-like states, and overdriven / melting regimes under environmental changes.

M10.7. Reproducibility and folder structure

The key directories created by the pipeline are:

outputs/dynamic_tau_v5/
– raw global sweep runs (metrics.csv, snapshots, meta)

outputs/dynamic_tau_v5_qridge/
– focused Q-ridge runs used for fine-grained analysis

outputs/dynamic_tau_v5_env/
– environment-variant runs for selected candidates

plots/proto_life_v5/
– summary CSVs and aggregate plots, including:

runs_summary_v5.csv

runs_summary_v5_qridge.csv

runs_summary_v5_qridge_with_internal.csv

env_tests_summary.csv

phase maps (e.g. coherence vs α–β)

coherence/entropy scatterplots

best-run montages and internal cross-sections

Together, these scripts and artefacts constitute a complete, reproducible pipeline to:

Generate dynamic τ–coupled reaction–diffusion runs over parameter space.

Identify regions of interest (Q-ridge) where life-like behaviour occurs.

Quantify morphology, internal dynamics, and τ structure.

Test robustness under environmental perturbations.

All methods described in this document correspond directly to these publicly available scripts and the data products they generate.
