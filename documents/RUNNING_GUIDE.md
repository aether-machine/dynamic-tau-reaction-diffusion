# Running Guide (Dynamic-τ Reaction–Diffusion)

This repo explores Gray–Scott reaction–diffusion extended with an additional *memory / time-density* state (`tau`) and a search driver (BO/QD-style) to discover interesting spatiotemporal attractors (rings, blobs/packets, multi-compartments, stripes, etc.).

This document is a practical “how to run it” guide:
- install + environment
- run a search
- replay best runs with dense snapshots
- render montages / GIFs
- common pitfalls + troubleshooting

> **Note:** Script filenames (e.g. `run_search_v7_minimal_w_acf_v25.py`) evolve. Use the *latest* runner in `simulations/` unless you intentionally want an older experiment.

---

## 1) Setup

### 1.1 Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 1.2 Install dependencies
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

## 2) Core directory layout

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

## 3) Running a search

### 3.1 Basic “v7” search command (example)
Run from repo root:

```bash
python simulations/run_search_v7_minimal_w_acf_v25.py   --out_root outputs/v25_example   --workers 8 --budget 400 --init_random 80   --bins 24   --steps 3000 --nx 150 --ny 150   --dt 0.01 --log_every 20   --w_enabled 1 --w_gate 0.01 --w_tau_gain_max 0.15   --osc_fmin 0.05 --osc_fmax 1.5 --osc_min_cycles 2.0   --seed 1
```

### 3.2 Switching initial condition modes (examples)

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

## 4) Understanding outputs

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

If you want, we can add a short “Interpretation guide” section (what rings/packets/stripes usually indicate) and a curated list of “good default” parameter presets.
