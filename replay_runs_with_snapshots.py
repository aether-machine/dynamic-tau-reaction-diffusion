#!/usr/bin/env python3
"""
replay_runs_with_snapshots.py

Replay selected runs from an existing search output folder and write dense
snapshot sequences (state_XXXXXX.npz) suitable for GIF/montage rendering.

Key features:
- Selects top_k runs by score from boqd_log.csv (optionally include best_so_far.json)
- Replays each run into: <out_root>/replay/<method>_<runid>/
- Forces snapshot saving (save_snapshots=1, snap_every=N)
- Writes meta.json + replay_meta.json in each replay run folder so downstream tools
  can label GIFs (desc, seed settings, etc.)

Expected simulator API:
  module.run_simulation(cfg: dict, outdir: str) -> None

This matches dynamic_tau_v7.py / dynamic_tau_v7_seeddiv.py.

Usage:
  python replay_runs_with_snapshots.py \
    --out_root outputs/dynamic_tau_v7_acf_seeddiv_v24_ring2_rand \
    --top_k 50 --include_best 1 \
    --snap_every 25 --snapshot_format npz \
    --clean 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_json(obj: Any) -> Any:
    """Convert numpy scalars to python scalars for json.dump(default=...)."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        # best-effort; treat as missing if malformed
        return None


def _write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_safe_json)


def _infer_sim_module(out_root: str, explicit: Optional[str]) -> str:
    """
    Choose which simulator module to import.
    Priority:
      1) explicit CLI flag
      2) dynamic_tau_v7_seeddiv if importable
      3) dynamic_tau_v7
    """
    if explicit:
        return explicit
    for name in ("dynamic_tau_v7_seeddiv", "dynamic_tau_v7"):
        try:
            import_module(name)
            return name
        except Exception:
            continue
    # last resort; let import error surface clearly
    return "dynamic_tau_v7"


def _normalize_run_name(run_dir: str) -> str:
    """
    Convert source run_dir path to a stable replay folder name.

    Examples:
      .../qd/run_00106_9a103baf3b  -> qd_run_00106_9a103baf3b
      .../init/run_00004_deadbeef -> init_run_00004_deadbeef
      .../run_00004_deadbeef      -> run_00004_deadbeef
    """
    parts = run_dir.replace("\\", "/").split("/")
    # look for method markers
    method = None
    for m in ("qd", "init", "rand"):
        if m in parts:
            method = m
    base = parts[-1]
    if method and not base.startswith(method + "_"):
        return f"{method}_{base}".replace("run_", "run_", 1).replace(f"{method}_run_", f"{method}_run_")
    return base


def _build_cfg_from_row(row: pd.Series) -> dict:
    """
    Construct a cfg dict for run_simulation(cfg,outdir) from boqd_log.csv row.
    Only includes keys that the simulator expects; everything else is ignored.
    """
    cfg: Dict[str, Any] = {}

    # Core sim params
    for k in ("nx", "ny", "steps", "dt", "log_every", "seed"):
        if k in row and pd.notna(row[k]):
            cfg[k] = row[k].item() if hasattr(row[k], "item") else row[k]

    # Gray-Scott params
    for k in ("feed", "kill", "Da", "Db"):
        if k in row and pd.notna(row[k]):
            cfg[k] = float(row[k])

    # tau / memory dynamics
    for k in ("alpha", "beta", "gamma", "kappa_tau", "tau_noise"):
        if k in row and pd.notna(row[k]):
            cfg[k] = float(row[k])

    # organism indicator gain (naming in your logs)
    if "w_tau_gain" in row and pd.notna(row["w_tau_gain"]):
        cfg["w_tau_gain"] = float(row["w_tau_gain"])
    if "w_tau_gain_max" in row and pd.notna(row["w_tau_gain_max"]):
        cfg["w_tau_gain_max"] = float(row["w_tau_gain_max"])
    if "w_enabled" in row and pd.notna(row["w_enabled"]):
        cfg["w_enabled"] = int(row["w_enabled"])
    if "w_gate" in row and pd.notna(row["w_gate"]):
        cfg["w_gate"] = float(row["w_gate"])

    # seed diversity params (present in v24 logs/meta if propagated)
    for k in ("init_mode", "seed_pos_mode", "seed_count", "seed_sigma", "seed_ring_width", "seed_stripe_period", "seed_margin", "seed_radius"):
        if k in row and pd.notna(row[k]):
            v = row[k]
            if isinstance(v, str):
                cfg[k] = v
            elif k in ("seed_count", "seed_stripe_period", "seed_margin", "seed_radius"):
                cfg[k] = int(v)
            else:
                cfg[k] = float(v)

    return cfg


def _load_source_meta_and_cfg(src_run_dir: str, row: Optional[pd.Series]) -> Tuple[dict, dict]:
    """
    Returns (src_meta, cfg) best-effort.
    Priority:
      - src_run_dir/meta.json, using cfg embedded if present
      - else cfg from row
    """
    src_meta = _read_json(os.path.join(src_run_dir, "meta.json")) or {}
    cfg = {}
    if isinstance(src_meta, dict) and "cfg" in src_meta and isinstance(src_meta["cfg"], dict):
        cfg = deepcopy(src_meta["cfg"])
    elif row is not None:
        cfg = _build_cfg_from_row(row)
    return src_meta, cfg


def _select_top_runs(out_root: str, top_k: int, include_best: bool) -> List[Tuple[str, Optional[pd.Series]]]:
    """
    Select run directories to replay.
    Returns list of (run_dir, row_or_None) ordered by descending score (best first).
    """
    log_path = os.path.join(out_root, "boqd_log.csv")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Missing boqd_log.csv at {log_path}")

    df = pd.read_csv(log_path)

    # Prefer successful rows with finite score if present
    if "error" in df.columns:
        ok = df["error"].isna() | (df["error"].astype(str).str.len() == 0)
        df = df[ok]

    if "score" in df.columns:
        df = df[pd.to_numeric(df["score"], errors="coerce").notna()]
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    if "run_dir" not in df.columns:
        raise ValueError("boqd_log.csv missing required column 'run_dir'")

    # Sort by score desc if available, else preserve file order
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)

    # Ensure run_dir paths are absolute-ish: they are usually already relative to cwd.
    # We'll keep them as-is; downstream will resolve relative to current working dir.
    chosen: List[Tuple[str, Optional[pd.Series]]] = []
    seen = set()

    for _, row in df.iterrows():
        run_dir = str(row["run_dir"])
        if run_dir in seen:
            continue
        if not os.path.isdir(run_dir):
            # if run_dir is relative to out_root, try join
            alt = os.path.join(out_root, run_dir)
            if os.path.isdir(alt):
                run_dir = alt
            else:
                continue
        chosen.append((run_dir, row))
        seen.add(run_dir)
        if len(chosen) >= top_k:
            break

    if include_best:
        best = _read_json(os.path.join(out_root, "best_so_far.json"))
        if best and isinstance(best, dict) and "run_dir" in best:
            bd = str(best["run_dir"])
            if not os.path.isdir(bd):
                alt = os.path.join(out_root, bd)
                if os.path.isdir(alt):
                    bd = alt
            if os.path.isdir(bd) and bd not in seen:
                chosen.insert(0, (bd, None))
                seen.add(bd)

    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True, help="Search output root containing boqd_log.csv etc.")
    ap.add_argument("--top_k", type=int, default=50, help="Replay the top K runs by score.")
    ap.add_argument("--include_best", type=int, default=1, help="Also replay best_so_far.json run (if present).")
    ap.add_argument("--snap_every", type=int, default=25, help="Snapshot cadence in steps (state_XXXXXX.npz).")
    ap.add_argument("--snapshot_format", type=str, default="npz", choices=["npz","npy"], help="Snapshot format hint for simulator (npz preferred).")
    ap.add_argument("--clean", type=int, default=0, help="If 1, delete and recreate <out_root>/replay.")
    ap.add_argument("--sim_module", type=str, default=None, help="Optional simulator module to import (e.g., dynamic_tau_v7_seeddiv).")
    args = ap.parse_args()

    out_root = args.out_root.rstrip("/")

    replay_root = os.path.join(out_root, "replay")
    if args.clean and os.path.exists(replay_root):
        shutil.rmtree(replay_root)
    os.makedirs(replay_root, exist_ok=True)

    sim_mod_name = _infer_sim_module(out_root, args.sim_module)
    sim = import_module(sim_mod_name)

    if not hasattr(sim, "run_simulation"):
        raise AttributeError(f"Simulator module '{sim_mod_name}' has no run_simulation(cfg,outdir)")

    runs = _select_top_runs(out_root, int(args.top_k), bool(args.include_best))

    print(f"[replay] sim_module={sim_mod_name}")
    print(f"[replay] selected {len(runs)} runs")

    for idx, (src_run_dir, row) in enumerate(runs):
        name = _normalize_run_name(src_run_dir)
        dst_dir = os.path.join(replay_root, name)

        # Clean per-run dir if it exists (idempotency)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir, exist_ok=True)

        src_meta, cfg = _load_source_meta_and_cfg(src_run_dir, row)

        # Force snapshot settings
        cfg["save_states"] = True
        cfg["save_snapshots"] = True
        cfg["snap_every"] = int(args.snap_every)
        # keep mid/final saved too; simulator uses mid_state_step default to steps//2
        # leave snapshot_format as an informative meta field (sim writes npz)
        cfg["snapshot_format"] = args.snapshot_format

        # Ensure we have a seed (if missing, derive from src_meta/run name)
        if "seed" not in cfg:
            cfg["seed"] = int(src_meta.get("seed", 0))

        # Compose replay meta (merge source meta if present)
        replay_meta: dict = deepcopy(src_meta) if isinstance(src_meta, dict) else {}
        # Keep a clean cfg record (source cfg might exist; we override with replay cfg)
        replay_meta["cfg"] = deepcopy(cfg)
        replay_meta["replay_of"] = src_run_dir
        replay_meta["source_run_dir"] = src_run_dir  # compat for downstream visualizers
        replay_meta["replay_out_dir"] = dst_dir
        replay_meta["replay_snap_every"] = int(args.snap_every)
        replay_meta["replay_snapshot_format"] = args.snapshot_format
        # carry score/desc if we have them from row
        if row is not None:
            for k in ("score", "desc_0", "desc_1", "method", "model"):
                if k in row and pd.notna(row[k]):
                    replay_meta[k] = row[k].item() if hasattr(row[k], "item") else row[k]

        # Write meta upfront so failures leave breadcrumbs
        _write_json(os.path.join(dst_dir, "meta.json"), replay_meta)
        _write_json(os.path.join(dst_dir, "replay_meta.json"), replay_meta)

        print(f"[replay] ({idx+1}/{len(runs)}) {name}")
        # Run the simulation replay
        sim.run_simulation(cfg, dst_dir)

    # Also write an index for convenience
    index = {
        "out_root": out_root,
        "replay_root": replay_root,
        "top_k": int(args.top_k),
        "include_best": int(args.include_best),
        "snap_every": int(args.snap_every),
        "sim_module": sim_mod_name,
        "runs": [_normalize_run_name(r[0]) for r in runs],
    }
    _write_json(os.path.join(replay_root, "index.json"), index)

    print(f"[replay] done. wrote {len(runs)} runs into {replay_root}")


if __name__ == "__main__":
    main()
