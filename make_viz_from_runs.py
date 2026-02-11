#!/usr/bin/env python3
"""
make_viz_from_runs.py

Create image montages and simple animations (GIF/MP4) for "w" (or other fields)
from reaction–diffusion search runs.

It can:
- Read best runs from best_so_far.json and/or qd_elites.json
- Fall back to scanning run directories for state_*.npz files
- Render mid/final (or any available state_*.npz sequence) as:
  * PNG montage grid
  * Per-run side-by-side mid/final panels
  * GIF (and optionally MP4 if ffmpeg available)

Typical usage:
  python make_viz_from_runs.py --out_root outputs/dynamic_tau_v7_acf_localstats_v20 --top_k 12

Notes:
- Expects per-run directories like: <out_root>/{init,rand,qd}/run_xxxxx_xxxxxxxx/
- Expects NPZ state files named state_mid.npz/state_final.npz OR state_*.npz
- Tries to find a 2D field named 'w' (case-insensitive); otherwise uses the first 2D array in the NPZ.
"""

from __future__ import annotations
import argparse, json, os, sys, glob, math, shutil, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Matplotlib is used for rendering (no seaborn).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

@dataclass
class RunInfo:
    run_dir: str
    score: Optional[float] = None
    desc_0: Optional[float] = None
    desc_1: Optional[float] = None
    method: Optional[str] = None
    osc_signal: Optional[str] = None
    morph_class: Optional[str] = None

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def load_json(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to read json {path}: {e}", file=sys.stderr)
        return None

def gather_runs(out_root: str, top_k: int, include_best: bool, include_elites: bool,
                include_methods: List[str]) -> List[RunInfo]:
    runs: List[RunInfo] = []
    seen = set()

    if include_best:
        best = load_json(os.path.join(out_root, "best_so_far.json"))
        if best and isinstance(best, dict) and "run_dir" in best:
            rd = best["run_dir"]
            if rd and rd not in seen:
                seen.add(rd)
                runs.append(RunInfo(
                    run_dir=rd,
                    score=_safe_float(best.get("score")),
                    desc_0=_safe_float(best.get("desc_0")),
                    desc_1=_safe_float(best.get("desc_1")),
                    method=best.get("method"),
                    osc_signal=best.get("osc_signal"),
                ))

    if include_elites:
        elites = load_json(os.path.join(out_root, "qd_elites.json"))
        if elites and isinstance(elites, dict):
            for e in elites.get("elites", [])[:]:
                rd = e.get("run_dir")
                if not rd or rd in seen:
                    continue
                seen.add(rd)
                runs.append(RunInfo(
                    run_dir=rd,
                    score=_safe_float(e.get("score")),
                    desc_0=_safe_float(e.get("d0", e.get("desc_0"))),
                    desc_1=_safe_float(e.get("d1", e.get("desc_1"))),
                    method="qd",
                    osc_signal=(e.get("params", {}) or {}).get("osc_signal"),
                ))

    # Fallback: scan filesystem for run dirs and sort by score from meta.json if present.
    if len(runs) < top_k:
        candidate_dirs = []
        for m in include_methods:
            candidate_dirs.extend(glob.glob(os.path.join(out_root, m, "run_*")))
        meta_scored: List[Tuple[float, RunInfo]] = []
        for rd in candidate_dirs:
            if rd in seen:
                continue
            meta = load_json(os.path.join(rd, "meta.json"))
            if meta and isinstance(meta, dict):
                score = _safe_float(meta.get("score"))
                if score is None:
                    continue
                ri = RunInfo(
                    run_dir=rd,
                    score=score,
                    desc_0=_safe_float(meta.get("desc_0")),
                    desc_1=_safe_float(meta.get("desc_1")),
                    method=meta.get("method"),
                    osc_signal=meta.get("osc_signal"),
                    morph_class=(meta.get("morph_class") if isinstance(meta.get("morph_class"), str) else None),
                )
                meta_scored.append((score, ri))
        meta_scored.sort(key=lambda t: t[0], reverse=True)
        for _, ri in meta_scored:
            if len(runs) >= top_k:
                break
            if ri.run_dir not in seen:
                seen.add(ri.run_dir)
                runs.append(ri)


    # Extra fallback: "replay" layout writes per-run dirs directly under out_root like:
    #   <out_root>/{qd,init,rand}_run_xxxxx_xxxxxxxx/
    # If we still don't have enough runs, scan one level deep for any directories containing state_*.npz.
    if len(runs) < top_k:
        candidate_dirs = []
        # One-level scan for directories whose basename contains "run_"
        for p in glob.glob(os.path.join(out_root, "*run_*")):
            if os.path.isdir(p):
                candidate_dirs.append(p)

        meta_scored: List[Tuple[float, RunInfo]] = []
        for rd in candidate_dirs:
            if rd in seen:
                continue
            # If this is a replay dir, it may contain replay_meta.json pointing to the original run_dir
            replay_meta = load_json(os.path.join(rd, "replay_meta.json"))
            src_run_dir = None
            if replay_meta and isinstance(replay_meta, dict):
                src_run_dir = replay_meta.get("source_run_dir") or replay_meta.get("replay_of")

            meta = None
            if src_run_dir and os.path.exists(os.path.join(src_run_dir, "meta.json")):
                meta = load_json(os.path.join(src_run_dir, "meta.json"))
            if meta is None:
                meta = load_json(os.path.join(rd, "meta.json"))

            if meta and isinstance(meta, dict):
                score = _safe_float(meta.get("score"))
                if score is None:
                    # allow unscored runs if they at least have state files
                    score = -1e18
                ri = RunInfo(
                    run_dir=rd,
                    score=score,
                    desc_0=_safe_float(meta.get("desc_0")),
                    desc_1=_safe_float(meta.get("desc_1")),
                    method=meta.get("method"),
                    osc_signal=meta.get("osc_signal"),
                    morph_class=(meta.get("morph_class") if isinstance(meta.get("morph_class"), str) else None),
                )
                meta_scored.append((score, ri))
            else:
                # As last resort, include runs that have any state files
                if list_state_files(rd, field=args.field):
                    meta_scored.append((-1e18, RunInfo(run_dir=rd)))

        meta_scored.sort(key=lambda t: t[0], reverse=True)
        for _, ri in meta_scored:
            if len(runs) >= top_k:
                break
            if ri.run_dir not in seen:
                seen.add(ri.run_dir)
                runs.append(ri)
    runs.sort(key=lambda r: (r.score if r.score is not None else -1e18), reverse=True)
    return runs[:top_k]

def pick_field_from_npz(npz: np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    keys = list(npz.keys())
    key_map = {k.lower(): k for k in keys}
    target = field.lower()

    # Common naming: simulator stores organism field as N, while analysis calls it w.
    if target == "w" and "n" in key_map:
        return np.asarray(npz[key_map["n"]])

    if target in key_map:
        return np.asarray(npz[key_map[target]])
    for k in keys:
        if target in k.lower():
            arr = np.asarray(npz[k])
            if arr.ndim == 2:
                return arr
    for k in keys:
        arr = np.asarray(npz[k])
        if arr.ndim == 2:
            return arr
    raise KeyError(f"No 2D arrays found in NPZ keys={keys}")


def load_field_from_state(path: str, field: str) -> np.ndarray:
    """
    Load a 2D field from either:
      - .npz produced by np.savez (supports keys A,B,tau,N, etc.)
      - .npy produced by np.save (either a raw 2D array, or a pickled dict/obj)
    """
    if path.endswith(".npz"):
        with np.load(path) as npz:
            return pick_field_from_npz(npz, field)

    # .npy
    arr = np.load(path, allow_pickle=True)

    # Common case: raw 2D array
    if isinstance(arr, np.ndarray) and arr.ndim == 2:
        return arr

    # Common case: stacked 3D array, e.g. shape (4, ny, nx) or (3, ny, nx)
    # Heuristic order: [A, B, tau, N] if 4; [A, B, N] if 3.
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] in (3, 4):
        order = ["a", "b", "tau", "n"] if arr.shape[0] == 4 else ["a", "b", "n"]
        target = field.lower()
        if target == "w":
            target = "n"
        if target in order:
            idx = order.index(target)
            return np.asarray(arr[idx])
        # fallback: last slice (often N)
        return np.asarray(arr[-1])

    # Sometimes saved as an object containing a dict of fields
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict):
            # normalize key access similar to NPZ
            key_map = {str(k).lower(): k for k in obj.keys()}
            target = field.lower()
            if target == "w" and "n" in key_map:
                return np.asarray(obj[key_map["n"]])
            if target in key_map:
                return np.asarray(obj[key_map[target]])
            for k in obj.keys():
                if target in str(k).lower():
                    v = np.asarray(obj[k])
                    if v.ndim == 2:
                        return v
            # fallback: first 2D array in dict
            for k, v in obj.items():
                v = np.asarray(v)
                if v.ndim == 2:
                    return v

    raise KeyError(f"Unsupported .npy snapshot format for {path}")

def _normalize_field_key(field: str) -> str:
    """Map requested field name to snapshot prefix / npz key."""
    f = (field or "").strip()
    if f.lower() in ("w", "n"):
        return "N"
    if f.lower() in ("a",):
        return "A"
    if f.lower() in ("b",):
        return "B"
    if f.lower() in ("tau", "t"):
        return "tau"
    return f

def list_state_files(run_dir: str, field: str = "N") -> List[str]:
    """
    Return an ordered list of frame files for visualization.

    Priority:
      1) Dense snapshot sequence in state_*.npz / state_*.npy (numeric, e.g. state_000025.npz)
         (NOTE: if only state_mid/state_final exist, we DO NOT treat that as a "dense sequence")
      2) Per-field snapshot sequence like B_000025.npy, tau_000025.npy, A_000025.npy, etc.
      3) Fallback to state_mid.npz/state_final.npz
    """
    # 1) Dense sequence: state_<number>.{npz,npy}
    dense = sorted(set(glob.glob(os.path.join(run_dir, "state_*.npz")))) + \
            sorted(set(glob.glob(os.path.join(run_dir, "state_*.npy"))))
    dense = sorted(set(dense))

    def is_mid_final(p: str) -> bool:
        b = os.path.basename(p)
        return b in ("state_mid.npz", "state_final.npz")

    # Exclude mid/final; those are not "dense"
    dense_num = [p for p in dense if (not is_mid_final(p)) and re.search(r"(\d+)", os.path.basename(p))]

    def sort_key(path: str):
        base = os.path.basename(path)
        m = re.search(r"(\d+)", base)
        return int(m.group(1)) if m else 10**9

    if dense_num:
        dense_num.sort(key=sort_key)
        return dense_num

    # 2) Per-field snapshots: e.g. B_000025.npy
    key = _normalize_field_key(field)
    per = sorted(set(glob.glob(os.path.join(run_dir, f"{key}_*.npy"))))
    if per:
        per.sort(key=sort_key)
        return per

    # 3) Fallback: mid/final
    out: List[str] = []
    mid = os.path.join(run_dir, "state_mid.npz")
    fin = os.path.join(run_dir, "state_final.npz")
    if os.path.exists(mid):
        out.append(mid)
    if os.path.exists(fin):
        out.append(fin)
    return out
def render_panel(ax, arr2d: np.ndarray, title: str, vmin: Optional[float], vmax: Optional[float]):
    im = ax.imshow(arr2d, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im

def save_side_by_side(run: RunInfo, field: str, out_dir: str, global_vmin: Optional[float], global_vmax: Optional[float],
                      per_run_scale: bool) -> Optional[str]:
    rd = run.run_dir
    mid_path = os.path.join(rd, "state_mid.npz")
    final_path = os.path.join(rd, "state_final.npz")

    seq = list_state_files(rd, field=field)
    if not os.path.exists(mid_path) or not os.path.exists(final_path):
        if len(seq) >= 2:
            mid_path, final_path = seq[0], seq[-1]
        else:
            print(f"[warn] no state files in {rd}", file=sys.stderr)
            return None

    A = load_field_from_state(mid_path, field)
    B = load_field_from_state(final_path, field)

    if per_run_scale:
        vmin = float(np.nanmin([A.min(), B.min()]))
        vmax = float(np.nanmax([A.max(), B.max()]))
    else:
        vmin, vmax = global_vmin, global_vmax

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    im0 = render_panel(axes[0], A, f"mid ({os.path.basename(mid_path)})", vmin, vmax)
    im1 = render_panel(axes[1], B, f"final ({os.path.basename(final_path)})", vmin, vmax)

    info = f"{os.path.basename(rd)}"
    if run.score is not None:
        info += f" | score={run.score:.4g}"
    if run.desc_0 is not None and run.desc_1 is not None:
        info += f" | desc=({run.desc_0:.3f},{run.desc_1:.3f})"
    if run.osc_signal:
        info += f" | osc_signal={run.osc_signal}"
    if run.morph_class:
        info += f" | class={run.morph_class}"
    fig.suptitle(info, fontsize=10)

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.ax.tick_params(labelsize=8)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{os.path.basename(rd)}_{field}_mid_final.png")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def save_montage_grid(runs: List[RunInfo], field: str, out_dir: str, global_vmin: Optional[float], global_vmax: Optional[float],
                      per_run_scale: bool, max_cols: int = 4) -> Optional[str]:
    if not runs:
        return None

    panels = []
    labels = []
    for r in runs:
        rd = r.run_dir
        mid_path = os.path.join(rd, "state_mid.npz")
        final_path = os.path.join(rd, "state_final.npz")
        seq = list_state_files(rd, field=field)
        if not os.path.exists(mid_path) or not os.path.exists(final_path):
            if len(seq) >= 2:
                mid_path, final_path = seq[0], seq[-1]
            else:
                continue
        try:
            A = load_field_from_state(mid_path, field)
            B = load_field_from_state(final_path, field)
        except Exception as e:
            print(f"[warn] failed loading {rd}: {e}", file=sys.stderr)
            continue
        panels.append((A, B))
        lb = f"{os.path.basename(rd)}\nscore={r.score:.3g}" if r.score is not None else os.path.basename(rd)
        labels.append(lb)

    if not panels:
        return None

    n = len(panels)
    cols = min(max_cols, n)
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(4*cols, 3.2*rows), dpi=150)

    for idx, ((A, B), lb) in enumerate(zip(panels, labels)):
        r = idx // cols
        c = idx % cols
        ax1 = fig.add_subplot(rows*2, cols, r*2*cols + c + 1)
        ax2 = fig.add_subplot(rows*2, cols, (r*2+1)*cols + c + 1)

        if per_run_scale:
            _vmin = float(np.nanmin([A.min(), B.min()]))
            _vmax = float(np.nanmax([A.max(), B.max()]))
        else:
            _vmin, _vmax = global_vmin, global_vmax

        render_panel(ax1, A, lb + "\nmid", _vmin, _vmax)
        render_panel(ax2, B, "final", _vmin, _vmax)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"montage_top{len(panels)}_{field}.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def save_gif(run: RunInfo, field: str, out_dir: str, global_vmin: Optional[float], global_vmax: Optional[float],
             per_run_scale: bool, max_frames: int = 200, fps: int = 10) -> Optional[str]:
    rd = run.run_dir
    files = list_state_files(rd, field=field)
    if len(files) < 2:
        return None

    if len(files) > max_frames:
        step = int(math.ceil(len(files) / max_frames))
        files = files[::step]

    frames = []
    used_files = []
    for p in files:
        try:
            frames.append(load_field_from_state(p, field))
            used_files.append(p)
        except Exception as e:
            print(f"[warn] skip frame {p}: {e}", file=sys.stderr)

    if len(frames) < 2:
        return None

    if per_run_scale:
        vmin = float(np.nanmin([f.min() for f in frames]))
        vmax = float(np.nanmax([f.max() for f in frames]))
    else:
        vmin, vmax = global_vmin, global_vmax

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
    im = ax.imshow(frames[0], origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])

    info = f"{os.path.basename(rd)}"
    if run.score is not None:
        info += f" | score={run.score:.4g}"
    if run.desc_0 is not None and run.desc_1 is not None:
        info += f" | desc=({run.desc_0:.3f},{run.desc_1:.3f})"
    if run.osc_signal:
        info += f" | osc_signal={run.osc_signal}"
    if run.morph_class:
        info += f" | class={run.morph_class}"
    ax.set_title(info, fontsize=9)

    def update(i):
        im.set_data(frames[i])
        ax.set_xlabel(os.path.basename(used_files[i]), fontsize=8)
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=int(1000/fps), blit=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{os.path.basename(rd)}_{field}.gif")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path

def maybe_save_mp4_from_gif(gif_path: str) -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    mp4_path = os.path.splitext(gif_path)[0] + ".mp4"
    cmd = f'"{ffmpeg}" -y -i "{gif_path}" -vf "fps=30,format=yuv420p" -movflags +faststart "{mp4_path}"'
    rc = os.system(cmd)
    return mp4_path if rc == 0 and os.path.exists(mp4_path) else None

def compute_global_scale(runs: List[RunInfo], field: str) -> Tuple[Optional[float], Optional[float]]:
    mins, maxs = [], []
    for r in runs:
        rd = r.run_dir
        candidates = [os.path.join(rd, "state_mid.npz"), os.path.join(rd, "state_final.npz")]
        if not all(os.path.exists(p) for p in candidates):
            seq = list_state_files(rd, field=field)
            if len(seq) >= 2:
                candidates = [seq[0], seq[-1]]
            else:
                continue
        try:
            for p in candidates:
                A = load_field_from_state(p, field)
                mins.append(float(np.nanmin(A)))
                maxs.append(float(np.nanmax(A)))
        except Exception:
            continue
    if not mins or not maxs:
        return (None, None)
    return (float(np.nanmin(mins)), float(np.nanmax(maxs)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Root output directory (contains boqd_log.csv, qd_elites.json, best_so_far.json)")
    ap.add_argument("--top_k", type=int, default=12, help="How many runs to visualize (sorted by score where available)")
    ap.add_argument("--field", type=str, default="w", help="Which field name to render from NPZ (default: w)")
    ap.add_argument("--include_best", type=int, default=1, help="Include best_so_far.json (1/0)")
    ap.add_argument("--include_elites", type=int, default=1, help="Include qd_elites.json elites (1/0)")
    ap.add_argument("--methods", type=str, default="qd,init,rand", help="Which method folders to scan as fallback")
    ap.add_argument("--out_dir", type=str, default="", help="Where to write visualizations (default: <out_root>/viz)")
    ap.add_argument("--per_run_scale", type=int, default=0, help="If 1, scale each run independently; else use a global scale across runs")
    ap.add_argument("--make_gifs", type=int, default=1, help="If 1, also write a GIF per run (uses any state_*.npz sequence; falls back to mid/final)")
    ap.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    ap.add_argument("--max_frames", type=int, default=200, help="Max frames per GIF (sequence will be downsampled)")
    ap.add_argument("--also_mp4", type=int, default=0, help="If 1, attempt GIF->MP4 conversion via ffmpeg if available")
    args = ap.parse_args()

    out_root = args.out_root
    out_dir = args.out_dir or os.path.join(out_root, "viz")
    os.makedirs(out_dir, exist_ok=True)

    runs = gather_runs(
        out_root=out_root,
        top_k=args.top_k,
        include_best=bool(args.include_best),
        include_elites=bool(args.include_elites),
        include_methods=[m.strip() for m in args.methods.split(",") if m.strip()],
    )
    if not runs:
        print("[error] no runs found to visualize", file=sys.stderr)
        sys.exit(2)

    if args.per_run_scale:
        gvmin, gvmax = (None, None)
    else:
        gvmin, gvmax = compute_global_scale(runs, args.field)
        print(f"[info] global scale for field={args.field}: vmin={gvmin}, vmax={gvmax}")

    panel_dir = os.path.join(out_dir, "panels")
    paths = []
    for r in runs:
        p = save_side_by_side(r, args.field, panel_dir, gvmin, gvmax, per_run_scale=bool(args.per_run_scale))
        if p:
            paths.append(p)

    montage = save_montage_grid(runs, args.field, out_dir, gvmin, gvmax, per_run_scale=bool(args.per_run_scale), max_cols=4)

    gif_dir = os.path.join(out_dir, "gifs")
    gifs = []
    if args.make_gifs:
        for r in runs:
            g = save_gif(r, args.field, gif_dir, gvmin, gvmax, per_run_scale=bool(args.per_run_scale),
                         max_frames=args.max_frames, fps=args.fps)
            if g:
                gifs.append(g)
                if args.also_mp4:
                    maybe_save_mp4_from_gif(g)

    index_path = os.path.join(out_dir, "index.txt")
    with open(index_path, "w") as f:
        f.write(f"Top {len(runs)} runs visualized from: {out_root}\n")
        f.write(f"Field: {args.field}\n")
        f.write(f"Montage: {montage}\n\n")
        f.write("Panels:\n")
        for p in paths:
            f.write(p + "\n")
        f.write("\nGIFs:\n")
        for g in gifs:
            f.write(g + "\n")

    print("[done] wrote:")
    if montage:
        print("  montage:", montage)
    print("  panels:", panel_dir)
    if gifs:
        print("  gifs:", gif_dir)
    print("  index:", index_path)

if __name__ == "__main__":
    main()
