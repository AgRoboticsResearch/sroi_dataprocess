#!/usr/bin/env python3
"""
Repair Trajectory Script — align SLAM poses to frames through the index ledger.

Kernel script: called by batches/traj_repair_batch.sh (robust outer layer).
Inputs per episode (produced by the patched ORB-SLAM3 build):
  times.txt                  N lines, one hw timestamp per frame
  CameraTrajectory.txt       K lines x 12 cols (K <= N; compacted by tracking loss)
  CameraTrajectory_idx.txt   K lines x 1 col  (frame index of each pose row)

What it does (order is fixed by design: align -> mark holes -> smooth):
  1. Place each pose row into its true frame slot via the ledger.
  2. Every missing frame becomes NaN + state "invalid" — NO interpolation.
     Rationale (corpus-verified 2026-10-16): ORB-SLAM3 pushes a pose for every
     frame once initialization succeeds (OK || RECENTLY_LOST both append), so
     gaps are initialization-head gaps only (6/1138 episodes, all at the head,
     zero mid-episode). Inventing poses is unjustified; consumers head-trim.
  3. Whittaker smoothing (lambda from config) per contiguous finite segment;
     NaN rows are hard boundaries and stay NaN. Rotation goes through rot6d +
     Gram-Schmidt reprojection (never smooth rotation matrices directly).
  4. Atomically rewrite in place (tmp file -> fsync -> os.replace):
       CameraTrajectory.txt      N lines x 12 cols  (NaN rows for invalid frames)
       CameraTrajectory_idx.txt  N lines x 3 cols:  <frame> <state> <seg_len>
     state: raw | invalid. seg_len: length of the invalid run (0 for raw). The raw K-row state is fully
     reconstructible from these two files (drop non-raw rows), so no raw
     backup file is kept.

"Which generation is this file?" -> after repair, rows == frame count and the
idx has 3 columns; that is also the idempotent skip criterion. Re-running SLAM
with skip_existing=false regresses the files to K rows / 1 col -- re-run repair.

Usage:
    python3 repair_trajectory.py <episode-or-session> [--recursive] \
        [--config configs/traj_repair.json] [--force] [--no-smooth]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

TRAJ_NAME = "CameraTrajectory.txt"
IDX_NAME = "CameraTrajectory_idx.txt"
TIMES_NAME = "times.txt"

DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "traj_repair.json")

DEFAULTS = {"whittaker_lambda": 1.0}


def load_config(path):
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


# ----------------------------------------------------------------------------
# Whittaker smoother (same formulation as analysis_qc/smooth_trajectories.py)
# ----------------------------------------------------------------------------

def whittaker(p: np.ndarray, lam: float) -> np.ndarray:
    """Smooth columns of p (n, d) with a second-difference penalty."""
    n = len(p)
    D = np.zeros((n - 2, n))
    idx = np.arange(n - 2)
    D[idx, idx] = 1.0
    D[idx, idx + 1] = -2.0
    D[idx, idx + 2] = 1.0
    return np.linalg.solve(np.eye(n) + lam * (D.T @ D), p)


def rot6d_from_R(R: np.ndarray) -> np.ndarray:
    """R (n,3,3) -> (n,6): first two columns."""
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)


def R_from_rot6d(r6: np.ndarray) -> np.ndarray:
    """r6 (n,6) -> (n,3,3) via Gram-Schmidt reprojection."""
    a, b = r6[:, :3], r6[:, 3:]
    a1 = a / np.linalg.norm(a, axis=1, keepdims=True)
    b1 = b - (b * a1).sum(1, keepdims=True) * a1
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    c1 = np.cross(a1, b1)
    return np.stack([a1, b1, c1], axis=2)


def smooth_segment(pos: np.ndarray, R: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth one contiguous valid segment (positions + rotation via rot6d)."""
    if len(pos) < 3:
        return pos, R  # too short to smooth; harmless
    pos_s = whittaker(pos, lam)
    r6_s = whittaker(rot6d_from_R(R), lam)
    return pos_s, R_from_rot6d(r6_s)


# ----------------------------------------------------------------------------
# Repair
# ----------------------------------------------------------------------------

def find_holes(ids: np.ndarray, n: int) -> list[tuple[int, int, str]]:
    """Return list of (start, end_exclusive, where) for missing index runs."""
    holes = []
    present = np.zeros(n, dtype=bool)
    present[ids] = True
    i = 0
    while i < n:
        if not present[i]:
            j = i
            while j < n and not present[j]:
                j += 1
            where = "head" if i == 0 else ("tail" if j == n else "interior")
            holes.append((i, j, where))
            i = j
        else:
            i += 1
    return holes


def repair_episode(ep: Path, cfg: dict, smooth: bool = True, force: bool = False) -> str:
    """Repair one episode directory. Returns 'ok' | 'skip' | 'error: ...'."""
    traj_fp, idx_fp, times_fp = ep / TRAJ_NAME, ep / IDX_NAME, ep / TIMES_NAME
    for fp in (times_fp, traj_fp, idx_fp):
        if not fp.exists():
            return f"error: missing {fp.name}"

    times = np.loadtxt(times_fp, ndmin=1)
    n = len(times)
    try:
        traj = np.loadtxt(traj_fp, ndmin=2)
    except Exception as e:
        return f"error: unparsable {TRAJ_NAME} ({e}) -- rerun SLAM"
    if traj.size == 0:
        return "error: empty CameraTrajectory.txt -- rerun SLAM"

    if traj.ndim != 2 or traj.shape[1] != 12:
        return f"error: {TRAJ_NAME} has {traj.shape if traj.ndim == 2 else '1D'} cols (want 12)"
    k = len(traj)

    # Read the ledger format-tolerantly: 1 col = raw SLAM generation,
    # 3 cols ("<frame> <state> <seg_len>") = already repaired.
    idx_lines = [l for l in idx_fp.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not idx_lines:
        return "error: empty ledger -- rerun SLAM"
    ncol = len(idx_lines[0].split())
    if ncol == 1:
        ids = np.array([int(round(float(l))) for l in idx_lines], dtype=np.int64)
    elif ncol == 3:
        if k != n:
            return f"error: repaired ledger but trajectory has {k} rows != {n} frames"
        if not force:
            return "skip"
        # Reconstruct the raw K-row state from the ledger (drop non-raw rows).
        rows = [l.split() for l in idx_lines]
        raw_mask = np.array([r[1] == "raw" for r in rows])
        traj, ids = traj[raw_mask], np.array([int(r[0]) for r in rows], dtype=np.int64)[raw_mask]
        k = len(traj)
    else:
        return "error: unexpected ledger column count ({ncol}) -- rerun SLAM to regenerate"

    if len(ids) != k:
        return f"error: ledger rows ({len(ids)}) != trajectory rows ({k}) -- rerun SLAM"
    if len(np.unique(ids)) != len(ids):
        return "error: duplicate frame ids in ledger -- rerun SLAM"
    if len(ids) > 1 and np.any(np.diff(ids) <= 0):
        return "error: ledger ids not strictly increasing -- rerun SLAM"
    if ids.size and (ids.min() < 0 or ids.max() >= n):
        return f"error: ledger id out of range [0,{n - 1}] -- rerun SLAM"

    # 1. Place raw rows into true slots.
    poses = np.full((n, 4, 4), np.nan)
    bottom = np.array([0.0, 0.0, 0.0, 1.0])
    poses[ids, :3, :4] = traj.reshape(-1, 3, 4)
    poses[ids, 3, :] = bottom
    state = np.array(["invalid"] * n, dtype=object)
    state[ids] = "raw"

    # 2. Mark holes: every missing frame stays NaN + "invalid" (no interpolation).
    for start, end, where in find_holes(ids, n):
        pass  # poses were initialized to NaN; state already "invalid"

    # Segment lengths for invalid runs (0 for raw).
    seg_len = np.zeros(n, dtype=np.int64)
    i = 0
    while i < n:
        if state[i] != "invalid":
            i += 1
            continue
        j = i
        while j < n and state[j] == "invalid":
            j += 1
        seg_len[i:j] = j - i
        i = j

    # 3. Segment-wise Whittaker smoothing over finite (raw) runs.
    if smooth:
        lam = float(cfg["whittaker_lambda"])
        valid = state == "raw"
        i = 0
        while i < n:
            if not valid[i]:
                i += 1
                continue
            j = i
            while j < n and valid[j]:
                j += 1
            pos_s, R_s = smooth_segment(poses[i:j, :3, 3], poses[i:j, :3, :3], lam)
            poses[i:j, :3, 3] = pos_s
            poses[i:j, :3, :3] = R_s
            i = j

    # Self-check: invalid rows must be all-NaN, others finite.
    invalid_rows = state == "invalid"
    if invalid_rows.any() and not np.isnan(poses[invalid_rows, :3, :]).all():
        return "error: internal check failed (invalid rows not all-NaN)"
    if not np.isfinite(poses[~invalid_rows, :3, :]).all():
        return "error: internal check failed (non-finite values in valid rows)"

    # 4. Atomic rewrite of both files (tmp -> replace, ledger last).
    traj_out = poses[:, :3, :4].reshape(n, 12)
    write_atomic(traj_fp, "".join(
        " ".join(f"{v:.9f}" for v in row) + "\n" for row in traj_out))
    ledger = "".join(
        f"{i} {state[i]} {seg_len[i]}\n" for i in range(n))
    write_atomic(idx_fp, ledger)
    return "ok"


def write_atomic(fp: Path, text: str):
    tmp = fp.with_name(fp.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, fp)


def iter_episodes(root: Path):
    if root.name.startswith("episode_") and (root / TRAJ_NAME).exists():
        yield root
    else:
        for d in sorted(root.rglob("episode_*")):
            if d.is_dir() and (d / TRAJ_NAME).exists():
                yield d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("root", type=str, help="episode dir or session/data root")
    ap.add_argument("--recursive", action="store_true", help="search episode_* dirs recursively")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true", help="re-repair even if already repaired")
    ap.add_argument("--no-smooth", action="store_true", help="skip Whittaker smoothing")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: {root} does not exist")
        sys.exit(1)

    cfg = load_config(args.config)
    n_ok = n_skip = n_err = 0
    for ep in iter_episodes(root):
        res = repair_episode(ep, cfg, smooth=not args.no_smooth, force=args.force)
        tag = res.split(":")[0]
        print(f"{ep.parent.name}/{ep.name}: {res}")
        if tag == "ok":
            n_ok += 1
        elif tag == "skip":
            n_skip += 1
        else:
            n_err += 1
    print(f"repair summary: {n_ok} ok, {n_skip} skipped, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
