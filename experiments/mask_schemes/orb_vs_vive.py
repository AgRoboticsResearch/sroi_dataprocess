#!/usr/bin/env python3
"""
Compare an ORB-SLAM trajectory with Vive motion capture: alignment and scale.

Per episode inputs:
  - CameraTrajectory.txt (ORB-SLAM3 KITTI output in the -png directory)
  - vive_poses.jsonl (Vive motion capture at about 240 Hz in the -mp4 directory)
  - times.txt (camera-frame UNIX timestamps in the -mp4 directory)

Method:
  1. Interpolate Vive positions at camera-frame timestamps.
  2. If the ORB and camera frame counts differ, resample both by normalized progress
     to equal length, then apply positional Umeyama Sim3 alignment. Scale s maps ORB
     to Vive: s near 1 is consistent, s > 1 means ORB is small, and s < 1 means large.
  3. Cross-check scale using a correspondence-independent bounding-box diagonal ratio.

Outputs in <png_session>_orbviz/:
  - Interactive HTML per episode: Vive, rigid-aligned ORB, and scale-aligned ORB
  - summary.csv with scale, RMSE, path length, bounding-box ratio, and frame counts

Usage:
  python orb_vs_vive.py <png_session>
  (replaces -png with -mp4 automatically when locating Vive data and timestamps)
"""
import sys, json, csv
from pathlib import Path
import numpy as np


def load_kitti_positions(p):
    """Load KITTI rows as row-major 3x4 [R|t] matrices and return translations."""
    a = np.atleast_2d(np.loadtxt(p))
    if a.shape[1] == 13:
        a = a[:, 1:]
    a = a.reshape(-1, 3, 4)
    return a[:, :3, 3]              # (M,3)


def load_vive(p):
    times, pos = [], []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            times.append(d["time"])
            pos.append(d["pos"])
    return np.array(times), np.array(pos)   # (K,) (K,3)


def load_times(p):
    return np.loadtxt(p).reshape(-1)        # (N,)


def umeyama(model, data, with_scale=True):
    """Align model to data; return (s, R, t) such that data ~= s * R * model + t."""
    mu_m = model.mean(0); mu_d = data.mean(0)
    mz = model - mu_m; dz = data - mu_d
    n = len(model)
    H = mz.T @ dz / n
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = Vt.T @ D @ U.T
    if with_scale:
        var_m = np.sum(np.var(model, axis=0))
        s = np.trace(np.diag(S) @ D) / var_m if var_m > 1e-12 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_m
    return s, R, t


def resample_to_length(P, L):
    """Resample an (n, 3) trajectory to L indexed points for pointwise alignment."""
    n = len(P)
    if n == L:
        return P
    idx = np.linspace(0, n - 1, L)
    return np.column_stack([np.interp(idx, np.arange(n), P[:, d]) for d in range(3)])


def bbox_diag(P):
    return float(np.linalg.norm(P.max(0) - P.min(0)))


def path_len(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def process_episode(orb_txt, vive_jsonl, times_txt, out_html, name):
    P_orb = load_kitti_positions(orb_txt)                 # (M,3)
    T_vive, P_vive_full = load_vive(vive_jsonl)           # (K,),(K,3)
    T_cam = load_times(times_txt)                         # (N,)

    # Interpolate Vive positions at camera timestamps; out-of-range samples become NaN.
    Pv_at_cam = np.column_stack([
        np.interp(T_cam, T_vive, P_vive_full[:, d],
                  left=np.nan, right=np.nan) for d in range(3)])  # (N,3)
    good = np.isfinite(Pv_at_cam).all(1)
    Pv = Pv_at_cam[good]
    # Pair equal-length inputs directly; otherwise resample, assuming both span the episode.
    if len(P_orb) == len(Pv):
        A, B = P_orb, Pv
    else:
        L = min(len(P_orb), len(Pv))
        A = resample_to_length(P_orb, L)
        B = resample_to_length(Pv, L)

    # Rigid alignment without scale, then Sim3 alignment with scale.
    s1, R1, t1 = umeyama(A, B, with_scale=False)
    s, R, t = umeyama(A, B, with_scale=True)
    A_rigid = (R1 @ A.T).T + t1                 # Blue: aligned while preserving ORB scale
    A_scaled = (s * (R @ A.T).T) + t            # Green: scaled alignment should match Vive
    rmse = float(np.sqrt(np.mean(np.sum((A_scaled - B) ** 2, axis=1))))

    # Correspondence-independent scale cross-check.
    diag_orb = bbox_diag(P_orb); diag_vive = bbox_diag(Pv)
    bbox_ratio = diag_orb / diag_vive if diag_vive > 1e-9 else float("nan")

    # ---- Interactive HTML ----
    if out_html is not None:
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=B[:, 0], y=B[:, 1], z=B[:, 2], mode="lines+markers",
                                       line=dict(width=4, color="red"), marker=dict(size=3),
                                       name=f"Vive reference ({len(B)})"))
            fig.add_trace(go.Scatter3d(x=A_rigid[:, 0], y=A_rigid[:, 1], z=A_rigid[:, 2], mode="lines",
                                       line=dict(width=3, color="blue"), name="ORB rigid alignment (no scale)"))
            fig.add_trace(go.Scatter3d(x=A_scaled[:, 0], y=A_scaled[:, 1], z=A_scaled[:, 2], mode="lines",
                                       line=dict(width=3, color="green"), name=f"ORB scaled alignment (s={s:.3f})"))
            fig.update_layout(
                title=(f"{name} | M_ORB={len(P_orb)} N_cam={len(T_cam)} | "
                       f"Sim3 s={s:.3f}  RMSE={rmse*1000:.1f}mm | bbox ratio={bbox_ratio:.2f}"),
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
                width=980, height=760)
            fig.write_html(out_html)
        except Exception as e:
            print(f"  (skipping HTML for {name}: {e})")

    return dict(episode=name, M_orb=len(P_orb), N_cam=len(T_cam), n_pairs=len(A),
                sim3_scale=s, rmse_mm=rmse * 1000, path_orb=path_len(P_orb),
                path_vive=path_len(Pv), bbox_ratio=bbox_ratio)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    args = [a for a in sys.argv[1:] if a != "--no-html"]
    png_session = Path(args[0]).resolve()
    if len(args) > 1:
        mp4_session = Path(args[1]).resolve()  # Explicit -mp4 path for derived mask trees
    else:
        mp4_session = Path(str(png_session).replace("-png", "-mp4"))
    if not mp4_session.exists():
        print(f"-mp4 directory does not exist: {mp4_session}"); sys.exit(1)

    outdir = png_session.parent / (png_session.name.replace("-png", "-orbviz"))
    outdir.mkdir(exist_ok=True)
    no_html = "--no-html" in sys.argv

    eps = sorted(d for d in png_session.iterdir() if d.is_dir() and d.name.startswith("episode_"))
    print(f"{png_session.name}: {len(eps)} episodes  ->  {outdir}\n")

    rows = []
    for ep in eps:
        orb_txt = ep / "CameraTrajectory.txt"
        vive = mp4_session / ep.name / "vive_poses.jsonl"
        times = mp4_session / ep.name / "times.txt"
        if not orb_txt.exists():
            print(f"  {ep.name}: no CameraTrajectory.txt; skipping")
            continue
        if not vive.exists() or not times.exists():
            print(f"  {ep.name}: missing Vive data or timestamps; skipping")
            continue
        html = None if no_html else outdir / f"{ep.name}.html"
        try:
            r = process_episode(orb_txt, vive, times, html, ep.name)
            rows.append(r)
            print(f"  {ep.name}: s={r['sim3_scale']:.3f}  RMSE={r['rmse_mm']:.1f}mm  "
                  f"bbox ratio={r['bbox_ratio']:.2f}  (M={r['M_orb']} N={r['N_cam']})")
        except Exception as e:
            print(f"  {ep.name}: error: {e}")

    if rows:
        csvp = outdir / "summary.csv"
        with open(csvp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        s = np.array([r["sim3_scale"] for r in rows])
        br = np.array([r["bbox_ratio"] for r in rows])
        print("\n================ Summary ================")
        print(f"episodes: {len(rows)}")
        print(f"Sim3 scale s: mean={s.mean():.3f}  median={np.median(s):.3f}  std={s.std():.3f}")
        print("  (s > 1: ORB is small; s < 1: ORB is large; s ~= 1: consistent)")
        print(f"bbox diagonal ratio: mean={br.mean():.3f}  median={np.median(br):.3f}")
        print(f"CSV: {csvp}")


if __name__ == "__main__":
    main()
