#!/usr/bin/env python3
"""Compare maskgripper/raw against maskhalf using rigid RMSE and tracking completeness.
Reports episodes where maskhalf loses frames but maskgripper tracks the full sequence.
Usage: deviation_vs_baseline.py <schemes_dir> [threshold_mm=15]
"""
import sys, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import orb_vs_vive as ovv

schemes = Path(sys.argv[1])
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
eps = sorted(d for d in schemes.iterdir() if d.is_dir() and d.name.startswith("episode_"))


def loadt(ep, sch):
    t = ep / sch / "CameraTrajectoryTransformed.txt"
    if not t.exists():
        t = ep / sch / "CameraTrajectory.txt"
    if not t.exists():
        return None
    try:
        points = ovv.load_kitti_positions(t)
    except (OSError, ValueError):
        return None
    return points if len(points) else None


def ninput(ep):
    t = ep / "times.txt"
    if t.exists():
        try:
            return int(len(np.loadtxt(t).reshape(-1)))
        except Exception:
            pass
    return None


def dev(A, B):
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return None
    if len(A) != len(B):
        L = min(len(A), len(B))
        A, B = ovv.resample_to_length(A, L), ovv.resample_to_length(B, L)
    s, R, t = ovv.umeyama(A, B, with_scale=False)
    return float(np.sqrt(np.mean(np.sum(((R @ A.T).T + t - B) ** 2, axis=1))))


def bbox(P):
    return float(np.linalg.norm(P.max(0) - P.min(0))) if P is not None and len(P) else 0.0


rows = []
for ep in eps:
    ni = ninput(ep)
    base = loadt(ep, "maskhalf"); mg = loadt(ep, "maskgripper"); raw = loadt(ep, "raw")
    rows.append(dict(ep=ep.name, nin=ni, mh_n=len(base) if base is not None else 0,
                     mg_n=len(mg) if mg is not None else 0,
                     raw_n=len(raw) if raw is not None else 0, d_mg=dev(mg, base), d_raw=dev(raw, base),
                     mh_bbox=bbox(base), mg_bbox=bbox(mg)))

outdir = schemes / "_compare"; outdir.mkdir(exist_ok=True)
csvp = outdir / "deviation.csv"
with open(csvp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode", "nin", "maskhalf_n", "maskgripper_n", "raw_n",
                "rmse_mg_vs_mh_mm", "rmse_raw_vs_mh_mm", "mh_bbox_cm", "mg_bbox_cm"])
    for r in rows:
        w.writerow([r["ep"], r["nin"], r["mh_n"], r["mg_n"], r["raw_n"],
                    f"{r['d_mg']*1000:.1f}" if r["d_mg"] is not None else "NA",
                    f"{r['d_raw']*1000:.1f}" if r["d_raw"] is not None else "NA",
                    f"{r['mh_bbox']*100:.1f}", f"{r['mg_bbox']*100:.1f}"])

comp = lambda r, k: (100 * r[k] / r["nin"]) if r["nin"] else 0
mh_c = np.array([comp(r, "mh_n") for r in rows])
mg_c = np.array([comp(r, "mg_n") for r in rows])
print(f"{schemes.name}: n={len(rows)}  (thr={thr}mm)")
if rows:
    print(f"  completeness: maskhalf mean={mh_c.mean():.0f}% (losses={int((mh_c < 100).sum())}) | "
          f"maskgripper mean={mg_c.mean():.0f}% (losses={int((mg_c < 100).sum())})")
else:
    print("  completeness: no episodes")
lost = [r for r in rows if r["nin"] and r["mh_n"] < r["nin"] and r["mg_n"] >= r["nin"]]
print(f"  maskhalf lost frames while maskgripper tracked fully: {len(lost)} episode(s)")
for r in lost:
    print(f"      {r['ep']}: maskhalf {r['mh_n']}/{r['nin']}  maskgripper {r['mg_n']}/{r['nin']}")
mg = np.array([r["d_mg"] for r in rows if r["d_mg"] is not None])
ra = np.array([r["d_raw"] for r in rows if r["d_raw"] is not None])
mg_med = np.median(mg) * 1000 if len(mg) else float("nan")
ra_med = np.median(ra) * 1000 if len(ra) else float("nan")
over_thr = int((mg * 1000 > thr).sum()) if len(mg) else 0
print(f"  deviation: maskgripper vs maskhalf median={mg_med:.1f}mm (>{thr:g}mm: {over_thr}) | "
      f"raw vs maskhalf med={ra_med:.1f}mm")
print(f"CSV: {csvp}")
