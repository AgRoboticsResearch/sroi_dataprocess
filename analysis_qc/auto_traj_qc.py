#!/usr/bin/env python3
"""Automated trajectory QC — programmatic version of the manual R1 review.

SUPERSEDED as a verdict source: F1/E0/E1/V1 now live in visualization/qc.py
(first_frame_projection / traj_frame_jump categories; HSV red detection;
thresholds in configs/qc_filters.json). This script's remaining purpose is the
`tune` mode: rendering detection-window overlays so thresholds can be tuned
against real frames. `full` mode still runs but uses the legacy RGB red test.

Per episode (reuses the projection pipeline of projection_qc_batch.py):
  F1  frame-count consistency: len(CameraTrajectory.txt) == #color frames == len(gripper_distances)
  E0  strawberry near START: red-pixel count in circle around projected tip at frame 0
  E1  strawberry below END:   circle centered at (end_px + (0, DY)), red-pixel count >= N_RED
  V1  max per-frame tip step (mm) <= V_MAX  (trajectory glitch => single-frame jumps)

Usage:
  python3 auto_traj_qc.py tune <session-png>/<episode>   # overlay for tuning
"""
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

EXTRINSICS = Path("/home/ss/sroi_dataprocess/configs/camera_gripper_extrinsics_sroi_v2_d405.json")
ROOTS = [Path("/home/ss/data/1000_onesb_labpicking"), Path("/home/ss/data/120_onesb_supplementary")]
MD_LIST = Path("/home/ss/data/QC工作汇总_20261013/轨迹QC/剔除清单_EXCLUDED.md")
OUT_CSV = Path("/home/ss/data/rebuild_qc_20261013/auto_traj_qc.csv")
OUT_OVERLAY = Path("/home/ss/data/rebuild_qc_20261013/auto_qc_overlay")

# --- tunables (adjust against overlays) ---
DY = 35          # px, search-window center offset below the projected end point
R_END = 40       # px, end-window radius
R_START = 40     # px, start-window radius
N_RED = 150      # min red pixels inside a window to call "strawberry present"
V_MAX = 25.0     # mm per frame, tip-step cap
RED = dict(r_min=70, rg=35, rb=35)  # BGR red test: r>r_min and r-g>rg and r-b>rb


def load_extrinsics():
    j = json.load(open(EXTRINSICS))
    T_opt_link = np.array(j["T_optical_camera"], dtype=float)
    T_link_tip = np.array(j["T_camera_gripper_tip"], dtype=float)
    return (T_opt_link @ T_link_tip)[:3, 3]


def red_mask(img):
    b, g, r = cv2.split(img.astype(np.int16))
    return ((r > RED["r_min"]) & (r - g > RED["rg"]) & (r - b > RED["rb"]))


def circle_count(mask, cx, cy, rad):
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    return int((mask & ((xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad)).sum())


def episode_metrics(ep: Path, tip_in_opt):
    """Return metrics dict, or None if files missing."""
    traj_fp = ep / "CameraTrajectory.txt"
    img_fp = ep / "color_000000.png"
    info_fp = ep / "camera_info_color.json"
    grip_fp = ep / "gripper_distances.txt"
    if not (traj_fp.exists() and img_fp.exists() and info_fp.exists()):
        return None
    traj = np.loadtxt(traj_fp).reshape(-1, 3, 4)
    R, t = traj[:, :, :3], traj[:, :, 3]
    pts_w = (R @ tip_in_opt[:, None]).squeeze(2) + t

    steps = np.linalg.norm(np.diff(pts_w, axis=0), axis=1) * 1000.0
    vmax = float(steps.max()) if len(steps) else 0.0

    info = json.load(open(info_fp))
    K = np.array(info["K"], dtype=float).reshape(3, 3)
    D = np.array(info["D"], dtype=float).ravel()
    pts2d, _ = cv2.projectPoints(pts_w.astype(np.float64), np.zeros(3), np.zeros(3), K, D)
    pts2d = pts2d.reshape(-1, 2)
    h, w = cv2.imread(str(img_fp)).shape[:2]
    start_px = tuple(np.clip(pts2d[0], [0, 0], [w - 1, h - 1]).astype(int))
    end_px = tuple(np.clip(pts2d[-1], [0, 0], [w - 1, h - 1]).astype(int))

    n_img = len(list(ep.glob("color_*.png")))
    n_grip = len(grip_fp.read_text().strip().splitlines()) if grip_fp.exists() else -1

    img = cv2.imread(str(img_fp))
    m = red_mask(img)
    red_start = circle_count(m, *start_px, R_START)
    end_cx = int(np.clip(end_px[0], 0, w - 1))
    end_cy = int(np.clip(end_px[1] + DY, 0, h - 1))
    red_end = circle_count(m, end_cx, end_cy, R_END)

    return dict(n_traj=len(traj), n_img=n_img, n_grip=n_grip, vmax=vmax,
                red_start=red_start, red_end=red_end,
                start_px=start_px, end_px=end_px)


def overlay(ep: Path, tip_in_opt, out_fp: Path):
    """Draw detection windows + red-mask highlight on the first frame."""
    met = episode_metrics(ep, tip_in_opt)
    if met is None:
        return
    img = cv2.imread(str(ep / "color_000000.png"))
    m = red_mask(img)
    img[m] = (255, 0, 255)  # magenta = pixels counted as red
    sx, sy = met["start_px"]
    cv2.circle(img, (sx, sy), R_START, (0, 255, 0), 2)
    cv2.putText(img, str(met["red_start"]), (sx + R_START + 4, sy), 0, 0.6, (0, 255, 0), 2)
    ex, ey = met["end_px"]
    cv2.drawMarker(img, (ex, ey), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(img, (ex, ey + DY), R_END, (0, 255, 255), 2)
    cv2.putText(img, str(met["red_end"]), (ex + R_END + 4, ey + DY), 0, 0.6, (0, 255, 255), 2)
    cv2.putText(img, f"vmax={met['vmax']:.1f}mm  n={met['n_traj']}/{met['n_img']}/{met['n_grip']}",
                (8, 22), 0, 0.6, (255, 255, 255), 2)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_fp), img)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    tip_in_opt = load_extrinsics()
    r1 = set(re.findall(r"^(\d{8}_\d{6}-png/episode_\d+)$", MD_LIST.read_text(), re.M))

    eps = []
    roots = [Path(p) for p in sys.argv[2:]] or ROOTS  # optional: override episode roots
    for root in roots:
        for sess in sorted(root.glob("*-png")):
            eps += sorted(p for p in sess.iterdir() if p.is_dir() and p.name.startswith("episode_"))

    if mode == "tune":
        for key in sys.argv[2:]:
            sess, epn = key.split("/")
            ep = Path(ROOTS[0].parent) / "1000_onesb_labpicking" / sess / epn
            if not ep.exists():
                ep = Path(ROOTS[1].parent) / "120_onesb_supplementary" / sess / epn
            overlay(ep, tip_in_opt, OUT_OVERLAY / f"{sess}_{epn}.png")
            print("saved", OUT_OVERLAY / f"{sess}_{epn}.png")
        return

    rows = []
    for k, ep in enumerate(eps, 1):
        key = f"{ep.parent.name}/{ep.name}"
        met = episode_metrics(ep, tip_in_opt)
        if met is None:
            rows.append((key, False, "missing_files", *([0] * 5)))
            continue
        f1 = met["n_traj"] == met["n_img"] == met["n_grip"]
        e0 = met["red_start"] >= N_RED
        e1 = met["red_end"] >= N_RED
        v1 = met["vmax"] <= V_MAX
        ok = f1 and e0 and e1 and v1
        why = "+".join(n for n, p in [("F1", f1), ("E0", e0), ("E1", e1), ("V1", v1)] if not p) or "-"
        rows.append((key, ok, why, met["n_traj"], met["n_img"], met["n_grip"],
                     round(met["vmax"], 1), met["red_start"], met["red_end"]))
        if k % 200 == 0:
            print(f"  {k}/{len(eps)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "ok", "fail", "n_traj", "n_img", "n_grip", "vmax_mm", "red_start", "red_end"])
        w.writerows(rows)

    # confusion vs manual R1
    tp = sum(1 for r in rows if r[0] in r1 and not r[1])          # manual BAD, auto BAD
    fn = [r[0] for r in rows if r[0] in r1 and r[1]]              # manual BAD, auto PASS (missed)
    fp = [r for r in rows if r[0] not in r1 and not r[1]]         # manual OK, auto BAD (overkill)
    print(f"\ntotal {len(rows)} | auto-BAD {sum(1 for r in rows if not r[1])}")
    print(f"R1 recall: {tp}/{len(r1)}  missed: {fn}")
    print(f"overkill (manual-OK but auto-BAD): {len(fp)}")
    for r in fp[:30]:
        print("  ", r[0], r[2], f"vmax={r[6]} red_s={r[7]} red_e={r[8]}")
    # distribution context for tuning
    vmaxes = sorted(r[6] for r in rows if r[6])
    print(f"\nvmax p50={vmaxes[len(vmaxes)//2]} p99={vmaxes[int(len(vmaxes)*0.99)]} max={vmaxes[-1]}")
    fc = [(r[0], r[3], r[4], r[5]) for r in rows if r[3] != r[4] or r[4] != r[5]]
    print(f"frame-count mismatches: {len(fc)}")
    for t in fc[:20]:
        print("  ", t)


if __name__ == "__main__":
    main()
