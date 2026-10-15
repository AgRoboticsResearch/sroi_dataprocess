#!/usr/bin/env python3
"""Render an N-panel RGB comparison video for each episode, one panel per scheme.
Each panel overlays the projected fingertip path from green to red, with a green current
point and blue endpoint. Episode videos are concatenated into ALL_compare.mp4.
Transform all trajectories first. Usage: viz_scheme_compare.py <schemes_dir> [--schemes ...]
"""
import sys, argparse, subprocess
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualization"))
import visualize_traj_video as vtv  # Reuse trajectory loading and projection helpers


def render_episode(ep_dir, schemes, fps, codec, out_mp4, tip_kin):
    color = sorted([*ep_dir.glob("color_*.png"), *ep_dir.glob("color_*.jpg")])
    K = vtv.load_K(ep_dir / "camera_info_color.json")
    poses_per = {}
    for sch in schemes:
        sd = ep_dir / sch
        t = sd / "CameraTrajectoryTransformed.txt"
        if not t.exists():
            t = sd / "CameraTrajectory.txt"
        poses_per[sch] = vtv.load_poses(t)
    if K is None or not color or not any(p is not None for p in poses_per.values()):
        return False
    n = len(color)
    if n < 2:
        return False
    color = color[:n]
    for s in schemes:
        if poses_per[s] is not None:
            poses_per[s] = poses_per[s][:n]

    first_bgr = cv2.imread(str(color[0]))
    if first_bgr is None:
        raise RuntimeError(f"failed to read image: {color[0]}")
    first = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    H, W = first.shape[:2]
    nS = len(schemes)
    fig, axes = plt.subplots(1, nS, figsize=(4.4 * nS, 4.3), dpi=100)
    if nS == 1:
        axes = [axes]
    im_objs, fut_lcs, start_dots, stop_dots = [], [], [], []
    for ax, sch in zip(axes, schemes):
        im_objs.append(ax.imshow(first)); ax.set_xticks([]); ax.set_yticks([])
        tracked = len(poses_per[sch]) if poses_per[sch] is not None else 0
        ax.set_title(f"{sch} ({tracked}/{len(color)} frames)",
                     fontsize=12, fontweight="bold")
        lc = LineCollection([], linewidths=2.0, capstyle="round", zorder=4)
        ax.add_collection(lc); fut_lcs.append(lc)
        start_dots.append(ax.scatter([], [], c=["#00FF00"], s=45, zorder=6, edgecolors="black", linewidths=0.4))
        stop_dots.append(ax.scatter([], [], c=["#0000FF"], s=45, zorder=6, edgecolors="black", linewidths=0.4))
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.05)
    fig.suptitle(f"{ep_dir.parent.name}/{ep_dir.name}", fontsize=11, y=0.97)

    cw, ch = fig.canvas.get_width_height()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{cw}x{ch}", "-r", str(fps), "-i", "-",
           "-c:v", codec, "-pix_fmt", "yuv420p", "-r", str(fps), str(out_mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for t in range(n):
            frame = cv2.imread(str(color[t]))
            if frame is None:
                raise RuntimeError(f"failed to read image: {color[t]}")
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for i, sch in enumerate(schemes):
                im_objs[i].set_data(img)
                if poses_per[sch] is not None and t < len(poses_per[sch]):
                    px, py = vtv.project_future(poses_per[sch], t, K, tip_kin)
                    pts = np.column_stack([px, py])
                    segs = (np.stack([pts[:-1], pts[1:]], axis=1)
                            if len(pts) >= 2 else np.empty((0, 2, 2)))
                    fut_lcs[i].set_segments(segs)
                    fut_lcs[i].set_colors(vtv._green_red_gradient(len(segs)))
                    start_dots[i].set_offsets([pts[0]] if len(pts) else [[np.nan, np.nan]])
                    spt = (pts[-1] if len(pts) and np.isfinite(pts[-1]).all()
                           else np.array([np.nan, np.nan]))
                    stop_dots[i].set_offsets([spt])
                else:
                    fut_lcs[i].set_segments([])
                    start_dots[i].set_offsets([[np.nan, np.nan]])
                    stop_dots[i].set_offsets([[np.nan, np.nan]])
            fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
    except Exception:
        proc.kill(); plt.close(fig); raise
    out, err = proc.communicate(); plt.close(fig)
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schemes_dir")
    ap.add_argument("--schemes", nargs="+", default=["raw", "maskhalf", "maskgripper"])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--codec", default="libopenh264")
    ap.add_argument(
        "--extrinsics-config",
        type=Path,
        default=vtv.DEFAULT_EXTRINSICS_CONFIG,
    )
    args = ap.parse_args()
    try:
        tip_kin = vtv.load_tip_kin(args.extrinsics_config)
    except ValueError as error:
        ap.error(str(error))
    sd = Path(args.schemes_dir)
    outdir = sd / "_compare"  # Keep comparison output inside the schemes tree
    outdir.mkdir(exist_ok=True)
    eps = sorted(d for d in sd.iterdir() if d.is_dir() and d.name.startswith("episode_"))
    print(f"{sd.name}: {len(eps)} episodes x {args.schemes} -> {outdir}", flush=True)
    ok = 0
    for i, ep in enumerate(eps, 1):
        mp4 = outdir / f"{ep.name}.mp4"
        if mp4.exists() and mp4.stat().st_size > 10000:
            print(f"[{i}/{len(eps)}] {ep.name} skip", flush=True); continue
        if render_episode(ep, args.schemes, args.fps, args.codec, mp4, tip_kin):
            ok += 1; print(f"[{i}/{len(eps)}] {ep.name} ok", flush=True)
        else:
            print(f"[{i}/{len(eps)}] {ep.name} skip(no data)", flush=True)
    print("concatenating...", flush=True)
    present = [e for e in eps if (outdir / f"{e.name}.mp4").exists()]
    lf = outdir / "_concat.txt"
    with open(lf, "w") as f:
        for e in present:
            f.write(f"file '{(outdir / f'{e.name}.mp4').resolve()}'\n")
    final = outdir / "ALL_compare.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lf),
                    "-c", "copy", str(final)], capture_output=True)
    print(f"DONE: {final} ({ok} episodes)", flush=True)


if __name__ == "__main__":
    main()
