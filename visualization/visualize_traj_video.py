#!/usr/bin/env python3
"""
Side-by-side video: camera color frames (left), an animated 3D ORB-SLAM3 trajectory
(middle), and the gripper-openness curve (right), for manual trajectory QC.
Encodes H.264 MP4 via ffmpeg (libopenh264, since libx264 isn't installed here).

Per episode, reads CameraTrajectoryTransformed.txt (falls back to CameraTrajectory.txt)
and the color_*.png frames, and writes <episode>/traj_sidebyside.mp4. The 3D panel
uses a FIXED axis extent computed from the full trajectory (stable view, no rescaling)
and progressively reveals the path with the current pose highlighted. The gripper
panel shows the full gripper_distances.txt curve with a dot tracking the current frame
(or "no gripper data" when the file is absent).

Usage (from repo root):
    # single episode
    python visualization/visualize_traj_video.py .../session-png/episode_001
    # a whole session / parent tree
    python visualization/visualize_traj_video.py .../session-png --recursive
    python visualization/visualize_traj_video.py .../pngs --recursive
    # use the calibration for a specific camera/gripper assembly
    python visualization/visualize_traj_video.py .../session-png --recursive \
        --extrinsics-config configs/camera_gripper_extrinsics_sroi_v2_d405.json
    # QC-badged videos for a whole processed tree: badge + filename prefix come
    # from qc.py's qc.csv, colors from configs/qc_filters.json (requires --output)
    python visualization/visualize_traj_video.py .../pngs --recursive \
        --output .../traj_videos --qc-csv .../pngs/qc.csv
Options: --fps 30 --codec libopenh264 --skip-existing --max-episodes N
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
import numpy as np

# Projection extrinsics are loaded from a standalone rig config so different
# camera/gripper assemblies can provide their own measured transforms.
DEFAULT_EXTRINSICS_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "camera_gripper_extrinsics_sroi_v2_d405.json"
)


def _load_rigid_transform(config: dict, key: str, config_path: Path) -> np.ndarray:
    try:
        transform = np.asarray(config[key], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{config_path}: {key} must be a numeric 4x4 matrix") from error
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{config_path}: {key} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{config_path}: {key} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{config_path}: {key} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{config_path}: {key} rotation determinant must be +1")
    return transform


def load_tip_kin(config_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load optical<-camera and camera<-gripper-tip transforms from JSON."""
    config_path = Path(config_path).resolve()
    try:
        with config_path.open() as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load extrinsics config {config_path}: {error}") from error
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"{config_path}: schema_version must be 1")
    return (
        _load_rigid_transform(config, "T_optical_camera", config_path),
        _load_rigid_transform(config, "T_camera_gripper_tip", config_path),
    )


TIP_KIN = load_tip_kin(DEFAULT_EXTRINSICS_CONFIG)


def load_poses(traj_path: Path) -> np.ndarray | None:
    """Load a KITTI-format trajectory -> (N, 4, 4) camera-in-world poses (meters), or None."""
    if not traj_path.exists():
        return None
    try:
        raw = np.loadtxt(traj_path, comments="#")
    except Exception:
        return None
    if raw.size == 0:
        return None
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] == 13:
        raw = raw[:, 1:]
    if raw.shape[1] < 12:
        return None
    poses34 = raw[:, :12].reshape(-1, 3, 4)
    bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(poses34), 1, 1))
    return np.concatenate([poses34, bottom], axis=1)  # (N, 4, 4)


def load_gripper(ep_dir: Path) -> np.ndarray | None:
    """Load normalized gripper distances (one float per frame), or None."""
    path = ep_dir / "gripper_distances.txt"
    if not path.exists():
        return None
    try:
        raw = np.loadtxt(path)
    except Exception:
        return None
    if raw.size == 0:
        return None
    return np.atleast_1d(raw)


def load_K(cam_info_path: Path) -> np.ndarray | None:
    """Load the color camera intrinsics (3x3) from camera_info_color.json, or None."""
    import json
    if not cam_info_path.exists():
        return None
    try:
        K = np.array(json.load(cam_info_path.open())["K"], dtype=float).reshape(3, 3)
    except Exception:
        return None
    return K


def project_future(
    poses: np.ndarray,
    t: int,
    K: np.ndarray,
    tip_kin: tuple[np.ndarray, np.ndarray] = TIP_KIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the gripper-tip path from the CURRENT frame t through the end into
    frame t's image (includes the current tip so the trail starts "now").

    Mirrors visualize_predictions.py: poses are treated as camera_link poses; the
    relative motion T_rel = inv(poses[t]) @ poses[k] is composed with the configured
    camera-to-tip transforms, giving the gripper tip in the current optical frame:
        p_tip = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]
    then projected with the (optical) color intrinsics K. Returns (px, py) for k=t..end
    in pixel coords (NaN where behind the camera).
    """
    T_t_inv = np.linalg.inv(poses[t])
    T_opt_cam, T_cam_ee = tip_kin
    pts = np.empty((len(poses) - t, 3))
    for i, k in enumerate(range(t, len(poses))):
        T_rel = T_t_inv @ poses[k]
        pts[i] = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]
    ox, oy, oz = pts[:, 0], pts[:, 1], pts[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    px = fx * ox / oz + cx
    py = fy * oy / oz + cy
    behind = oz <= 1e-3
    px = np.where(behind, np.nan, px)
    py = np.where(behind, np.nan, py)
    return px, py


def _green_red_gradient(n_seg: int) -> list[tuple[float, float, float]]:
    """vispred-style per-segment colors: green (start) -> red (end)."""
    if n_seg <= 1:
        return [(0.0, 1.0, 0.0)]
    return [((i / (n_seg - 1)), (1.0 - i / (n_seg - 1)), 0.0) for i in range(n_seg)]


QC_FILTERS_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "qc_filters.json"
)


def load_qc_statuses(qc_csv: Path) -> dict[Path, str]:
    """episode-dir -> QC category, from a visualization/qc.py CSV."""
    with qc_csv.open() as f:
        rows = list(csv.DictReader(f))
    statuses = {}
    for row in rows:
        episode = row.get("path") or ""
        category = row.get("category") or ""
        if episode and category:
            statuses[Path(episode).resolve()] = category
    if not statuses:
        raise ValueError(f"{qc_csv}: no usable rows (need 'path' and 'category' columns)")
    return statuses


def load_qc_colors() -> dict[str, str]:
    """category -> badge color, matching qc.py's PDF color coding."""
    with QC_FILTERS_CONFIG.open() as f:
        categories = json.load(f)["categories"]
    return {k: v["color"] for k, v in categories.items() if k != "_doc"}


def _out_mp4_path(ep_dir: Path, out_dir: Path | None, prefix: str | None = None) -> Path:
    """Output mp4 path for an episode: <out_dir>/<prefix_><session>_<episode>.mp4
    if out_dir is given (all videos in one folder; the optional prefix is e.g. the
    QC category), else <episode>/traj_sidebyside.mp4."""
    if out_dir is not None:
        stem = ep_dir.parent.name.replace("-png", "") + "_" + ep_dir.name
        if prefix:
            stem = f"{prefix}_{stem}"
        return out_dir / f"{stem}.mp4"
    return ep_dir / "traj_sidebyside.mp4"


def render_episode_video(ep_dir: Path, fps: int, codec: str, dpi: int = 100,
                         out_dir: Path | None = None,
                         badge: str | None = None, badge_color: str = "#2e7d32",
                         prefix: str | None = None,
                         tip_kin: tuple[np.ndarray, np.ndarray] = TIP_KIN) -> bool:
    """Render one side-by-side traj video. Returns True on success.

    Writes to <out_dir>/<prefix_><session>_<episode>.mp4 when out_dir is set (all
    videos in one folder; prefix is e.g. the QC category), otherwise
    <episode>/traj_sidebyside.mp4 next to the data.
    """
    poses = load_poses(ep_dir / "CameraTrajectoryTransformed.txt")
    if poses is None:
        poses = load_poses(ep_dir / "CameraTrajectory.txt")
    if poses is None:
        print(f"  skip {ep_dir.name}: no trajectory")
        return False
    images = sorted([*ep_dir.glob("color_*.png"), *ep_dir.glob("color_*.jpg")])
    if not images:
        # Stereo-only (oak) episodes: the left camera IS the color camera.
        images = sorted([*ep_dir.glob("left_*.png"), *ep_dir.glob("left_*.jpg")])
    if not images:
        print(f"  skip {ep_dir.name}: no color frames")
        return False
    K = load_K(ep_dir / "camera_info_color.json")
    if K is None:
        K = load_K(ep_dir / "camera_info_left.json")
    n = min(len(poses), len(images))
    if n < 2:
        print(f"  skip {ep_dir.name}: too few frames ({n})")
        return False
    poses = poses[:n]
    images = images[:n]
    pos = poses[:, :3, 3]

    # Full-traj extent (fixed axes, padded) for a stable 3D view.
    pad = max((pos.max(axis=0) - pos.min(axis=0)).max() * 0.08, 0.02)
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    path_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

    # Figure: left = color frame (+ projected future traj), middle = 3D trajectory,
    # right = the full gripper-openness curve with a moving current-frame marker.
    fig = plt.figure(figsize=(16.2, 4.8), dpi=dpi)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.1, 0.75])
    ax_img = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")
    ax_grip = fig.add_subplot(grid[0, 2])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.20)

    if badge:  # optional status badge (colored block + text), top-left of the video
        fig.text(0.005, 0.985, " " + badge + " ", color="white", fontsize=12,
                 fontweight="bold", va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.35", fc=badge_color, ec="black", lw=1.0))

    first = cv.imread(str(images[0]))
    if first is None:
        print(f"  skip {ep_dir.name}: failed to read {images[0]}")
        plt.close(fig)
        return False
    im_obj = ax_img.imshow(cv.cvtColor(first, cv.COLOR_BGR2RGB))
    ax_img.set_xticks([]); ax_img.set_yticks([])
    title = ax_img.set_title(f"{ep_dir.parent.name}/{ep_dir.name}  f0/{n}", fontsize=10)
    # projected gripper-tip path overlay: green->red gradient (vispred style) +
    # green start dot (current tip) + blue stop dot (final tip). Updated per frame.
    fut_lc = LineCollection([], linewidths=2.2, capstyle="round", zorder=4)
    ax_img.add_collection(fut_lc)
    start_dot = ax_img.scatter([], [], c=["#00FF00"], s=55, zorder=6, edgecolors="black", linewidths=0.5)
    stop_dot = ax_img.scatter([], [], c=["#0000FF"], s=55, zorder=6, edgecolors="black", linewidths=0.5)
    ax_img.set_xlim(0, first.shape[1]); ax_img.set_ylim(first.shape[0], 0)  # image pixel coords

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    line, = ax3d.plot([x[0]], [y[0]], [z[0]], color="#0072B2", lw=2.0)  # blue (CB-safe)
    head = ax3d.scatter([x[0]], [y[0]], [z[0]], c=["#D55E00"], s=42, zorder=5)  # vermillion
    ax3d.scatter([x[0]], [y[0]], [z[0]], c=["#009E73"], s=30, marker="o", zorder=4)  # start=green
    ax3d.set_xlim(lo[0], hi[0]); ax3d.set_ylim(lo[1], hi[1]); ax3d.set_zlim(lo[2], hi[2])
    ax3d.set_xlabel("X (m)", fontsize=8); ax3d.set_ylabel("Y (m)", fontsize=8); ax3d.set_zlabel("Z (m)", fontsize=8)
    ax3d.tick_params(labelsize=7)
    try:
        ax3d.set_box_aspect((hi[0]-lo[0], hi[1]-lo[1], max(hi[2]-lo[2], 1e-3)))
    except Exception:
        pass
    ax3d.view_init(elev=22, azim=-60)
    ax3d.set_title(f"3D trajectory  ({path_len*100:.1f} cm)", fontsize=10)

    # Gripper panel: full curve always visible, dot tracks the current frame.
    grip = load_gripper(ep_dir)
    grip = grip[:n] if grip is not None else None
    if grip is not None:
        ax_grip.plot(np.arange(len(grip)), grip, color="#6a1b9a", lw=1.6)
        grip_dot = ax_grip.scatter([0], [grip[0]], c=["#D55E00"], s=42, zorder=5,
                                   edgecolors="black", linewidths=0.5)
        ax_grip.set_xlim(0, max(n - 1, 1))
        ax_grip.set_ylim(-0.05, 1.05)
        ax_grip.set_yticks([0.0, 0.5, 1.0])
    else:
        grip_dot = None
        ax_grip.text(0.5, 0.5, "no gripper data", ha="center", va="center",
                     transform=ax_grip.transAxes, fontsize=9, color="#757575")
    ax_grip.set_title("gripper (0=closed, 1=open)", fontsize=10)
    ax_grip.set_xlabel("frame", fontsize=8)
    ax_grip.tick_params(labelsize=7)

    canvas_w, canvas_h = fig.canvas.get_width_height()
    out_path = _out_mp4_path(ep_dir, out_dir, prefix)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{canvas_w}x{canvas_h}", "-r", str(fps), "-i", "-",
        "-c:v", codec, "-pix_fmt", "yuv420p", "-r", str(fps),
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        for t in range(n):
            img = cv.imread(str(images[t]))
            if img is None:
                raise RuntimeError(f"failed to read image: {images[t]}")
            im_obj.set_data(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            title.set_text(f"{ep_dir.parent.name}/{ep_dir.name}  f{t}/{n-1}")
            # 3D panel: revealed path up to t + current head
            line.set_data(x[:t + 1], y[:t + 1])
            line.set_3d_properties(z[:t + 1])
            head._offsets3d = ([x[t]], [y[t]], [z[t]])
            # RGB panel: project gripper-tip path (current..end) into the frame,
            # drawn as a green->red gradient with green start (current tip) + blue stop.
            if K is not None:
                px, py = project_future(poses, t, K, tip_kin)
                pts = np.column_stack([px, py])
                segs = np.stack([pts[:-1], pts[1:]], axis=1) if len(pts) >= 2 else np.empty((0, 2, 2))
                fut_lc.set_segments(segs)
                fut_lc.set_colors(_green_red_gradient(len(segs)))
                start_dot.set_offsets([pts[0]] if len(pts) else [[np.nan, np.nan]])
                stop_pt = pts[-1] if len(pts) and np.isfinite(pts[-1]).all() else np.array([np.nan, np.nan])
                stop_dot.set_offsets([stop_pt])
            else:
                fut_lc.set_segments([])
                start_dot.set_offsets([[np.nan, np.nan]]); stop_dot.set_offsets([[np.nan, np.nan]])
            if grip_dot is not None and t < len(grip):
                grip_dot.set_offsets([[t, grip[t]]])
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]  # (H, W, 3)
            proc.stdin.write(rgb.tobytes())
    except Exception:
        proc.kill()
        plt.close(fig)
        raise
    out, err = proc.communicate()  # closes stdin + waits
    plt.close(fig)
    if proc.returncode != 0:
        print(f"  ffmpeg error for {ep_dir.name}: {err.decode()[:300]}")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Episode dir, or session/parent with --recursive")
    p.add_argument("--recursive", action="store_true", help="Process all episode_* dirs under path")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--codec", default="libopenh264", help="ffmpeg video codec (default libopenh264 = H.264)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--max-episodes", type=int, default=0, help="Stop after N episodes (0 = no limit)")
    p.add_argument("--output", type=Path, default=None,
                   help="Write ALL videos into this folder as <session>_<episode>.mp4 (default: <episode>/traj_sidebyside.mp4)")
    p.add_argument("--badge", default=None, help="Optional status badge text drawn top-left of the video")
    p.add_argument("--badge-color", default="#2e7d32", help="Badge background color (default green)")
    p.add_argument("--qc-csv", type=Path, default=None,
                   help="qc.py output CSV: render each episode with its QC category as a "
                        "colored badge and filename prefix (requires --output; episodes "
                        "missing from the CSV are skipped)")
    p.add_argument(
        "--extrinsics-config",
        type=Path,
        default=DEFAULT_EXTRINSICS_CONFIG,
        help=f"Camera-to-gripper-tip JSON config (default: {DEFAULT_EXTRINSICS_CONFIG})",
    )
    args = p.parse_args(argv)

    try:
        tip_kin = load_tip_kin(args.extrinsics_config)
    except ValueError as error:
        p.error(str(error))

    qc_statuses = None
    qc_colors = {}
    if args.qc_csv is not None:
        if args.output is None:
            p.error("--qc-csv requires --output (badged videos go to one folder)")
        try:
            qc_statuses = load_qc_statuses(args.qc_csv)
        except (OSError, ValueError) as error:
            p.error(str(error))
        qc_colors = load_qc_colors()

    if not args.path.exists():
        print(f"not found: {args.path}", file=sys.stderr); return 1

    if args.recursive:
        eps = sorted([d for d in args.path.rglob("episode_*") if d.is_dir()])
    elif (args.path / "CameraTrajectoryTransformed.txt").exists() or (args.path / "CameraTrajectory.txt").exists():
        eps = [args.path]  # single episode dir
    else:
        eps = sorted([d for d in args.path.glob("episode_*") if d.is_dir()])  # session dir

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {args.output} ({len(eps)} episodes)")

    done = 0; ok = 0
    for ep in eps:
        badge, badge_color, prefix = args.badge, args.badge_color, None
        if qc_statuses is not None:
            category = qc_statuses.get(ep.resolve())
            if category is None:
                print(f"  skip {ep.parent.name}/{ep.name}: not in {args.qc_csv.name}")
                continue
            badge, badge_color, prefix = category.upper(), qc_colors.get(category, "#757575"), category
        out = _out_mp4_path(ep, args.output, prefix)
        if args.skip_existing and out.exists():
            continue
        if args.max_episodes and ok >= args.max_episodes:
            break
        print(f"[{done+1}/{len(eps)}] {ep.parent.name}/{ep.name}")
        render_kwargs = {} if badge is None else {"badge": badge, "badge_color": badge_color}
        if prefix is not None:
            render_kwargs["prefix"] = prefix
        if render_episode_video(
            ep,
            args.fps,
            args.codec,
            out_dir=args.output,
            **render_kwargs,
            tip_kin=tip_kin,
        ):
            ok += 1
        done += 1
    print(f"Done: {ok} video(s) rendered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
