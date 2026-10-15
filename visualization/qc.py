#!/usr/bin/env python3
"""
QC tool for ORB-SLAM + gripper estimation outputs.

Scans a session tree (default: <root>/*-png/episode_*/) and produces:
  - <out>.csv : per-episode metrics (path length, gripper std, category, ...)
  - <out>.pdf : one summary page + one page per session. Each episode shows
                [RGB+projected gripper path][3D trajectory][gripper] (the default
                "rich" view, mirroring the traj videos); --simple reverts to a
                2D top-down trajectory + gripper. Color-coded by category.

Designed to be re-runnable as data updates. Idempotent — overwrites outputs.

CLI usage:
    # Default: scan all *-png sessions, write qc.csv + qc.pdf in cwd
    python visualization/qc.py /path/to/20260611

    # Custom output prefix (writes <prefix>.csv and <prefix>.pdf)
    python visualization/qc.py /path/to/20260611 -o /tmp/qc

    # Only one kind of output
    python visualization/qc.py /path/to/20260611 --csv-only
    python visualization/qc.py /path/to/20260611 --pdf-only

    # Glob for sessions with a different pattern
    python visualization/qc.py /path/to/data -p '*-jpeg'

    # Override thresholds / colors / layout from a custom JSON
    python visualization/qc.py /path/to/data --filters /path/to/qc_filters.json

Config:
    All thresholds, category metadata, colors, and page layout live in
    configs/qc_filters.json (next to this script). Edit that file to tune
    behavior; pass --filters <path> to swap in a different config per run.
    The JSON is auto-loaded at import time, so importers don't need an
    explicit load_filters() call unless they want to override the path.

API usage:
    from visualization.qc import scan_sessions, write_csv, render_pdf
    sessions = scan_sessions(Path("/path/to/data"))
    write_csv(sessions, Path("qc.csv"))
    render_pdf(sessions, Path("qc.pdf"))

    # Override filters programmatically:
    from visualization.qc import load_filters
    load_filters(Path("/path/to/custom.json"))  # before scan_sessions

Categories (per episode), evaluated in priority order (first match wins;
evidence columns keep every check's numbers regardless of the final label):
    not_processed           - trajectory/gripper stream missing or too short
                              (plumbing guard: an upstream step did not run)
    traj_long_gap           - trajectory ledger has 'invalid' rows (long tracking loss)
    gripper_missing         - gripper_distances.txt contains any NaN (incl. all-NaN)
    traj_frame_jump         - per-frame step above threshold on the SMOOTHED trajectory
    first_frame_projection  - projected tip path on the first color frame finds no
                              strawberry near the start tip and/or below the end point
    gripper_not_closed      - mean of final finite readings is above the threshold
    ok                      - everything passed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the "3d" projection)
import numpy as np

try:
    import cv2  # projection verdict needs it; CSV-only mode works without
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


# ============================================================================
# Config — all tunable values live in configs/qc_filters.json
# ============================================================================

DEFAULT_FILTERS_PATH = Path(__file__).resolve().parent.parent / "configs" / "qc_filters.json"

# Module-level config vars. Populated by load_filters() below; read by the rest
# of the module. Names mirror the old hardcoded constants so existing call sites
# and importers (scan_sessions, classify, render_pdf) work unchanged.
STATIC_PATH_M: float = 0.0
FLAT_GRIPPER_STD: float = 0.0
MIN_TRAJ_FRAMES: int = 2
EXTENT_PADDING_RATIO: float = 0.05
EXTENT_PADDING_MIN_CM: float = 0.5
GRIPPER_END_OPEN_THRESHOLD: float = 0.2
GRIPPER_END_TAIL: int = 10
# Projection verdict + frame-jump threshold (from cfg["projection"])
V_MAX_MM: float = 25.0
PROJ_DY: int = 35
PROJ_R_START: int = 40
PROJ_R_END: int = 40
PROJ_N_RED: int = 150
RED_HUE1_MAX: int = 10
RED_HUE2_MIN: int = 160
RED_S_MIN: int = 60
RED_V_MIN: int = 60
EXTRINSICS_CONFIG: str = ""
CATEGORY_COLORS: dict[str, str] = {}
CATEGORY_DEFAULT_COLOR: str = "#999999"
PROBLEM_CATEGORIES: set[str] = set()
CATEGORY_SORT_ORDER: dict[str, int] = {}
TRAJ_COLOR: str = ""
GRIP_COLOR: str = ""
PANEL_BG: str = ""
GRIPPER_YLIM: tuple[float, float] = (-0.05, 1.05)
EPISODES_PER_ROW: int = 3
FIGSIZE_SESSION: tuple[float, float] = (11.0, 2.4)
FIGSIZE_SUMMARY: tuple[float, float] = (11.0, 0.4)
FIGSIZE_SUMMARY_BASE_HEIGHT: float = 1.5
OUTER_HSPACE: float = 0.55
OUTER_WSPACE: float = 0.30
INNER_WSPACE: float = 0.30
OUTER_MARGINS: tuple[float, float, float, float] = (0.04, 0.98, 0.95, 0.04)
# Rich (default) layout: per-episode page = [RGB+projection][3D trajectory][gripper].
EPISODES_PER_ROW_RICH: int = 2
FIGSIZE_SESSION_RICH: tuple[float, float] = (13.0, 3.2)
INNER_WSPACE_RICH: float = 0.25


def load_filters(path: Path | None = None) -> None:
    """Load QC filter rules from JSON into module-level config vars.

    Uses DEFAULT_FILTERS_PATH (configs/qc_filters.json next to this script) when
    path is None. Reassigns STATIC_PATH_M, CATEGORY_COLORS, etc. so the rest of
    the module picks up the new values on the next call.
    """
    global STATIC_PATH_M, FLAT_GRIPPER_STD, MIN_TRAJ_FRAMES
    global EXTENT_PADDING_RATIO, EXTENT_PADDING_MIN_CM
    global GRIPPER_END_OPEN_THRESHOLD, GRIPPER_END_TAIL
    global V_MAX_MM, PROJ_DY, PROJ_R_START, PROJ_R_END, PROJ_N_RED
    global RED_HUE1_MAX, RED_HUE2_MIN, RED_S_MIN, RED_V_MIN, EXTRINSICS_CONFIG
    global CATEGORY_COLORS, CATEGORY_DEFAULT_COLOR, PROBLEM_CATEGORIES, CATEGORY_SORT_ORDER
    global TRAJ_COLOR, GRIP_COLOR, PANEL_BG, GRIPPER_YLIM
    global EPISODES_PER_ROW, FIGSIZE_SESSION, FIGSIZE_SUMMARY, FIGSIZE_SUMMARY_BASE_HEIGHT
    global OUTER_HSPACE, OUTER_WSPACE, INNER_WSPACE, OUTER_MARGINS
    global EPISODES_PER_ROW_RICH, FIGSIZE_SESSION_RICH, INNER_WSPACE_RICH

    filters_path = path or DEFAULT_FILTERS_PATH
    if not filters_path.exists():
        raise FileNotFoundError(
            f"QC filters file not found: {filters_path}\n"
            f"Expected configs/qc_filters.json next to visualization/qc.py."
        )
    with filters_path.open() as f:
        cfg = json.load(f)

    t = cfg["thresholds"]
    STATIC_PATH_M = float(t.get("static_path_m", 0.1))       # evidence only now
    FLAT_GRIPPER_STD = float(t.get("flat_gripper_std", 0.05))  # unused, kept for compat
    MIN_TRAJ_FRAMES = int(t.get("min_traj_frames", 2))
    EXTENT_PADDING_RATIO = float(t.get("extent_padding_ratio", 0.05))
    EXTENT_PADDING_MIN_CM = float(t.get("extent_padding_min_cm", 0.5))
    GRIPPER_END_OPEN_THRESHOLD = float(t.get("gripper_end_open_threshold", 0.2))
    GRIPPER_END_TAIL = int(t.get("gripper_end_tail", 10))
    if GRIPPER_END_TAIL < 1:
        raise ValueError("thresholds.gripper_end_tail must be at least 1")

    p = cfg.get("projection", {})
    V_MAX_MM = float(p.get("v_max_mm", 25.0))
    PROJ_DY = int(p.get("dy_px", 35))
    PROJ_R_START = int(p.get("r_start_px", 40))
    PROJ_R_END = int(p.get("r_end_px", 40))
    PROJ_N_RED = int(p.get("n_red_min", 150))
    hsv = p.get("red_hsv", {})
    RED_HUE1_MAX = int(hsv.get("hue1_max", 10))
    RED_HUE2_MIN = int(hsv.get("hue2_min", 160))
    RED_S_MIN = int(hsv.get("s_min", 60))
    RED_V_MIN = int(hsv.get("v_min", 60))
    EXTRINSICS_CONFIG = str(p.get("extrinsics_config", ""))

    cats = {k: v for k, v in cfg["categories"].items() if isinstance(v, dict)}
    CATEGORY_COLORS = {k: v["color"] for k, v in cats.items()}
    PROBLEM_CATEGORIES = {k for k, v in cats.items() if v.get("problem")}
    CATEGORY_SORT_ORDER = {k: v["priority"] for k, v in cats.items()}

    d = cfg["display"]
    CATEGORY_DEFAULT_COLOR = d.get("category_default_color", "#999999")
    TRAJ_COLOR = d["traj_color"]
    GRIP_COLOR = d["gripper_color"]
    PANEL_BG = d["panel_background"]
    GRIPPER_YLIM = tuple(d.get("gripper_ylim", [-0.05, 1.05]))
    EPISODES_PER_ROW = int(d.get("episodes_per_row", 3))
    FIGSIZE_SESSION = tuple(d.get("figsize_session", [11.0, 2.4]))
    FIGSIZE_SUMMARY = tuple(d.get("figsize_summary", [11.0, 0.4]))
    FIGSIZE_SUMMARY_BASE_HEIGHT = float(d.get("figsize_summary_base_height", 1.5))
    OUTER_HSPACE = float(d.get("outer_hspace", 0.55))
    OUTER_WSPACE = float(d.get("outer_wspace", 0.30))
    INNER_WSPACE = float(d.get("inner_wspace", 0.30))
    OUTER_MARGINS = tuple(d.get("outer_margins", [0.04, 0.98, 0.95, 0.04]))
    EPISODES_PER_ROW_RICH = int(d.get("episodes_per_row_rich", 2))
    FIGSIZE_SESSION_RICH = tuple(d.get("figsize_session_rich", [13.0, 3.2]))
    INNER_WSPACE_RICH = float(d.get("inner_wspace_rich", 0.25))


# Auto-load at import so the importable API works without an explicit call.
load_filters()


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class EpisodeQC:
    session: str
    episode: str
    path: str
    category: str
    n_traj_frames: int = 0
    path_length_cm: float = 0.0
    xy_span_cm: float = 0.0
    z_span_cm: float = 0.0
    n_grip_frames: int = 0
    n_nan_gripper: int = 0
    gripper_std: float = 0.0
    gripper_range: float = 0.0
    gripper_end: float | None = None
    n_invalid_traj: int = -1
    vmax_mm: float = 0.0
    red_start: int | None = None
    red_end: int | None = None
    # Raw arrays (only kept if keep_arrays=True on scan)
    pos: np.ndarray | None = field(default=None, repr=False)
    grip: np.ndarray | None = field(default=None, repr=False)


# ============================================================================
# Loading + classification
# ============================================================================

def _load_trajectory(traj_path: Path) -> np.ndarray | None:
    """Load CameraTrajectoryTransformed.txt -> (N, 3) xyz positions, or None."""
    if not traj_path.exists():
        return None
    try:
        raw = np.loadtxt(traj_path, comments="#")
    except Exception:
        return None
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 12:
        return None
    if raw.shape[1] > 12:
        # Lines carry a leading timestamp; drop it so columns stay aligned.
        # (Taking raw[:, :12] instead silently shifted everything: the
        # "translation" became a rotation-matrix column and path lengths
        # came out as nonsense, e.g. 108cm for a 48cm trajectory.)
        raw = raw[:, -12:]
    poses = raw.reshape(-1, 3, 4)
    return poses[:, :3, 3]


def _load_gripper(grip_path: Path) -> np.ndarray | None:
    """Load gripper_distances.txt -> (N,) array, or None."""
    if not grip_path.exists():
        return None
    try:
        v = np.loadtxt(grip_path)
    except Exception:
        return None
    if v.size == 0:
        return None
    return np.atleast_1d(v)


def _gripper_end_value(grip: np.ndarray | None) -> float | None:
    """Mean of the last GRIPPER_END_TAIL finite gripper readings, or None.

    Convention: 0 = closed, 1 = open. Trailing NaN frames (e.g. the tag occluded
    once the fruit is in the gripper) are skipped by masking to finite values.
    """
    if grip is None:
        return None
    finite = grip[np.isfinite(grip)]
    if len(finite) == 0:
        return None
    tail = finite[-GRIPPER_END_TAIL:]
    return float(tail.mean())


def _path_length(pos: np.ndarray | None) -> float:
    if pos is None or len(pos) < 2:
        return 0.0
    return float(np.nansum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def _load_poses(traj_path: Path) -> np.ndarray | None:
    """Load CameraTrajectoryTransformed.txt -> (N, 4, 4) camera_link poses, or None."""
    if not traj_path.exists():
        return None
    try:
        raw = np.loadtxt(traj_path, comments="#")
    except Exception:
        return None
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 12:
        return None
    if raw.shape[1] > 12:
        raw = raw[:, -12:]
    poses34 = raw.reshape(-1, 3, 4)
    bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(poses34), 1, 1))
    return np.concatenate([poses34, bottom], axis=1)


def _load_ledger_invalid(episode_dir: Path) -> int:
    """Count 'invalid' rows in CameraTrajectory_idx.txt (long-gap verdicts).

    Returns -1 when the ledger is missing/unreadable (upstream repair not run).
    """
    fp = episode_dir / "CameraTrajectory_idx.txt"
    if not fp.exists():
        return -1
    try:
        n = sum(1 for l in fp.read_text().splitlines()
                if l.strip() and not l.startswith("#") and l.split()[1] == "invalid")
    except Exception:
        return -1
    return n


def _red_mask_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of red pixels via HSV hue ranges (robust to the yellow
    false positives of the old RGB test)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    return (((h <= RED_HUE1_MAX) | (h >= RED_HUE2_MIN))
            & (hsv[:, :, 1] >= RED_S_MIN) & (hsv[:, :, 2] >= RED_V_MIN))


def _circle_count(mask: np.ndarray, cx: int, cy: int, rad: int) -> int:
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    return int((mask & ((xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad)).sum())


def _projection_evidence(episode_dir: Path, poses: np.ndarray) -> tuple[float, int | None, int | None]:
    """First-frame projection verdict inputs + per-frame max step.

    Treats the gripper tip as a static point in the camera frame and projects
    its world path (relative to frame 0) back onto the FIRST color frame with
    the recorded intrinsics. Returns (vmax_mm, red_start, red_end); the red
    counts are None when the needed files/cv2 are unavailable (evidence row
    stays empty, category decided by the remaining checks).
    """
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) * 1000.0
    vmax = float(np.nanmax(steps)) if len(steps) else 0.0

    img_fp = episode_dir / "color_000000.png"
    info_fp = episode_dir / "camera_info_color.json"
    if not (HAVE_CV2 and img_fp.exists() and info_fp.exists() and EXTRINSICS_CONFIG):
        return vmax, None, None
    ext_path = Path(EXTRINSICS_CONFIG)
    if not ext_path.is_absolute():
        ext_path = Path(__file__).resolve().parent.parent / ext_path
    try:
        ext = json.load(open(ext_path))
        info = json.load(open(info_fp))
        img = cv2.imread(str(img_fp))
        if img is None:
            return vmax, None, None
        T_oc = np.array(ext["T_optical_camera"], dtype=float)[:3, :3]
        tip_link = np.array(ext["T_camera_gripper_tip"], dtype=float)[:3, 3]
        K = np.array(info["K"], dtype=float).reshape(3, 3)
        D = np.array(info["D"], dtype=float).ravel()

        rel = np.linalg.inv(poses[0]) @ poses          # camera_link, relative to frame 0
        pts_link = (rel[:, :3, :3] @ tip_link[:, None]).squeeze(2) + rel[:, :3, 3]
        pts_opt = pts_link @ T_oc.T                     # into optical convention for K
        pts2d, _ = cv2.projectPoints(pts_opt.astype(np.float64),
                                     np.zeros(3), np.zeros(3), K, D)
        pts2d = pts2d.reshape(-1, 2)
        finite = np.isfinite(pts2d).all(axis=1)
        pts2d = pts2d[finite]
        if len(pts2d) < 2:
            return vmax, None, None
        h, w = img.shape[:2]
        m = _red_mask_hsv(img)
        sx, sy = np.clip(pts2d[0], [0, 0], [w - 1, h - 1]).astype(int)
        red_start = _circle_count(m, int(sx), int(sy), PROJ_R_START)
        ex, ey = np.clip(pts2d[-1], [0, 0], [w - 1, h - 1]).astype(int)
        end_cx = int(np.clip(ex, 0, w - 1))
        end_cy = int(np.clip(ey + PROJ_DY, 0, h - 1))
        red_end = _circle_count(m, end_cx, end_cy, PROJ_R_END)
        return vmax, red_start, red_end
    except Exception:
        return vmax, None, None


def classify(pos: np.ndarray | None, grip: np.ndarray | None,
             n_invalid_traj: int = 0, vmax_mm: float = 0.0,
             red_start: int | None = None, red_end: int | None = None) -> str:
    """Task-level verdict. Evidence values ride along only to gate the checks
    that need them; every number is ALSO written to the CSV as evidence."""
    if pos is None or len(pos) < MIN_TRAJ_FRAMES or grip is None or len(grip) == 0:
        return "not_processed"
    if n_invalid_traj != 0:  # >0 = long gap present; -1 = repair not run
        return "traj_long_gap" if n_invalid_traj > 0 else "not_processed"
    if not np.isfinite(grip).all():
        return "gripper_missing"
    if vmax_mm > V_MAX_MM:
        return "traj_frame_jump"
    # Verdict uses the END window only: parameter sweeps over the whole corpus
    # (1138 episodes, 2026-10-16) showed the START near-tip window has no
    # discriminative power (median 0 red pixels for BOTH manual-ok and
    # manual-bad — the hand usually starts away from fruit), while the end
    # window separates cleanly (manual-ok: ~2% below threshold). red_start
    # stays in the CSV as evidence.
    if red_end is not None and red_end < PROJ_N_RED:
        return "first_frame_projection"
    end = _gripper_end_value(grip)
    if end is not None and end > GRIPPER_END_OPEN_THRESHOLD:
        return "gripper_not_closed"
    return "ok"


def scan_episode(episode_dir: Path, keep_arrays: bool = True) -> EpisodeQC:
    """Compute QC metrics for a single episode directory."""
    poses = _load_poses(episode_dir / "CameraTrajectoryTransformed.txt")
    pos = poses[:, :3, 3] if poses is not None else None
    grip = _load_gripper(episode_dir / "gripper_distances.txt")
    n_invalid = _load_ledger_invalid(episode_dir)

    vmax_mm, red_start, red_end = 0.0, None, None
    if poses is not None and len(poses) >= 2:
        vmax_mm, red_start, red_end = _projection_evidence(episode_dir, poses)
    cat = classify(pos, grip, n_invalid, vmax_mm, red_start, red_end)

    e = EpisodeQC(
        session=episode_dir.parent.name,
        episode=episode_dir.name,
        path=str(episode_dir),
        category=cat,
        n_traj_frames=int(len(pos)) if pos is not None else 0,
        path_length_cm=_path_length(pos) * 100,
        xy_span_cm=float(np.nanmax(np.ptp(pos[:, :2], axis=0))) * 100 if pos is not None and len(pos) else 0.0,
        z_span_cm=float(np.nanmax(np.ptp(pos[:, 2]))) * 100 if pos is not None and len(pos) else 0.0,
        n_grip_frames=int(len(grip)) if grip is not None else 0,
        n_nan_gripper=int((~np.isfinite(grip)).sum()) if grip is not None else 0,
        n_invalid_traj=n_invalid,
        vmax_mm=vmax_mm,
        red_start=red_start,
        red_end=red_end,
    )
    if grip is not None and len(grip):
        finite = grip[np.isfinite(grip)]
        if len(finite):
            e.gripper_std = float(finite.std())
            e.gripper_range = float(finite.max() - finite.min())
            e.gripper_end = _gripper_end_value(grip)
    if keep_arrays:
        e.pos = pos
        e.grip = grip
    return e


def scan_sessions(root: Path, pattern: str = "*-png", keep_arrays: bool = True) -> dict[str, list[EpisodeQC]]:
    """Scan a session tree, returning {session_name: [EpisodeQC, ...]}."""
    sessions = sorted([d for d in root.glob(pattern) if d.is_dir()])
    out: dict[str, list[EpisodeQC]] = {}
    for s in sessions:
        eps = []
        for ep_dir in sorted(s.glob("episode_*")):
            if ep_dir.is_dir():
                eps.append(scan_episode(ep_dir, keep_arrays=keep_arrays))
        if eps:
            out[s.name] = eps
    return out


# ============================================================================
# CSV output
# ============================================================================

CSV_FIELDS = ["session", "episode", "category",
              "path_length_cm", "xy_span_cm", "z_span_cm", "n_traj_frames",
              "n_nan_gripper", "n_grip_frames", "gripper_std", "gripper_range",
              "gripper_end", "n_invalid_traj", "vmax_mm", "red_start", "red_end",
              "path"]


def write_csv(session_data: dict[str, list[EpisodeQC]], output: Path) -> None:
    """Write per-episode metrics to CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for session_name in sorted(session_data):
            for e in session_data[session_name]:
                w.writerow({k: getattr(e, k) for k in CSV_FIELDS})


# ============================================================================
# PDF output
# ============================================================================

def _compute_traj_extent(session_data: dict[str, list[EpisodeQC]]) -> tuple[float, float, float, float] | None:
    """Global (x_min, x_max, y_min, y_max) in cm across every episode trajectory.

    Trajectories are shifted to start at origin (matching _plot_episode), so the
    extent always includes 0. Returns None if no episode has a usable trajectory.
    """
    x_min = x_max = y_min = y_max = 0.0
    found = False
    for eps in session_data.values():
        for e in eps:
            if e.pos is None or len(e.pos) < MIN_TRAJ_FRAMES:
                continue
            xy = e.pos[:, :2] - e.pos[0, :2]
            x_min = min(x_min, float(xy[:, 0].min()) * 100)
            x_max = max(x_max, float(xy[:, 0].max()) * 100)
            y_min = min(y_min, float(xy[:, 1].min()) * 100)
            y_max = max(y_max, float(xy[:, 1].max()) * 100)
            found = True
    if not found:
        return None
    pad_x = max(EXTENT_PADDING_MIN_CM, (x_max - x_min) * EXTENT_PADDING_RATIO)
    pad_y = max(EXTENT_PADDING_MIN_CM, (y_max - y_min) * EXTENT_PADDING_RATIO)
    return (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)


def _plot_episode(axes, e: EpisodeQC, extent: tuple[float, float, float, float] | None = None):
    ax_traj, ax_grip = axes
    cat_color = CATEGORY_COLORS.get(e.category, CATEGORY_DEFAULT_COLOR)

    # Match panel background so the pair reads as one group
    ax_traj.set_facecolor(PANEL_BG)
    ax_grip.set_facecolor(PANEL_BG)

    # Trajectory — always blue inside the panel
    if e.pos is not None and len(e.pos) >= 2:
        xy = e.pos[:, :2] - e.pos[0, :2]
        ax_traj.plot(xy[:, 0] * 100, xy[:, 1] * 100, "-", linewidth=1.2, color=TRAJ_COLOR)
        ax_traj.scatter([xy[0, 0] * 100], [xy[0, 1] * 100], c="green", s=25, zorder=3)
        ax_traj.scatter([xy[-1, 0] * 100], [xy[-1, 1] * 100], c="red", s=25, zorder=3, marker="x")
        if extent is not None:
            # Shared scale across the whole PDF — fixed limits, box adjusts to keep aspect equal
            ax_traj.set_xlim(extent[0], extent[1])
            ax_traj.set_ylim(extent[2], extent[3])
            ax_traj.set_aspect("equal", adjustable="box")
        else:
            ax_traj.set_aspect("equal", adjustable="datalim")
    else:
        ax_traj.text(0.5, 0.5, "no traj", ha="center", va="center",
                     transform=ax_traj.transAxes, fontsize=9)
    ax_traj.set_title(f"{e.episode}  ({e.path_length_cm:.1f} cm, {e.category})",
                      fontsize=9, color=cat_color)
    ax_traj.tick_params(labelsize=7)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlabel("x (cm)", fontsize=8)
    ax_traj.set_ylabel("y (cm)", fontsize=8)

    # Gripper — always orange inside the panel
    if e.grip is not None and len(e.grip) > 0:
        t = np.arange(len(e.grip))
        mask = np.isfinite(e.grip)
        ax_grip.plot(t[mask], e.grip[mask], "-", linewidth=1.0, color=GRIP_COLOR)
        if (~mask).any():
            ax_grip.plot(t[~mask], np.zeros((~mask).sum()), "rx", markersize=4)
        ax_grip.set_ylim(*GRIPPER_YLIM)
    else:
        ax_grip.text(0.5, 0.5, "no grip", ha="center", va="center",
                     transform=ax_grip.transAxes, fontsize=9)
    ax_grip.tick_params(labelsize=7)
    ax_grip.grid(True, alpha=0.3)
    ax_grip.set_xlabel("frame", fontsize=8)
    ax_grip.set_ylabel("gripper", fontsize=8)


# ----------------------------------------------------------------------------
# Rich (default) per-episode panels: RGB+projection | 3D trajectory | gripper.
# These mirror visualization/visualize_traj_video.py's left panel at frame 0 and
# right panel at the last frame, reusing that module's loaders + projection math.
# ----------------------------------------------------------------------------

_VIDEO_TOOL_CACHE = None


def _video_tool():
    """Lazily import cv2 and the sibling visualize_traj_video module.

    Only needed for the rich PDF path, so importing here keeps --csv-only light.
    Both files live in this directory; qc.py runs as `python visualization/qc.py`
    (sys.path[0] is already this dir), but the bootstrap also covers the
    `from visualization.qc import ...` API path.
    """
    global _VIDEO_TOOL_CACHE
    if _VIDEO_TOOL_CACHE is not None:
        return _VIDEO_TOOL_CACHE
    import sys
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import cv2  # type: ignore
    import visualize_traj_video as vt  # type: ignore
    _VIDEO_TOOL_CACHE = (vt, cv2)
    return _VIDEO_TOOL_CACHE


def _draw_rgb_projection(ax_img, e: EpisodeQC, tip_kin, vt, cv2) -> None:
    """Color frame 0 with the full gripper-tip path projected onto it (= the traj
    video's left panel at its first frame). Green->red gradient + green start + blue stop.
    """
    ep = Path(e.path)
    images = sorted([*ep.glob("color_*.png"), *ep.glob("color_*.jpg")])
    first = cv2.imread(str(images[0])) if images else None
    if first is None:
        ax_img.text(0.5, 0.5, "no frame", ha="center", va="center",
                    transform=ax_img.transAxes, fontsize=9)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        return
    ax_img.imshow(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    h, w = first.shape[:2]

    poses = vt.load_poses(ep / "CameraTrajectoryTransformed.txt")
    if poses is None:
        poses = vt.load_poses(ep / "CameraTrajectory.txt")
    K = vt.load_K(ep / "camera_info_color.json")
    if poses is not None and len(poses) >= 2 and K is not None:
        px, py = vt.project_future(poses, 0, K, tip_kin)
        pts = np.column_stack([px, py])
        if len(pts) >= 2:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection([], linewidths=2.0, capstyle="round", zorder=4)
            lc.set_segments(segs)
            lc.set_colors(vt._green_red_gradient(len(segs)))
            ax_img.add_collection(lc)
        if len(pts):
            ax_img.scatter([pts[0, 0]], [pts[0, 1]], c=["#00FF00"], s=40,
                           zorder=6, edgecolors="black", linewidths=0.5)
            if np.isfinite(pts[-1]).all():
                ax_img.scatter([pts[-1, 0]], [pts[-1, 1]], c=["#0000FF"], s=40,
                               zorder=6, edgecolors="black", linewidths=0.5)
    ax_img.set_xlim(0, w)
    ax_img.set_ylim(h, 0)  # image pixel coords (y increases downward)


def _draw_3d_traj(ax3d, e: EpisodeQC) -> None:
    """Full 3D trajectory (= the traj video's right panel at its last frame).

    Per-episode fixed extent, blue path, green start, vermillion head, same view as
    the video. Rasterized to keep the multi-page PDF size bounded.
    """
    pos = e.pos
    if pos is None or len(pos) < 2:
        ax3d.text2D(0.5, 0.5, "no traj", ha="center", va="center",
                    transform=ax3d.transAxes, fontsize=9)
        ax3d.set_axis_off()
        return
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    pad = max((pos.max(axis=0) - pos.min(axis=0)).max() * 0.08, 0.02)
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    ax3d.plot(x, y, z, color="#0072B2", lw=1.5, rasterized=True)
    ax3d.scatter([x[0]], [y[0]], [z[0]], c=["#009E73"], s=18, rasterized=True)  # start
    ax3d.scatter([x[-1]], [y[-1]], [z[-1]], c=["#D55E00"], s=22, zorder=5, rasterized=True)  # head
    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(lo[2], hi[2])
    ax3d.set_xlabel("X (m)", fontsize=7)
    ax3d.set_ylabel("Y (m)", fontsize=7)
    ax3d.set_zlabel("Z (m)", fontsize=7)
    ax3d.tick_params(labelsize=6)
    try:
        ax3d.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], max(hi[2] - lo[2], 1e-3)))
    except Exception:
        pass
    ax3d.view_init(elev=22, azim=-60)
    ax3d.set_rasterized(True)


def _plot_episode_rich(axes, e: EpisodeQC, tip_kin, vt, cv2) -> None:
    """Rich 3-panel per-episode row: RGB+projection | 3D trajectory | gripper."""
    ax_img, ax3d, ax_grip = axes
    cat_color = CATEGORY_COLORS.get(e.category, CATEGORY_DEFAULT_COLOR)
    for ax in (ax_img, ax3d, ax_grip):
        ax.set_facecolor(PANEL_BG)

    _draw_rgb_projection(ax_img, e, tip_kin, vt, cv2)
    ax_img.set_title(f"{e.episode}  ({e.path_length_cm:.1f} cm, {e.category})",
                     fontsize=8, color=cat_color)

    _draw_3d_traj(ax3d, e)
    ax3d.set_title(f"3D  ({e.path_length_cm:.1f} cm)", fontsize=8, color=cat_color)

    # Gripper — identical to the simple layout ("the gripper figure as now").
    if e.grip is not None and len(e.grip) > 0:
        t = np.arange(len(e.grip))
        mask = np.isfinite(e.grip)
        ax_grip.plot(t[mask], e.grip[mask], "-", linewidth=1.0, color=GRIP_COLOR)
        if (~mask).any():
            ax_grip.plot(t[~mask], np.zeros((~mask).sum()), "rx", markersize=4)
        ax_grip.set_ylim(*GRIPPER_YLIM)
    else:
        ax_grip.text(0.5, 0.5, "no grip", ha="center", va="center",
                     transform=ax_grip.transAxes, fontsize=9)
    ax_grip.tick_params(labelsize=7)
    ax_grip.grid(True, alpha=0.3)
    ax_grip.set_xlabel("frame", fontsize=8)
    ax_grip.set_ylabel("gripper", fontsize=8)


def _render_summary_page(pdf: PdfPages, session_data: dict[str, list[EpisodeQC]]):
    sessions = sorted(session_data.keys())
    n = len(sessions)
    fig, ax = plt.subplots(figsize=(FIGSIZE_SUMMARY[0], FIGSIZE_SUMMARY[1] * n + FIGSIZE_SUMMARY_BASE_HEIGHT))
    ax.axis("off")

    rows = []
    for s in sessions:
        eps = session_data[s]
        cats = [e.category for e in eps]
        paths = [e.path_length_cm for e in eps]
        n_ok = sum(1 for c in cats if c == "ok")
        n_bad = sum(1 for c in cats if c in PROBLEM_CATEGORIES)
        n_unproc = len(cats) - n_ok - n_bad
        rows.append([
            s,
            f"{len(eps)}",
            f"{n_ok}",
            f"{n_bad}",
            f"{n_unproc}",
            f"{np.mean(paths):.1f}" if paths else "0",
            f"{max(paths):.1f}" if paths else "0",
        ])

    cols = ["session", "eps", "ok", "problem", "unprocessed", "avg_path_cm", "max_path_cm"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#e0e0e0")
        table[(0, j)].set_text_props(weight="bold")
    for i, s in enumerate(sessions, start=1):
        cats = [e.category for e in session_data[s]]
        if all(c == "ok" for c in cats):
            color = "#c8e6c9"
        elif any(c in PROBLEM_CATEGORIES for c in cats):
            color = "#ffcdd2"
        else:
            color = "#ffe0b2"
        table[(i, 0)].set_facecolor(color)

    total_eps = sum(len(v) for v in session_data.values())
    ax.set_title(f"QC Summary — {n} sessions, {total_eps} episodes",
                 fontsize=13, pad=20)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_session_page(pdf: PdfPages, session_name: str, episodes: list[EpisodeQC],
                         extent: tuple[float, float, float, float] | None = None,
                         *, simple: bool = False, tip_kin=None):
    from matplotlib import gridspec

    episodes = sorted(episodes, key=lambda e: (CATEGORY_SORT_ORDER.get(e.category, 9),
                                                -e.path_length_cm))
    n = len(episodes)
    if simple:
        eps_per_row, figsize, inner_w, inner_wspace = (
            EPISODES_PER_ROW, FIGSIZE_SESSION, 2, INNER_WSPACE)
        vt = cv2 = None
    else:
        eps_per_row, figsize, inner_w, inner_wspace = (
            EPISODES_PER_ROW_RICH, FIGSIZE_SESSION_RICH, 3, INNER_WSPACE_RICH)
        vt, cv2 = _video_tool()
    nrows = int(np.ceil(n / eps_per_row))
    fig = plt.figure(figsize=(figsize[0], figsize[1] * nrows))

    # Outer grid: one cell per episode. Generous wspace/hspace to make panels
    # read as distinct groups.
    outer = gridspec.GridSpec(
        nrows, eps_per_row, figure=fig,
        hspace=OUTER_HSPACE, wspace=OUTER_WSPACE,
        left=OUTER_MARGINS[0], right=OUTER_MARGINS[1],
        top=OUTER_MARGINS[2], bottom=OUTER_MARGINS[3],
    )

    cat_counts = defaultdict(int)
    for e in episodes:
        cat_counts[e.category] += 1
    cat_summary = "  ".join(f"{c}:{n}" for c, n in sorted(cat_counts.items()))
    fig.suptitle(f"{session_name}  —  {n} episodes  —  {cat_summary}",
                 fontsize=12, y=0.985)

    for i, e in enumerate(episodes):
        row, col = divmod(i, eps_per_row)
        # Inner 1xN grid: the episode's panels, tighter spacing inside the group.
        inner = gridspec.GridSpecFromSubplotSpec(
            1, inner_w, subplot_spec=outer[row, col], wspace=inner_wspace,
        )
        if simple:
            ax_traj = fig.add_subplot(inner[0, 0])
            ax_grip = fig.add_subplot(inner[0, 1])
            _plot_episode((ax_traj, ax_grip), e, extent=extent)
        else:
            ax_img = fig.add_subplot(inner[0, 0])
            ax3d = fig.add_subplot(inner[0, 1], projection="3d")
            ax_grip = fig.add_subplot(inner[0, 2])
            _plot_episode_rich((ax_img, ax3d, ax_grip), e, tip_kin, vt, cv2)

    # Hide unused cells
    for i in range(n, nrows * eps_per_row):
        row, col = divmod(i, eps_per_row)
        ax = fig.add_subplot(outer[row, col])
        ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_pdf(session_data: dict[str, list[EpisodeQC]], output: Path,
               *, simple: bool = False, extrinsics_config: Path | None = None) -> None:
    """Render multi-page PDF: summary + one page per session.

    Default (rich): each episode is [RGB+projection][3D trajectory][gripper], mirroring
    the side-by-side traj videos. Pass simple=True for the fast 2D top-down layout.

    In the simple layout all trajectory plots share one xy extent (computed across every
    episode in session_data) so motion is visually comparable across pages.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    extent = _compute_traj_extent(session_data)
    tip_kin = None
    if not simple:
        vt, _ = _video_tool()
        tip_kin = vt.load_tip_kin(extrinsics_config or vt.DEFAULT_EXTRINSICS_CONFIG)
    with PdfPages(output) as pdf:
        _render_summary_page(pdf, session_data)
        # Sessions with most problems first
        def issue_score(item):
            eps = item[1]
            return sum(1 for e in eps if e.category in PROBLEM_CATEGORIES)
        for name, eps in sorted(session_data.items(), key=lambda kv: (-issue_score(kv), kv[0])):
            _render_session_page(pdf, name, eps, extent=extent,
                                 simple=simple, tip_kin=tip_kin)


# ============================================================================
# CLI
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root", type=Path, help="Directory containing session folders")
    p.add_argument("-o", "--output", type=Path, default=Path("qc"),
                   help="Output prefix; writes <prefix>.csv and <prefix>.pdf (default: ./qc)")
    p.add_argument("-p", "--pattern", default="*-png",
                   help="Session glob pattern (default: *-png)")
    p.add_argument("--csv-only", action="store_true", help="Skip PDF generation")
    p.add_argument("--pdf-only", action="store_true", help="Skip CSV generation")
    p.add_argument("--filters", type=Path, default=None,
                   help="Path to a qc_filters.json override (default: configs/qc_filters.json next to this script)")
    p.add_argument("--simple", action="store_true",
                   help="Use the fast 2D top-down layout (no RGB/3D panels); default is the rich 3-panel page")
    p.add_argument("--extrinsics-config", type=Path, default=None,
                   help="Camera-to-gripper-tip JSON for the RGB projection overlay "
                        "(default: configs/camera_gripper_extrinsics_sroi_v2_d405.json)")
    args = p.parse_args(argv)

    if args.filters is not None:
        load_filters(args.filters)

    if not args.root.exists():
        print(f"error: {args.root} does not exist", file=sys.stderr)
        return 1

    keep_arrays = not args.csv_only  # only need arrays for PDF
    session_data = scan_sessions(args.root, pattern=args.pattern, keep_arrays=keep_arrays)
    if not session_data:
        print(f"error: no sessions matching {args.pattern} under {args.root}", file=sys.stderr)
        return 1

    total_eps = sum(len(v) for v in session_data.values())
    print(f"Scanned {len(session_data)} session(s), {total_eps} episode(s)")

    if not args.pdf_only:
        csv_path = args.output.with_suffix(".csv")
        write_csv(session_data, csv_path)
        print(f"  wrote {csv_path}")

    if not args.csv_only:
        # Ensure we have arrays if user passed --pdf-only without prior scan
        if not keep_arrays:
            session_data = scan_sessions(args.root, pattern=args.pattern, keep_arrays=True)
        pdf_path = args.output.with_suffix(".pdf")
        render_pdf(session_data, pdf_path,
                   simple=args.simple, extrinsics_config=args.extrinsics_config)
        print(f"  wrote {pdf_path}")

    # Print category breakdown
    counts = defaultdict(int)
    for eps in session_data.values():
        for e in eps:
            counts[e.category] += 1
    print("Category counts:")
    for c, n in sorted(counts.items()):
        print(f"  {c:18s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
