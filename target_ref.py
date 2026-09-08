#!/usr/bin/env python3
"""Hindsight target conditioning: derive a per-episode target-strawberry marker.

Given an episode's camera trajectory (ORB-SLAM, transformed), normalized gripper
signal, camera intrinsics, and camera<->gripper-tip extrinsics, this module finds
the frame where the final sustained gripper closure starts (the grasp), projects
the gripper-tip position at that moment back into the INITIAL image, and renders
a rough circle below that point — a label for the target strawberry, not the
picking location (the policy learns where/how to grasp from the demonstrations).

The same draw_target_marker() is used at training time (hindsight point from the
demonstration) and at deployment (user clicks the fruit), so style, resolution,
and coordinate conventions match exactly.

No object detection / segmentation / tracking is used anywhere — the target
point is derived purely geometrically from the demonstration.

Loader helpers intentionally mirror visualization/visualize_traj_video.py so this
module stays self-contained (numpy + Pillow only) and can be copied to a
deployment machine on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_EXTRINSICS_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "camera_gripper_extrinsics_sroi_v2_d405.json"
)

MARKER_COLOR_RGB = (0, 255, 255)  # cyan — contrasts with red fruit and green leaves

# QC status values returned by episode_target_point / compute_episode_target.
STATUS_OK = "ok"
STATUS_NO_TRAJECTORY = "no_trajectory"
STATUS_NO_GRIPPER = "no_gripper"
STATUS_NO_SUSTAINED_CLOSURE = "no_sustained_closure"
STATUS_NO_INTRINSICS = "no_intrinsics"
STATUS_DEPTH_INVALID = "depth_invalid"
STATUS_OUT_OF_FRAME = "out_of_frame"
STATUS_CLOSURE_AT_START = "closure_at_start"

# Plausibility bounds for the projected depth (meters). A strawberry pick happens
# well inside 1.5 m; anything beyond indicates unstable VO.
MAX_DEPTH_M = 1.5


# ---------------------------------------------------------------------------
# Loaders (mirror visualization/visualize_traj_video.py)
# ---------------------------------------------------------------------------

def load_poses(traj_path: Path | str) -> np.ndarray | None:
    """Load a KITTI-format trajectory -> (N, 4, 4) camera-in-world poses (meters), or None."""
    traj_path = Path(traj_path)
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
    return np.concatenate([poses34, bottom], axis=1)


def load_gripper(grip_path: Path | str) -> np.ndarray | None:
    """Load normalized gripper distances (one float per frame), or None.

    Convention: 0 = closed, 1 = open (see gripper_normalization.py).
    """
    grip_path = Path(grip_path)
    if not grip_path.exists():
        return None
    try:
        raw = np.loadtxt(grip_path)
    except Exception:
        return None
    if raw.size == 0:
        return None
    return np.atleast_1d(raw)


def load_intrinsics(ep_dir: Path | str) -> np.ndarray | None:
    """Load the color camera intrinsics (3x3) from camera_info_color.json, or None.

    Falls back to camera_info_left.json for stereo-only episodes (left IS color).
    """
    ep_dir = Path(ep_dir)
    for name in ("camera_info_color.json", "camera_info_left.json"):
        path = ep_dir / name
        if not path.exists():
            continue
        try:
            return np.array(json.load(path.open())["K"], dtype=float).reshape(3, 3)
        except Exception:
            continue
    return None


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
    """Load (T_optical_camera, T_camera_gripper_tip) from the rig extrinsics JSON."""
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


# ---------------------------------------------------------------------------
# Grasp-frame detection
# ---------------------------------------------------------------------------

def detect_grasp_frame(
    grip: np.ndarray,
    fps: float = 30.0,
    closed_threshold: float = 0.5,
    hold_s: float = 0.5,
    smooth: int = 5,
) -> int | None:
    """Start index of the final sustained closed-gripper segment, or None.

    The gripper convention is 0 = closed, 1 = open. NaNs (tag occluded, typically
    after the fruit is grabbed) are filled by interpolation / nearest-valid
    extension so a trailing NaN run does not break the final closed segment.

    A short moving average plus a minimum run length (hold_s * fps) prevents a
    noisy single-frame transition from being selected. The LAST qualifying closed
    run is used and its START returned — not the last frame — so the post-grasp
    pull-away does not shift the pick location.
    """
    g = np.asarray(grip, dtype=float).ravel()
    if g.size == 0:
        return None
    finite = np.isfinite(g)
    if not finite.any():
        return None
    if not finite.all():
        # Linear interpolation for interior gaps; nearest-valid extension at the ends.
        idx = np.arange(g.size)
        g = np.interp(idx, idx[finite], g[finite])
    if smooth > 1:
        kernel = np.ones(int(smooth), dtype=float) / int(smooth)
        g = np.convolve(g, kernel, mode="same")

    closed = g < closed_threshold
    min_run = max(1, int(round(hold_s * fps)))

    # Collect runs of consecutive closed frames.
    padded = np.concatenate([[False], closed, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    runs = list(zip(edges[0::2], edges[1::2]))  # (start, end_exclusive) in `closed`
    qualifying = [(s, e) for s, e in runs if e - s >= min_run]
    if not qualifying:
        return None
    return int(qualifying[-1][0])


# ---------------------------------------------------------------------------
# Target point: grasp-frame gripper tip projected into the initial image
# ---------------------------------------------------------------------------

def project_tip_into_initial(
    poses: np.ndarray,
    t_g: int,
    K: np.ndarray,
    tip_kin: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float, float]:
    """Project the gripper tip at frame t_g into the INITIAL (frame 0) image.

    Mirrors visualize_traj_video.project_future at t=0:
        p_tip = (T_optical_camera @ inv(poses[0]) @ poses[t_g] @ T_camera_gripper_tip)[:3, 3]
        u = fx * x/z + cx,  v = fy * y/z + cy
    Returns (u, v, z) in pixels / meters. z <= 0 means behind the initial camera.
    """
    T_opt_cam, T_cam_ee = tip_kin
    T_rel = np.linalg.inv(poses[0]) @ poses[t_g]
    p = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]
    z = float(p[2])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = float(fx * p[0] / z + cx) if abs(z) > 1e-12 else float("nan")
    v = float(fy * p[1] / z + cy) if abs(z) > 1e-12 else float("nan")
    return u, v, z


def episode_target_point(
    poses: np.ndarray | None,
    grip: np.ndarray | None,
    K: np.ndarray | None,
    tip_kin: tuple[np.ndarray, np.ndarray],
    fps: float = 30.0,
    image_shape: tuple[int, int] | None = None,
    closed_threshold: float = 0.5,
    hold_s: float = 0.5,
    smooth: int = 5,
) -> dict:
    """Compute the episode's hindsight target point with QC status.

    image_shape is (height, width) of the initial image; required for the
    out-of-frame check. Returns a dict with keys: status, t_g, u, v, z.
    """
    result = {"status": STATUS_OK, "t_g": None, "u": None, "v": None, "z": None}

    if poses is None or len(poses) < 2:
        result["status"] = STATUS_NO_TRAJECTORY
        return result
    if grip is None or len(grip) == 0:
        result["status"] = STATUS_NO_GRIPPER
        return result
    if K is None:
        result["status"] = STATUS_NO_INTRINSICS
        return result

    n = min(len(poses), len(grip))
    t_g = detect_grasp_frame(
        grip[:n], fps=fps, closed_threshold=closed_threshold, hold_s=hold_s, smooth=smooth
    )
    result["t_g"] = t_g
    if t_g is None:
        result["status"] = STATUS_NO_SUSTAINED_CLOSURE
        return result
    if t_g == 0:
        # Gripper already closed when the episode starts: no pick event is
        # observed, and the marker would just point at the resting gripper.
        result["status"] = STATUS_CLOSURE_AT_START
        return result

    u, v, z = project_tip_into_initial(poses[:n], t_g, K, tip_kin)
    result.update(u=u, v=v, z=z)
    if not np.isfinite(z) or z <= 1e-3 or z > MAX_DEPTH_M:
        result["status"] = STATUS_DEPTH_INVALID
        return result
    if image_shape is not None:
        h, w = image_shape
        if not (0.0 <= u < w and 0.0 <= v < h):
            result["status"] = STATUS_OUT_OF_FRAME
            return result
    return result


def compute_episode_target(
    ep_dir: Path | str,
    tip_kin: tuple[np.ndarray, np.ndarray],
    fps: float = 30.0,
    closed_threshold: float = 0.5,
    hold_s: float = 0.5,
    smooth: int = 5,
) -> dict:
    """Load one episode directory and compute its target point (adds image_shape)."""
    ep_dir = Path(ep_dir)
    poses = load_poses(ep_dir / "CameraTrajectoryTransformed.txt")
    if poses is None:
        poses = load_poses(ep_dir / "CameraTrajectory.txt")
    grip = load_gripper(ep_dir / "gripper_distances.txt")
    K = load_intrinsics(ep_dir)

    image_shape = None
    first_image = first_color_image(ep_dir)
    if first_image is not None:
        with Image.open(first_image) as img:
            w, h = img.size
        image_shape = (h, w)

    result = episode_target_point(
        poses, grip, K, tip_kin, fps=fps, image_shape=image_shape,
        closed_threshold=closed_threshold, hold_s=hold_s, smooth=smooth,
    )
    result["image_shape"] = image_shape
    result["first_image"] = str(first_image) if first_image is not None else None
    return result


def first_color_image(ep_dir: Path | str) -> Path | None:
    """First camera frame of an episode (color_*, falling back to left_*)."""
    ep_dir = Path(ep_dir)
    files = sorted([*ep_dir.glob("color_*.png"), *ep_dir.glob("color_*.jpg")])
    if not files:
        files = sorted([*ep_dir.glob("left_*.png"), *ep_dir.glob("left_*.jpg")])
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Marker rendering (shared by training-data generation and deployment)
# ---------------------------------------------------------------------------

def draw_target_marker(
    image_rgb: np.ndarray,
    u: float,
    v: float,
    radius_px: int | None = None,
    offset_px: int | None = None,
    style: str = "circle",
) -> np.ndarray:
    """Return a copy of image_rgb (H, W, 3 uint8) with the target marker drawn.

    The marker is a rough circle around the target strawberry — an object label,
    not a picking location. Center = (u, v + offset_px): by default offset_px =
    radius_px, so the circle's top edge touches the projected picking point and
    its body covers the fruit hanging below it. At deployment the user clicks the
    fruit directly, so offset_px=0 centers the circle on the click.

    radius_px/offset_px default to 30 px scaled by min(h, w)/480.
    style: "circle" (default), "bbox" (square), or "point" (small filled dot).
    """
    out = Image.fromarray(image_rgb).copy()
    draw = ImageDraw.Draw(out)
    h, w = out.height, out.width
    scale = min(h, w) / 480.0
    if radius_px is None:
        radius_px = int(round(30 * scale))
    if offset_px is None:
        offset_px = radius_px
    thickness = max(2, int(round(3 * scale)))

    cx, cy = int(round(u)), int(round(v + offset_px))
    black = (0, 0, 0)
    color = MARKER_COLOR_RGB

    if style == "circle":
        box = [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px]
        draw.ellipse(box, outline=black, width=thickness + 2)
        draw.ellipse(box, outline=color, width=thickness)
    elif style == "bbox":
        r = radius_px
        box = [cx - r, cy - r, cx + r, cy + r]
        draw.rectangle(box, outline=black, width=thickness + 2)
        draw.rectangle(box, outline=color, width=thickness)
    elif style == "point":
        dot_r = max(4, int(round(6 * scale)))
        box = [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r]
        draw.ellipse([b - 2 for b in box], fill=black)
        draw.ellipse(box, fill=color)
    else:
        raise ValueError(f"unknown marker style: {style!r} (expected circle|bbox|point)")
    return np.array(out)
