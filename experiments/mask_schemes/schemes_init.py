#!/usr/bin/env python3
"""Arrange existing raw and maskhalf outputs into a clean schemes tree.
Shared intrinsics, timestamps, and color frames stay at the episode root. Each scheme
subdirectory contains its own stereo images and trajectories, with symlinks to shared
times.txt and YAML files.

Usage: schemes_init.py <src_png> <src_mask292> <dst_schemes>
"""
import os, shutil, sys
from pathlib import Path

src_png = Path(sys.argv[1])
src_mask = Path(sys.argv[2])
dst = Path(sys.argv[3])
META = ["camera_info_color.json", "camera_info_left.json", "camera_info_right.json",
        "orb_slam_realsense_d405.yaml", "times.txt", "timestamps.json",
        "vive_poses.jsonl", "vive_reference.json"]
TRAJ = ["CameraTrajectory.txt", "CameraTrajectoryTransformed.txt"]


def populate(sub_dir: Path, img_src: Path, traj_src: Path):
    sub_dir.mkdir(parents=True, exist_ok=True)
    for p in img_src.iterdir():
        if p.name.startswith("left_") or p.name.startswith("right_"):
            shutil.copy2(p, sub_dir / p.name)
    for t in TRAJ:
        s = traj_src / t
        if s.exists() and not (sub_dir / t).exists():
            shutil.copy2(s, sub_dir / t)
    for shared in ["times.txt", "orb_slam_realsense_d405.yaml"]:  # Link shared files
        link = sub_dir / shared
        if not link.exists():
            os.symlink(f"../{shared}", link)


eps = sorted(d.name for d in src_png.iterdir() if d.is_dir() and d.name.startswith("episode_"))
for ep in eps:
    root = dst / ep
    root.mkdir(parents=True, exist_ok=True)
    for m in META:
        s = src_png / ep / m
        if s.exists() and not (root / m).exists():
            shutil.copy2(s, root / m)
    for c in (src_png / ep).glob("color_*.png"):
        shutil.copy2(c, root / c.name)
    populate(root / "raw", src_png / ep, src_png / ep)
    populate(root / "maskhalf", src_mask / ep, src_mask / ep)
print(f"done -> {dst} ({len(eps)} episodes)")
