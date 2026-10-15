#!/usr/bin/env python3
"""Run ORB-SLAM3 in batch for one scheme in a schemes tree.
Reads stereo images plus linked YAML/timestamps from each episode scheme directory.
Trajectories are stored in that directory; existing trajectories are skipped.
Usage: run_orb_scheme.py <schemes_dir> <scheme_name> [--orbslam-dir PATH]
"""
import argparse, os, subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("schemes_dir", type=Path)
parser.add_argument("scheme_name")
parser.add_argument("--orbslam-dir", type=Path,
                    default=Path(os.environ.get("ORB_SLAM3_DIR", Path.home() / "code" / "ORB_SLAM3")))
args = parser.parse_args()
ORB = args.orbslam_dir.resolve()
STEREO = ORB / "Examples" / "Stereo" / "stereo_kitti"
VOCAB = ORB / "Vocabulary" / "ORBvoc.txt"
if not STEREO.is_file():
    parser.error(f"stereo_kitti not found: {STEREO}")
if not os.access(STEREO, os.X_OK):
    parser.error(f"stereo_kitti is not executable: {STEREO}")
if not VOCAB.is_file():
    parser.error(f"ORB vocabulary not found: {VOCAB}")
schemes = args.schemes_dir.resolve()
scheme = args.scheme_name

eps = sorted(d for d in schemes.iterdir() if d.is_dir() and d.name.startswith("episode_"))
ok = skip = fail = 0
for i, ep in enumerate(eps, 1):
    sd = ep / scheme
    if not sd.exists() or not (sd / "left_000000.png").exists():
        print(f"[{i}/{len(eps)}] {ep.name}/{scheme}: no images, skipping", flush=True); skip += 1; continue
    traj = sd / "CameraTrajectory.txt"
    if traj.exists():
        print(f"[{i}/{len(eps)}] {ep.name}/{scheme}: trajectory exists, skipping", flush=True); skip += 1; continue
    cfg = sd / "orb_slam_realsense_d405.yaml"
    if not cfg.is_file():
        print(f"[{i}/{len(eps)}] {ep.name}/{scheme}: missing config {cfg}", flush=True)
        fail += 1
        continue
    r = subprocess.run([str(STEREO), str(VOCAB), str(cfg), str(sd), "false"],
                       capture_output=True, text=True)
    if r.returncode == 0 and traj.exists():
        ok += 1; print(f"[{i}/{len(eps)}] {ep.name}/{scheme}: done", flush=True)
    else:
        fail += 1; print(f"[{i}/{len(eps)}] {ep.name}/{scheme}: FAIL {r.stderr[-120:]}", flush=True)
print(f"=== {scheme}: ok={ok} skip={skip} fail={fail} / {len(eps)} ===")
