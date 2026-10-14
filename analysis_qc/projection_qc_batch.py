#!/usr/bin/env python3
"""Batch trajectory-projection QC for the onesb data (masked/trap trajectories).

For every episode under the given session roots:
  - load the SLAM trajectory (CameraTrajectory.txt, optical frame, anchored at frame 0)
  - transform the fingertip through the fixed hand-eye (T_optical_camera @ T_camera_tip)
  - project the full fingertip path into the FIRST color frame (K + D from camera_info_color.json)
  - draw gradient polyline (green start -> red end), big endpoint marker, episode label
Output: one flat folder of <session>_<episode>.png for manual endpoint-on-strawberry review.

Usage: python3 projection_qc_batch.py <out_dir> <session_png_root> [<more roots>...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

EXTRINSICS = Path("/home/ss/sroi_dataprocess/configs/camera_gripper_extrinsics_sroi_v2_d405.json")


def load_extrinsics():
    j = json.load(open(EXTRINSICS))
    T_opt_link = np.array(j["T_optical_camera"], dtype=float)
    T_link_tip = np.array(j["T_camera_gripper_tip"], dtype=float)
    tip_in_opt = (T_opt_link @ T_link_tip)[:3, 3]
    return tip_in_opt


def project_episode(ep_dir: Path, tip_in_opt: np.ndarray):
    traj_fp = ep_dir / "CameraTrajectory.txt"
    img_fp = ep_dir / "color_000000.png"
    info_fp = ep_dir / "camera_info_color.json"
    if not (traj_fp.exists() and img_fp.exists() and info_fp.exists()):
        return None, "missing files"

    traj = np.loadtxt(traj_fp).reshape(-1, 3, 4)
    R, t = traj[:, :, :3], traj[:, :, 3]
    pts_w = (R @ tip_in_opt[:, None]).squeeze(2) + t  # fingertip path in frame-0 optical frame

    info = json.load(open(info_fp))
    K = np.array(info["K"], dtype=float).reshape(3, 3)
    D = np.array(info["D"], dtype=float).ravel()
    pts2d, _ = cv2.projectPoints(pts_w.astype(np.float64), np.zeros(3), np.zeros(3), K, D)
    pts2d = pts2d.reshape(-1, 2)
    valid = pts_w[:, 2] > 0.05  # keep points in front of the camera

    img = cv2.imread(str(img_fp))
    n = len(pts2d)
    for i in range(n - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        c = (int(255 * i / max(1, n - 1)), 60, int(255 * (1 - i / max(1, n - 1))))
        cv2.line(img, tuple(pts2d[i].astype(int)), tuple(pts2d[i + 1].astype(int)), c, 2)
    if valid[0]:
        cv2.circle(img, tuple(pts2d[0].astype(int)), 6, (0, 255, 0), -1)
    if valid[-1]:
        x, y = tuple(pts2d[-1].astype(int))
        cv2.circle(img, (x, y), 9, (0, 0, 255), 2)
        cv2.line(img, (x - 13, y), (x + 13, y), (0, 0, 255), 2)
        cv2.line(img, (x, y - 13), (x, y + 13), (0, 0, 255), 2)
    cv2.putText(img, f"{ep_dir.parent.name}/{ep_dir.name}  frames={n}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img, None


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    tip_in_opt = load_extrinsics()

    eps = []
    for root in sys.argv[2:]:
        for sess in sorted(Path(root).glob("*-png")):
            eps += sorted(p for p in sess.iterdir() if p.is_dir() and p.name.startswith("episode_"))

    skipped = []
    for k, ep in enumerate(eps, 1):
        img, err = project_episode(ep, tip_in_opt)
        name = f"{ep.parent.name}_{ep.name}"
        if img is None:
            skipped.append(f"{name}: {err}")
            continue
        cv2.imwrite(str(out_dir / f"{name}.png"), img)
        if k % 100 == 0:
            print(f"  {k}/{len(eps)}")
    print(f"done: {len(eps) - len(skipped)} images -> {out_dir}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped[:20]:
            print("  ", s)


if __name__ == "__main__":
    main()
