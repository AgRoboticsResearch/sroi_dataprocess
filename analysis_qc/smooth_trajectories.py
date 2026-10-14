#!/usr/bin/env python3
"""Apply Whittaker (lambda=1) smoothing to SLAM trajectories: position xyz
and rotation (rot6d + Gram-Schmidt reprojection). Gripper untouched (own pipeline).

For each episode dir under <root>/*-png/episode_*:
  - CameraTrajectory.txt        -> smoothed in place (12-col format preserved)
  - CameraTrajectory_unsmoothed.txt <- backup of the original
Then the caller re-runs transform_trajectory.py to regenerate Transformed.

Usage: python3 smooth_trajectories.py <episode_root>
"""
import sys
from pathlib import Path

import numpy as np

LAMBDA = 1.0


def whittaker(p, lam=LAMBDA):
    n = len(p)
    D = np.zeros((n - 2, n))
    idx = np.arange(n - 2)
    D[idx, idx] = 1.0
    D[idx, idx + 1] = -2.0
    D[idx, idx + 2] = 1.0
    return np.linalg.solve(np.eye(n) + lam * (D.T @ D), p)


def rot6d_from_R(R):          # R: (n,3,3) -> (n,6): first two columns
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)


def R_from_rot6d(r6):         # r6: (n,6) -> (n,3,3) via Gram-Schmidt
    a, b = r6[:, :3], r6[:, 3:]
    a1 = a / np.linalg.norm(a, axis=1, keepdims=True)
    b1 = b - (b * a1).sum(1, keepdims=True) * a1
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    c1 = np.cross(a1, b1)
    return np.stack([a1, b1, c1], axis=2)


def smooth_episode(ep: Path):
    traj_fp = ep / "CameraTrajectory.txt"
    bak_fp = ep / "CameraTrajectory_unsmoothed.txt"
    if not traj_fp.exists():
        return None
    traj = np.loadtxt(traj_fp)          # (n,12) row-major: [R00 R01 R02 tx R10 ... tz]
    if traj.ndim == 1:
        traj = traj[None, :]
    n = len(traj)
    if n < 4:
        return (ep.name, 0, 0.0, 0.0)
    if not bak_fp.exists():
        np.savetxt(bak_fp, traj, fmt="%.9f")

    M = traj.reshape(n, 3, 4)           # 3x4 [R|t] per frame
    R = M[:, :, :3]
    t = M[:, :, 3]

    t_s = whittaker(t)
    r6_s = whittaker(rot6d_from_R(R))
    R_s = R_from_rot6d(r6_s)

    M_s = np.concatenate([R_s, t_s[:, :, None]], axis=2)  # (n,3,4)
    np.savetxt(traj_fp, M_s.reshape(n, 12), fmt="%.9f")

    pos_mod = np.linalg.norm(t_s - t, axis=1).max() * 1000
    ang_mod = np.degrees(np.linalg.norm(
        np.einsum('nji,njk->nik', R_s, R) - np.eye(3), axis=(1, 2))).max() / np.sqrt(2)
    return (ep.name, n, pos_mod, ang_mod)


def main():
    root = Path(sys.argv[1])
    eps = sorted(p for sess in sorted(root.glob("*-png")) for p in sess.iterdir()
                 if p.is_dir() and p.name.startswith("episode_"))
    mods = []
    for k, ep in enumerate(eps, 1):
        r = smooth_episode(ep)
        if r:
            mods.append(r)
        if k % 25 == 0:
            print(f"  {k}/{len(eps)}")
    pm = np.array([m[2] for m in mods])
    am = np.array([m[3] for m in mods])
    print(f"{len(mods)} 条平滑完成(λ={LAMBDA})")
    print(f"位置修改量 mm : p50={np.median(pm):.3f} p95={np.percentile(pm,95):.3f} max={pm.max():.2f} ({mods[int(np.argmax(pm))][0]})")
    print(f"旋转修改量 °  : p50={np.median(am):.3f} p95={np.percentile(am,95):.3f} max={am.max():.2f} ({mods[int(np.argmax(am))][0]})")


if __name__ == "__main__":
    main()
