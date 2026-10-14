#!/usr/bin/env python3
"""Whittaker smoother lambda scan on val trajectories (3D position).

Outputs into ~/data/val_independent/smoothing_scan/:
  - stats per lambda over all 100 episodes: modification |x-p| p50/p95/max,
    vmax before/after, steps>15mm survivors
  - 3D overlay figures for representative episodes at chosen lambdas
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

ROOT = Path('/home/ss/data/val_independent/20260714_160922-png')
OUT = Path('/home/ss/data/val_independent/smoothing_scan')
OUT.mkdir(exist_ok=True)
LAMBDAS = [1, 10, 100, 1000]

ext = json.load(open('/home/ss/sroi_dataprocess/configs/camera_gripper_extrinsics_sroi_v2_d405.json'))
T = np.array(ext['T_optical_camera']) @ np.array(ext['T_camera_gripper_tip'])
TIPC = T[:3, 3]


def load_pts(ep):
    traj = np.loadtxt(ep / 'CameraTrajectory.txt').reshape(-1, 3, 4)
    return np.einsum('nij,j->ni', traj[:, :, :3], TIPC) + traj[:, :, 3]


def whittaker(p, lam):
    n = len(p)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1, -2, 1
    A = np.eye(n) + lam * (D.T @ D)
    return np.linalg.solve(A, p)


def steps(pts):
    return np.linalg.norm(np.diff(pts, axis=0), axis=1) * 1000


eps = sorted(ROOT.glob('episode_*'))
stats = {lam: {'mod': [], 'vmax': [], 'n_over': []} for lam in LAMBDAS}
raw_vmax = []
cache = {}
for ep in eps:
    pts = load_pts(ep)
    cache[ep.name] = pts
    s0 = steps(pts)
    raw_vmax.append(s0.max())
    for lam in LAMBDAS:
        sm = whittaker(pts, lam)
        d = np.linalg.norm(sm - pts, axis=1) * 1000
        s = steps(sm)
        stats[lam]['mod'].append([np.median(d), np.percentile(d, 95), d.max()])
        stats[lam]['vmax'].append(s.max())
        stats[lam]['n_over'].append(int((s > 15).sum()))

print(f'原始: vmax p50={np.median(raw_vmax):.1f} p95={np.percentile(raw_vmax,95):.1f} | steps>15mm 总数={sum(1 for v in raw_vmax if v>15)}条有')
for lam in LAMBDAS:
    mod = np.array(stats[lam]['mod'])
    vm = np.array(stats[lam]['vmax'])
    no = stats[lam]['n_over']
    print(f'λ={lam:5d} | 修改量mm p50={np.median(mod[:,0]):.3f} p95={np.median(mod[:,1]):.3f} max中位={np.median(mod[:,2]):.2f} '
          f'| 平滑后vmax p50={np.median(vm):.1f} p95={np.percentile(vm,95):.1f} | >15mm步残留={sum(1 for x in vm if x>15)}条')

# 3D overlays: 091 (step) + fast-motion episode + one normal
fast = max(eps, key=lambda e: steps(cache[e.name]).max() if '091' not in e.name else 0)
normal = ROOT / 'episode_001'
for name, lam_show in [('episode_091', 100), (fast.name, 100), (normal.name, 100)]:
    pts = cache[name]
    sm = whittaker(pts, lam_show)
    fig = plt.figure(figsize=(14, 5), dpi=110)
    for k in range(2):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        ax.plot(pts[:, 0] * 1000, pts[:, 1] * 1000, pts[:, 2] * 1000, color='#999', lw=1, label='原始')
        ax.plot(sm[:, 0] * 1000, sm[:, 1] * 1000, sm[:, 2] * 1000, color='#d62728', lw=2, label=f'平滑 λ={lam_show}')
        ax.scatter(*pts[0] * 1000, color='green', s=40)
        ax.scatter(*pts[-1] * 1000, color='blue', s=40)
        ax.view_init(elev=22, azim=35 + k * 60)
        ax.set_xlabel('x mm'); ax.set_ylabel('y mm'); ax.set_zlabel('z mm')
        ax.legend(fontsize=8)
    fig.suptitle(f'{name}  vmax原始={steps(pts).max():.1f} → 平滑={steps(sm).max():.1f} mm/帧', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f'3d_{name}_lam{lam_show}.png'); plt.close(fig)

# steps curve for 091 at all lambdas
fig, ax = plt.subplots(figsize=(11, 4), dpi=110)
ax.plot(steps(cache['episode_091']), color='#999', lw=1.2, label='原始')
for lam in LAMBDAS:
    ax.plot(steps(whittaker(cache['episode_091'], lam)), lw=1.2, label=f'λ={lam}')
ax.axhline(15, color='k', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('帧'); ax.set_ylabel('步距 mm/帧'); ax.set_title('episode_091 平滑前后逐帧步距')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / 'steps_091.png'); plt.close(fig)
print(f'\n图已存 {OUT}/')
