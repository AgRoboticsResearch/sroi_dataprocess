#!/usr/bin/env python3
"""Compare old (backup tar) vs new (rerun) gripper recovery across all episodes.

Outputs into ~/data/_gripper_v2_validation/对比_20261013/:
  - 汇总.csv                per-episode: coverage, max_gap, n_nan, RMS/max diff vs old
  - 重点对比/               overlay plots for: NaN episodes, spike episodes,
                            top-30 by RMS difference, strong-light trio
  - 新图全部/               flat copies of the new per-episode plots
"""
import glob, json, os, tarfile, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

TAR = None  # 旧输出备份tar已删除；旧值改从 onesb_1120_trap parquet 的夹爪列读取
OLD_PARQUET = os.path.expanduser('~/data/lerobot/onesb_1120_trap/data/chunk-000/file-000.parquet')
OUT = os.path.expanduser('~/data/QC工作汇总_20261013/夹爪v2/对比_20261013')
os.makedirs(OUT + '/重点对比', exist_ok=True)
os.makedirs(OUT + '/新图全部', exist_ok=True)

# 1) 读旧值: onesb_1120_trap parquet 的 observation.ee 夹爪列(第7维),
#    按 episode_index 对齐到源episode顺序。映射经帧数指纹校验(2026-10-13):
#    1123/1135 位置精确对齐;12处不匹配集中在R1剔除条目(轨迹重跑过)与强光
#    161913(旧管线部分漏检,尾部3条无旧值)——这些条目的新旧对比图已在
#    重点对比/ 里留档,无需再生。
import pandas as pd
_df = pd.read_parquet(OLD_PARQUET)
old = {}
for ep_idx, grp in _df.groupby('episode_index'):
    g = np.array([float(r[-1]) for r in grp['observation.ee']])
    old[ep_idx] = g
# episode_index -> "<session>/episode_XXX" 映射(按转换顺序)
_keys = sorted(glob.glob('/home/ss/data/1000_onesb_labpicking/*-png/episode_*/gripper_distances.txt')
             + glob.glob('/home/ss/data/120_onesb_supplementary/*-png/episode_*/gripper_distances.txt'))
_key_by_idx = {i: '/'.join(fp.split('/')[-3:-1]) for i, fp in enumerate(_keys)}
old = {_key_by_idx.get(i, str(i)): g for i, g in old.items()}

# 2) 扫新输出
rows, nan_eps, spike_eps = [], [], []
new_vals = {}
for fp in sorted(glob.glob('/home/ss/data/1000_onesb_labpicking/*-png/episode_*/gripper_distances.txt')
               + glob.glob('/home/ss/data/120_onesb_supplementary/*-png/episode_*/gripper_distances.txt')):
    key = '/'.join(fp.split('/')[-3:-1])
    g = np.loadtxt(fp)
    new_vals[key] = g
    s_fp = fp.replace('gripper_distances.txt', 'gripper_detection_stats.json')
    s = json.load(open(s_fp)) if os.path.exists(s_fp) else {}
    n_nan = int(np.isnan(g).sum())
    gl = s.get('gap_lengths', [])
    rms = mx = 0.0
    if key in old:
        o, n = old[key], g
        m2 = min(len(n), len(o))  # 帧数防护:只比共同前缀
        m = ~np.isnan(n[:m2]) & ~np.isnan(o[:m2])
        if m.any():
            rms = float(np.sqrt(np.mean((n[:m2][m]-o[:m2][m])**2)))
            mx = float(np.max(np.abs(n[:m2][m]-o[:m2][m])))
    rows.append((key, s.get('coverage', -1), max(gl) if gl else 0, n_nan, rms, mx))
    if n_nan: nan_eps.append(key)
    if s.get('spikes_removed'): spike_eps.append(key)

# 3) 汇总CSV
with open(OUT + '/汇总.csv', 'w') as f:
    f.write('episode,coverage,max_gap,n_nan,rms_vs_old,maxdiff_vs_old\n')
    for r in rows: f.write(f'{r[0]},{r[1]:.4f},{r[2]},{r[3]},{r[4]:.4f},{r[5]:.3f}\n')

# 4) 重点对比: NaN集 + 尖峰集 + RMS top30
top30 = sorted(rows, key=lambda r: -r[4])[:30]
focus = list(dict.fromkeys(nan_eps + spike_eps + [r[0] for r in top30]))
for key in focus:
    if key not in old: continue
    o, n = old[key], new_vals[key]
    fig, ax = plt.subplots(figsize=(11, 4), dpi=110)
    ax.plot(o, '#3b6fb6', lw=1.3, label='旧方案')
    ax.plot(n, '#d62728', lw=1.3, alpha=0.85, label='新方案(20261013)')
    row = next(r for r in rows if r[0] == key)
    ax.set_title(f"{key}  RMS={row[4]:.3f} maxdiff={row[5]:.2f} NaN={row[3]} 最长洞={row[2]}", fontsize=9)
    ax.set_xlabel('frame'); ax.set_ylabel('gripper'); ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/重点对比/{key.replace('/','_')}.png"); plt.close(fig)

# 5) 新图全部(平铺)
import shutil
for fp in sorted(glob.glob('/home/ss/data/1000_onesb_labpicking/*-png/episode_*/gripper_distances.jpg')
               + glob.glob('/home/ss/data/120_onesb_supplementary/*-png/episode_*/gripper_distances.jpg')):
    key = '_'.join(fp.split('/')[-3:-1]).replace('-png', '')
    shutil.copy(fp, f"{OUT}/新图全部/{key}.jpg")

arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
print(f"共 {len(rows)} 条 | 覆盖率均值 {arr[:,0].mean():.4f} 最低 {arr[:,0].min():.3f}")
print(f"含NaN(长遮挡,建议下游剔除): {len(nan_eps)} 条")
for k in nan_eps: print("   ", k)
print(f"剔尖峰: {len(spike_eps)} 条 | RMS中位数 {np.median(arr[:,3]):.4f} | 重点对比图 {len(focus)} 张")
print(f"新图全部: {len(glob.glob(OUT+'/新图全部/*.jpg'))} 张")
