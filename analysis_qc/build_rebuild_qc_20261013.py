#!/usr/bin/env python3
"""Generate qc.csv for the 2026-10-13 onesb dataset rebuild.

Three exclusion rules, applied to all episodes under the two raw roots:
  R1  manual trajectory-QC list (20) — parsed from 剔除清单_EXCLUDED.md
      (~/data/traj_projection_qc_onesb_20261011_BAD)
  R2  gripper long occlusion: any NaN in gripper_distances.txt (>10-frame
      gap left as NaN by the v2 recovery — jaw unknowable for 0.7s+)
  R3  no closing action: median of the last 5 valid gripper values >= 0.25
      (episode ends with the jaw still open = accidental recording; today
      redundant with R1, kept as the automated safety net)

Output: qc.csv (session,episode,category,reason) — category is "ok" or
"excluded"; feed to sroi_to_lerobot.py --qc_csv with --qc_categories ok.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

MD_LIST = Path('/home/ss/data/QC工作汇总_20261013/轨迹QC/剔除清单_EXCLUDED.md')
ROOTS = [Path('/home/ss/data/one_strawberry_labpicking'), Path('/home/ss/data/120_onesb_supplementary')]
OUT = Path('/home/ss/data/rebuild_qc_20261013/qc.csv')
TAIL_THRESHOLD = 0.25

# R1: parse the machine-readable tail of the manual QC markdown
r1 = set(re.findall(r'^(\d{8}_\d{6}-png/episode_\d+)$', MD_LIST.read_text(), re.M))
print(f"R1 手动轨迹QC清单: {len(r1)} 条 (来自 {MD_LIST.name})")

rows = []
counts = {'R1': 0, 'R2': 0, 'R3': 0, 'ok': 0}
for root in ROOTS:
    for sess in sorted(root.glob('*-png')):
        for ep in sorted(sess.glob('episode_*')):
            key = f"{sess.name}/{ep.name}"
            reasons = []
            if key in r1:
                reasons.append('R1_manual_traj_qc')
            g = np.loadtxt(ep / 'gripper_distances.txt')
            if np.isnan(g).any():
                reasons.append('R2_gripper_long_occlusion')
            v = g[~np.isnan(g)]
            if len(v) and float(np.median(v[-5:])) >= TAIL_THRESHOLD:
                reasons.append('R3_no_closing_action')
            if reasons:
                cat = 'excluded'
                for r in reasons:
                    counts[r.split('_')[0]] += 1
            else:
                cat = 'ok'
                counts['ok'] += 1
            rows.append({'session': sess.name, 'episode': ep.name,
                         'category': cat, 'reason': '+'.join(reasons)})

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['session', 'episode', 'category', 'reason'])
    w.writeheader()
    w.writerows(rows)

total = len(rows)
print(f"共 {total} 条 | ok {counts['ok']} | excluded {total - counts['ok']}")
print(f"按规则命中(可叠加): R1={counts['R1']} R2={counts['R2']} R3={counts['R3']}")
print(f"写出 {OUT}")
print("\nexcluded 明细:")
for r in rows:
    if r['category'] == 'excluded':
        print(f"  {r['session']}/{r['episode']}  {r['reason']}")
