#!/usr/bin/env python3
"""全集置信度字段扫描：为 v3 置信度闸（hamming==0 / decision_margin≥30）定参。

对 1138 集每帧做 v1 窄带检测但**不设尺寸闸**，记录带内全部检出的
hamming / decision_margin / 双对角线 / 是否落在 v2 窗条内。据此统计：
- 尺寸合格的检出里，有多少 ham≥1 或 dm<30（会被闸杀掉，其中多少是真 tag）
- 闸后真 tag 的保留率（窗内 + 尺寸合格 + 过闸）
输出：confidence_scan.csv（一行一个检出）
"""
import csv
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pupil_apriltags import Detector

cfg = json.loads((REPO / "configs/sroi_v2_d405.json").read_text())
LID, RID = cfg["left_gripper_tag_id"], cfg["right_gripper_tag_id"]
NOM, THR = cfg["nominal_diag_distance"], cfg["threshold"]
Y0, Y1 = cfg["gripper_roi"], cfg["gripper_roi_bottom"]
LW, RW = tuple(cfg["left_tag_x_range"]), tuple(cfg["right_tag_x_range"])
det = Detector(families=cfg["tag_family"], nthreads=1, quad_decimate=1.0,
               quad_sigma=0, refine_edges=1, decode_sharpening=0.25, debug=0)
ROOTS = [Path("/home/ss/data/one_strawberry_labpicking"),
         Path("/home/ss/data/120_onesb_supplementary")]
OUT_CSV = Path(__file__).resolve().parent / "confidence_scan.csv"

out = open(OUT_CSV, "w", newline="")
w = csv.writer(out)
w.writerow(["episode", "frame", "tag_id", "x", "y", "hamming", "decision_margin",
            "d0", "d1", "size_ok", "in_window"])
eps = [p for rt in ROOTS for p in sorted(rt.glob("*/episode_*"))
       if list(p.glob("color_*.png"))]
print(f"扫描 {len(eps)} 集", flush=True)
for k, ep in enumerate(eps):
    name = f"{ep.parent.name}/{ep.name}"
    for i, p in enumerate(sorted(ep.glob("color_*.png"))):
        g = cv.cvtColor(cv.imread(str(p)), cv.COLOR_BGR2GRAY)
        g = g.copy()
        g[:Y0, :] = 0
        g[Y1:, :] = 0
        for res in det.detect(g):
            d0 = np.linalg.norm(res.corners[0] - res.corners[2])
            d1 = np.linalg.norm(res.corners[1] - res.corners[3])
            x0, x1 = (LW if res.tag_id == LID else RW)
            inw = x0 <= res.center[0] <= x1 and Y0 <= res.center[1] <= Y1
            w.writerow([name, i, res.tag_id, round(float(res.center[0]), 1),
                        round(float(res.center[1]), 1), res.hamming,
                        round(float(res.decision_margin), 1),
                        round(float(d0), 1), round(float(d1), 1),
                        int(abs(d0 - NOM) <= THR and abs(d1 - NOM) <= THR),
                        int(inw)])
    out.flush()
    if (k + 1) % 50 == 0:
        print(f"  进度 {k+1}/{len(eps)}", flush=True)
out.close()
print("done ->", OUT_CSV, flush=True)
