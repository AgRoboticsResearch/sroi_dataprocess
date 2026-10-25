#!/usr/bin/env python3
"""重绘尖峰帧注释图（v1 方案口径）：只保留窄带 395-450 行 ROI——带外压暗、
带边界细黄线；框出该帧 v1 实际配对输出距离的两个检出（含 ID/对角线标注）。
着色 = 位置是否落在该侧后来的 v2 窗条范围内（绿=范围内，红=范围外，即假货）；
仅供分析参照，v1 当年并无窗条。"""
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
OUTD = Path(__file__).resolve().parent / "frames_narrowband"
OUTD.mkdir(exist_ok=True)

def detect(gray):
    """v1 窄带检测：带外置零，每 ID 取最后一个合格检出。"""
    g = gray.copy()
    g[:Y0, :] = 0
    g[Y1:, :] = 0
    left = right = None
    for res in det.detect(g):
        ok = (abs(np.linalg.norm(res.corners[0] - res.corners[2]) - NOM) <= THR and
              abs(np.linalg.norm(res.corners[1] - res.corners[3]) - NOM) <= THR)
        if not ok:
            continue
        if res.tag_id == LID:
            left = res
        elif res.tag_id == RID:
            right = res
    return left, right

rows = [r for r in csv.DictReader(open(Path(__file__).resolve().parent /
                                       "narrowband_spike_scan.csv"))
        if int(r["n_spikes"]) > 0]
Z = 2
for r in rows:
    name = r["episode"]
    root = next(rt for rt in ROOTS if (rt / name).exists())
    fs = sorted((root / name).glob("color_*.png"))
    for f_str in r["spike_frames"].split(";"):
        i = int(f_str)
        img = cv.imread(str(fs[i]))
        vis = cv.resize(img, None, fx=Z, fy=Z, interpolation=cv.INTER_NEAREST)
        dark = (vis.astype(np.float32) * 0.35).astype(np.uint8)   # 带外压暗
        mask = np.ones(vis.shape[:2], bool)
        mask[Y0 * Z:Y1 * Z, :] = False
        vis[mask] = dark[mask]
        cv.line(vis, (0, Y0 * Z), (vis.shape[1], Y0 * Z), (0, 255, 255), 1)
        cv.line(vis, (0, Y1 * Z), (vis.shape[1], Y1 * Z), (0, 255, 255), 1)
        left, right = detect(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        for side, res in [("L", left), ("R", right)]:
            if res is None:
                continue
            x0, x1 = LW if side == "L" else RW
            x = res.center[0]
            col = (0, 255, 0) if x0 <= x <= x1 else (0, 0, 255)
            pts = np.int32(res.corners * Z)
            cv.polylines(vis, [pts], True, col, 2)
            dg = np.linalg.norm(res.corners[0] - res.corners[2])
            org = (int(pts[:, 0].min()), max(22, int(pts[:, 1].min()) - 8))
            cv.putText(vis, f"ID{res.tag_id} {dg:.0f}px", org,
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5)
            cv.putText(vis, f"ID{res.tag_id} {dg:.0f}px", org,
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        out = OUTD / f"{name.replace('/', '_')}_f{i:03d}.png"
        cv.imwrite(str(out), vis)
        print("saved", out.name)
