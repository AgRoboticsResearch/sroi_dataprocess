#!/usr/bin/env python3
"""21 个尖峰 episode 的 v1 / v3 处理结果叠加曲线。

v1 曲线 = 窄带方案逐帧检测（带内每 ID 取最后一个、双对角线 41±9）→ 前值填充
（首有效值作种子，快速运动段跟随上一值——v1 原方案如此，不做修饰）→ 用该集
v3 归一化锚点（gripper_normalization.json）归一化。
v3 曲线 = 数据集中现成的 gripper_distances.txt。
尖峰帧画竖虚线。输出 curves_v1_v3/<episode>.png
"""
import csv
import json
from pathlib import Path

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[2]
sys_path = REPO
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from pupil_apriltags import Detector  # noqa: E402

cfg = json.loads((REPO / "configs/sroi_v2_d405.json").read_text())
LID, RID = cfg["left_gripper_tag_id"], cfg["right_gripper_tag_id"]
NOM, THR = cfg["nominal_diag_distance"], cfg["threshold"]
Y0, Y1 = cfg["gripper_roi"], cfg["gripper_roi_bottom"]
det = Detector(families=cfg["tag_family"], nthreads=1, quad_decimate=1.0,
               quad_sigma=0, refine_edges=1, decode_sharpening=0.25, debug=0)
ROOTS = [Path("/home/ss/data/one_strawberry_labpicking"),
         Path("/home/ss/data/120_onesb_supplementary")]
OUTD = Path(__file__).resolve().parent / "curves_v1_v3"
OUTD.mkdir(exist_ok=True)

rows = [r for r in csv.DictReader(open(Path(__file__).resolve().parent /
                                       "narrowband_spike_scan.csv"))
        if int(r["n_spikes"]) > 0]
for r in rows:
    name = r["episode"]
    root = next(rt for rt in ROOTS if (rt / name).exists())
    ep = root / name
    fs = sorted(ep.glob("color_*.png"))
    spikes = [int(s) for s in r["spike_frames"].split(";")]

    # --- v1 处理：逐帧检测 → 前值填充 → 归一化 ---
    px = []
    for p in fs:
        g = cv.cvtColor(cv.imread(str(p)), cv.COLOR_BGR2GRAY)
        g = g.copy()
        g[:Y0, :] = 0
        g[Y1:, :] = 0
        l = r_ = None
        for res in det.detect(g):
            d0 = np.linalg.norm(res.corners[0] - res.corners[2])
            d1 = np.linalg.norm(res.corners[1] - res.corners[3])
            if abs(d0 - NOM) <= THR and abs(d1 - NOM) <= THR:
                if res.tag_id == LID:
                    l = res.center[0]      # 覆盖 = 取最后一个
                elif res.tag_id == RID:
                    r_ = res.center[0]
        px.append(abs(r_ - l) if l is not None and r_ is not None else np.nan)
    px = np.array(px)
    anchors = json.loads((ep / "gripper_normalization.json").read_text())["normalization"]
    lo, hi = anchors["min_distance_px"], anchors["max_distance_px"]
    v1 = np.full(len(px), np.nan)
    last = px[~np.isnan(px)][0]                # 首有效值作种子（v1 前值填充语义）
    for i in range(len(px)):
        if not np.isnan(px[i]):
            last = px[i]
        v1[i] = np.clip((last - lo) / (hi - lo), 0, 1)
    v3 = np.loadtxt(ep / "gripper_distances.txt")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(v1, color="#1f77b4", lw=1.3, label="v1 方案（窄带全行，前值填充）")
    ax.plot(v3, color="#d62728", lw=1.3, label="v3 方案（窗条+判据）")
    for s in spikes:
        ax.axvline(s, color="k", ls="--", lw=0.8)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("钳口开度（归一化）")
    ax.set_xlabel("帧号")
    ax.set_title(f"{name}  尖峰帧 {spikes}")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = OUTD / f"{name.replace('/', '_')}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out.name, flush=True)
