#!/usr/bin/env python3
"""窄带全行方案（2026-07-09 f4c8162 → 2026-10-14 v2 之间的仓库形态）严格保真复刻，
全集扫描定位会出现尖峰的 episode。

窄带方案语义（git 3814532 版 gripper_estimation_april_tag.py + f4c8162 版 config）：
  - 搜索带 = 行 395..450 全宽（带外上下都置零）
  - 每 ID 取**最后一个**合格检出（循环覆盖语义，非 v2 的取首个）
  - 双对角线 41±9 px；raw-only（无预处理链）；无对称/先验/行程界
  - 双 ID 齐 → 距离；缺 → None（旧法再前值填充；不影响有效样本尖峰判定）
尖峰判据 = v2 remove_spikes 平坦上下文规则（0.15 × (262−83) px，≤2 帧，两侧各 3 个
有效样本），即用户 10 月初人工观察的量化版。

用法：
  python3 scan_narrowband_spikes.py suspects   # 历史尖峰集 + 汇总.csv maxdiff 嫌疑
  python3 scan_narrowband_spikes.py all        # 两数据根全集（约 50 分钟，建议 nohup）
输出：narrowband_spike_scan.csv（追加写）+ stdout 摘要
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
from gripper_tag_core import remove_spikes

cfg = json.loads((REPO / "configs/sroi_v2_d405.json").read_text())
LID, RID = cfg["left_gripper_tag_id"], cfg["right_gripper_tag_id"]
NOM, THR = cfg["nominal_diag_distance"], cfg["threshold"]  # 41 ± 9
Y0, Y1 = cfg["gripper_roi"], cfg["gripper_roi_bottom"]     # 395..450
det = Detector(families=cfg["tag_family"], nthreads=1, quad_decimate=1.0,
               quad_sigma=0, refine_edges=1, decode_sharpening=0.25, debug=0)
ROOTS = [Path("/home/ss/data/one_strawberry_labpicking"),
         Path("/home/ss/data/120_onesb_supplementary")]
OUT_CSV = Path(__file__).resolve().parent / "narrowband_spike_scan.csv"
RANGE_PX = 262.0 - 83.0  # 定标行程（去尖峰判据的量纲基准）

def narrowband_detect(gray):
    """带内检测，每 ID 取最后一个合格检出（覆盖语义）。返回 (lx, ly, rx, ry)。"""
    gray = gray.copy()
    gray[:Y0, :] = 0
    gray[Y1:, :] = 0
    left = right = None
    for res in det.detect(gray):
        if res.tag_id == LID and check(res):
            left = res          # 覆盖 = 取最后一个
        elif res.tag_id == RID and check(res):
            right = res
    return left, right

def check(r):
    d0 = np.linalg.norm(r.corners[0] - r.corners[2])
    d1 = np.linalg.norm(r.corners[1] - r.corners[3])
    return abs(d0 - NOM) <= THR and abs(d1 - NOM) <= THR

def scan_episode(ep_dir):
    """返回 (n_frames, n_valid, spikes)；spikes = [dict(frame,value,l,r,desc)]"""
    frames = sorted(ep_dir.glob("color_*.png"))
    xs, details = [], {}
    for i, p in enumerate(frames):
        g = cv.cvtColor(cv.imread(str(p)), cv.COLOR_BGR2GRAY)
        l, r = narrowband_detect(g)
        if l is not None and r is not None:
            xs.append(abs(r.center[0] - l.center[0]))
            details[i] = (l.center, r.center, l, r)
        else:
            xs.append(None)
    indices = [i for i, x in enumerate(xs) if x is not None]
    values = [xs[i] for i in indices]
    drop = remove_spikes(indices, values, RANGE_PX)  # 平坦上下文尖峰判据
    spikes = []
    for j in drop:
        i = indices[j]
        (lc, rc, lres, rres) = details[i]
        spikes.append({
            "frame": i, "value_px": round(values[j], 1),
            "left": f"({lc[0]:.0f},{lc[1]:.0f},d{np.linalg.norm(lres.corners[0]-lres.corners[2]):.0f})",
            "right": f"({rc[0]:.0f},{rc[1]:.0f},d{np.linalg.norm(rres.corners[0]-rres.corners[2]):.0f})",
        })
    return len(frames), len(indices), spikes

def episodes():
    for root in ROOTS:
        for ep in sorted(root.glob("*/episode_*")):
            if list(ep.glob("color_*.png")):
                yield ep

def main(mode):
    if mode == "suspects":
        names = ["20260709_144719-png/episode_027", "20260709_153147-png/episode_009",
                 "20260709_155555-png/episode_031", "20260709_151502-png/episode_039"]
        for r in ROOTS:  # 汇总.csv maxdiff top-20 嫌疑
            pass
        import csv as _csv
        top = _csv.reader(open(Path.home() / "data/QC工作汇总_20261013/夹爪v2/对比_20261013/汇总.csv"))
        next(top)
        rows = sorted(top, key=lambda r: -float(r[5]))[:20]
        names += [f"{r[0]}" for r in rows]
        todo = []
        seen = set()
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            for root in ROOTS:
                if (root / n).exists():
                    todo.append(root / n)
    else:
        todo = list(episodes())
    print(f"扫描 {len(todo)} 个 episode", flush=True)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "n_frames", "n_valid", "n_spikes",
                    "spike_frames", "spike_details"])
        for k, ep in enumerate(todo):
            name = f"{ep.parent.name}/{ep.name}"
            n_f, n_v, spikes = scan_episode(ep)
            w.writerow([name, n_f, n_v, len(spikes),
                        ";".join(str(s["frame"]) for s in spikes),
                        ";".join(f"f{s['frame']}:{s['value_px']}px L{s['left']} R{s['right']}"
                                 for s in spikes)])
            f.flush()
            if spikes:
                print(f"[尖峰] {name}: {[(s['frame'], s['value_px']) for s in spikes]}", flush=True)
            if (k + 1) % 50 == 0:
                print(f"  进度 {k+1}/{len(todo)}", flush=True)
    print("done ->", OUT_CSV, flush=True)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "suspects")
