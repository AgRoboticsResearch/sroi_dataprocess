#!/usr/bin/env python3
"""qc_review_gui — 人工复核 QC 的本地图形界面工具。

用法:
    python3 visualization/qc_review_gui.py [--root 数据根] [--port 8765] [--verdicts pending]

浏览器打开 http://localhost:8765 。功能:
  - 左栏 episode 清单(按 verdict 过滤: pending / 全部), flags 徽章
  - 主区逐帧播放/拖动/步进, 轨迹投影叠加(过去渐变/未来灰, 近平面+视场裁剪),
    胶囊检测区轮廓, 可选红掩膜着色, 帧号/时间戳
  - 右栏: 当前帧位姿矩阵(3x4)、flags/证据、人工裁决(通过/拒绝/撤回)、
    start_frame(可一键取当前帧)、备注; 保存只写 qc.json 的人工字段
    (verdict/start_frame/review_note/reviewed_at), 程序字段永不触碰
  - 快捷键: 空格播放, ←/→ 步进, Shift+←/→ 跳10帧, PgUp/PgDn 上下集
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qc  # noqa: E402  — 同目录, 复用判据数学(投影/胶囊/HSV/加载)

# ---------------------------------------------------------------------------
# 数据索引与缓存
# ---------------------------------------------------------------------------

class Store:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.eps: dict[str, dict] = {}      # key -> {dir, qc}
        self.traj: dict[str, dict] = {}     # key -> poses/ledger/grip/times/K/D
        self.reload()

    def reload(self):
        eps = {}
        sessions = sorted(set(self.root.glob("*-png")) | set(self.root.glob("*/*-png")))
        for sess in sessions:
            if not sess.is_dir():
                continue
            for ep in sorted(sess.glob("episode_*")):
                fp = ep / qc.QC_JSON_NAME
                if not fp.exists():
                    continue
                try:
                    doc = json.loads(fp.read_text())
                except Exception:
                    continue
                eps[f"{sess.name}/{ep.name}"] = {"dir": ep, "qc": doc}
        with self.lock:
            self.eps = eps
        return len(eps)

    def trajectory(self, key: str) -> dict | None:
        with self.lock:
            if key in self.traj:
                return self.traj[key]
        d = self.eps.get(key)
        if d is None:
            return None
        ep = d["dir"]
        poses = qc._load_poses(ep / "CameraTrajectoryTransformed.txt")
        if poses is None:
            return None
        grip = np.atleast_1d(np.loadtxt(ep / "gripper_distances.txt"))
        ext = json.loads(Path(qc.EXTRINSICS_CONFIG).read_text()) if qc.EXTRINSICS_CONFIG else None
        info = json.loads((ep / "camera_info_color.json").read_text())
        K = np.array(info["K"], float).reshape(3, 3)
        D = np.array(info["D"], float).ravel()
        T_oc = np.array(ext["T_optical_camera"], float)[:3, :3]
        tip = np.array(ext["T_camera_gripper_tip"], float)[:3, 3]
        # 世界系(=首帧参考)下的指尖路径 + 末端点(胶囊几何用,不随显示帧变)
        rel0 = np.linalg.inv(poses[0]) @ poses
        pts_w = (rel0[:, :3, :3] @ tip[:, None]).squeeze(2) + rel0[:, :3, 3]
        times = np.atleast_1d(np.loadtxt(ep / "times.txt"))
        # 每帧自己的参考系投影(用户 2026-10-17 修正: 画在第 i 帧上的路径必须
        # 以第 i 帧为参考, 否则路径像刚体一样钉在相机上跟着走)
        p2f = np.full((len(poses), len(poses), 2), np.nan)
        okf = np.zeros((len(poses), len(poses)), dtype=bool)
        for i in range(len(poses)):
            if not np.isfinite(poses[i][:3, :3]).all():
                continue
            rel = np.linalg.inv(poses[i]) @ poses
            pts = (rel[:, :3, :3] @ tip[:, None]).squeeze(2) + rel[:, :3, 3]
            pts_opt = pts @ T_oc.T
            if not np.isfinite(pts_opt).all():
                fin = np.isfinite(pts_opt).all(axis=1)
            else:
                fin = np.ones(len(poses), dtype=bool)
            pj, _ = cv2.projectPoints(pts_opt[fin].astype(float), np.zeros(3), np.zeros(3), K, D)
            pj = pj.reshape(-1, 2)
            z = pts_opt[fin][:, 2]
            ang = np.degrees(np.arctan2(np.linalg.norm(pts_opt[fin][:, :2], axis=1), np.maximum(z, 1e-9)))
            ok = (z >= 0.03) & (ang <= 45) & np.isfinite(pj).all(axis=1)
            p2f[i][fin] = pj
            okf[i][fin] = ok
        tdoc = {"poses": poses, "grip": grip, "times": times, "K": K, "D": D,
                "p2f": p2f, "okf": okf, "pts_w_last": pts_w[-1], "dir": ep,
                "tip": tip, "T_oc": T_oc}
        with self.lock:
            self.traj[key] = tdoc
        return tdoc


STORE: Store | None = None
CACHE_DIR = Path("/tmp/qc_review_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_prerender: dict[str, dict] = {}   # key -> {"done": int, "total": int, "thread": T}
_prerender_lock = threading.Lock()


def _cache_dir(key: str) -> Path:
    d = CACHE_DIR / key.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d


def start_prerender(key: str):
    """Background-render every frame of an episode into the cache dir (default
    overlay combo fut=1 reg=1 mask=0) so scrubbing/playback is instant."""
    with _prerender_lock:
        st = _prerender.get(key)
        if st and st["thread"].is_alive():
            return
        t = STORE.trajectory(key)
        total = len(t["poses"]) if t else 0
        th = threading.Thread(target=_prerender_worker, args=(key, total), daemon=True)
        _prerender[key] = {"done": 0, "total": total, "thread": th}
        th.start()


def _prerender_worker(key: str, total: int):
    d = _cache_dir(key)
    for i in range(total):
        for ov in (1, 0):
            fp = d / f"f{i:05d}_o{ov}.jpg"
            if not fp.exists():
                try:
                    fp.write_bytes(render_frame(key, i, bool(ov)))
                except Exception as e:
                    print(f"[prerender] {key} f{i}: {e}", flush=True)
                    return
        with _prerender_lock:
            _prerender[key]["done"] = i + 1

STORE_PLACEHOLDER = None

# ---------------------------------------------------------------------------
# 帧渲染(服务端 cv2, 遵守渲染规矩)
# ---------------------------------------------------------------------------

def render_frame(key: str, i: int, overlay: bool) -> bytes:
    """Render one review frame.

    Tier A (overlay=False): video-style trajectory only — a single green->red
    gradient polyline of the REMAINING path (current tip -> final tip, in the
    current frame's reference), green dot at the tip, blue dot at the end.
    Tier B (overlay=True): tier A + HSV red-mask tint + capsule region outline.
    """
    t = STORE.trajectory(key)
    if t is None:
        raise FileNotFoundError(key)
    ep = t["dir"]
    imgs = sorted(ep.glob("color_*.png")) + sorted(ep.glob("color_*.jpg"))
    img = cv2.imread(str(imgs[i]))
    if img is None:
        raise FileNotFoundError(str(imgs[i]))
    vis = img.copy()
    m = qc._red_mask_hsv(img) if overlay else None
    if overlay and m is not None:
        ov = vis.copy()
        ov[m] = (255, 0, 255)
        vis = cv2.addWeighted(ov, 0.4, vis, 0.6, 0)

    p2, ok = t["p2f"][i], t["okf"][i]     # 第 i 帧参考系的投影
    n = len(p2)
    last = max(k for k in range(n) if ok[k]) if ok.any() else i
    # 剩余路径: 从当前帧到终点, 绿(现在)->红(终点) 单根渐变线(同视频)
    for a in range(i, last):
        if not (ok[a] and ok[a + 1]):
            continue
        f = (a - i) / max(1, last - i)
        cv2.polylines(vis, [p2[a:a + 2].astype(np.int32).reshape(-1, 1, 2)], False,
                      (0, int(255 * (1 - f)), int(255 * f)), 3)
    # 当前指尖绿点(钉在物理指尖上) + 终点蓝点
    if ok[i]:
        cv2.circle(vis, (int(round(p2[i][0])), int(round(p2[i][1]))), 6, (0, 255, 0), -1)
    if ok[last]:
        cv2.circle(vis, (int(round(p2[last][0])), int(round(p2[last][1]))), 6, (255, 0, 0), -1)
    # 档B: 胶囊检测区(终点已有蓝点,红叉冗余删去)
    if overlay and m is not None:
        rel = np.linalg.inv(t["poses"][i]) @ t["poses"]
        end3 = (rel[last, :3, :3] @ t["tip"]) + rel[last, :3, 3]
        end_opt = end3 @ t["T_oc"].T
        if np.isfinite(end_opt).all():
            geo = qc.capsule_geometry(end_opt, t["K"], t["D"])
            region = qc.capsule_region_mask(m.shape, geo["discs"])
            contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)
    okk, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>QC 复核台</title>
<style>
:root{--bg:#14161a;--panel:#1d2026;--panel2:#22262e;--line:#2c313a;--fg:#d8dce3;--dim:#8b93a0;
--acc:#4da3ff;--ok:#2e9e5b;--bad:#e05252;--pend:#d9a441;--mono:ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,"Noto Sans CJK SC",sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:14px;padding:8px 14px;background:var(--panel);border-bottom:1px solid var(--line)}
header h1{font-size:15px;font-weight:600}
header .dim{color:var(--dim);font-size:12px}
header select,header button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer}
header button:hover{border-color:var(--acc)}
#main{flex:1;display:flex;min-height:0}
#list{width:270px;background:var(--panel);border-right:1px solid var(--line);overflow-y:auto}
.ep{padding:7px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;flex-direction:column;gap:3px}
.ep:hover{background:var(--panel2)}
.ep.on{background:var(--panel2);box-shadow:inset 3px 0 0 var(--acc)}
.ep .name{font-size:12.5px;font-family:var(--mono);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ep .meta{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
.chip{font-size:10px;padding:1px 7px;border-radius:8px;font-weight:600}
.v-manual_ok{background:#173d27;color:#5ddb8d}.v-auto_ok{background:#16324a;color:#6fb7ff}
.v-rejected{background:#43191c;color:#ff8f8f}.v-pending{background:#40321a;color:#ecc06a}
.fdot{width:8px;height:8px;border-radius:50%;display:inline-block}
#stage{flex:1;display:flex;flex-direction:column;min-width:0}
#viewer{flex:1;display:flex;align-items:center;justify-content:center;background:#0d0e11;min-height:0;position:relative}
#viewer img{max-width:100%;max-height:100%;user-select:none}
#hud{position:absolute;top:10px;left:12px;font-family:var(--mono);font-size:12px;background:rgba(15,17,21,.75);padding:4px 10px;border-radius:6px;border:1px solid var(--line)}
#bar{display:flex;align-items:center;gap:8px;padding:8px 12px;background:var(--panel);border-top:1px solid var(--line)}
#bar button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer;font-size:13px}
#bar button:hover{border-color:var(--acc)}
#bar input[type=range]{flex:1;accent-color:var(--acc)}
#frameNo{font-family:var(--mono);min-width:86px;text-align:center}
label.tog{display:flex;align-items:center;gap:4px;font-size:12px;color:var(--dim);cursor:pointer}
#side{width:330px;background:var(--panel);border-left:1px solid var(--line);overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:12px}
.sec{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px}
.sec h3{font-size:12px;color:var(--dim);margin-bottom:8px;font-weight:600;letter-spacing:.5px}
table.pose{font-family:var(--mono);font-size:11.5px;border-collapse:collapse;width:100%}
table.pose td{padding:2px 6px;text-align:right;border:1px solid var(--line);white-space:nowrap}
.flags{display:flex;flex-wrap:wrap;gap:5px}
.evi{display:grid;grid-template-columns:1fr auto;gap:3px 10px;font-size:12px}
.evi span:nth-child(2){font-family:var(--mono);color:var(--dim)}
.vbtns{display:flex;gap:6px}
.vbtn{flex:1;padding:8px 0;border-radius:7px;border:1px solid var(--line);cursor:pointer;font-weight:600;font-size:13px;background:var(--panel);color:var(--fg)}
.vbtn.ok:hover,.vbtn.on.ok{background:#173d27;color:#5ddb8d;border-color:var(--ok)}
.vbtn.bad:hover,.vbtn.on.bad{background:#43191c;color:#ff8f8f;border-color:var(--bad)}
.vbtn.pend:hover,.vbtn.on.pend{background:#40321a;color:#ecc06a;border-color:var(--pend)}
.sfrow{display:flex;gap:6px;margin-top:8px}
.sfrow input{flex:1;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-family:var(--mono)}
textarea{width:100%;min-height:64px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12.5px;resize:vertical}
#save{width:100%;padding:9px 0;margin-top:8px;border-radius:7px;border:none;background:var(--acc);color:#0d1117;font-weight:700;cursor:pointer;font-size:14px}
#save:hover{filter:brightness(1.1)}
#toast{position:fixed;bottom:18px;right:18px;background:#173d27;color:#5ddb8d;padding:9px 16px;border-radius:8px;font-size:13px;opacity:0;transition:.25s;pointer-events:none}
#toast.show{opacity:1}
.kbd{font-size:11px;color:var(--dim)}
.cut{outline:3px solid var(--pend)}
</style></head><body>
<header>
  <h1>QC 复核台</h1><span class="dim" id="root"></span>
  <select id="filter"><option value="pending">待人工 (pending)</option><option value="all">全部</option>
  <option value="flagged">有任何flag</option><option value="manual">人工判过的</option></select>
  <span class="dim" id="count"></span><button onclick="reload()">刷新清单</button>
  <span class="kbd">空格播放 · ←/→步进 · Shift跳10 · PgUp/PgDn换集 · S保存</span>
</header>
<div id="main">
  <div id="list"></div>
  <div id="stage"><img id="frame" draggable="false"></div>
  <div id="side"></div>
</div>
<div id="bar">
  <button id="play">▶</button><button onclick="step(-1)">◀</button>
  <button onclick="step(1)">▶</button><span id="prog" class="kbd"></span>
  <input type="range" id="slider" min="0" value="0">
  <span id="frameNo">-</span>
  <label class="tog"><input type="checkbox" id="tOv" checked>掩膜与检测框</label>
</div>
</body>
<script>
let EPS=[],cur=-1,N=0,i=0,playing=false,timer=null,TR=null,QC=null,dirty=false;
const $=id=>document.getElementById(id);
const FCOL={not_processed:'#8b93a0',traj_missing:'#c62828',gripper_missing:'#ad1457',traj_frame_jump:'#ef6c00',first_frame_projection:'#6a1b9a',gripper_not_closed:'#00838f'};
async function loadList(){EPS=await (await fetch('/api/episodes?filter='+$('filter').value)).json();$('count').textContent=EPS.length+' 集';
  if(cur>=EPS.length)cur=-1; renderList(); if(cur<0&&EPS.length){cur=0;loadEp();}}
function renderList(){const L=$('list');L.innerHTML='';EPS.forEach((e,k)=>{const d=document.createElement('div');
  d.className='ep'+(k===cur?' on':'');const fl=Object.entries(e.flags||{}).filter(([k,v])=>v).map(([k])=>`<span class="fdot" title="${k}" style="background:${FCOL[k]||'#888'}"></span>`).join('');
  d.innerHTML=`<div class="name">${e.key.split('/')[1]} <span style="color:var(--dim);font-size:10px">${e.key.split('/')[0].slice(9,-4)}</span></div>
  <div class="meta"><span class="chip v-${e.verdict}">${e.verdict}</span>${fl}${e.start_frame?`<span style="color:var(--pend);font-size:10px">start=${e.start_frame}</span>`:''}</div>`;
  d.onclick=()=>{cur=k;renderList();loadEp()};L.appendChild(d);});}
async function loadEp(){const e=EPS[cur];if(!e)return;dirty=false;
  TR=await (await fetch('/api/traj/'+encodeURIComponent(e.key))).json();QC=await (await fetch('/api/qc/'+encodeURIComponent(e.key))).json();
  N=TR.n;i=0;$('slider').max=N-1;$('slider').value=0;stop();renderSide();show();
  fetch('/api/open/'+encodeURIComponent(e.key));pollProg();}
function pollProg(){if(cur<0||!EPS[cur])return;const e=EPS[cur];
  fetch('/api/progress/'+encodeURIComponent(e.key)).then(r=>r.json()).then(p=>{
    const el=$('prog');
    if(!el)return;
    if(p.total>0&&p.done<p.total){el.textContent=`渲染中 ${p.done}/${p.total}`;setTimeout(pollProg,700);}
    else el.textContent='';
  }).catch(()=>{});}
function src(){const e=EPS[cur];
  return '/frame/'+encodeURIComponent(e.key)+`?i=${i}&ov=${$('tOv').checked?1:0}`;}
function show(){$('frame').src=src();$('slider').value=i;$('frameNo').textContent=`${i} / ${N-1}`;
  const cut=(QC.start_frame||0)>i;$('frame').classList.toggle('cut',cut);
  if(TR.poses&&TR.poses[i]){const p=TR.poses[i];const tb=document.getElementById('poseT');
    if(tb){let h='';for(let r=0;r<3;r++){h+='<tr>';for(let c=0;c<4;c++)h+=`<td>${p[r*4+c]}</td>`;h+='</tr>';}tb.innerHTML=h;}}
  drawGrip();draw3d();}
function drawGrip(){const c=document.getElementById('gcurve');if(!c||!TR||!TR.grip)return;
  const ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#2c313a';ctx.strokeRect(0,0,W,H);
  ctx.strokeStyle='#e65100';ctx.beginPath();
  const g=TR.grip,n=g.length;
  for(let k=0;k<n;k++){const x=k/(n-1)*W;const y=H-8-(g[k]??0)*(H-16);
    if(g[k]==null){ctx.moveTo(x,y);continue}
    k===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
  ctx.stroke();
  const sf=(QC&&QC.start_frame)||0;if(sf>0){ctx.strokeStyle='#d9a441';ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(sf/(n-1)*W,0);ctx.lineTo(sf/(n-1)*W,H);ctx.stroke();ctx.setLineDash([]);}
  ctx.strokeStyle='#4da3ff';ctx.lineWidth=2;ctx.beginPath();
  ctx.moveTo(i/(n-1)*W,0);ctx.lineTo(i/(n-1)*W,H);ctx.stroke();ctx.lineWidth=1;
  const gv=document.getElementById('gval');
  if(gv){const v=g[i];gv.textContent=v==null?'开度 NaN':'开度 '+v.toFixed(3);}}
function draw3d(){const c=document.getElementById('traj3d');if(!c||!TR||!TR.xyz)return;
  const ctx=c.getContext('2d'),W=c.width,H=c.height;
  ctx.fillStyle='#fafafa';ctx.fillRect(0,0,W,H);
  const pts=TR.xyz,n=pts.length;if(!n)return;
  let lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];
  for(const p of pts)if(p)p.forEach((v,k)=>{lo[k]=Math.min(lo[k],v);hi[k]=Math.max(hi[k],v);});
  // 俯视(用户指定朝向): X→屏幕上方, Y→屏幕左侧; 统一米/像素(两轴同尺度,
  // 以画布能容纳的最大跨度为准——X远大于Y时如实显示为细长轨迹,不再各自归一)
  const pr=(()=>{const sp=Math.max(hi[0]-lo[0],hi[1]-lo[1])||1e-6;
    const cx=(lo[0]+hi[0])/2, cy=(lo[1]+hi[1])/2, k=Math.min(W,H)*0.8/sp;
    return p=>{if(!p)return null;
      return [W/2-(p[1]-cy)*k, H/2-(p[0]-cx)*k]};})();
  // 全程淡线 + 渐进亮线
  ctx.strokeStyle='#cfcfcf';ctx.beginPath();let m=0;
  for(const p of pts){const q=pr(p);if(!q){m=1;continue}m?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);m=0;}
  ctx.stroke();
  ctx.strokeStyle='#1565c0';ctx.lineWidth=2;ctx.beginPath();m=0;
  for(let k=0;k<=i&&k<n;k++){const q=pr(pts[k]);if(!q){m=1;continue}m?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);m=0;}
  ctx.stroke();ctx.lineWidth=1;
  const a=pr(pts.find(Boolean));if(a){ctx.fillStyle='#1b7a3d';ctx.beginPath();ctx.arc(a[0],a[1],4,0,7);ctx.fill();}
  const cur=pr(pts[i]);if(cur){ctx.fillStyle='#0d5bb5';ctx.beginPath();ctx.arc(cur[0],cur[1],4,0,7);ctx.fill();}
  for(let k=n-1;k>=0;k--){const q=pr(pts[k]);if(q){ctx.fillStyle='#e05252';ctx.beginPath();ctx.arc(q[0],q[1],4,0,7);ctx.fill();break;}}
}
document.addEventListener('click',e=>{const c=e.target.closest&&e.target.closest('#gcurve');
  if(c&&TR){const r=c.getBoundingClientRect();const f=Math.round((e.clientX-r.left)/r.width*(N-1));
  i=Math.max(0,Math.min(N-1,f));show();}});
function step(d){i=Math.max(0,Math.min(N-1,i+d));show();}
function play(){if(!playing&&i>=N-1)i=0;playing=!playing;$('play').textContent=playing?'❚❚':'▶';
  if(playing){timer=setInterval(()=>{if(i>=N-1){stop();return;}i++;show();},1000/30);}else clearInterval(timer);}
function stop(){playing=false;clearInterval(timer);$('play').textContent='▶';}
function renderSide(){const e=EPS[cur];const fl=Object.entries(QC.flags||{}).filter(([k,v])=>v);
  const evi=QC.evidence||{};
  $('side').innerHTML=`
  <div class="sec"><h3>轨迹俯视 (X↑ 前, Y← 左, Z 压掉)</h3><canvas id="traj3d" width="290" height="200" style="width:100%"></canvas></div>
  <div class="sec"><h3>夹爪曲线 <span id="gval" style="float:right;font-family:var(--mono);color:var(--acc)"></span></h3><canvas id="gcurve" width="290" height="90" style="width:100%;cursor:crosshair"></canvas></div>
  <div class="sec"><h3>当前帧位姿 (camera_link 3×4)</h3><table class="pose"><tbody id="poseT"></tbody></table></div>
  <div class="sec"><h3>FLAGS</h3><div class="flags">${fl.length?fl.map(([k])=>`<span class="chip" style="background:${FCOL[k]||'#333'}22;color:${FCOL[k]}">${k}</span>`).join(''):'<span style="color:var(--ok)">无</span>'}</div>
    <div class="evi" style="margin-top:8px">
    <span>red_end / 阈值</span><span>${evi.red_end??'-'} / ${evi.red_end_min??'-'}</span>
    <span>vmax</span><span>${evi.vmax_mm??'-'} mm</span>
    <span>invalid 行</span><span>${evi.n_invalid_traj??'-'}</span>
    <span>gripper_end</span><span>${evi.gripper_end??'-'}</span>
    <span>path_length</span><span>${evi.path_length_cm??'-'} cm</span></div></div>
  <div class="sec"><h3>人工裁决</h3>
    <div class="vbtns">
      <button class="vbtn ok ${QC.verdict==='manual_ok'?'on':''}" onclick="setV('manual_ok')">✓ 通过</button>
      <button class="vbtn bad ${QC.verdict==='rejected'?'on':''}" onclick="setV('rejected')">✕ 拒绝</button>
      <button class="vbtn pend ${QC.verdict==='pending'?'on':''}" onclick="setV('pending')">撤回待审</button>
    </div>
    <div class="sfrow"><input id="sf" type="number" min="0" max="${N-1}" value="${QC.start_frame||0}" title="起始帧">
      <button onclick="$('sf').value=i;dirty=true" title="把起始帧设为当前显示的帧">设为当前帧</button>
      <span title="选择该 episode 进入数据集的起始帧。起始帧之前通常是: 初始化缺帧(NaN)或画面中尚未出现草莓的帧——这些帧没有训练价值,人工选定从此帧开始。" style="cursor:help;color:var(--dim);border:1px solid var(--line);border-radius:50%;width:18px;height:18px;display:inline-flex;align-items:center;justify-content:center;font-size:11px">?</span></div>
    <textarea id="note" placeholder="人工备注…">${QC.review_note||''}</textarea>
    <button id="save" onclick="save()">保存 (S)</button></div>`;}
function setV(v){QC.verdict=v;dirty=true;
  document.querySelectorAll('.vbtn').forEach(b=>b.classList.remove('on'));
  const map={manual_ok:0,rejected:1,pending:2};const bs=document.querySelectorAll('.vbtn');
  if(bs[map[v]])bs[map[v]].classList.add('on');}
async function save(){const e=EPS[cur];
  const body={verdict:QC.verdict,start_frame:parseInt($('sf').value||0),review_note:$('note').value};
  const r=await (await fetch('/api/qc/'+encodeURIComponent(e.key),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(r.ok){toast('已保存');QC=await (await fetch('/api/qc/'+encodeURIComponent(e.key))).json();dirty=false;
    const el=EPS[cur];el.verdict=body.verdict;el.start_frame=body.start_frame;renderList();renderSide();}else toast('保存失败');}
function toast(m){const t=$('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),1500);t.id='toast';}
async function reload(){await fetch('/api/reload');await loadList();toast('清单已刷新');}
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
  if(e.code==='Space'){e.preventDefault();play();}
  else if(e.key==='ArrowLeft')step(-1);else if(e.key==='ArrowRight')step(1);
  else if(e.key==='PageUp'&&cur>0){cur--;renderList();loadEp();}else if(e.key==='PageDown'&&cur<EPS.length-1){cur++;renderList();loadEp();}
  else if(e.key==='s'||e.key==='S')save();});
$('slider').oninput=e=>{i=parseInt(e.target.value);show();};
$('tOv').onchange=()=>show();
$('filter').onchange=()=>{cur=0;loadList();};
$('play').onclick=play;
const t2=document.createElement('div');t2.id='toast';document.body.appendChild(t2);
window.onerror=(m,src,l,c)=>{const b=document.createElement('div');
b.style.cssText='position:fixed;top:0;left:0;right:0;background:#8a1c1c;color:#fff;padding:6px 12px;font:12px monospace;z-index:99';
b.textContent='JS错误: '+m+' @'+l+':'+c;document.body.appendChild(b);};
window.addEventListener('unhandledrejection',e=>{const b=document.createElement('div');
b.style.cssText='position:fixed;top:26px;left:0;right:0;background:#7a5a1c;color:#fff;padding:6px 12px;font:12px monospace;z-index:99';
b.textContent='Promise错误: '+(e.reason&&e.reason.message||e.reason);document.body.appendChild(b);});
(async()=>{$('root').textContent=(await (await fetch('/api/root')).json()).root;await loadList();})();
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        print("[http]", fmt % a, flush=True)

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/" :
                b = PAGE.encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            elif u.path == "/api/root":
                self._json({"root": str(STORE.root)})
            elif u.path == "/api/reload":
                self._json({"episodes": STORE.reload()})
            elif u.path == "/api/episodes":
                flt = q.get("filter", ["pending"])[0]
                out = []
                for key, d in sorted(STORE.eps.items()):
                    doc = d["qc"]
                    v = doc.get("verdict")
                    flags = doc.get("flags", {})
                    if flt == "pending" and v != qc.VERDICT_PENDING:
                        continue
                    if flt == "flagged" and not any(flags.values()):
                        continue
                    if flt == "manual" and v not in (qc.VERDICT_MANUAL_OK, qc.VERDICT_REJECTED):
                        continue
                    out.append({"key": key, "verdict": v, "flags": flags,
                                "start_frame": doc.get("start_frame", 0)})
                self._json(out)
            elif u.path.startswith("/api/qc/"):
                key = unquote(u.path[len("/api/qc/"):])
                self._json(STORE.eps[key]["qc"])
            elif u.path.startswith("/api/traj/"):
                key = unquote(u.path[len("/api/traj/"):])
                t = STORE.trajectory(key)
                poses = t["poses"]
                self._json({
                    "n": len(poses),
                    "times": [f"{float(x):.3f}" for x in t.get("times", [])],
                    "grip": [round(float(x), 3) if np.isfinite(x) else None for x in t["grip"]],
                    "poses": [[round(float(v), 6) if np.isfinite(v) else None
                               for v in p[:3].reshape(-1)] for p in poses],
                    "xyz": [[round(float(v), 4) if np.isfinite(v) else None
                             for v in p[:3, 3]] for p in poses],
                })
            elif u.path.startswith("/api/open/"):
                key = unquote(u.path[len("/api/open/"):])
                start_prerender(key)
                self._json({"ok": True})
            elif u.path.startswith("/api/progress/"):
                key = unquote(u.path[len("/api/progress/"):])
                with _prerender_lock:
                    st = _prerender.get(key, {"done": 0, "total": 0})
                self._json({"done": st["done"], "total": st["total"]})
            elif u.path == "/path3d.jpg":
                key = unquote(q.get("key", [""])[0])
                fp = _cache_dir(key) / "path3d.jpg"
                if fp.exists():
                    b = fp.read_bytes()
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "max-age=3600")
                    self.send_header("Content-Length", str(len(b))); self.end_headers()
                    self.wfile.write(b)
                else:
                    self.send_error(404)
            elif u.path.startswith("/frame/"):
                key = unquote(u.path[len("/frame/"):])
                i = int(q.get("i", ["0"])[0])
                ov = q.get("ov", ["1"])[0] == "1"
                fp = _cache_dir(key) / f"f{i:05d}_o{int(ov)}.jpg"
                b = fp.read_bytes() if fp.exists() else None
                if b is None:
                    b = render_frame(key, i, ov)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            else:
                self.send_error(404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path.startswith("/api/qc/"):
            key = unquote(u.path[len("/api/qc/"):])
            d = STORE.eps.get(key)
            if d is None:
                return self._json({"ok": False, "error": "no such episode"}, 404)
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            fp = d["dir"] / qc.QC_JSON_NAME
            doc = json.loads(fp.read_text())
            # 只写人工字段 —— 程序字段(flags/evidence)永不触碰
            for k in ("verdict", "start_frame", "review_note"):
                if k in body:
                    doc[k] = body[k]
            doc["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            tmp = fp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
            os.replace(tmp, fp)
            d["qc"] = doc
            return self._json({"ok": True})
        self.send_error(404)


def main():
    global STORE
    ap = argparse.ArgumentParser(description="QC 人工复核图形界面")
    ap.add_argument("--root", default="/home/ss/data/1000_onesb_labpicking")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    STORE = Store(Path(args.root))
    print(f"QC 复核台: http://localhost:{args.port}  (root={args.root}, {len(STORE.eps)} 集)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
