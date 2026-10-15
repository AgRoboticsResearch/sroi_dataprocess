# patches/ — 外部依赖的补丁

本目录存放**不属于本仓库、但本仓库依赖**的外部程序的本地改动。

目前只有一个：**ORB-SLAM3**。

---

## ORB-SLAM3

### 为什么需要这个补丁

ORB-SLAM3 **不在本仓库里**。它是官方仓库的一份 clone，自己编译出
`Examples/Stereo/stereo_kitti` 给本管线用。我们在这份 clone 上打了补丁，
但**从未提交**——只存在于本机工作区。换机器 / 重装 / 重新 clone，
这些改动会**全部消失**，而且不是可有可无的：

- 不打补丁 → `stereo_kitti` **找不到我们的图片**（图片命名约定不同）→ 直接跑不了
- 不打补丁 → Sophus 的 `-Werror` 让 gcc-13 编译**直接失败**
- 不打补丁 → **没有帧号账本**（`CameraTrajectory_idx.txt`），轨迹行和帧的
  对齐无从谈起，下游修复工具直接拒收

所以这个补丁是本管线的**硬依赖**。

### 基线

| | |
|---|---|
| 仓库 | `https://github.com/UZ-SLAMLab/ORB_SLAM3.git` |
| 基线 commit | `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`（upstream，2022-02-10） |
| 补丁文件 | `orbslam3-sroi.patch`（5 个文件，41+/19- 行改动） |

> ⚠️ 仓库 URL 只能用 `UZ-SLAMLab`（无连字符）。组织名曾是 `UZ-SLAM-Lab`，
> 旧的带连字符 URL 会 **404**，而 git 会误报成"要你输入用户名"。

### 打补丁 + 编译

```bash
# 1. clone 到 ~/code/ORB_SLAM3 并对齐基线 commit
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ~/code/ORB_SLAM3
cd ~/code/ORB_SLAM3 && git checkout 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4

# 2. 打补丁
git apply /path/to/sroi_dataprocess/patches/orbslam3-sroi.patch

# 3. 编译（Pangolin 是前提，见下）
export Pangolin_DIR=$HOME/code/Pangolin/install/lib/cmake/Pangolin
export CMAKE_CXX_FLAGS="-include cstdint -include limits -include memory -include algorithm -include functional"
bash build.sh

# 4. 运行前必须设库路径，否则找不到 .so
export LD_LIBRARY_PATH=$HOME/code/ORB_SLAM3/lib:$HOME/code/Pangolin/install/lib
```

**Pangolin 前提**：本机把 Pangolin v0.6 装在 `~/code/Pangolin/install`
（**不是** `/usr/local`——这个账号没有免密 sudo）。
`CMAKE_CXX_FLAGS` 里那串 `-include` 不是可选项：g++13 不再传递性包含这些标准头，
不加就会报 `uint32_t` / `numeric_limits` / `std::unique_ptr` 未定义。

### 补丁改了什么

`Examples/Stereo/stereo_kitti.cc`

1. `argc != 4` → `!= 5`，多解析一个第 4 参数 `bUseViewer`（batch 会传）
2. `System(...,STEREO,true)` → `System(...,STEREO,bUseViewer)`
   （官方写死开窗口，批处理会强弹 GUI）
3. `LoadImages`：图片前缀 `image_0/` + `image_1/` → `left_` + `right_`
   （**最关键**：我们的 episode 目录里直接放 `left_%06d.png` / `right_%06d.png`，没有子目录）
4. 保存目标 `"CameraTrajectory.txt"`（CWD）→ `string(argv[3]) + "/CameraTrajectory.txt"`
   （写进输入序列目录，batch 才能把轨迹收回 episode 目录）
5. 删掉按 `times.txt` 间隔的实时 `usleep` 限速 → 全速离线跑
   （录制的 `times.txt` 间距不可靠，保留限速会让单条 episode 跑几十分钟）

`Thirdparty/Sophus/CMakeLists.txt`

6. 去掉 `-Werror`：gcc-13 对 Eigen/SSE 的 `-Warray-bounds` 误报会变成致命错误

`include/Tracking.h` + `src/Tracking.cc` + `src/System.cc`（帧号账本，2026-10-16 加）

7. `Tracking` 新增 `mlFrameIds`（与 `mlRelativeFramePoses` 等四个 list 平行，
   `Tracking.cc` 每次 push 的同时记录 `mCurrentFrame.mnId`）。
   **帧号必须用 `mnId`**：那四个 list 是被压缩过的（跟丢帧根本没进 list），
   循环计数器给的是"看着连续、实际错位"的号；时间戳也不行——退化分支
   `mlFrameTimes.push_back(mlFrameTimes.back())` 恰好在跟丢时复制上一帧时间戳。
   离线 stereo 模式下 `mnId` 每张图恰好 +1，等于图像的绝对帧号。
8. `SaveTrajectoryKITTI` 在写 12 列位姿的同时，把 "<文件名>_idx.txt" 写成
   **每行一个帧号**（与位姿行严格一一对应；`if(!pKF) continue;` 跳过的行
   两边同步跳过）。下游 `repair_trajectory.py` 只认账本对齐，不认行号。

### 只重编 `stereo_kitti`

改完 `stereo_kitti.cc` 后不必全量重编：

```bash
cmake --build build --target stereo_kitti -j4
```

---

## 以后加新补丁的规矩

- **一个外部依赖一个 patch 文件**，基线 commit 写在表格里
- 补丁必须能 `git apply` 干净地打在基线 commit 上
- 改完外部依赖后**立刻更新本文件**——这里是它唯一的记录处
