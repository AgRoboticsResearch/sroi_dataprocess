# AGENTS.md

Guidance for AI agents working in this repository.

## Overview

Standalone Python scripts + shell batches + JSON configs forming the **SROI strawberry-picking data pipeline**:
record RealSense stereo IR + color → decode → ORB-SLAM3 trajectory recovery (frame ledger + repair + smoothing)
→ gripper action recovery from AprilTags → QC → convert to a LeRobot v3.0 dataset. No package, no build system,
no test suite — each entry point is invoked directly from the repo root.

```
                  ┌─ (2) trajectory recovery ─ SLAM(masked, emits frame ledger) → repair(align/interp/smooth) → transform ─→ CameraTrajectoryTransformed.txt
(1) decode ─→ PNG streams ┤
                  └─ (3) gripper recovery ─ AprilTag detection → completion → interpolation → normalization ─────→ gripper_distances.txt
                            ↓ merge
                          (4) QC — the single verdict point ─→ qc.csv (5 categories + evidence columns)
                            ↓
                          (5) convert to LeRobot (executes the verdict + trims the quasi-static head)
```

All raw data is processed to the end; filtering happens only at QC (4) and conversion (5).

## Entry points (do NOT call the kernel `*.py` scripts directly)

**Iron rule: run exactly the commands below. The entry scripts own robustness — hard validation +
non-zero exit codes. A non-zero exit means the step is incomplete: stop and investigate.**
Each entry script picks its own Python interpreter (see Environment), independent of the active shell.

```bash
# 1. Decode mp4 → PNG (needs PyAV; the batch auto-selects the conda `lerobot` env)
bash batches/decode_batch.sh <day-dir | <session>-mp4> [png_root] [--jobs 4]

# 2. ORB-SLAM3 + trapezoid gripper mask → per episode CameraTrajectory.txt (K×12)
#    + CameraTrajectory_idx.txt (K frame ids)
bash batches/orbslam_batch_local.sh <session>-png ~/code/ORB_SLAM3 true false \
    --mask-config configs/gripper_mask_sroi_v2_d405.json

# 3. Trajectory repair: ledger alignment → short-hole interpolation (linear pos + slerp rot;
#    head freeze / tail extrapolation) → long holes stay NaN (invalid) → segment-wise
#    Whittaker smoothing (λ=1, pos + rot6d + Gram-Schmidt) → atomic rewrite to N×12 + 3-col ledger.
#    Params: configs/traj_repair.json
bash batches/traj_repair_batch.sh <session-png | data root> [--force] [--no-smooth]

# 4. Coordinate transform (optical → camera_link X-fwd/Y-left/Z-up; timestamps via the ledger's
#    frame ids; hard-fails if repair has not run)
bash batches/transform_batch.sh <session-png | data root>

# 5. Gripper recovery (session-level entry: the normalization range is pooled per session)
python3 gripper_estimation_april_tag.py <session-png> --recursive
#    Zero-detection episodes produce an all-NaN series of the right length (verdict left to QC).

# 6. QC → qc.csv (+ optional PDF). All rules/thresholds live in configs/qc_filters.json.
python3 visualization/qc.py <data root> --csv-only -o <output prefix>

# 7. Convert to LeRobot (executes the QC verdict; unequal stream counts = non-zero exit, never silent truncation)
bash batches/convert_batch.sh <data root> <repo_id> [out root] [--qc-csv <qc.csv> | --no-qc]
```

**Re-run safety**: decode (resumes partial), SLAM (skips when output exists), repair (idempotent;
`--force` reconstructs the raw state from the ledger). **Discipline**: re-running SLAM with
`skip_existing=false` regresses the files to the raw K-row generation → repair and transform MUST be re-run.

## Trajectory output contract (how to tell which generation a file is in)

| File | Fresh from SLAM | After repair |
|---|---|---|
| `CameraTrajectory.txt` | K rows × 12 cols (gapped) | **N rows × 12 cols** (N == frame count; long-gap rows = NaN) |
| `CameraTrajectory_idx.txt` | K rows × 1 col (frame ids) | N rows × 3 cols (`frame state seg_len`; state ∈ raw/interp/invalid) |
| `CameraTrajectoryTransformed.txt` | — | N rows × 13 cols (timestamp + 12) — **the only trajectory file downstream should read** |

`rows == frames` and a 3-column ledger = repaired. The raw K-row state is fully reconstructible
(drop non-raw rows), so no backup file is kept.

## QC categories (first match wins; every check's numbers go to the CSV as evidence)

| category | rule |
|---|---|
| `not_processed` | trajectory/gripper stream missing, too short, or ledger absent (plumbing guard) |
| `traj_long_gap` | ledger contains `invalid` rows (tracking loss beyond gap_max_frames) |
| `gripper_missing` | gripper_distances.txt contains ANY NaN (including all-NaN) |
| `traj_frame_jump` | per-frame step of the SMOOTHED trajectory above projection.v_max_mm |
| `first_frame_projection` | CAPSULE model: camera pitched 32° down (30° mount + ~2° habit, no roll) → world-vertical g=(0,cos,sin) in the optical frame. Pick point sits 5-20mm above the calyx → berry center 25-40mm below the endpoint: a 40mm region sphere swept along g forms the detection CAPSULE, projected as a union of sampled discs. Threshold decoupled from the region: a 30mm all-red berry projected at the "30mm below endpoint" depth × red_area_fraction (0.3, occlusion allowance). HSV H≤10/≥160, s≥60, v≥60 (grid sweep showed tightening is strictly worse). Rendering: merged thin contour; near-plane (z≥3cm) + FOV (≤45°) clipping |
| `gripper_not_closed` | mean of the last N finite gripper readings above threshold (still open) |
| `ok` | all checks passed |

Thresholds and the HSV parameters live in `configs/qc_filters.json`; tune them against overlays from
`analysis_qc/auto_traj_qc.py tune <session>/<episode>` (that script's only remaining purpose).

## Conventions you must know

- **camera_link IS the tool link, everywhere.** Training `observation.ee`, deployment, and
  `visualization/ik_feasibility.py` all use the **camera pose**; the extrinsics JSON
  (`T_camera_gripper_tip`) is used **only** for visualization projection. Never multiply it into the pipeline.
- **`observation.ee` is currently unused by training**: the lerobot fei branch (0.5.2) derives state
  from `action[t-1:t+1]` (`UmiDeriveStateFromActionStep`). The column is kept as the record of the
  current-frame absolute pose.
- **Frame identity == `Frame::mnId`** (ORB-SLAM3's static counter; equals the 0-based image index for
  offline stereo). Neither a loop counter (the pose list is compacted) nor timestamps (the degenerate
  branch duplicates the previous timestamp) can serve as frame identity.
- Start-trim of the quasi-static recording head is on by default in the converter (`--no-trim-start` to
  disable; counts in `meta/start_trim.json`). The ~1.5 s end pause is intentional training signal — keep it.

## Why the gripper mask exists (most important historical finding)

The gripper + AprilTags at the bottom of the D405 frame are high-contrast moving features that break
ORB-SLAM tracking (~2/3 of episodes collapse without the mask). The trapezoid mask
(`configs/gripper_mask_sroi_v2_d405.json`) covers only the gripper region; the batch builds a temporary
masked copy, so the recorded images are never touched and full RGB stays available for training.
Experimental evidence: `experiments/mask_schemes/`.

## Old-scheme comparison (for ablations)

- Trajectory unmasked: step 2 without `--mask-config`.
- Legacy gripper behavior: step 5 with `--interp forward_fill --no-symmetry --no-preprocess --spike-threshold 0`.

## Directory map

```
batches/       robust outer layer (.sh) + decode/ORBSLAM mask helpers
configs/       rig configs (mask/extrinsics/sroi_v2_d405) + rule params (traj_repair/qc_filters/ik_filters)
analysis_qc/   offline analysis tools (auto_traj_qc tune overlays, smoothing λ scan, comparisons)
visualization/ qc.py, trajectory-projection videos, IK feasibility
experiments/   method-comparison writeups + experiment code (video encoding; mask schemes in mask_schemes/)
design/        design conventions (rig_configs.md: rig JSON schemas + calibration workflow)
patches/       external-dependency patches (ORB-SLAM3: baseline commit + apply/build recipe)
april_tags/    printable AprilTag artwork for the jaws (same placement on every rig; zero code references — DO NOT delete)
orb_slam_yaml/ ORB-SLAM templates (the recorder reads these to emit per-episode yaml)
lerobot/       LeRobot conversion (sroi_to_lerobot.py + a local copy of the lerobot package)
```

## Environment (entry scripts select automatically; matters only when calling kernels by hand)

| env | has | used for |
|---|---|---|
| system python3 | cv2/numpy/scipy/matplotlib/pupil_apriltags | masking, repair, transform, gripper, QC, visualization |
| conda `lerobot` | torch/av/lerobot/cv2 (**no matplotlib**) | decode (1), LeRobot conversion (5) |

ffmpeg has libx264 (no libopenh264) → always `--codec libx264`. A third env, conda `lerobot_fork`,
is for running lerobot fei-branch policies — unrelated to data processing.

## External dependency

ORB-SLAM3 lives at `~/code/ORB_SLAM3`: an official clone that **must carry
`patches/orbslam3-sroi.patch`** (without it: our image naming is not found, Sophus fails to compile,
and there is no frame ledger). Full recipe: `patches/README.md`.

## Dependencies

`requirements.txt` covers the pipeline's own needs. Complete set:

```bash
pip install pyrealsense2 opencv-python numpy tqdm pupil-apriltags scipy PyYAML
pip install av            # decode
pip install lerobot       # conversion
pip install plotly matplotlib   # visualization tools
```
