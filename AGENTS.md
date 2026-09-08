# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

This repo is a collection of standalone Python scripts (plus shell batches and configs) that form the **SROI robot data pipeline**: record RealSense stereo IR + color → run ORB-SLAM3 → transform camera trajectory → extract gripper distance from AprilTags → convert everything to a LeRobot v3.0 training dataset. There is no package, build system, or test suite — each script is invoked directly from the repo root.

The pipeline has two entry points:

- **Pipeline A — Direct recording** (`record_realsense.py`): live capture from a D405/D435i camera. This is the primary path for new data.
- **Pipeline B — Legacy ROS bags** (`extract_stereo_rosbags.py`): convert pre-recorded `.bag` files. Steps 2+ are shared.

The canonical end-to-end recipe lives in `README.md` (Pipeline A: Step 1 record → decode → ORB-SLAM3 → transform → gripper → LeRobot). Read it first before running anything.

## Key commands

```bash
# Record (PNG / JPEG / MP4; MP4 needs `pip install av`)
python record_realsense.py -o ~/Desktop/435 --camera realsense_d405
python record_realsense.py -o ~/Desktop/435 --camera realsense_d405 --encode-video --mem --headless

# Decode MP4 sessions back to PNG (always required before ORB-SLAM)
python decode_videos.py ~/Desktop/435/20260520_143052-mp4 --recursive
# Decode to custom root instead of sibling -png folder
python decode_videos.py <input> --recursive -o /path/to/output

# Run ORB-SLAM3 on every episode
# Host install (e.g. ~/code/ORB_SLAM3) — pass the install dir explicitly:
batches/orbslam_batch_local.sh /path/to/session-png ~/code/ORB_SLAM3 [skip_existing=true] [visualization=false] [--mask-config PATH]
# Inside the ORB_SLAM3 Docker container (hardcoded /ORB_SLAM3 paths):
batches/orbslam_batch_d405.sh /path/to/session-png [skip_existing=true] [visualization=false] [--mask-config PATH]

# Render trajectory projection using the matching camera/gripper extrinsics
python visualization/visualize_traj_video.py /path/to/session-png --recursive \
    --extrinsics-config configs/camera_gripper_extrinsics_sroi_v2_d405.json

# Transform ORB-SLAM camera trajectory into X-forward initial-camera frame
python transform_trajectory.py --recursive /path/to/session-png

# Gripper distance from AprilTag (D405 config by default; D435 needs --configs)
python gripper_estimation_april_tag.py --recursive /path/to/session-png
python gripper_estimation_april_tag.py /path/to/episode --configs configs/sroi_v1_d435.json

# QC: per-episode metrics + multi-page PDF (summary + one page per session)
python visualization/qc.py /path/to/session-tree -o /tmp/qc
# Same, CSV-only or PDF-only:
python visualization/qc.py /path/to/session-tree --csv-only -o /tmp/qc
# Tune what counts as static / flat-gripper by importing scan_sessions, etc.

# Convert to LeRobot v3.0 dataset
python lerobot/sroi_to_lerobot.py \
    --data_path /path/to/session-png \
    --repo_id sroi/lab_picking --fps 30 \
    --root /mnt/data0/data/lerobot \
    --task "pick the red strawberry"
```

### Legacy (Pipeline B)

```bash
python rosbag_segment.py input.bag -o /path/to/output
python extract_stereo_rosbags.py /path/to/bag.bag /output/folder/ realsense_d435i --compressed
python create_orb_slam_yaml.py /output/folder/ realsense_d435i
# then continue from ORB-SLAM3 above
```

## Architecture notes that span multiple files

### Session / episode layout (shared convention)
All steps agree on this on-disk shape and use the suffixes to know which step a folder is in:

```
{timestamp}-{format}/        # session, e.g. 20260520_143052-png  (or -mp4 / -jpeg)
  episode_001/
    left_000000.png  right_000000.png  color_000000.png   # frame streams
    times.txt                                                 # hw timestamps, one float per line
    timestamps.json                                           # frame_count + fps + from/to
    camera_info_{left,right,color}.json                       # ROS-style intrinsics
    orb_slam_realsense_d405.yaml                              # written during recording
    CameraTrajectory.txt                                      # raw ORB-SLAM output
    CameraTrajectoryTransformed.txt                           # after transform_trajectory.py
    gripper_distances.txt                                     # after gripper_estimation_april_tag.py
```

`decode_videos.py` rewrites `*-mp4` sessions as sibling `*-png` folders; the downstream tools expect the PNG form.

### Recording produces everything downstream needs
`record_realsense.py` is more than a recorder: it also writes the ORB-SLAM YAML for the recorded camera (D405 uses fixed calibration, D435i uses live calibration), so the ORB-SLAM step is fully driven by recording output. Switching cameras (D405 vs D435i) requires the matching `configs/sroi_v*.json` for gripper estimation and a different `batches/orbslam_batch_*.sh` — don't mix them.

### Rig-specific mask and projection configs
Gripper masking is opt-in in the standard ORB batch scripts. Passing `--mask-config` creates temporary masked stereo frames, runs ORB-SLAM3 on them, copies `CameraTrajectory.txt` back to the original episode, and removes the temporary input; source images are not changed. Omitting the option preserves the original unmasked path. Projection visualization separately loads camera-to-gripper-tip transforms through `--extrinsics-config`. The supplied SROI v2 D405 files are `configs/gripper_mask_sroi_v2_d405.json` and `configs/camera_gripper_extrinsics_sroi_v2_d405.json`; use matching files for another physical rig. Schema and validation details are in `doc/rig_configs.md`.

### Encoding modes trade off disk vs memory
`record_realsense.py` defaults to PNG. `--encode-video` writes MP4 (smallest, requires `av`). Three encoding paths:
- **PNG/JPEG to disk** (default): write frames as you go.
- **MP4 from disk** (`--encode-video`): write PNGs, then encode at end of episode (blocks).
- **MP4 from RAM** (`--encode-video --mem`): keep frames in memory, skip the disk round-trip (faster, watch memory on long episodes).
- **MP4 in realtime** (`--encode-video --realtime-encoding`): background encoder threads consume frames during recording. Useful on the Pi where end-of-episode blocking disrupts the control loop.

`--status-file <path>` writes a JSON status blob (state/episode/frame/fps/format) atomically every ~1 s for an external UI to poll. See `doc/video_encoding_approaches.md` for the design comparison against lerobot upstream encoders.

### `record_realsense.py` is shared with the UPI repo
The canonical recorder lives at `/home/zfei/code/UPI/raspberry_pi/upi_python/record_realsense.py` (deployed on the Pi). The copy in this repo is synced from there — when changing recorder behavior, update UPI first and copy across, or the two will drift.

### What feeds LeRobot
`lerobot/sroi_to_lerobot.py` reads three things per episode — `CameraTrajectoryTransformed.txt` (4x4 poses → `ee.xyz` + `ee.wx,wy,wz` rotvec), `gripper_distances.txt`, and `color_*.png`/`color_*.jpg` — and emits a LeRobot v3.0 dataset with a fixed feature schema (`observation.images.camera` video + `action` 7-vector; `observation.state` is intentionally omitted/derived downstream). It accepts a single episode dir, one session dir, or a parent folder containing many decoded `*-png` sessions. Use `--episodes` only for single-session subsetting; use `--qc_csv` + `--qc_categories` to filter episodes across sessions and build one unified dataset. Schema and examples are documented in `lerobot/README.md`.

**Opt-in target conditioning** (`--target-conditioning`): adds a constant per-episode `observation.images.target_ref` video — the initial frame with a circle around the hindsight-labeled target strawberry (final sustained gripper closure → gripper tip → projected into the initial image; `target_ref.py` at repo root, no detection/segmentation). Invalid episodes are skipped into `meta/target_qc.csv`. QC the labels with `visualization/visualize_target_ref.py`; render deployment markers with `lerobot/render_target_ref.py`; e2e test: `lerobot/test_target_conditioning_e2e.py`. ACT/π0.5 pick the new image feature up automatically (every `observation.images.*` key is a visual input).

### Two LeRobot converters exist
- `lerobot/sroi_to_lerobot.py` — main converter for SROI camera-trajectory data (Pipeline A/B output).
- `piper_sroi_convert/convert_to_lerobot.py` — for Piper arm data where the trajectory is already in end-effector space (`EETrajectory.txt`, 12 floats/line). It uses the upstream `lerobot` package layout and a different input shape; do not substitute one for the other.

### `misc/` — one-off repair scripts
`misc/fix_d405_baseline.py` and `misc/regenerate_d405_yaml.py` repair legacy D405 recordings where the stereo baseline was written 1000× too small (pyrealsense2 returns mm; an older recorder treated it as m, so ORB-SLAM3 saw a ~50-micron baseline and the trajectory collapsed to identity). The recorder is now patched with a sanity check, so newly-recorded data is fine — these scripts exist for the pre-fix corpus. Both are idempotent. Run `fix_d405_baseline.py` first; if its first-pass `yaml.dump()` mangled the OpenCV YAML format (stripped `!!opencv-matrix`), run `regenerate_d405_yaml.py` to rebuild the yaml from `camera_info_{left,right}.json`.

### QC tool reads ORB-SLAM + gripper outputs
`visualization/qc.py` scans a session tree and emits a per-episode CSV plus a multi-page PDF (summary table + one page per session). It classifies each episode as `ok`, `static`, `flat_gripper`, `no_gripper`, `no_traj`, or `all_nan_gripper`, color-codes by category, sorts problem episodes to the top, and uses a shared xy scale for trajectory plots. QC thresholds, colors, category priority/problem flags, and layout settings live in `configs/qc_filters.json` (override with `--filters`). The PDF/CSV are idempotent — safe to re-run as data updates. The resulting `qc.csv` can be passed to `lerobot/sroi_to_lerobot.py --qc_csv ... --qc_categories ok` to filter training data.

## Dependencies

There is no `setup.py`/`pyproject.toml`; install per-script as needed. `requirements.txt` covers only `extract_stereo_rosbags.py`. The complete set is in `README.md`:

```bash
pip install pyrealsense2 opencv-python numpy tqdm pupil-apriltags scipy PyYAML
pip install av            # for --encode-video and decode_videos.py
pip install rosbags       # for extract_stereo_rosbags.py (Pipeline B)
pip install lerobot       # for sroi_to_lerobot.py
pip install plotly matplotlib   # visualization tools
```

The LeRobot converter auto-adds its local `lerobot/` package to `sys.path` when run, so the repo's copy wins over any installed version.

## Repo quirks

- `batches/orbslam_batch_d405.sh` and `orbslam_batch.sh` assume the ORB_SLAM3 Docker container (`/ORB_SLAM3` paths); `orbslam_batch_local.sh` takes the install dir as `$2` and runs on the host.
- `extract_rgbd/` and `notebooks/` contain older/experimental variants and Jupyter scratch — prefer the top-level scripts for new work.
- The host ORB_SLAM3 install lives at `/home/zfei/code/ORB_SLAM3`; `stereo_kitti` is invoked directly when using the local batch.
