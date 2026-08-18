# SROI ROS Bag Utilities

> 🌐 Builds the SROI datasets — see the **[Datasets page](https://agroboticsresearch.github.io/sroi_datasets/)** and **[SROI V2](https://agroboticsresearch.github.io/sroi_v2/)**.

Tools for recording, processing, and converting SROI robot data for ORB-SLAM and LeRobot training pipelines.

---

## Pipeline A: Record → LeRobot (Direct Recording)

Record directly from a RealSense camera and convert to a LeRobot training dataset.

### Step 1: Record

```bash
# PNG mode (default, lossless)
python record_realsense.py -o ~/Desktop/435 --camera realsense_d405

# JPEG mode (smaller files)
python record_realsense.py -o ~/Desktop/435 --camera realsense_d405 --image-format jpeg

# MP4 mode (smallest, auto-encodes after each episode)
python record_realsense.py -o ~/Desktop/435 --camera realsense_d405 --encode-video

# Headless SSH session (no display needed)
python record_realsense.py -o ~/Desktop/435 --headless

# Combine: headless + MP4
python record_realsense.py -o ~/Desktop/435 --headless --encode-video --camera realsense_d405
```

Controls (GUI window or terminal in headless): `r` record, `s` stop, `q` quit

Session folders are named `{timestamp}-{format}` (e.g. `20260520_143052-mp4`).

### Step 2: Decode MP4 (only if recorded with --encode-video)

For MP4 recordings, PNG frames are regenerated here. To keep the raw MP4 store
pristine, decode into a **separate PNG root** (one wipeable tree) instead of
`-png` siblings next to the `-mp4` folders:

```bash
# Recommended: decode a whole recording day into <day>/_png/<ts>-png/...
# (raw *-mp4 folders are never modified; PNGs are regenerable from them)
batches/decode_batch.sh ~/Desktop/435/20260520_143052
# Custom root + parallelism:
batches/decode_batch.sh ~/Desktop/435/20260520_143052 /tmp/pngout --jobs 8
# Single session also works:
batches/decode_batch.sh ~/Desktop/435/20260520_143052/20260520_143052-mp4
```

Then run ORB-SLAM3 / transform / gripper / QC (Steps 3–6) against the PNG root
(e.g. `~/Desktop/435/20260520_143052/_png`).

Direct `decode_videos.py` still works for one-off or custom roots:

```bash
# Legacy/convenience behavior: creates a sibling -png folder
python decode_videos.py ~/Desktop/435/20260520_143052-mp4 --recursive
# Custom root
python decode_videos.py ~/Desktop/435/20260520_143052-mp4 --recursive -o /path/to/png_root
```

> **SROI_V2 dataset layout (what the lab datasets use):** decoded PNGs live in a
> **standalone root that is a sibling of the MP4 store at dataset level** — e.g.
> `1000-onesb-lab-picking/` (raw `*-mp4`) next to `1000-onesb-lab-picking-pngs/`
> (all decoded `*-png` sessions), and likewise `20260803-occluded-cases/` →
> `20260803-occluded-cases-pngs/`. The MP4 store stays pristine and the PNG tree
> is one wipeable/regenerable folder. Achieve it with:
> `batches/decode_batch.sh <dataset>/<name> <dataset>/<name>-pngs --jobs N`, or
> per session with `python decode_videos.py <session>-mp4 --recursive -o <name>-pngs`.
> Run ORB-SLAM / transform / gripper / QC against the `-pngs` root, never against
> `-png` siblings inside the MP4 store.

### Step 3: Run ORB-SLAM3

Note: The ORB-SLAM YAML is generated automatically during recording for both D405 (fixed calibration) and D435i (from live calibration).

**Masked SLAM is the standard pipeline for SROI v2.** The gripper/fingers are masked out
of the stereo frames before tracking so the camera trajectory does not include the moving
gripper — this is what the LeRobot datasets are built from. Always pass `--mask-config`
unless you have a specific reason not to.

```bash
# Single episode (unmasked, manual)
cd ~/code/ORB_SLAM3
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt \
    ~/Desktop/435/20260520_143052-png/episode_001/orb_slam_realsense_d405.yaml \
    ~/Desktop/435/20260520_143052-png/episode_001 false

# STANDARD: batch all episodes WITH the gripper mask (host install)
batches/orbslam_batch_local.sh ~/Desktop/435/20260520_143052-png ~/code/ORB_SLAM3 \
    true false --mask-config configs/gripper_mask_sroi_v2_d405.json

# Same masked run, but parallelize N episodes at once (default 4; speeds up large sessions)
batches/orbslam_batch_local.sh ~/Desktop/435/20260520_143052-png ~/code/ORB_SLAM3 \
    true false --mask-config configs/gripper_mask_sroi_v2_d405.json --jobs 8

# Inside the ORB_SLAM3 Docker container (masked)
./batches/orbslam_batch_d405.sh ~/Desktop/435/20260520_143052-png true false \
    --mask-config /codes/sroi_rosbag_utilities/configs/gripper_mask_sroi_v2_d405.json

# Unmasked run (NOT what the datasets use — legacy/ablation only)
batches/orbslam_batch_local.sh ~/Desktop/435/20260520_143052-png ~/code/ORB_SLAM3
```

When `--mask-config` is provided, the batch creates a temporary masked copy of each
episode's stereo frames, runs ORB-SLAM3 on that copy, saves `CameraTrajectory.txt` back
to the original episode, and removes the temporary images. Each episode runs in its own
subshell with its own masked temp dir, so `--jobs N` concurrent runs are fully isolated
and the original recordings are never modified. Omitting `--mask-config` preserves the
previous unmasked behavior.

Mask JSON files define the expected image size, per-eye polygons, and an optional
full-width `black_below_row`. The mask affects only the stereo trajectory; the
color-based AprilTag gripper signal (Step 5) is never masked. Use a different config for
a different gripper/camera assembly; `configs/gripper_mask_sroi_v2_d405.json` contains
the current 640x480 D405 geometry. See [`doc/rig_configs.md`](doc/rig_configs.md) for the
schema, validation rules, temporary-file behavior, and instructions for adding another
rig.

### Step 4: Transform Trajectory

```bash
# Single episode
python transform_trajectory.py ~/Desktop/435/20260520_143052-png/episode_001

# All episodes recursively
python transform_trajectory.py --recursive ~/Desktop/435/20260520_143052-png
```

### Step 5: Extract Gripper

```bash
# Default: D405 config
python gripper_estimation_april_tag.py ~/Desktop/435/20260520_143052-png/episode_001

# D435 config
python gripper_estimation_april_tag.py /output/folder/ --configs configs/sroi_v1_d435.json

# All episodes in a session share one min/max, robustly detected from the pooled data
python gripper_estimation_april_tag.py --recursive ~/Desktop/435/20260520_143052-png
```

Gripper distance is normalized to `[0, 1]`. With `--recursive`, every episode in that run
is normalized against one shared min/max (robust clustered-extremum over pooled raw
distances) instead of each episode picking its own range. To set the range manually
instead of auto-detecting it, add `"gripper_min_distance"` / `"gripper_max_distance"`
(both required) to the camera config JSON passed via `--configs`.

> **Shared normalization range (what the lab datasets use):** all SROI_V2 datasets
> normalize against the gripper range recovered from the 1000-onesb dataset so
> action vectors stay comparable across LeRobot datasets. Pass
> `--configs configs/sroi_v2_d405_1000_onesb_range.json` (holds the
> `gripper_min_distance`/`gripper_max_distance` from
> `sroiv2_strawberry_picking_lab/1000-onesb-lab-picking-pngs`). Check the
> `gripper_distance_distribution.png` histogram after each run to confirm the
> fixed range still covers the new data.

Each run also writes `gripper_distance_distribution.png` next to the input path — a
histogram of the pooled raw distances with the selected min/max marked, so you can
sanity-check the range before trusting the normalized `gripper_distances.txt` files.

Every successfully processed episode also stores `gripper_normalization.json` beside
`gripper_distances.txt`, and the run root receives the same manifest. It records the
exact min/max pixels, `0 = closed` / `1 = open` convention, formula, manual-vs-auto
range source (including a reference dataset when configured), detector settings,
median filter, and run scope. Keep this manifest with the processed data.

Add `--median_filter 3` (must be odd) to smooth single-frame AprilTag detection noise
in the raw distances before the min/max is computed and episodes are normalized.
Default is `1` (no filtering).

### Optional: Trajectory projection video

Each video shows the RGB frame with the projected gripper-tip path (left), the
animated 3D trajectory (middle), and the full gripper-openness curve with a dot
tracking the current frame (right). The camera-to-gripper-tip transforms used by
the projection overlay are loaded from a standalone JSON config:

```bash
python visualization/visualize_traj_video.py /path/to/session-png --recursive \
    --extrinsics-config configs/camera_gripper_extrinsics_sroi_v2_d405.json
```

Use a different extrinsics config for a different camera/gripper assembly.
The matrix convention and calibration checks are documented in
[`doc/rig_configs.md`](doc/rig_configs.md).

For reviewing a QC'd tree, pass `--qc-csv` (requires `--output`) to render every
episode into one folder with its QC category as a colored on-video badge and a
filename prefix (`ok_...`, `static_...`, ...), colors matching the QC PDF:

```bash
python visualization/visualize_traj_video.py /path/to/pngs --recursive \
    --output /path/to/traj_videos --qc-csv /path/to/pngs/qc.csv
```

### Step 6: QC and Convert to LeRobot Dataset

For a single decoded session folder:

```bash
python lerobot/sroi_to_lerobot.py \
    --data_path ~/Desktop/435/20260520_143052-png \
    --repo_id sroi/lab_picking \
    --fps 30 \
    --root /mnt/data0/data/sroi/sroi_lab_picking \
    --task "pick the red strawberry"
```

For a full recording day containing many `*-png` session folders, first run QC,
then use the QC CSV to keep only selected categories (usually `ok`) while
converting all sessions into one LeRobot dataset:

```bash
# Writes /path/to/day/qc.csv and /path/to/day/qc.pdf
python visualization/qc.py /path/to/day -o /path/to/day/qc
```

Gripper values use `0 = closed` and `1 = open`. QC assigns
`gripper_not_closed` when the mean of the final 10 finite gripper readings is
above `0.2`; tune `gripper_end_tail` and `gripper_end_open_threshold` in
`configs/qc_filters.json` (or a file passed with `--filters`).

The per-episode QC page (default "rich" view) shows three panels per episode,
mirroring the side-by-side trajectory videos: the RGB frame with the projected
gripper-tip path (first video frame, left), the 3D trajectory (last video frame,
right), and the gripper signal. Pass `--simple` for the fast 2D top-down trajectory
+ gripper layout; pass `--extrinsics-config` to use a different camera/gripper rig
for the projection overlay.

```bash
python lerobot/sroi_to_lerobot.py \
    --data_path /path/to/day \
    --repo_id sroi/lab_picking \
    --fps 30 \
    --root /mnt/data0/data/sroi/sroi_lab_picking \
    --task "pick the red strawberry" \
    --qc_csv /path/to/day/qc.csv \
    --qc_categories ok \
    --num_workers 4
```

`--root` must point to a dataset directory that does not already exist;
`LeRobotDataset.create()` creates it fresh.

### Step 6b: IK Feasibility Check (experimental, optional)

`visualization/ik_feasibility.py` is an **experimental** layer that runs the same
Piper inverse kinematics the real deploy uses (`examples/umi_relative_ee/deploy_umi_relative_ee_piper.py`,
placo-backed) over each recorded trajectory and reports whether the arm could track
it. It writes `ik_feasibility.csv` + `ik_feasibility.pdf` (one summary page + one
page per episode: joint traces vs URDF limits, desired-vs-reached EE xyz/rpy, and
pos/rot error vs tolerance). It runs on the same `*-png` tree as QC.

It needs the `py310` conda env (placo) and the `sroi-piper` submodule URDF
(initialize it once from the lerobot checkout: `git submodule update --init sroi-piper`):

```bash
# From the repo root, py310 interpreter:
/home/zfei/anaconda3/envs/py310/bin/python visualization/ik_feasibility.py \
    /path/to/day -o /path/to/day/ik_feasibility
```

Seed, tolerances, URDF path, joint limits, and category thresholds live in
`configs/ik_filters.json`; override with `--filters`, `--seed`, `--urdf`.

**By default this does NOT filter the dataset** — it only produces a report. The CSV
keeps the `session, episode, category` columns (categories: `ik_ok`, `ik_unreachable`,
`ik_joint_limit`, `ik_branch_jump`, `ik_no_traj`), so you *can* opt into IK-based
filtering by passing it to the converter, but this is deliberate and off by default:

```bash
# OPT-IN: also gate the LeRobot dataset on the IK verdict (not the default)
python lerobot/sroi_to_lerobot.py \
    --data_path /path/to/day \
    --repo_id sroi/lab_picking \
    --fps 30 \
    --root /mnt/data0/data/sroi/sroi_lab_picking \
    --task "pick the red strawberry" \
    --qc_csv /path/to/day/ik_feasibility.csv \
    --qc_categories ik_ok
```

Caveat: the IK verdict is seed-dependent (the arm is assumed to start from the
`start` seed, the deploy's ready pose) and the anchor is a necessary-not-sufficient
heuristic — a pass means "the deploy IK would accept this trajectory from that seed
within tolerance", not a guarantee of dynamic executability. See the module docstring.

### Optional: Merge LeRobot Datasets

Use the official LeRobot edit tool when combining multiple completed local
datasets into one dataset. The official checkout currently needs the `py312`
conda environment and its source tree on `PYTHONPATH`:

```bash
cd /home/zfei/code/lerobots/lerobot_official

PYTHONPATH=/home/zfei/code/lerobots/lerobot_official/src \
/home/zfei/anaconda3/envs/py312/bin/python -m lerobot.scripts.lerobot_edit_dataset \
    --new_repo_id sroi/sroi_v2_merged_official \
    --new_root /mnt/data1/data/lerobot/sroi_v2_merged_official \
    --operation.type merge \
    --operation.repo_ids "['sroi/lerobot_sroi_v2', 'sroi/sroi_v2_20260611']" \
    --operation.roots "['/mnt/data1/data/lerobot/lerobot_sroi_v2', '/mnt/data1/data/lerobot/sroi_v2_20260611']"
```

`--operation.repo_ids` labels the input datasets; `--operation.roots` provides
the exact local dataset folders. `--new_root` must not already exist. Keep a
small provenance note (for example `MERGE_INFO.md` and `merge_info.json`) in the
merged dataset folder listing the input datasets, frame counts, and merge
command.

---

## Pipeline B: ROS Bag → LeRobot (Legacy)

Process pre-recorded ROS bag files.

### Step 1: Segment ROS Bags

```bash
python rosbag_segment.py input.bag -o /path/to/output
```

### Step 2: Extract Images from Bags

```bash
python3 extract_stereo_rosbags.py /path/to/bag.bag /output/folder/ realsense_d435i --compressed
```

### Step 3: Create ORB-SLAM Config (D435i from bags only)

```bash
python3 create_orb_slam_yaml.py /output/folder/ realsense_d435i
```

### Step 4: Extract Gripper

```bash
python gripper_estimation_april_tag.py /output/folder/
```

Then continue with ORB-SLAM3 (Step 3 in Pipeline A) and follow remaining steps.

---

## Pipeline B′: SROI v1 ROS1 bags → v2 data (conversion)

For the recorded v1 corpus (`oak_d_sr` + `realsense_d435i` rigs, e.g.
`/mnt/data1/sroi/sroi_v1`): `v1_to_v2/convert_v1_bags.py` converts the bags
into v2-style sessions — camera_info always from the bag (explicit fallback +
`CALIBRATION_NOTE.txt` for the dec-2024 oak bags that lack it, resolution
checked against the pixels), per-episode EuRoC `imu.txt`, and the tf_static
extrinsics. The full-corpus run uses `--mp4-only` (PNG trees would need
~265 GiB); PNG trees are regenerated per batch with `decode_videos.py`.
The complete runbook — dry-run, full command, resume, verification,
regenerating PNG trees — lives in `v1_to_v2/README.md`.

---

## Dependencies

```bash
pip install pyrealsense2 opencv-python numpy tqdm pupil-apriltags scipy PyYAML
pip install av  # for --encode-video and decode_videos.py
pip install rosbags  # for extract_stereo_rosbags.py (Pipeline B)
pip install lerobot  # for sroi_to_lerobot.py
pip install plotly matplotlib  # for visualization tools
```
