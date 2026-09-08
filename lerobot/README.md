# SROI to LeRobot Dataset Converter

This converter transforms SROI-format robot end-effector trajectory recordings into a LeRobot v3.0 compatible dataset.

The converter reads end-effector poses, gripper distances and per-frame camera images and produces a single episode that follows the LeRobot v3.0 feature contract used by the `so100`/EE pipelines.

## Highlights

- Produces LeRobot v3.0 datasets (features schema compatible with `ForwardKinematicsJointsToEE` / EE pipelines)
- Stores end-effector pose as rotation-vector (`ee.wx, ee.wy, ee.wz`) + translation (`ee.x, ee.y, ee.z`) and gripper position
- Saves `observation.images.camera` as video frames (CHW) and writes Parquet metadata
- Action and observation feature shapes match the recording example in `examples/so100_to_so100_EE/record.py`

## Data Layout Expected

`--data_path` can point at one of three levels:

```
# 1. Single episode
episode_001/
├── CameraTrajectoryTransformed.txt   # timestamps + 3x4 transform per frame
├── gripper_distances.txt             # one normalized gripper measurement per line
├── gripper_normalization.json        # exact min/max, convention, source, detector settings
├── color_000000.png                  # camera images (color_*.png or color_*.jpg)
└── ...

# 2. Single session
20260520_143052-png/
├── episode_001/
├── episode_002/
└── ...

# 3. Multi-session parent
20260611/
├── 20260611_151828-mp4/              # ignored by the converter
├── 20260611_151828-png/              # converted
├── 20260611_155158-png/              # converted
└── ...
```

In multi-session mode, the converter only uses session folders that contain
convertible decoded episodes (`episode_*` with transformed trajectory, gripper
file, and color frames). Original `*-mp4` folders are ignored.

## Usage

Run the converter directly from anywhere (it will add the local `lerobot` package to `sys.path`).

Single episode or single session conversion:

```bash
python /home/zfei/code/sroi_rosbag_utilities/lerobot/sroi_to_lerobot.py \
    --data_path "/path/to/your/session-png" \
    --repo_id "your_username/your_dataset_name" \
    --fps 30 \
    --root "/tmp/lerobot_datasets/my_dataset" \
    --task "End-effector manipulation task"
```

Convert a full day / parent folder containing many decoded `*-png` sessions into one dataset:

```bash
python /home/zfei/code/sroi_rosbag_utilities/lerobot/sroi_to_lerobot.py \
    --data_path "/path/to/20260611" \
    --repo_id "your_username/your_dataset_name" \
    --fps 30 \
    --root "/tmp/lerobot_datasets/my_dataset" \
    --task "pick the strawberry" \
    --num_workers 4
```

Filter conversion using QC results from `visualization/qc.py` (recommended for large recording batches):

```bash
python /home/zfei/code/sroi_rosbag_utilities/visualization/qc.py \
    "/path/to/20260611" \
    -o "/path/to/20260611/qc"

python /home/zfei/code/sroi_rosbag_utilities/lerobot/sroi_to_lerobot.py \
    --data_path "/path/to/20260611" \
    --repo_id "your_username/your_dataset_name" \
    --fps 30 \
    --root "/tmp/lerobot_datasets/my_dataset" \
    --task "pick the strawberry" \
    --qc_csv "/path/to/20260611/qc.csv" \
    --qc_categories "ok" \
    --num_workers 4
```

`--qc_categories` is comma-separated, e.g. `ok,flat_gripper`. `--episodes` is still available for single-session conversion, but it is intentionally rejected in multi-session mode because names like `episode_001` occur in every session.

To push the created dataset to the Hugging Face Hub add `--push_to_hub` and provide a `--repo_id` that you own.

There is also an example wrapper `example_convert.py` in the same folder for quick local testing.

## CLI Arguments

- `--data_path` (required): Path to a single episode, a single session folder, or a parent folder containing many decoded `*-png` sessions.
- `--repo_id` (required): Dataset repo id used for `LeRobotDataset.create` (e.g., `username/dataset`).
- `--fps`: Frames per second for the dataset (default: `30`).
- `--root`: Filesystem root where the dataset is written. It must not already exist; `LeRobotDataset.create()` creates it fresh.
- `--push_to_hub`: If set, pushes the saved dataset to the HF Hub.
- `--task`: Task description string stored with each frame.
- `--episodes`: Comma-separated episode names to convert from a single session, or `all` (default). Mutually exclusive with `--qc_csv`. Not allowed in multi-session mode.
- `--num_workers`: Parallel image-loading workers (default: `4`; use `1` for sequential loading).
- `--qc_csv`: Optional path to `qc.csv` from `visualization/qc.py`; filters episodes by QC category for each session.
- `--qc_categories`: Comma-separated QC categories to keep when `--qc_csv` is set (default: `ok`).
- `--target-conditioning`: Opt-in target-conditioned mode; see below. Off by default — the standard dataset is unchanged.
- `--extrinsics-config`: Camera-to-gripper-tip JSON for target projection (default: `configs/camera_gripper_extrinsics_sroi_v2_d405.json`).
- `--closed-threshold` / `--grasp-hold-seconds`: Grasp detection tuning (defaults `0.5` / `0.5 s`).
- `--marker-style` / `--marker-radius-px` / `--marker-offset-px`: Marker rendering (defaults `circle` / 30 px at 480p / = radius).

## Target-conditioned datasets (`observation.images.target_ref`)

Opt-in mode (`--target-conditioning`) for disambiguating *which* strawberry to pick when several
are visible. Each episode gets a second, **constant** image feature:

```
observation.images.camera     = current frame I_t          (unchanged)
observation.images.target_ref = initial frame I_0 + circle around the target strawberry
```

The label is derived purely geometrically from the demonstration (no detection/segmentation):

1. `target_ref.detect_grasp_frame` finds the start of the **final sustained gripper closure**
   (gripper convention 0=closed/1=open, NaN-filled, moving-average smoothing, ≥0.5 s hold) — not
   the last frame, so the post-grasp pull-away does not matter.
2. The gripper-tip pose at that frame is projected into the initial image with the same math as
   `visualization/visualize_traj_video.py`: `p = (T_optical_camera @ inv(poses[0]) @ poses[t_g] @
   T_camera_gripper_tip)[:3,3]`, then pinhole projection with `camera_info_color.json`.
3. A rough cyan circle (default radius 30 px at 480p) is drawn **below** the projected picking
   point so it covers the fruit. It is an object label ("this strawberry"), not a picking
   location — the policy still learns where/how to grasp.

QC: episodes with no sustained closure, invalid depth (z ≤ 0 or z > 1.5 m), or an out-of-frame
projection are **skipped** and logged with their status to `meta/target_qc.csv`; per-episode
details land in `meta/target_ref/<episode>.json`. Inspect labels before training:

```bash
python visualization/visualize_target_ref.py /path/to/data --recursive --num-samples 24 -o /tmp/target_qc.pdf
```

Conversion:

```bash
python lerobot/sroi_to_lerobot.py \
    --data_path /path/to/session-png --repo_id sroi/lab_picking_targeted \
    --fps 30 --root /mnt/data1/data/lerobot/lab_picking_targeted \
    --task "pick the marked strawberry" --target-conditioning
```

**Training (ACT / π0.5): no config change needed.** Any `observation.images.*` video feature
auto-becomes a VISUAL policy input (`dataset_to_policy_features`); ACT encodes it with the shared
ResNet backbone and concatenates the tokens, π0.5 embeds it like every other camera. Keep
`consecutive_frames=1` (default) so the static reference is not frame-stacked. For the policy to
actually use `target_ref`, the training data must contain meaningful target variation: in
multi-strawberry scenes, record otherwise-similar episodes picking *different* fruit, or the
network can ignore the marker and learn a positional bias. Single-strawberry episodes remain
usable as-is.

**Deployment:** capture the initial image, let the user click the desired strawberry, render the
identical marker, and freeze the result for the whole episode:

```bash
python lerobot/render_target_ref.py --image init.png --u 320 --v 240 -o target_ref.png
```

(the click is on the fruit itself, so the circle is centered on the click — offset 0.)

**Evaluation:** keep the existing picking metric and add, bucketed by 1 / 2 / 3 / 4+ visible
strawberries: intended-target success, any-strawberry success, wrong-target rate, and
failure-to-commit rate. Strongest test: reset the same scene and run it once with target_ref
marking fruit A and once marking fruit B — the same policy should pick the marked one.

**Test:** `lerobot/test_target_conditioning_e2e.py` converts a synthetic episode (known grasp
frame) plus optionally a real one (`--episode-dir`) and checks that target_ref exists, is
constant across the episode, lands at the expected pixel, and that the default mode is
unaffected.

## Merging Completed LeRobot Datasets

After creating separate local LeRobot datasets, merge them with the official
LeRobot edit tool from `/home/zfei/code/lerobots/lerobot_official`. Use the
`py312` conda environment because the official checkout requires Python 3.12.

Example: merge two local datasets under `/mnt/data1/data/lerobot` into a new
dataset folder:

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

Notes:

- `--new_root` is the output dataset folder and must not already exist.
- `--operation.roots` are exact local source dataset folders.
- `--operation.repo_ids` are labels for those source datasets and must have the
  same length/order as `--operation.roots`.
- Keep merge provenance in the output dataset folder (for example
  `MERGE_INFO.md` and `merge_info.json`) with source paths, source episode/frame
  counts, output counts, and the exact command.
- A quick sanity check is to compare `meta/info.json` from each source and the
  merged output; expected merged counts are the sum of source episodes/frames.

## Produced LeRobot v3.0 Feature Schema

The converter creates the following core features (matching the pipeline expectations in `record.py`):

- `observation.images.camera`:
    - `dtype`: `video`
    - `shape`: `(channels, height, width)` (CHW per-frame images)
    - `names`: `['channels', 'height', 'width']`

- `observation.state`:
    - `dtype`: `float32`
    - `shape`: `[7]`
    - `names`: `['ee.x','ee.y','ee.z','ee.wx','ee.wy','ee.wz','ee.gripper_pos']`

- `action`:
    - `dtype`: `float32`
    - `shape`: `[7]`
    - `names`: `['ee.x','ee.y','ee.z','ee.wx','ee.wy','ee.wz','ee.gripper_pos']`

Notes:

- The `ee.wx/wy/wz` fields store the rotation vector (axis-angle) representation (the same `rotvec` used in the LeRobot pipelines), not Euler angles. This matches the `ForwardKinematicsJointsToEE` and `InverseKinematicsEEToJoints` processors.
- `action` stores the *target* end-effector pose for the next timestep (so policies operating on EE-space can be trained directly).

## Processing Steps (high level)

1. Load `CameraTrajectoryTransformed.txt` and parse the per-frame 3x4 transforms.
2. Convert each 3x4 into a 4x4 pose matrix and extract translation (`ee.x, ee.y, ee.z`) and rotation as a rotation-vector (`ee.wx, ee.wy, ee.wz`).
3. Read `gripper_distances.txt` and align length with poses and images.
4. Copy each episode's `gripper_normalization.json` into
   `meta/gripper_normalization/` and write a dataset-level
   `meta/gripper_normalization.json` grouped by distinct parameter set.
5. Load `color_*.png` images, convert to CHW and write them as video frames into the LeRobot dataset.
6. For each frame, create `observation.state` from the current pose+gripper and `action` from the next frame's pose+gripper (last frame repeats the final pose).
7. Save the episode (Parquet + video files) and optionally push to the hub.

## Error handling & validation

- The script validates that the `--data_path` exists and prints a clear error if not.
- It truncates arrays to the shortest modality (poses, gripper, images) and warns if images are missing.
- It reports progress during conversion and raises exceptions for unrecoverable issues.

## Example output

```
Original trajectory shape: (125, 13)
Dataset length: 125 frames
Adding frames to dataset...
Processed 10/125 frames
...
Dataset created with 125 frames
Dataset episodes: 1
Conversion successful!
```

## Dependencies

- `numpy`
- `Pillow` (PIL)
- `lerobot` (local codebase in `../lerobot`)

Run the converter from any location — the script will add the local `lerobot` package to `sys.path` so that the current repo is used.
