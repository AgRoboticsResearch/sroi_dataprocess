# v1 → v2 conversion

Converts the SROI v1 ROS1 bags (`oak_d_sr` and `realsense_d435i` rigs) into
v2-style sessions — the layout `record_realsense.py` produces — so the rest of
the pipeline (ORB-SLAM3 batch → transform → gripper → QC → LeRobot) runs on
them unchanged.

**The canonical output for the full corpus is the MP4 store** (`--mp4-only`):
PNG trees for all 42 bags would need ~265 GiB, the MP4 store is ~18 GiB.
PNG trees are regenerated per batch with `decode_videos.py` when a processing
step needs them — the same store/wipeable-PNG split the SROI_V2 lab datasets
use (`1000-onesb-lab-picking/` ↔ `1000-onesb-lab-picking-pngs/`).

## Files

- `convert_v1_bags.py` — the converter (one bag or a whole directory tree).
- `export_camera_info.py` — dumps a bag's camera_info to JSON; used once to
  create the oak fallback below.
- `fallback/oak_d_sr_camera_info.json` — left/right camera_info exported from
  `oak_d_sr_2025-04-14-08-21-26.bag` (same physical rig as the 2024-12-30
  bags, whose values also match the hand-made `orb_slam_yaml/oak_d_sr.yaml`).
  That source bag was a test recording and has been deleted — this JSON is
  the surviving copy and still backs every dec-2024 oak conversion.

## Full-corpus procedure (run 2026-08-17)

From the repo root, against `/mnt/data1/sroi/sroi_v1` (40 bags: 17 ningbo
realsense + 23 shunba oak-2024-12-30):

```bash
# 0) Check the inputs and the plan (writes nothing). Confirm the bag count,
#    episode segmentation and that every dec-2024 oak bag needs the fallback.
python3 v1_to_v2/convert_v1_bags.py /mnt/data1/sroi/sroi_v1 \
    --fallback-camera-info v1_to_v2/fallback/oak_d_sr_camera_info.json --dry-run

# 1) Full conversion → MP4 store only (~18 GiB; ~7% of the PNG equivalent).
#    --skip-existing makes reruns resume: episodes with timestamps.json are
#    skipped (and get their imu.txt backfilled if it predates IMU saving).
python3 v1_to_v2/convert_v1_bags.py /mnt/data1/sroi/sroi_v1 \
    --fallback-camera-info v1_to_v2/fallback/oak_d_sr_camera_info.json \
    --mp4-only --skip-existing --jobs 8
```

Output mirrors the source layout under `<input>/v1_to_v2`:

```
v1_to_v2/ningbo_sroi_rosbags/<YYYYmmdd_HHMMSS>-mp4/episode_001..N
v1_to_v2/shunba-shanjie/20241230/<YYYYmmdd_HHMMSS>-mp4/...   (+ CALIBRATION_NOTE.txt)
v1_to_v2/shunba-shanjie/<YYYYmmdd_HHMMSS>-mp4/...
```

Each episode holds `left.mp4`/`right.mp4`(`color.mp4`), `times.txt`
(hardware timestamps), `timestamps.json`, `camera_info_*.json`,
`orb_slam_<camera_type>.yaml`, and `imu.txt`. The session dir adds
`conversion.json` (provenance: segments, sync stats, calibration source,
per-episode counts), `imu_info.json` + `extrinsics.json` (realsense only),
and `CALIBRATION_NOTE.txt` where the fallback was used.

Why not PNG for everything: measured per-frame costs from the test
conversions are ~2.2 MiB/frame for oak color PNG pairs and ~240 KiB/frame for
realsense triplets → ~265 GiB total, which does not fit next to the bags.
MP4 (libx264, crf 23 — same settings as `record_realsense.py`) is ~10–30
KiB/frame. The PNG mode (`--encode-mp4`) remains available for single bags
and small batches.

## Regenerating PNG trees for processing

`decode_videos.py` decodes a session (or a whole tree with `--recursive`)
into a sibling `-png` folder, or into any custom root with `-o`; it copies
the metadata (camera_info, times, yaml, `imu.txt`, `CALIBRATION_NOTE.txt`)
along. Example batch flow, mirroring the SROI_V2 datasets:

```bash
# decode the whole store into a dataset-level PNG root (wipeable)
python3 decode_videos.py /mnt/data1/sroi/sroi_v1/v1_to_v2/shunba-shanjie \
    --recursive -o /mnt/data1/sroi/sroi_v1/v1_to_v2/shunba-shanjie-pngs
# ... run ORB-SLAM/transform/gripper/QC on the -pngs tree ...
# ... then wipe the -pngs tree; the -mp4 store stays pristine.
```

## What the v1 bags look like (surveyed 2026-08-17)

| group | camera_info in bag | streams | IMU | episodes |
|---|---|---|---|---|
| ningbo `rs435_*.bag` (848x480 @30), 17 bags | yes (infra1/infra2/color) | mono8 IR jpeg + bgr color jpeg | `/camera/imu` ~200 Hz + IMUInfo + tf_static | `is_action` segments (3–6 s each) |
| shunba `20241230/oak_d_sr_*.bag` (1280x720 @30), 23 bags | **none** | bgr jpeg left/right only | `/oak/imu/data` ~400 Hz | `is_action` present but never True → whole bag = 1 episode |

Full-run result (2026-08-17): 42/42 bags ok → 407 episodes, 146,447 frames.
Two of those bags (`oak_d_sr_2025-04-14-08-21-26` and
`oak_d_sr_2024-12-30-07-15-40`) were test recordings, not real data, and were
deleted afterwards together with their sessions — the store now holds 40
sessions / 405 episodes / 144,885 frames, 18 GiB (episodes are action-gated,
so far fewer frames than the bags contain in total).

(Originally 40 synced bags; two truncated `.bag.active` leftovers —
`rs435_2026-02-05-03-44-02` and `rs435_2026-02-05-03-59-22` — were repaired
into complete bags before the full run, making 42, and two shunba test bags
were deleted after it (see above), leaving 40. The converter skips
`.bag.active`/`.bag.XXXX` temp files automatically.)

## Rules the converter enforces

- **Camera info always comes from the bag being converted.** The fallback file
  is only consulted when the bag has no camera_info topics, must be passed
  explicitly, and every episode then carries a `CALIBRATION_NOTE.txt`
  stating the original camera_info is missing and where the numbers came from.
- **Calibration must match the image pixels.** The first frame of every stream
  is decoded and its resolution compared against the camera_info (and the
  principal point bounds-checked). A mismatch — mislabeled bag or a fallback
  from a different resolution vintage — aborts that bag with an error instead
  of writing unusable calibration.
- **Resolution is verified per role** (`left`, `right`, `color`), so a color
  stream at a different size than IR is caught too.

## OAK specifics

- Both OAK cameras are color cameras (no IR, no separate color stream), so
  episodes contain only `left_/right_*` files. The downstream consumers
  (`gripper_estimation_april_tag.py`, `lerobot/sroi_to_lerobot.py`,
  `visualization/visualize_traj_video.py`) fall back to `left_*` frames when
  `color_*` is absent — no duplicated color files on disk.
- The depthai driver puts the stereo baseline offset on the **left** P matrix
  (right-reference convention), so the signed baseline is negative; the yaml
  uses the magnitude (20.16 mm) and the log notes the sign flip. RealSense
  bags follow the standard convention (baseline on right P, 49.95 mm).
- v1 OAK images are `image_raw` (unrectified; the rectified topics used by the
  old v1 pipeline were not recorded). `stereo_kitti` assumes rectified pairs,
  so expect OAK trajectories to be less accurate than the realsense ones —
  this is inherent to the recorded data, not the conversion.
- `batches/orbslam_batch_local.sh` globs `orb_slam_*.yaml` per episode
  (was: hardcoded `orb_slam_realsense_d405.yaml`), so the generated
  `orb_slam_realsense_d435i.yaml` / `orb_slam_oak_d_sr.yaml` are picked up.

## IMU + extrinsics (saved for later; not used by the pipeline yet)

Every v1 bag carries one fused `sensor_msgs/Imu` stream. Each episode gets an
EuRoC-style `imu.txt`:

```
# timestamp_ns gx gy gz ax ay az (header stamps, EuRoC imu.txt order)
1770265477843616962 0.066322513 0.073303826 -0.479965538 -0.029419949 -9.129990578 0.029419949
```

- Timestamps are the original header stamps in nanoseconds, sliced to the
  episode window ±2 s. They were verified to share the image header clock on
  all three vintages (rs435: identical; oak apr-2025: both −50 s vs bag time,
  constant; oak dec-2024: both ≈0), so `imu.txt` lines up with `times.txt`
  as-is.
- `<session>/imu_info.json` holds the realsense `IMUInfo` messages (frame
  ids, noise/bias variances, extrinsics `data` field). On these bags the
  driver left the variances at 0 — if a stereo-inertial run ever happens,
  use datasheet noise values for `IMU.NoiseGyro`/`NoiseAcc`.
- `<session>/extrinsics.json` holds the `/tf_static` tree; on the realsense
  bags this is where the real IMU-to-camera transform lives
  (`camera_link -> camera_gyro/accel_frame`, t ≈ (−0.0117, −0.0055, +0.0051) m)
  alongside the infra baseline (`camera_link -> camera_infra2_frame`, −50 mm).
  OAK bags publish none of this; the oak stereo baseline comes from the P
  matrices as above.
- Wiring this into ORB-SLAM3 (`stereo_inertial_kitti`-style) is future work;
  nothing downstream reads `imu.txt` yet.

## Test matrix (one bag per camera type and site, 2026-08-17)

| bag | site | camera | result |
|---|---|---|---|
| `rs435_2026-02-04-08-48-37.bag` | ningbo | realsense | 11 episodes / 3917 frames synced, info from bag, baseline 49.95 mm, PNG+MP4, `decode_videos.py` roundtrip OK, ORB-SLAM3 batch 11/11 ok |
| `oak_d_sr_2025-04-14-08-21-26.bag` | shunba | oak (2025) | 1 episode / 138 frames, info from bag, baseline 20.16 mm |
| `oak_d_sr_2024-12-30-07-15-40.bag` | shunba | oak (2024) | 1 episode / 1424 frames via fallback + `CALIBRATION_NOTE.txt` |
| `rs435_2026-02-05-04-24-37.bag` (mp4-only mode) | ningbo | realsense | 1 episode / 333 frames, `imu.txt` 200 Hz, `extrinsics.json` (10 tf_static transforms), roundtrip OK |
| `oak_d_sr_2024-12-30-07-33-48.bag` (mp4-only mode) | shunba | oak (2024) | 1 episode / 1214 frames via fallback, `imu.txt` 400 Hz, note carried into mp4 tree |

(The `oak_d_sr_2025-04-14` and `oak_d_sr_2024-12-30-07-15-40` rows used during
validation refer to the two test bags that were later deleted — the sessions
they produced are gone from the store.)

After conversion, continue with the standard pipeline on a decoded PNG tree:
`batches/orbslam_batch_local.sh <session-png> ~/code/ORB_SLAM3 --mask-config
configs/gripper_mask_sroi_v1_d435.json` (ningbo rs435 IR mask, ported from the
v2 D405 mask by x-scaling 640→848 — see `v2_mask_overlay_check.png` in the
output tree for how it sits on v1 frames; `apply_gripper_mask.py` also
auto-rescales any mask config onto other frame sizes now) →
`transform_trajectory.py --recursive` → `gripper_estimation_april_tag.py
--recursive --configs configs/sroi_v1_d435.json` (realsense; OAK needs a
matching gripper config) → `visualization/qc.py`. Note the ORB-SLAM batch takes
a single session dir (episodes one level down), so loop it over the `-pngs`
sessions. For reviewing the QC verdicts per episode, render badged review
videos from the qc.csv (category as on-video badge + filename prefix):
`visualization/visualize_traj_video.py <pngs-root> --recursive --output
<traj_videos> --qc-csv <pngs-root>/qc.csv --codec libx264`. The tip-path
projection inside those uses the v2 D405 extrinsics config by default —
approximate for the v1 rig (no v1 extrinsics config exists yet).
