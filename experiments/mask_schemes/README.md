# Mask scheme comparison (raw / maskhalf / maskgripper)

This directory compares three pre-ORB-SLAM3 masking strategies and contains the
standalone tooling used for the experiment.

## Pipeline integration status

The gripper mask is integrated into the canonical ORB batch scripts as an **opt-in**
step. Existing commands remain unmasked. Pass a rig-specific config to enable masking:

```bash
batches/orbslam_batch_local.sh <session> <orbslam-dir> true false \
    --mask-config configs/gripper_mask_sroi_v2_d405.json
```

The batch creates temporary masked stereo frames, runs ORB on them, copies the trajectory
back to the original episode, and removes the temporary input. Source images are never
modified. The mask and extrinsics schemas are documented in
[`../../design/rig_configs.md`](../../design/rig_configs.md). The other scripts in this directory
remain useful for multi-scheme experiments.

## Preliminary experiment findings

The original experiment covered 287 episodes from four Vive sessions and two sessions
without Vive. It reported the following results:

| Scheme | Description | Reported result |
|---|---|---|
| `raw` | No mask | Frequent trajectory collapse |
| `maskhalf` | Black out the bottom 38/97 of the image (cutoff 292) | Prevented collapse, with occasional tracking loss |
| `maskgripper` | Per-eye gripper trapezoid plus the bottom 13 rows | Similar quality with fewer reported tracking losses |

- For 278 episodes where both masked schemes tracked fully, the reported median
  maskgripper-versus-maskhalf deviation was 0.5 mm.
- The original summary reported 9 maskhalf losses, 5 maskgripper losses, and 4 episodes
  recovered by maskgripper.

PR #8 fixed accounting bugs that previously omitted complete maskhalf failures and
zero-valued results. Re-run the comparison before treating the historical counts above
as final.

The working hypothesis is that the gripper and AprilTags at the bottom of the frame
produce high-contrast moving features that violate ORB-SLAM's static-scene assumption.

## Manual masking usage

Apply the per-eye gripper mask before running ORB-SLAM:
```bash
python3 experiments/mask_schemes/mask_gripper_trapezoid.py <schemes/session/episode-directory> \
    --config configs/gripper_mask_sroi_v2_d405.json
```

For a schemes directory, the script reads `episode_*/raw/`. For a normal session or
single episode, it reads images from the episode root. Output is always written to
`episode_*/maskgripper/`; it does not overwrite the source images.

Mask geometry for the validated 640x480 D405 rigs:

- Left eye: [(236,292), (417,292), (525,467), (127,467)], plus rows 467 and below.
- Right eye: [(182,292), (365,292), (444,467), (46,467)], plus rows 467 and below.
- The right-eye polygon is shifted left by stereo disparity.

## Full comparison workflow

```bash
# 1) Build raw/maskhalf/maskgripper trees and run ORB for each scheme
MASK_DATA=/path/to/data bash experiments/mask_schemes/build_session_schemes.sh 20260709_XXXXXX

# 2) Transform each scheme trajectory for RGB projection
python3 transform_trajectory.py <session>-schemes --recursive

# 3) Render side-by-side comparison videos with this rig's extrinsics
python3 experiments/mask_schemes/viz_scheme_compare.py <session>-schemes \
    --extrinsics-config configs/camera_gripper_extrinsics_sroi_v2_d405.json

# 4) Calculate completeness and deviation against maskhalf
python3 experiments/mask_schemes/deviation_vs_baseline.py <session>-schemes

# 5) Aggregate results across sessions
python3 experiments/mask_schemes/aggregate_schemes.py /path/to/data
```

Outputs are written under `<session>-schemes/_compare/`.

## Files

- `mask_gripper_trapezoid.py`: creates config-driven maskgripper images.
- `mask_session.py`: creates the maskhalf comparison images.
- `schemes_init.py`: builds the per-episode schemes layout.
- `build_session_schemes.sh`: runs the standalone end-to-end comparison workflow.
- `run_orb_scheme.py`: runs ORB-SLAM3 for one scheme.
- `viz_scheme_compare.py`: renders multi-scheme comparison videos.
- `deviation_vs_baseline.py`: computes per-session completeness and deviation.
- `aggregate_schemes.py`: aggregates comparison CSV files across sessions.
- `orb_vs_vive.py`: aligns ORB trajectories with Vive motion capture.

## Notes

- Projection extrinsics are loaded from `configs/camera_gripper_extrinsics_sroi_v2_d405.json`.
  Use a separate config and verify the projected green point for every different rig.
- The default data root is `/home/ss/data/1000_onesb_labpicking`. Override it with
  `MASK_DATA`, or pass a data-root argument to `aggregate_schemes.py`.
- `run_orb_scheme.py` defaults to `~/code/ORB_SLAM3`. Override it with
  `ORB_SLAM3_DIR` or `--orbslam-dir`.
- Vive motion capture can become unreliable under occlusion; exclude occluded episodes
  before treating it as a reference.
