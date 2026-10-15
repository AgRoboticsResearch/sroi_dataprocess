# Gripper Mask and Projection Extrinsics

Mask geometry and camera-to-gripper-tip projection transforms are rig-specific. Keep
them in separate JSON files so a recording can be processed with the calibration for
the camera/gripper assembly that produced it.

The supplied SROI v2 D405 examples are:

- `configs/gripper_mask_sroi_v2_d405.json`
- `configs/camera_gripper_extrinsics_sroi_v2_d405.json`

Copy and rename these files when calibrating another camera, resolution, mounting
position, or gripper. Do not edit the D405 files to represent a different rig.

## Optional gripper mask in the ORB-SLAM pipeline

The standard batch commands remain unmasked unless `--mask-config` is passed. The
option comes after the existing positional arguments:

```bash
# Host ORB-SLAM3 installation
batches/orbslam_batch_local.sh /path/to/session-png ~/code/ORB_SLAM3 \
    true false --mask-config configs/gripper_mask_sroi_v2_d405.json

# D435i / other rigs: copy the D405 config, change the geometry, pass it the same way
```

When masking is enabled, each episode is processed as follows:

1. The stereo images are copied to a temporary directory and masked there.
2. `times.txt` and the ORB-SLAM YAML are linked into the temporary input.
3. ORB-SLAM3 runs against the temporary images.
4. `CameraTrajectory.txt` is copied atomically back to the source episode.
5. The temporary directory is removed, including after an interrupt.

The recorded images are never modified. With no `--mask-config`, ORB-SLAM3 receives
the original episode path exactly as before. When `skip_existing=true`, an episode
that already has `CameraTrajectory.txt` is skipped before temporary masked images are
created. Use `skip_existing=false` to regenerate an existing trajectory with a mask.

The standalone equivalent for one episode is:

```bash
python apply_gripper_mask.py /path/to/episode /tmp/masked-episode \
    --config configs/gripper_mask_sroi_v2_d405.json
```

### Mask JSON schema

```json
{
  "schema_version": 1,
  "image_size": {"width": 640, "height": 480},
  "streams": {
    "left": {
      "polygons": [[[236, 292], [417, 292], [525, 467], [127, 467]]],
      "black_below_row": 467
    },
    "right": {
      "polygons": [[[182, 292], [365, 292], [444, 467], [46, 467]]],
      "black_below_row": 467
    }
  }
}
```

- Polygon points are integer pixel coordinates `[x, y]`, with the origin at the
  top-left. Each polygon needs at least three in-bounds points.
- Each eye may contain multiple polygons.
- `black_below_row` is optional. If set to `r`, rows `r` through the bottom of the
  image are blacked out. Use `null` to disable the horizontal cutoff.
- Each eye must define at least one polygon or a cutoff.
- Input images must exactly match `image_size`. Frame names must be sequential from
  `left_000000` and `right_000000`, stereo counts must match, and the number of
  nonempty lines in `times.txt` must equal the frame count.
- PNG, JPEG, and JPG input frames are accepted. Temporary output is always PNG for
  `stereo_kitti`; grayscale and color channel layouts are preserved.

To calibrate a new mask, view representative left and right IR frames, choose polygons
that cover the gripper through its full motion, and test the config on a temporary
episode before running a full session. Stereo disparity usually means the two eyes
need different coordinates.

## Camera-to-gripper-tip projection extrinsics

The trajectory video overlays the future gripper-tip path on the color frames. Load
the transform calibration explicitly when processing a rig:

```bash
python visualization/visualize_traj_video.py /path/to/session-png --recursive \
    --extrinsics-config configs/camera_gripper_extrinsics_sroi_v2_d405.json

python experiments/mask_schemes/viz_scheme_compare.py /path/to/session-schemes \
    --extrinsics-config configs/camera_gripper_extrinsics_sroi_v2_d405.json
```

Both commands default to the supplied SROI v2 D405 config when the option is omitted.
That default is only appropriate for recordings made with the matching assembly.

### Extrinsics JSON schema

```json
{
  "schema_version": 1,
  "transform_convention": "T_a_b maps coordinates from frame b into frame a",
  "frames": {
    "optical": "camera_optical_link",
    "camera": "camera_link",
    "tip": "gripper_tip"
  },
  "T_optical_camera": [
    [2.6794896412773997e-08, -0.9999999999999996, 0.0, 0.0],
    [2.6794896357262843e-08, 0.0, -0.9999999999999998, 0.0],
    [0.9999999999999993, 2.6794896412773968e-08, 2.6794896468285145e-08, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "T_camera_gripper_tip": [
    [0.8660254, 0.0, -0.5, 0.145],
    [0.0, 1.0, 0.0, 0.0],
    [0.5, 0.0, 0.8660254, -0.03],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

For readability, tiny near-zero values in this example are shown as `0.0`; use the
full measured values in the supplied config. `T_a_b` maps a point written in frame
`b` into frame `a`:

- `T_optical_camera` maps camera-link coordinates into the optical frame.
- `T_camera_gripper_tip` gives the gripper-tip pose in the camera-link frame.

Each transform must be a finite rigid 4x4 homogeneous matrix. The last row must be
`[0, 0, 0, 1]`; the 3x3 rotation must be orthonormal with determinant +1. After
calibrating a new file, render a short episode and verify the projected green tip path
against the physical gripper before using it for QC.

The mask config and extrinsics config are independent: the first changes the stereo
images supplied to ORB-SLAM3, while the second is used only for trajectory projection.
In practice, both should be versioned and named for the same physical rig.
