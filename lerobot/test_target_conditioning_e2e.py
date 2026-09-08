#!/usr/bin/env python3
"""End-to-end test for the target-conditioning conversion mode.

Runs without any real data: builds a small synthetic episode (known trajectory,
known grasp frame), converts it twice — once with --target-conditioning, once
without — and verifies:

  1. target mode: observation.images.target_ref exists with the right shape;
  2. the target_ref frames are (near-)constant across the whole episode;
  3. the detected grasp frame and projected (u, v) match independent computation;
  4. the marker is visible at the expected location (differs from the plain
     camera frame there, matches elsewhere);
  5. meta/target_ref/episode_001.json and meta/target_qc.csv are written;
  6. default mode (no flag): the dataset has NO target_ref feature — the
     standard pipeline is unaffected.

Optionally also converts a real episode (--episode-dir) with the same checks
minus the known ground truth.

Run with the env that has lerobot installed (no cv2 needed):
    /home/zfei/anaconda3/envs/py312/bin/python lerobot/test_target_conditioning_e2e.py
    /home/zfei/anaconda3/envs/py312/bin/python lerobot/test_target_conditioning_e2e.py \
        --episode-dir /path/to/session-png/episode_002 --keep

Exit code 0 = all checks passed.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lerobot"))

import target_ref as tr
import sroi_to_lerobot as conv

N_FRAMES = 40
GRASP_FRAME = 20
H, W = 480, 640
FPS = 30
K_SYNTH = [392.0, 0.0, 320.0, 0.0, 392.0, 240.0, 0.0, 0.0, 1.0]

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def build_synthetic_episode(root: Path) -> Path:
    """session/episode_001 with known grasp frame and near-identity poses."""
    ep = root / "synthetic-png" / "episode_001"
    ep.mkdir(parents=True)

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    for i in range(N_FRAMES):
        noise = rng.integers(0, 8, size=(H, W, 1), dtype=np.uint8)
        Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8)).save(
            ep / f"color_{i:06d}.png"
        )

    # Poses: identity at t=0, then a slow drift so poses are not all identical.
    with (ep / "CameraTrajectoryTransformed.txt").open("w") as f:
        for i in range(N_FRAMES):
            pose = np.eye(4)
            pose[0, 3] = i * 0.0005  # 0.5 mm/frame drift in x
            pose[1, 3] = i * 0.0002
            flat = " ".join(f"{v:.6f}" for v in pose[:3].flatten())
            f.write(f"{i / FPS:.6f} {flat}\n")

    grip = np.ones(N_FRAMES)
    grip[GRASP_FRAME:] = 0.1
    grip[-3:] = np.nan  # trailing NaNs (tag occluded after grasp)
    np.savetxt(ep / "gripper_distances.txt", grip, fmt="%.6f")

    with (ep / "camera_info_color.json").open("w") as f:
        json.dump({"width": W, "height": H, "K": K_SYNTH}, f)
    return ep.parent  # session dir


def expected_target(ep_dir: Path) -> dict:
    """Independent recomputation of the expected target point."""
    tip_kin = tr.load_tip_kin(tr.DEFAULT_EXTRINSICS_CONFIG)
    return tr.compute_episode_target(ep_dir, tip_kin, fps=FPS)


def check_converted_dataset(root: Path, ep_dir: Path, expected: dict | None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("test/target_ref_e2e", root=root)
    check("target_ref feature present", tr_feature_in(ds), f"features={list(ds.features)}")

    n = len(ds)
    idxs = sorted({0, n // 2, n - 1})
    frames = [ds[i]["observation.images.target_ref"].numpy() for i in idxs]
    # Same source image is stored per frame; video codecs are lossy, so allow
    # small decode noise but reject any genuinely changing content.
    max_diffs = [float(np.abs(frames[0] - fr).max()) for fr in frames[1:]]
    mean_diffs = [float(np.abs(frames[0] - fr).mean()) for fr in frames[1:]]
    const = all(m < 0.2 for m in max_diffs) and all(m < 0.01 for m in mean_diffs)
    check("target_ref constant across episode", const,
          f"max abs diff vs frame 0: {max_diffs}, mean: {mean_diffs}")

    meta_path = root / "meta" / "target_ref" / f"{ep_dir.name}.json"
    check("per-episode meta json written", meta_path.exists(), str(meta_path))
    if meta_path.exists() and expected is not None:
        meta = json.load(meta_path.open())
        check("grasp frame matches", meta["grasp_frame"] == expected["t_g"],
              f"meta={meta['grasp_frame']} expected={expected['t_g']}")
        uv_ok = (abs(meta["u"] - expected["u"]) < 1.0 and abs(meta["v"] - expected["v"]) < 1.0)
        check("projected (u, v) matches", uv_ok,
              f"meta=({meta['u']:.1f}, {meta['v']:.1f}) expected=({expected['u']:.1f}, {expected['v']:.1f})")
        check("(u, v) in image bounds", 0 <= meta["u"] < W and 0 <= meta["v"] < H)

    qc_path = root / "meta" / "target_qc.csv"
    check("target_qc.csv written", qc_path.exists(), str(qc_path))

    # Marker visibility: target_ref vs plain camera frame (both decoded, lossy) —
    # the difference must concentrate around the marker, not the rest of the image.
    if expected is not None and expected["status"] == tr.STATUS_OK:
        t0 = ds[0]["observation.images.target_ref"].numpy()  # (3, H, W) in [0, 1]
        c0 = ds[0]["observation.images.camera"].numpy()
        diff = np.abs(t0 - c0).mean(axis=0)  # (H, W)
        r = 30
        off = r
        cx, cy = int(round(expected["u"])), int(round(expected["v"] + off))
        inside = np.zeros((H, W), dtype=bool)
        yy, xx = np.mgrid[0:H, 0:W]
        ring = np.abs(np.hypot(xx - cx, yy - cy) - r) <= 6
        inside |= ring
        in_mean = float(diff[inside].mean())
        out_mean = float(diff[~inside].mean())
        check("marker visible at expected ring", in_mean > 3 * max(out_mean, 1e-6),
              f"ring mean diff={in_mean:.4f} vs rest={out_mean:.4f}")


def tr_feature_in(ds) -> bool:
    return "observation.images.target_ref" in ds.features


def run_conversion(data_path: Path, root: Path, target_conditioning: bool) -> None:
    conv.create_lerobot_dataset(
        data_path=data_path,
        repo_id="test/target_ref_e2e",
        fps=FPS,
        root=str(root),
        single_task="pick the strawberry",
        num_workers=1,
        target_conditioning=target_conditioning,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dir", type=Path, default=None,
                    help="Optional real episode to convert as an extra check")
    ap.add_argument("--keep", action="store_true", help="Keep the temp directory")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="target_ref_e2e_"))
    print(f"work dir: {tmp}")
    try:
        print("\n== 1. synthetic episode, target conditioning ON ==")
        session = build_synthetic_episode(tmp / "src")
        ep_dir = session / "episode_001"
        expected = expected_target(ep_dir)
        check("synthetic grasp frame detected", expected["t_g"] == GRASP_FRAME,
              f"t_g={expected['t_g']} expected={GRASP_FRAME}")
        check("synthetic target status ok", expected["status"] == tr.STATUS_OK,
              f"u={expected['u']:.1f} v={expected['v']:.1f} z={expected['z']:.3f}")

        ds_root = tmp / "ds_target"
        run_conversion(session, ds_root, target_conditioning=True)
        check_converted_dataset(ds_root, ep_dir, expected)

        print("\n== 2. synthetic episode, default mode (flag OFF) ==")
        plain_root = tmp / "ds_plain"
        run_conversion(session, plain_root, target_conditioning=False)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds_plain = LeRobotDataset("test/target_ref_e2e", root=plain_root)
        check("no target_ref feature by default", not tr_feature_in(ds_plain),
              f"features={list(ds_plain.features)}")
        check("no target_qc.csv by default",
              not (plain_root / "meta" / "target_qc.csv").exists())

        if args.episode_dir is not None:
            print(f"\n== 3. real episode: {args.episode_dir} ==")
            expected_real = expected_target(args.episode_dir)
            check("real target status ok", expected_real["status"] == tr.STATUS_OK,
                  f"status={expected_real['status']} t_g={expected_real['t_g']} "
                  f"u={expected_real['u']} v={expected_real['v']}")
            real_root = tmp / "ds_real"
            run_conversion(args.episode_dir, real_root, target_conditioning=True)
            check_converted_dataset(real_root, args.episode_dir, expected_real)
    finally:
        if args.keep:
            print(f"\nkept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
