#!/usr/bin/env python3

"""
Convert SROI robot trajectory data to LeRobot Dataset format.

This script converts end-effector trajectory data, gripper distances, and camera images
from SROI format to a LeRobot dataset that can be used for training policies.

Usage:
    python sroi_to_lerobot.py --data_path /path/to/data --repo_id username/dataset_name
"""

import argparse
import csv
import json
import logging
import numpy as np
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.rotation import Rotation


GRIPPER_NORMALIZATION_MANIFEST = "gripper_normalization.json"

# --- Start-segment (quasi-static head) trimming --------------------------------
# Recordings open with a quasi-static head: the hand hovers in place (jitter
# with occasional slow drift) before committing to the pick. Two-stage rule
# (user-designed 2026-10-13, validated on 1138 episodes):
#   1. ANCHOR   - the first frame whose net displacement from frame 0 reaches
#                 `gate_mm`. Generous enough that jitter/drift cannot trigger it.
#   2. BACK-WALK- from a seeded step at/after the anchor, walk backwards while
#                 each step qualifies: magnitude >= `step_min_mm` AND direction
#                 within `angle_max_deg` of the step above it. The first
#                 non-qualifying step marks the start of the consistent run;
#                 data is kept from that frame. Oscillating hover-with-drift
#                 fails the angle test and stays trimmed; genuine (possibly
#                 curvy) approach keeps its acceleration ramp.
# Episode that never reaches the gate is kept whole; the trim is capped at
# `max_fraction` of the episode as a safety net. A short quasi-static lead-in
# is preserved on purpose (TRIM_KEEP_EXTRA_FRAMES before the found start):
# recovery-style episodes naturally open with a brief pause, and a few still
# frames cost nothing at execution time.
TRIM_GATE_MM = 10.0        # anchor: net displacement from the first frame
TRIM_STEP_MIN_MM = 0.8     # qualifying back-walk step magnitude
TRIM_ANGLE_MAX_DEG = 10.0  # max angle between adjacent back-walk steps
TRIM_SEED_WINDOW = 15      # frames after the anchor to look for a seed step
TRIM_MAX_FRACTION = 0.3    # safety cap: never trim more than this fraction
TRIM_KEEP_EXTRA_FRAMES = 3  # keep a brief quasi-static lead-in before the found start


def find_start_trim_count(
    positions: np.ndarray,
    gate_mm: float = TRIM_GATE_MM,
    step_min_mm: float = TRIM_STEP_MIN_MM,
    angle_max_deg: float = TRIM_ANGLE_MAX_DEG,
    seed_window: int = TRIM_SEED_WINDOW,
    max_fraction: float = TRIM_MAX_FRACTION,
) -> int:
    """Number of leading quasi-static frames to drop (see block comment above).

    Returns 0 when the episode is too short, never leaves the start vicinity,
    or no seed step exists after the anchor (conservative: keep everything).
    The count is reduced by TRIM_KEEP_EXTRA_FRAMES (floored at 0) so the kept
    head includes a brief quasi-static lead-in.
    """
    n = len(positions)
    if n < 8:
        return 0
    d0 = np.linalg.norm(positions - positions[0], axis=1) * 1000.0  # mm
    hit = np.nonzero(d0 >= gate_mm)[0]
    if not len(hit):
        return 0  # never leaves the start vicinity: keep everything
    m = int(hit[0])
    steps = np.diff(positions, axis=0) * 1000.0
    sn = np.linalg.norm(steps, axis=1)
    # seed: first step of at least step_min_mm within `seed_window` after anchor
    seed = None
    for i in range(m, min(n - 1, m + seed_window)):
        if sn[i] >= step_min_mm:
            seed = i
            break
    if seed is None:
        return 0
    j = seed
    while j - 1 >= 0:
        if sn[j - 1] < step_min_mm:
            break
        a, b = steps[j], steps[j - 1]
        cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        if np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))) > angle_max_deg:
            break
        j -= 1
    return max(0, min(j, int(max_fraction * n)) - TRIM_KEEP_EXTRA_FRAMES)


def _copy_gripper_normalization(
    episode_path: Path,
    dataset_root: Path,
    episode_key: str,
) -> dict | None:
    """Copy one episode's normalization manifest into LeRobot metadata."""
    source = episode_path / GRIPPER_NORMALIZATION_MANIFEST
    if not source.exists():
        return None
    with source.open() as f:
        payload = json.load(f)
    destination_dir = dataset_root / "meta" / "gripper_normalization"
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_dir / f"{episode_key}.json")
    return payload


def _write_gripper_normalization_summary(
    dataset_root: Path,
    records: list[tuple[str, dict]],
    missing: list[str],
) -> Path:
    """Write a dataset-level index, grouping episodes by parameter set."""
    grouped: dict[str, dict] = {}
    for episode_label, manifest in records:
        key = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        group = grouped.setdefault(
            key,
            {"parameters": manifest, "episodes": []},
        )
        group["episodes"].append(episode_label)

    summary = {
        "schema_version": 1,
        "manifest_name_in_source_episodes": GRIPPER_NORMALIZATION_MANIFEST,
        "normalization_set_count": len(grouped),
        "normalization_sets": list(grouped.values()),
        "episodes_missing_manifest": missing,
    }
    destination = dataset_root / "meta" / GRIPPER_NORMALIZATION_MANIFEST
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    os.replace(temporary, destination)
    return destination


QC_JSON_NAME = "qc.json"
QC_PASS_VERDICTS = {"auto_ok", "manual_ok"}


def _load_episode_qc(ep_dir: Path) -> dict | None:
    """Read the episode-local qc.json (source of truth written by
    visualization/qc.py; human review updates verdict/start_frame in place).

    Returns None when the file is missing (QC has not run for this episode).
    """
    fp = ep_dir / QC_JSON_NAME
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text())
    except Exception as e:
        raise ValueError(f"{fp}: unparsable qc.json ({e})") from e


def _collect_episodes(
    data_path: Path,
    episodes: str,
) -> tuple[list[tuple[str, Path]], bool]:
    """Auto-detect single-episode / single-session / multi-session mode and return
    the work list.

    Returns (candidates, is_multi_session) where candidates is a list of
    (session_name, episode_dir) tuples in deterministic sorted order. QC
    filtering is NOT applied here — the per-episode qc.json verdict is checked
    in the processing loop.
    """
    # Mode 1: single episode directory (backward compat)
    if (data_path / "CameraTrajectoryTransformed.txt").exists():
        return [(data_path.parent.name, data_path)], False

    # Mode 2: single session (episode_* dirs directly under data_path)
    direct = sorted([d for d in data_path.iterdir() if _is_episode_dir(d)])
    if direct:
        kept = _apply_session_filter(data_path.name, direct, None, None, episodes)
        return [(data_path.name, ep) for ep in kept], False

    # Mode 3: multi-session (subdirs each containing decoded episode_* folders).
    # A recording root can contain both *-mp4 and *-png siblings. Only decoded
    # PNG/JPEG episode folders are convertible; MP4 sessions are ignored here.
    session_dirs = sorted([d for d in data_path.iterdir() if _is_session_dir(d)])
    if episodes != "all":
        raise ValueError(
            "--episodes is ambiguous in multi-session mode. Drop --episodes to convert all "
            "qualifying episodes across sessions, or point --data_path at a specific session."
        )
    candidates: list[tuple[str, Path]] = []
    for s in session_dirs:
        eps = sorted([d for d in s.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        if not eps:
            continue
        kept = _apply_session_filter(s.name, eps, None, None, "all")
        for ep in kept:
            candidates.append((s.name, ep))

    if not candidates:
        raise ValueError(
            f"No episode_* directories found in {data_path} or its subdirectories. "
            f"Expected --data_path to point at an episode dir, a session dir, or a parent of session dirs."
        )
    return candidates, True


def _is_episode_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("episode_")


def _color_image_files(path: Path) -> list[Path]:
    """Sorted camera frames of an episode.

    Stereo-only (oak) episodes have no color_* files — their left camera IS
    the color camera — so fall back to left_* frames there.
    """
    files = sorted([*path.glob("color_*.png"), *path.glob("color_*.jpg")])
    if not files:
        files = sorted([*path.glob("left_*.png"), *path.glob("left_*.jpg")])
    return files


def _is_convertible_episode(path: Path) -> bool:
    if not _is_episode_dir(path):
        return False
    has_pose = (path / "CameraTrajectoryTransformed.txt").exists()
    has_gripper = (path / "gripper_distances.txt").exists()
    has_image = bool(_color_image_files(path))
    return has_pose and has_gripper and has_image


def _is_session_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(_is_convertible_episode(ep) for ep in path.glob("episode_*"))


def _apply_session_filter(
    session_name: str,
    episode_dirs: list[Path],
    qc_data: dict[str, dict[str, str]] | None,
    keep: set[str] | None,
    episodes: str,
) -> list[Path]:
    """Filter a single session's episode_dirs by QC categories or explicit --episodes list."""
    if qc_data is not None:
        if keep is None:
            raise ValueError("keep_categories must be set when qc_data is provided")
        if session_name not in qc_data:
            raise ValueError(
                f"Session {session_name!r} not found in qc.csv "
                f"(sessions present: {sorted(qc_data)[:3]}...)."
            )
        sess_qc = qc_data[session_name]
        kept = [ep for ep in episode_dirs if sess_qc.get(ep.name) in keep]
        print(f"QC filter {session_name}: kept {len(kept)}/{len(episode_dirs)} (categories={sorted(keep)})")
        return kept
    if episodes != "all":
        wanted = {e.strip() for e in episodes.split(",")}
        kept = [ep for ep in episode_dirs if ep.name in wanted]
        missing = wanted - {ep.name for ep in episode_dirs}
        if missing:
            raise ValueError(f"Episodes not found in {session_name}: {sorted(missing)}")
        return kept
    return episode_dirs


def load_trajectory_data(traj_path: str):
    """
    Load and process trajectory data from CameraTrajectoryTransformed.txt
    
    Args:
        traj_path: Path to trajectory file
        
    Returns:
        timestamps: Array of timestamps
        poses: Array of 4x4 transformation matrices
        positions: Array of positions (x, y, z)
        rotvecs: Array of rotation vectors (wx, wy, wz)
    """
    # Load trajectory data, skipping comment lines
    traj = np.loadtxt(traj_path, delimiter=" ")
    print("Original trajectory shape:", traj.shape)
    
    timestamps = traj[:, 0]
    traj = traj[:, 1:]
    traj = traj.reshape(-1, 3, 4)

    # Append [0 0 0 1] to each 3x4 matrix to make it 4x4
    append_row = np.zeros((traj.shape[0], 1, 4))
    append_row[:, 0, 3] = 1
    poses = np.concatenate((traj, append_row), axis=1)
    
    # Extract positions and orientations
    positions = poses[:, :3, 3]
    rotvecs = np.array([Rotation.from_matrix(p[:3, :3]).as_rotvec() for p in poses])
    
    return timestamps, poses, positions, rotvecs


def load_gripper_data(gripper_path: str):
    """
    Load gripper distance data.
    
    Args:
        gripper_path: Path to gripper_distances.txt
        
    Returns:
        gripper_distances: Array of gripper distances
    """
    gripper_distances = np.loadtxt(gripper_path)
    return gripper_distances


def _load_image(path_str: str) -> np.ndarray:
    """Load a single image from path string. Used by process pool workers."""
    return np.array(Image.open(path_str))


def load_images_parallel(image_files, num_workers: int) -> list[np.ndarray]:
    """Load images in parallel using a process pool."""
    paths = [str(f) for f in image_files]
    if num_workers <= 1:
        return [_load_image(p) for p in paths]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(_load_image, paths, chunksize=max(1, len(paths) // (num_workers * 4))))


def create_lerobot_dataset(
    data_path: str | Path,
    repo_id: str,
    fps: int = 30,
    root: str | None = None,
    push_to_hub: bool = False,
    single_task: str = "End-effector manipulation task",
    episodes: str = "all",
    num_workers: int = 4,
    trim_start: bool = True,
):
    """
    Convert SROI data to LeRobot dataset format.

    Args:
        data_path: Path to one of:
            - a single episode directory (has CameraTrajectoryTransformed.txt)
            - a single session directory (contains episode_* subdirs)
            - a parent of session directories (contains *-png/episode_* trees)
        repo_id: Repository ID for the dataset
        fps: Frames per second
        root: Root directory for dataset storage (must not exist; converter creates it)
        push_to_hub: Whether to push to Hugging Face Hub
        single_task: Task description
        episodes: Comma-separated episode names (single-session mode only);
            "all" converts every episode.
        num_workers: Parallel workers for image loading
        QC: per-episode qc.json next to the data is the source of truth
            (verdict auto_ok/manual_ok pass; start_frame slices the head).
        trim_start: Drop the quasi-static head frames of each episode (see
            find_start_trim_count). Counts are recorded in meta/start_trim.json.
    """
    data_path = Path(data_path)
    candidates, is_multi_session = _collect_episodes(data_path, episodes)
    mode = "multi-session" if is_multi_session else "single-session/episode"
    print(f"Mode: {mode}, {len(candidates)} episode(s) to convert")

    # Process the first episode to determine image shape for features
    first_ep_path = candidates[0][1]
    sample_images = _color_image_files(first_ep_path)
    if not sample_images:
        raise ValueError(f"No images found in {first_ep_path}")
    sample_image = np.array(Image.open(str(sample_images[0])))
    h, w = sample_image.shape[:2]

    # Define dataset features
    # observation.state is omitted — it's derived from the action column
    # during training via DeriveStateFromActionStep (Pattern B).
    features = {
        "observation.images.camera": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        # 当前帧 EE 绝对位姿+夹爪(UMI 口径 state[t]=action[t-1] 的精确化:
        # 头样本的基准/速度参考不再依赖 padding 近似)。训练侧派生步优先消费。
        "observation.ee": {
            "dtype": "float32",
            "names": [
                "ee.x",
                "ee.y",
                "ee.z",
                "ee.wx",
                "ee.wy",
                "ee.wz",
                "ee.gripper_pos",
            ],
            "shape": (7,),
        },
        "action": {
            "dtype": "float32",
            "names": [
                "ee.x",
                "ee.y",
                "ee.z",
                "ee.wx",
                "ee.wy",
                "ee.wz",
                "ee.gripper_pos"
            ],
            "shape": (7,)
        },
    }

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type="so100",
        features=features,
        use_videos=True,
        image_writer_threads=1,
        image_writer_processes=1 ,
    )

    normalization_records: list[tuple[str, dict]] = []
    normalization_missing: list[str] = []
    start_trims: dict[str, int] = {}
    nan_head_trims: dict[str, int] = {}
    manual_start_frames: dict[str, int] = {}
    stream_failures: list[str] = []
    qc_skipped: list[str] = []

    # Process each episode
    for ep_idx, (session_name, ep_path) in enumerate(candidates):
        label = f"{session_name}/{ep_path.name}" if is_multi_session else ep_path.name
        print(f"\n--- Processing episode {ep_idx + 1}/{len(candidates)}: {label} ---")

        # Per-episode QC gate (source of truth: episode-local qc.json written
        # by visualization/qc.py; human review updates verdict/start_frame).
        try:
            ep_qc = _load_episode_qc(ep_path)
        except ValueError as e:
            print(f"ERROR: {label}: {e}")
            stream_failures.append(f"{label}: unparsable qc.json")
            continue
        if ep_qc is None:
            print(f"ERROR: {label}: no qc.json (run visualization/qc.py first)")
            stream_failures.append(f"{label}: missing qc.json")
            continue
        verdict = ep_qc.get("verdict")
        if verdict not in QC_PASS_VERDICTS:
            qc_skipped.append(f"{label} [{verdict}]")
            continue
        start_frame = max(0, int(ep_qc.get("start_frame") or 0))

        # Load trajectory data
        traj_path = ep_path / "CameraTrajectoryTransformed.txt"
        if not traj_path.exists():
            print(f"ERROR: {label} - no trajectory file found (run SLAM + repair + transform first)")
            stream_failures.append(f"{label}: no CameraTrajectoryTransformed.txt")
            continue
        timestamps, poses, positions, rotvecs = load_trajectory_data(str(traj_path))

        # Load gripper data
        gripper_path = ep_path / "gripper_distances.txt"
        if not gripper_path.exists():
            print(f"ERROR: {label} - no gripper file found (run gripper estimation first)")
            stream_failures.append(f"{label}: no gripper_distances.txt")
            continue
        gripper_distances = load_gripper_data(str(gripper_path))

        image_files = _color_image_files(ep_path)

        # Stream-count assertion: upstream (decode / trajectory repair / gripper
        # estimation) guarantees all three streams have exactly the frame count.
        # A mismatch means an upstream step is stale or incomplete — fail loudly
        # instead of silently truncating a long episode to the shortest stream.
        if not (len(timestamps) == len(gripper_distances) == len(image_files)):
            msg = (f"stream counts differ: traj={len(timestamps)} "
                   f"gripper={len(gripper_distances)} images={len(image_files)} — "
                   "rerun the upstream steps for this episode")
            print(f"ERROR: {label}: {msg}")
            stream_failures.append(f"{label}: {msg}")
            continue
        min_length = len(timestamps)

        # Manual start frame (human review): slice every stream from the
        # chosen frame (0 for normal episodes). Applies BEFORE the NaN
        # fallback and the quasi-static trim, which then judge relative to
        # the new effective start.
        if start_frame > 0:
            if start_frame >= min_length:
                msg = f"start_frame {start_frame} >= episode length {min_length}"
                print(f"ERROR: {label}: {msg}")
                stream_failures.append(f"{label}: {msg}")
                continue
            timestamps = timestamps[start_frame:]
            positions = positions[start_frame:]
            rotvecs = rotvecs[start_frame:]
            gripper_distances = gripper_distances[start_frame:]
            image_files = image_files[start_frame:]
            min_length -= start_frame
            manual_start_frames[label] = start_frame
            print(f"Sliced to manual start_frame={start_frame}; {min_length} remain")

        # Leading-NaN head trim (user, 2026-10-16): episodes whose trajectory
        # has fewer valid poses than frames are initialization-head gaps
        # (corpus-verified: ORB-SLAM3 never drops frames after init). Cut the
        # leading NaN rows from every stream and record the cut; a NaN AFTER
        # the first valid row would be a mid-episode gap — that episode fails
        # (never silently shipped as NaN actions).
        finite = np.isfinite(positions).all(axis=1)
        first_valid = int(np.argmax(finite)) if finite.any() else min_length
        if first_valid > 0 or not finite.any():
            if not finite.any() or not finite[first_valid:].all():
                msg = ("all-NaN trajectory" if not finite.any()
                       else f"NaN rows after first valid frame (mid-gap at "
                            f"{int(np.argmax(~finite[first_valid:])) + first_valid})")
                print(f"ERROR: {label}: {msg}")
                stream_failures.append(f"{label}: {msg}")
                continue
            timestamps = timestamps[first_valid:]
            positions = positions[first_valid:]
            rotvecs = rotvecs[first_valid:]
            gripper_distances = gripper_distances[first_valid:]
            image_files = image_files[first_valid:]
            min_length -= first_valid
            nan_head_trims[label] = first_valid
            print(f"Trimmed {first_valid} leading-NaN frame(s) (init-head gap); "
                  f"{min_length} remain")

        print(f"Episode length: {min_length} frames")

        # Trim the quasi-static recording head (pause before the hand starts
        # moving); keeps every stream aligned by slicing off the same frames.
        trim = find_start_trim_count(positions[:min_length]) if trim_start else 0
        if trim > 0:
            timestamps = timestamps[trim:]
            positions = positions[trim:]
            rotvecs = rotvecs[trim:]
            gripper_distances = gripper_distances[trim:]
            image_files = image_files[trim:]
            min_length -= trim
            start_trims[label] = trim
            print(f"Trimmed {trim} quasi-static head frame(s); {min_length} remain")

        print(f"Loading {min_length} images with {num_workers} worker(s)...")

        images = load_images_parallel(image_files[:min_length], num_workers)

        print("Adding frames to dataset...")

        # Add frames to dataset
        for i in range(min_length):
            image = images[i]

            # 当前帧 EE 绝对位姿+夹爪(与图像/action 同帧对齐)。
            observation_ee = np.array([
                positions[i, 0],  # x
                positions[i, 1],  # y
                positions[i, 2],  # z
                rotvecs[i, 0],    # wx
                rotvecs[i, 1],    # wy
                rotvecs[i, 2],    # wz
                gripper_distances[i],  # gripper_pos
            ], dtype=np.float32)

            # Action (next state)
            # For the last frame, we repeat the last state
            next_idx = min(i + 1, min_length - 1)
            action = np.array([
                positions[next_idx, 0],  # x
                positions[next_idx, 1],  # y
                positions[next_idx, 2],  # z
                rotvecs[next_idx, 0],    # wx
                rotvecs[next_idx, 1],    # wy
                rotvecs[next_idx, 2],    # wz
                gripper_distances[next_idx],  # gripper_pos
            ], dtype=np.float32)

            # Build frame
            frame = {
                "observation.images.camera": image,
                "observation.ee": observation_ee,
                "action": action,
                "task": single_task,
            }

            dataset.add_frame(frame)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{min_length} frames")

        # Save episode
        dataset.save_episode()
        print(f"Episode {label} saved")

        norm_key = f"{session_name}__{ep_path.name}" if is_multi_session else ep_path.name
        norm_manifest = _copy_gripper_normalization(ep_path, dataset.root, norm_key)
        if norm_manifest is None:
            normalization_missing.append(label)
            print(f"Warning: no {GRIPPER_NORMALIZATION_MANIFEST} for {label}")
        else:
            normalization_records.append((label, norm_manifest))

        # Copy camera info files. Namespace by session in multi-session mode so
        # episode_001 from session A doesn't clobber episode_001 from session B.
        camera_info_files = sorted(ep_path.glob("camera_info_*.json"))
        if camera_info_files:
            cam_key = f"{session_name}__{ep_path.name}" if is_multi_session else ep_path.name
            cam_dir = dataset.root / "meta" / "camera_info" / cam_key
            cam_dir.mkdir(parents=True, exist_ok=True)
            for cif in camera_info_files:
                shutil.copy2(cif, cam_dir / cif.name)
            print(f"Copied {len(camera_info_files)} camera info file(s) to {cam_dir}")

    print(f"\nDataset created with {dataset.num_episodes} episode(s)")
    print(f"Dataset features: {list(dataset.features.keys())}")
    if qc_skipped:
        print(f"\nQC skipped {len(qc_skipped)} episode(s) (not failures):")
        for k in qc_skipped[:20]:
            print(f"  - {k}")
        if len(qc_skipped) > 20:
            print(f"  ... and {len(qc_skipped) - 20} more")
    if stream_failures:
        print(f"\nERROR: {len(stream_failures)} episode(s) skipped with stream problems:")
        for f in stream_failures:
            print(f"  - {f}")
        dataset.finalize()
        raise SystemExit(
            f"Conversion finished with {len(stream_failures)} stream-count failure(s); exiting non-zero.")
    summary_path = _write_gripper_normalization_summary(
        dataset.root, normalization_records, normalization_missing
    )
    if trim_start:
        trim_summary = {
            "rule": {
                "anchor_gate_mm": TRIM_GATE_MM,
                "step_min_mm": TRIM_STEP_MIN_MM,
                "angle_max_deg": TRIM_ANGLE_MAX_DEG,
                "seed_window_frames": TRIM_SEED_WINDOW,
                "max_fraction": TRIM_MAX_FRACTION,
            },
            "episodes_trimmed": len(start_trims),
            "start_trim_frames": start_trims,
            "nan_head_trim": {
                "episodes_trimmed": len(nan_head_trims),
                "frames_cut": nan_head_trims,
                "note": "initialization-head NaN rows cut before conversion",
            },
            "manual_start_frames": manual_start_frames,
            "qc_skipped_episodes": len(qc_skipped),
        }
        trim_path = dataset.root / "meta" / "start_trim.json"
        with trim_path.open("w") as f:
            json.dump(trim_summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Start-trim metadata: {trim_path} ({len(start_trims)} episode(s) trimmed)")
    print(f"Gripper normalization metadata: {summary_path}")

    dataset.finalize()

    # Push to hub if requested
    if push_to_hub:
        print("Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub()
        print("Dataset pushed successfully!")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert SROI data to LeRobot dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to SROI data: a single episode dir, a session dir (containing episode_*), "
             "or a parent of session dirs (multi-session mode — all sessions get unified into one dataset)."
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True,
        help="Repository ID for the dataset (e.g., 'username/dataset_name')"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30,
        help="Frames per second for the dataset (default: 30)"
    )
    parser.add_argument(
        "--root", 
        type=str, 
        default=None,
        help="Root directory for dataset storage (default: current directory)"
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true",
        help="Push dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="End-effector manipulation task",
        help="Task description for the dataset"
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Comma-separated list of episode subdirectory names to convert, or 'all' (default: all)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for image loading (default: 4, use 1 for sequential)"
    )
    parser.add_argument(
        "--no_trim_start",
        action="store_true",
        help="Disable trimming of the quasi-static recording head (default: trim on; "
             "counts recorded in meta/start_trim.json)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Validate data path
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")

    # Create dataset
    dataset = create_lerobot_dataset(
        data_path=args.data_path,
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        push_to_hub=args.push_to_hub,
        single_task=args.task,
        episodes=args.episodes,
        num_workers=args.num_workers,
        trim_start=not args.no_trim_start,
    )
    
    print("Conversion completed successfully!")
    return dataset


if __name__ == "__main__":
    main()
