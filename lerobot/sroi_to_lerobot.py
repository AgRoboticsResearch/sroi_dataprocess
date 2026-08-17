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


def _load_qc_csv(qc_csv: str | Path) -> dict[str, dict[str, str]]:
    """Read qc.csv into {session_name: {episode_name: category}}.

    qc.csv columns (output of visualization/qc.py): session, episode, category, ...
    """
    qc_path = Path(qc_csv)
    if not qc_path.exists():
        raise FileNotFoundError(f"QC CSV not found: {qc_path}")
    out: dict[str, dict[str, str]] = {}
    with qc_path.open(newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["session"], {})[row["episode"]] = row["category"]
    return out


def _collect_episodes(
    data_path: Path,
    qc_csv: str | Path | None,
    qc_categories: str,
    episodes: str,
) -> tuple[list[tuple[str, Path]], bool]:
    """Auto-detect single-episode / single-session / multi-session mode and return
    the work list.

    Returns (candidates, is_multi_session) where candidates is a list of
    (session_name, episode_dir) tuples in deterministic sorted order. QC filter
    is applied per-session when qc_csv is provided.
    """
    if qc_csv is not None and episodes != "all":
        raise ValueError("--qc_csv and --episodes are mutually exclusive.")

    # Mode 1: single episode directory (backward compat)
    if (data_path / "CameraTrajectoryTransformed.txt").exists():
        if qc_csv is not None:
            print(f"Warning: --qc_csv ignored; {data_path.name} is a single episode.")
        return [(data_path.parent.name, data_path)], False

    qc_data = _load_qc_csv(qc_csv) if qc_csv else None
    keep = {c.strip() for c in qc_categories.split(",") if c.strip()} if qc_csv else None

    # Mode 2: single session (episode_* dirs directly under data_path)
    direct = sorted([d for d in data_path.iterdir() if _is_episode_dir(d)])
    if direct:
        kept = _apply_session_filter(data_path.name, direct, qc_data, keep, episodes)
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
        kept = _apply_session_filter(s.name, eps, qc_data, keep, "all")
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
    qc_csv: str | None = None,
    qc_categories: str = "ok",
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
            "all" converts every episode. Mutually exclusive with --qc_csv.
            Ambiguous in multi-session mode (use --qc_csv instead).
        num_workers: Parallel workers for image loading
        qc_csv: Optional path to a qc.csv from visualization/qc.py. Episodes are
            filtered per session to those whose category is in qc_categories.
        qc_categories: Comma-separated category names to keep when qc_csv is set
            (default: "ok"). E.g. "ok,flat_gripper" to also keep no-actuation runs.
    """
    data_path = Path(data_path)
    candidates, is_multi_session = _collect_episodes(data_path, qc_csv, qc_categories, episodes)
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

    # Process each episode
    for ep_idx, (session_name, ep_path) in enumerate(candidates):
        label = f"{session_name}/{ep_path.name}" if is_multi_session else ep_path.name
        print(f"\n--- Processing episode {ep_idx + 1}/{len(candidates)}: {label} ---")

        # Load trajectory data
        traj_path = ep_path / "CameraTrajectoryTransformed.txt"
        if not traj_path.exists():
            print(f"Warning: Skipping {label} - no trajectory file found")
            continue
        timestamps, poses, positions, rotvecs = load_trajectory_data(str(traj_path))

        # Load gripper data
        gripper_path = ep_path / "gripper_distances.txt"
        if not gripper_path.exists():
            print(f"Warning: Skipping {label} - no gripper file found")
            continue
        gripper_distances = load_gripper_data(str(gripper_path))

        # Ensure all data has the same length
        min_length = min(len(timestamps), len(gripper_distances))
        timestamps = timestamps[:min_length]
        positions = positions[:min_length]
        rotvecs = rotvecs[:min_length]
        gripper_distances = gripper_distances[:min_length]

        print(f"Episode length: {min_length} frames")

        # Check for image files
        image_files = _color_image_files(ep_path)
        if len(image_files) < min_length:
            print(f"Warning: Only {len(image_files)} images found, but {min_length} frames needed")
            min_length = min(min_length, len(image_files))
            # Truncate other arrays
            timestamps = timestamps[:min_length]
            positions = positions[:min_length]
            rotvecs = rotvecs[:min_length]
            gripper_distances = gripper_distances[:min_length]

        print(f"Loading {min_length} images with {num_workers} worker(s)...")

        images = load_images_parallel(image_files[:min_length], num_workers)

        print("Adding frames to dataset...")

        # Add frames to dataset
        for i in range(min_length):
            image = images[i]

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
    summary_path = _write_gripper_normalization_summary(
        dataset.root, normalization_records, normalization_missing
    )
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
        "--qc_csv",
        type=str,
        default=None,
        help="Path to a qc.csv from visualization/qc.py. Filters episodes by category for each "
             "session under --data_path. Mutually exclusive with --episodes."
    )
    parser.add_argument(
        "--qc_categories",
        type=str,
        default="ok",
        help="Comma-separated QC categories to keep when --qc_csv is set (default: 'ok'). "
             "Example: 'ok,flat_gripper'."
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
        qc_csv=args.qc_csv,
        qc_categories=args.qc_categories,
    )
    
    print("Conversion completed successfully!")
    return dataset


if __name__ == "__main__":
    main()
