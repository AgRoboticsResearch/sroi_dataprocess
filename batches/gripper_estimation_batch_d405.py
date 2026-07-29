#!/usr/bin/env python3
"""
Batch Gripper Estimation using AprilTags (RealSense D405 defaults).

D405-specific version with tuned parameters:
  gripper_roi=200, nominal_diag=42, threshold=15

Usage:
    python gripper_estimation_batch_d405.py -i test_output/
    python gripper_estimation_batch_d405.py -i test_output/ --skip-existing
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2 as cv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import glob
import copy
from pupil_apriltags import Detector
from tqdm import tqdm
from scipy.signal import medfilt

import numpy as np

from gripper_normalization import (
    MANIFEST_NAME,
    ROBUST_EXTREMUM_MIN_SAMPLES,
    ROBUST_EXTREMUM_THRESHOLD,
    build_manifest,
    write_manifest,
)


def check_tag_valid(tag, nominal_diag_distance=50, threshold=10):
    """Check if detected tag is valid based on diagonal distance."""
    diag_distance_1 = np.linalg.norm(tag.corners[0] - tag.corners[2])
    diag_distance_2 = np.linalg.norm(tag.corners[1] - tag.corners[3])
    if abs(diag_distance_1 - nominal_diag_distance) > threshold or abs(diag_distance_2 - nominal_diag_distance) > threshold:
        return False
    else:
        return True


def detect_april_tag(image, gripper_roi, at_detector, left_gripper_tag_id, right_gripper_tag_id, nominal_diag_distance, threshold, verbose=False):
    """Detect AprilTags and compute gripper distance."""
    gripper_distance_current = None
    left_gripper_tag = None
    right_gripper_tag = None

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray[0:gripper_roi, :] = 0
    ret = at_detector.detect(gray)
    if verbose:
        print("detected tag: ", ret)
    for res in ret:
        if res.tag_id == left_gripper_tag_id:
            if check_tag_valid(res, nominal_diag_distance=nominal_diag_distance, threshold=threshold):
                left_gripper_tag = res
        elif res.tag_id == right_gripper_tag_id:
            if check_tag_valid(res, nominal_diag_distance=nominal_diag_distance, threshold=threshold):
                right_gripper_tag = res

    if left_gripper_tag is not None and right_gripper_tag is not None:
        gripper_distance = abs(right_gripper_tag.center[0] - left_gripper_tag.center[0])
        return gripper_distance
    else:
        return None


def find_distances_max_min(distances, threshold=0.05, threshold_num=10):
    """Find valid max and min distances from the data."""
    distances_copy = copy.deepcopy(distances)
    ret_max_distance, ret_min_distance = None, None
    for _ in range(len(distances)):
        max_dist_id = np.argmax(distances)
        max_distance = distances[max_dist_id]
        data_in_range = distances[(distances > max_distance * (1 - threshold)) & (distances < max_distance * (1 + threshold))]
        if len(data_in_range) >= threshold_num:
            ret_max_distance = max_distance
            break
        distances[max_dist_id] = -1
    if ret_max_distance is None:
        ret_max_distance = np.max(distances_copy)

    distances = distances_copy
    for _ in range(len(distances)):
        min_dist_id = np.argmin(distances)
        min_distance = distances[min_dist_id]
        data_in_range = distances[(distances > min_distance * (1 - threshold)) & (distances < min_distance * (1 + threshold))]
        if len(data_in_range) >= threshold_num:
            ret_min_distance = min_distance
            break
        distances[min_dist_id] = np.inf
    if ret_min_distance is None:
        ret_min_distance = np.min(distances_copy)

    return ret_max_distance, ret_min_distance


def save_gripper_distance_plot(gripper_distances_normalized, output_path):
    """
    Create and save a time vs gripper distance plot.

    Args:
        gripper_distances_normalized (np.array): Normalized gripper distances
        output_path (Path): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    time_indices = np.arange(len(gripper_distances_normalized))
    plt.plot(time_indices, gripper_distances_normalized, 'b-', linewidth=1.5, label='Normalized Gripper Distance')
    plt.xlabel('Frame Index (Time)')
    plt.ylabel('Normalized Gripper Distance')
    plt.title('Time vs Gripper Distance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_gripper_range_plot(pooled_raw_distances, min_distance, max_distance, output_path):
    """Save a histogram of raw (pre-normalization) gripper distances for the whole
    batch processed in this run, with the selected min/max marked."""
    plt.figure(figsize=(10, 6))
    plt.hist(pooled_raw_distances, bins=100, color='steelblue', alpha=0.8)
    plt.axvline(min_distance, color='red', linestyle='--', linewidth=1.5,
                label=f'min = {min_distance:.2f}')
    plt.axvline(max_distance, color='green', linestyle='--', linewidth=1.5,
                label=f'max = {max_distance:.2f}')
    plt.xlabel('Raw Gripper Distance (px)')
    plt.ylabel('Frame Count')
    plt.title(f'Gripper Distance Distribution ({len(pooled_raw_distances)} frames)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved gripper distance distribution plot to", output_path)


def detect_folder(data_folder: Path, at_detector, args) -> dict:
    """
    Run AprilTag detection for a single folder and forward-fill gaps.

    Returns a dict with "status", "frame_count", "valid_detections", and
    (on success) "raw_distances" (np.array of raw, unnormalized pixel distances).
    """
    result = {
        "folder": str(data_folder),
        "status": "success",
        "frame_count": 0,
        "valid_detections": 0,
    }

    # Check for color images (PNG or JPEG)
    color_images = sorted(data_folder.glob("color_*.png")) or sorted(data_folder.glob("color_*.jpg"))
    if not color_images:
        result["status"] = "skipped"
        result["error"] = "No color images found"
        return result

    # Check if already processed
    output_file = data_folder / "gripper_distances.txt"
    if (args.skip_existing and output_file.exists()
            and (data_folder / MANIFEST_NAME).exists()):
        result["status"] = "skipped"
        result["error"] = "Already exists"
        return result

    try:
        image_nums = len(color_images)
        result["frame_count"] = image_nums

        gripper_distances = []
        for color_image_path in tqdm(color_images, desc=f"  Frames", leave=False):
            color_image = cv.imread(str(color_image_path))
            if color_image is None:
                gripper_distances.append(None)
                continue
            gripper_distance = detect_april_tag(
                color_image, args.gripper_roi, at_detector,
                args.left_tag_id, args.right_tag_id,
                args.nominal_diag, args.threshold,
                verbose=args.verbose
            )
            gripper_distances.append(gripper_distance)

        gripper_distance_valid = [x for x in gripper_distances if x is not None]
        result["valid_detections"] = len(gripper_distance_valid)

        if not gripper_distance_valid:
            result["status"] = "error"
            result["error"] = "No valid AprilTag detections"
            return result

        # Fill gaps: leading missing frames take the first valid detection
        # (nearest-in-time), interior/trailing gaps forward-fill from the
        # previous frame. Seeding the head with the episode max (old behavior)
        # amplified a single spurious detection into a long constant run that
        # defeated both the median filter and the clustered-extremum check.
        first_valid = gripper_distance_valid[0]
        gripper_distances_processed = []
        prev = first_valid
        for x in gripper_distances:
            if x is None:
                gripper_distances_processed.append(prev)
            else:
                gripper_distances_processed.append(x)
                prev = x

        raw = np.asarray(gripper_distances_processed, dtype=float)
        if args.median_filter > 1:
            raw = medfilt(raw, kernel_size=args.median_filter)
        result["raw_distances"] = raw

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def normalize_and_save_folder(
    data_folder: Path, raw_distances, min_distance, max_distance,
    normalization_manifest,
) -> None:
    """Scale raw distances and write outputs plus normalization provenance."""
    gripper_distances_normalized = (raw_distances - min_distance) / (max_distance - min_distance)
    gripper_distances_normalized[gripper_distances_normalized > 1] = 1
    gripper_distances_normalized[gripper_distances_normalized < 0] = 0

    output_file = data_folder / "gripper_distances.txt"
    np.savetxt(output_file, gripper_distances_normalized)
    write_manifest(data_folder, normalization_manifest)

    plot_path = data_folder / "gripper_distances.jpg"
    save_gripper_distance_plot(gripper_distances_normalized, plot_path)


def find_data_folders(input_dir: Path) -> list:
    """Find all folders that contain color images."""
    data_folders = []
    for item in sorted(input_dir.iterdir()):
        if item.is_dir() and item.name != "segment_bags":
            # Check if folder has color images
            color_images = list(item.glob("color_*.png")) or list(item.glob("color_*.jpg"))
            if color_images:
                data_folders.append(item)
    return data_folders


def main():
    parser = argparse.ArgumentParser(
        description="Batch process gripper estimation using AprilTags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all folders in test_output/
    python gripper_estimation_batch.py -i test_output/

    # Skip already processed folders
    python gripper_estimation_batch.py -i test_output/ --skip-existing

    # Custom AprilTag parameters
    python gripper_estimation_batch.py -i test_output/ --left-tag-id 0 --right-tag-id 15
        """
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory containing data folders")
    parser.add_argument("--skip-existing", action="store_true", help="Skip folders that already have gripper_distances.txt")
    parser.add_argument("--gripper-roi", type=int, default=200, help="ROI for gripper detection (default: 200)")
    parser.add_argument("--left-tag-id", type=int, default=0, help="AprilTag ID for left gripper (default: 0)")
    parser.add_argument("--right-tag-id", type=int, default=15, help="AprilTag ID for right gripper (default: 15)")
    parser.add_argument("--nominal-diag", type=int, default=42, help="Nominal diagonal distance for tag validation (default: 42)")
    parser.add_argument("--threshold", type=int, default=15, help="Threshold for tag validation (default: 15)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed detection info")
    parser.add_argument("--log", help="Path to save processing log as JSON")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON file with manual 'gripper_min_distance'/'gripper_max_distance' "
             "overrides (both required to take effect). Skips auto-detection of the range.",
    )
    parser.add_argument(
        "--median-filter",
        type=int,
        default=1,
        help="Median filter window size applied to raw gripper distances before "
             "computing min/max and normalizing, to smooth AprilTag detection noise "
             "(must be odd, e.g. 3; default 1 = no filtering)",
    )

    args = parser.parse_args()
    if args.median_filter < 1 or args.median_filter % 2 == 0:
        parser.error("--median-filter must be an odd number >= 1")

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    manual_min = manual_max = None
    range_cfg = {}
    if args.config:
        with open(args.config) as f:
            range_cfg = json.load(f)
        manual_min = range_cfg.get("gripper_min_distance")
        manual_max = range_cfg.get("gripper_max_distance")

    # Find data folders
    data_folders = find_data_folders(input_dir)
    if not data_folders:
        print(f"No data folders with color images found in {input_dir}")
        sys.exit(1)

    if args.skip_existing:
        complete = [
            folder for folder in data_folders
            if (folder / "gripper_distances.txt").exists()
            and (folder / MANIFEST_NAME).exists()
        ]
        if 0 < len(complete) < len(data_folders):
            print(
                "Partial existing batch detected; reprocessing every folder so "
                "all outputs share one normalization range and root manifest."
            )
            args.skip_existing = False

    print(f"Found {len(data_folders)} data folder(s)")

    # Create AprilTag detector
    at_detector = Detector(
        families="tag16h5",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    # Phase 1: detect raw distances for every folder
    all_results = []
    start_time = datetime.now()

    for folder in tqdm(data_folders, desc="Detecting folders"):
        print(f"\n{folder.name}:")
        result = detect_folder(folder, at_detector, args)
        all_results.append(result)

        if result["status"] == "success":
            print(f"  Detected: {result['frame_count']} frames, {result['valid_detections']} valid detections")
        elif result["status"] == "skipped":
            print(f"  Skipped: {result.get('error', '')}")
        else:
            print(f"  Error: {result.get('error', '')}")

    # Phase 2: normalize with one shared min/max (manual override, else pooled
    # clustered-extremum over every successfully-detected folder in this run)
    detected = [r for r in all_results if r["status"] == "success"]
    pooled = np.concatenate([r["raw_distances"] for r in detected]) if detected else None
    if manual_min is not None and manual_max is not None:
        min_distance, max_distance = manual_min, manual_max
        source_type = "manual_config"
        print(f"\nUsing manual gripper range from {args.config}: min={min_distance}, max={max_distance}")
    elif detected:
        source_type = "pooled_robust_extrema"
        max_distance, min_distance = find_distances_max_min(
            pooled,
            threshold=ROBUST_EXTREMUM_THRESHOLD,
            threshold_num=ROBUST_EXTREMUM_MIN_SAMPLES,
        )
        print(
            f"\nComputed gripper range from {len(detected)} folder(s): "
            f"min={min_distance}, max={max_distance}"
        )
    else:
        min_distance = max_distance = None

    if min_distance is not None:
        manifest_config = {
            "tag_family": "tag16h5",
            "left_gripper_tag_id": args.left_tag_id,
            "right_gripper_tag_id": args.right_tag_id,
            "gripper_roi": args.gripper_roi,
            "nominal_diag_distance": args.nominal_diag,
            "threshold": args.threshold,
            "gripper_range_source": range_cfg.get("gripper_range_source"),
        }
        normalization_manifest = build_manifest(
            min_distance=min_distance,
            max_distance=max_distance,
            source_type=source_type,
            config_path=args.config,
            config=manifest_config,
            median_filter_size=args.median_filter,
            episode_count=len(detected),
            frame_count=len(pooled),
        )
        dist_plot_path = input_dir / "gripper_distance_distribution.png"
        save_gripper_range_plot(pooled, min_distance, max_distance, dist_plot_path)
        write_manifest(input_dir, normalization_manifest)

        for result in detected:
            folder = Path(result["folder"])
            normalize_and_save_folder(
                folder, result.pop("raw_distances"), min_distance, max_distance,
                normalization_manifest,
            )
            result["min_distance"] = min_distance
            result["max_distance"] = max_distance

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    success_count = len(detected)
    skipped_count = sum(1 for r in all_results if r["status"] == "skipped")
    error_count = sum(1 for r in all_results if r["status"] == "error")
    total_frames = sum(r.get("frame_count", 0) for r in detected)

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Total folders: {len(data_folders)}")
    print(f"  Successful: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Elapsed time: {elapsed:.2f}s")

    # Save log
    if args.log:
        log_data = {
            "processed_at": datetime.now().isoformat(),
            "input_directory": str(input_dir),
            "total_folders": len(data_folders),
            "successful": success_count,
            "skipped": skipped_count,
            "errors": error_count,
            "total_frames": total_frames,
            "elapsed_seconds": elapsed,
            "results": all_results,
        }
        with open(args.log, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"  Log saved to: {args.log}")


if __name__ == "__main__":
    main()
