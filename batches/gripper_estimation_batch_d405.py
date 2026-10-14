#!/usr/bin/env python3
"""
Batch Gripper Estimation using AprilTags (RealSense D405 defaults).

D405-specific version with tuned parameters (defaults match configs/sroi_v2_d405.json
when passed via --config-file; flag defaults kept for backward compatibility):
  gripper_roi=200, nominal_diag=42, threshold=15

Detection and series recovery are shared with gripper_estimation_april_tag.py
via gripper_tag_core.py: preprocessing fallback chain, symmetric-jaw
constraint (single-tag mirror + both-tag cross-validation), flat-context
spike removal, and linear gap interpolation instead of legacy forward-fill.

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
from pupil_apriltags import Detector

try:
    from tqdm import tqdm
except ImportError:  # system python3 may lack tqdm; fall back to a plain iterable
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else ()
from scipy.signal import medfilt

import numpy as np

from gripper_normalization import (
    MANIFEST_NAME,
    ROBUST_EXTREMUM_MIN_SAMPLES,
    ROBUST_EXTREMUM_THRESHOLD,
    build_manifest,
    write_manifest,
)
from gripper_tag_core import (
    DEFAULT_PREPROCESS_CHAIN,
    DEFAULT_SPIKE_THRESHOLD,
    DEFAULT_SYMMETRY_TOL_PX,
    detect_episode,
    finalize_series,
    find_distances_max_min,
    run_center_fallback,
)

DETECTION_STATS_NAME = "gripper_detection_stats.json"


def save_gripper_distance_plot(gripper_distances_normalized, output_path):
    """
    Create and save a time vs normalized gripper distance plot.

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
    Run AprilTag detection (gripper_tag_core) for a single folder.

    Returns a dict with "status", "frame_count", "valid_detections", and
    (on success) "detection" (gripper_tag_core.EpisodeDetection) plus
    "center" (episode symmetry center or None).
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
        result["frame_count"] = len(color_images)
        det = detect_episode(
            color_images, at_detector,
            args.left_tag_id, args.right_tag_id,
            args.nominal_diag, args.threshold, args.gripper_roi,
            gripper_roi_bottom=args.gripper_roi_bottom,
            left_window=tuple(args.left_tag_x_range) if getattr(args, "left_tag_x_range", None) else None,
            right_window=tuple(args.right_tag_x_range) if getattr(args, "right_tag_x_range", None) else None,
            preprocess_chain=() if args.no_preprocess else DEFAULT_PREPROCESS_CHAIN,
        )
        n_both = sum(1 for l, r in zip(det.left_xs, det.right_xs)
                     if l is not None and r is not None)
        n_left = sum(1 for l in det.left_xs if l is not None)
        n_right = sum(1 for r in det.right_xs if r is not None)
        result["valid_detections"] = max(n_left, n_right)
        if max(n_left, n_right) == 0:
            result["status"] = "error"
            result["error"] = "No valid AprilTag detections"
            return result
        result["detection"] = det
        print(f"  both={n_both}  left_only={n_left - n_both}  right_only={n_right - n_both}"
              f"  preprocess_fallback={det.preprocess_frames}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def normalize_and_save_folder(
    data_folder: Path, raw_distances, min_distance, max_distance,
    normalization_manifest, detection_stats=None,
) -> None:
    """Scale raw distances and write outputs plus normalization provenance."""
    gripper_distances_normalized = (raw_distances - min_distance) / (max_distance - min_distance)
    gripper_distances_normalized[gripper_distances_normalized > 1] = 1
    gripper_distances_normalized[gripper_distances_normalized < 0] = 0

    output_file = data_folder / "gripper_distances.txt"
    np.savetxt(output_file, gripper_distances_normalized)
    write_manifest(data_folder, normalization_manifest)
    if detection_stats is not None:
        with open(data_folder / DETECTION_STATS_NAME, 'w') as f:
            json.dump(detection_stats, f, indent=2)

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
    parser.add_argument("--gripper-roi-bottom", type=int, default=None, help="Bottom ROI row (default: none)")
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
    parser.add_argument(
        "--interp",
        choices=("linear", "forward_fill"),
        default="linear",
        help="Gap fill mode: linear interpolation with nearest edges (default) or "
        "the legacy forward-fill (for A/B comparison)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable the strong-light preprocessing fallback chain",
    )
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="Disable the symmetric-jaw constraint (both-tag frames only, no "
        "cross-validation, no single-tag mirror estimates)",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=DEFAULT_SPIKE_THRESHOLD,
        help=f"Flat-context spike removal threshold as a fraction of the gripper "
             f"range (default {DEFAULT_SPIKE_THRESHOLD}); 0 disables removal",
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

    # Phase 1: detect raw tag positions for every folder
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

    # Phase 2: one shared min/max over the symmetry-constrained valid samples
    # (manual override, else pooled clustered-extremum), then per-folder
    # spike removal + gap interpolation + normalization
    detected = [r for r in all_results if r["status"] == "success"]
    run_center = run_center_fallback([r["detection"] for r in detected])
    pooled_parts = []
    for r in detected:
        center = r["detection"].center_px
        if center is None:
            center = run_center
        if args.no_symmetry:
            center = None
        indices, values, flags = r["detection"].series(center, tol=DEFAULT_SYMMETRY_TOL_PX)
        if values:
            pooled_parts.append(np.asarray(values, dtype=float))

    pooled = np.concatenate(pooled_parts) if pooled_parts else None
    if manual_min is not None and manual_max is not None:
        min_distance, max_distance = manual_min, manual_max
        source_type = "manual_config"
        print(f"\nUsing manual gripper range from {args.config}: min={min_distance}, max={max_distance}")
    elif pooled is not None and len(pooled_parts):
        source_type = "pooled_robust_extrema"
        max_distance, min_distance = find_distances_max_min(
            pooled,
            threshold=ROBUST_EXTREMUM_THRESHOLD,
            threshold_num=ROBUST_EXTREMUM_MIN_SAMPLES,
        )
        print(
            f"\nComputed gripper range from {len(pooled_parts)} folder(s): "
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
            "gripper_roi_bottom": args.gripper_roi_bottom,
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
            episode_count=len(pooled_parts),
            frame_count=len(pooled),
            interp_mode=args.interp,
            preprocess_chain=[] if args.no_preprocess else list(DEFAULT_PREPROCESS_CHAIN),
            symmetry_enabled=not args.no_symmetry,
            symmetry_tol_px=DEFAULT_SYMMETRY_TOL_PX,
            spike_threshold=args.spike_threshold,
        )
        dist_plot_path = input_dir / "gripper_distance_distribution.png"
        save_gripper_range_plot(pooled, min_distance, max_distance, dist_plot_path)
        write_manifest(input_dir, normalization_manifest)

        median_filter_fn = None
        if args.median_filter > 1:
            median_filter_fn = lambda a: medfilt(a, kernel_size=args.median_filter)

        for result in detected:
            folder = Path(result["folder"])
            det = result.pop("detection")
            center = det.center_px
            if center is None:
                center = run_center
            if args.no_symmetry:
                center = None
            raw, stats = finalize_series(
                det, min_distance, max_distance,
                center=center, symmetry_tol=DEFAULT_SYMMETRY_TOL_PX,
                interp_mode=args.interp,
                spike_threshold=args.spike_threshold,
                median_filter_fn=median_filter_fn,
            )
            if raw is None:
                result["status"] = "error"
                result["error"] = "No valid samples after cleanup"
                continue
            stats["interp_mode"] = args.interp
            normalize_and_save_folder(
                folder, raw, min_distance, max_distance,
                normalization_manifest, detection_stats=stats,
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
