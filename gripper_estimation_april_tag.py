"""
Gripper Distance Estimation via AprilTag Detection

Detects left/right gripper AprilTags in image sequences to estimate
gripper opening distance over time. The pixel distance between tag centers
is normalized to [0, 1] and saved to disk along with a plot.

Detection and series recovery live in gripper_tag_core.py and use the rig's
hardware constraints; see that module's docstring for the full rationale:

  * preprocessing fallback chain (raw -> CLAHE -> equalize_hist) recovers
    tags in strong-light episodes at no cost on good frames;
  * the symmetric-jaw constraint (left/right tags mirror around a fixed
    image center, measured stable to <0.5 px) both infers the distance when
    only one tag is visible and cross-validates both-tag frames;
  * flat-context spike removal kills isolated 1-2 frame misdetections that
    pass the tag-size check (real fast motion keeps its flat-context shape
    and survives);
  * gaps (missing, cross-validation-rejected, spikes) are filled by linear
    interpolation with nearest-value edges instead of the legacy
    forward-fill, which produced staircases during motion-blur dropouts
    (--interp forward_fill restores the legacy behavior for A/B checks).

With --recursive, all episodes in one run share a single min/max: the raw
distances from every episode are pooled and passed through the robust
clustered-extremum finder (find_distances_max_min) once, then each episode
is normalized against that shared range. Without --recursive, the "batch" is
just the one episode. To skip auto-detection entirely, set
"gripper_min_distance" / "gripper_max_distance" in the camera config JSON --
both must be present to take effect.

Every run also saves gripper_distance_distribution.png next to the input
path: a histogram of the pooled raw (pre-normalization) distances with the
selected min/max marked, for a quick sanity check of the chosen range.
It also writes gripper_normalization.json at the run root and beside every
gripper_distances.txt so normalization parameters and their source remain
attached to processed data, plus a per-episode gripper_detection_stats.json
(coverage, symmetry-center, both/mirror counts, removed spikes).

Use --median_filter (odd window size, e.g. 3) to smooth per-frame AprilTag
detection noise in the raw distances before min/max is computed and the
episode is normalized. Default is 1 (no filtering).

Supported camera configs live in configs/ as JSON files.
Use --configs to select one (defaults to D405).
"""

import json
import os
import sys
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from pupil_apriltags import Detector
import argparse

try:
    from tqdm import tqdm
except ImportError:  # system python3 may lack tqdm; fall back to a plain iterable
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else ()

from scipy.signal import medfilt

import numpy as np
from pathlib import Path

from gripper_normalization import (
    ROBUST_EXTREMUM_MIN_SAMPLES,
    ROBUST_EXTREMUM_THRESHOLD,
    build_manifest,
    write_manifest,
)
from gripper_tag_core import (
    DEFAULT_PREPROCESS_CHAIN,
    DEFAULT_PRIOR_CENTER_PX,
    DEFAULT_PRIOR_CENTER_TOL_PX,
    DEFAULT_SPIKE_THRESHOLD,
    DEFAULT_SYMMETRY_TOL_PX,
    DEFAULT_TAG_DISTANCE_BOUNDS,
    DEFAULT_CALIB_MIN_PX,
    DEFAULT_CALIB_MAX_PX,
    DEFAULT_ANCHOR_SANITY_PX,
    detect_episode,
    finalize_series,
    find_distances_max_min,
    run_center_fallback,
)

# Default config path relative to this script
DEFAULT_CONFIG = "configs/sroi_v2_d405.json"

DETECTION_STATS_NAME = "gripper_detection_stats.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate gripper distance from AprilTag detections in image sequences."
    )
    parser.add_argument("data_folder", help="Path to the image sequence folder")
    parser.add_argument(
        "--configs",
        default=DEFAULT_CONFIG,
        help=f"Path to camera config JSON (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Interactive calibration mode with GUI sliders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output config path for --calibrate (default: overwrite --configs)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process all episode subdirectories recursively",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame detection diagnostics",
    )
    parser.add_argument(
        "--median_filter",
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
        help="Disable the strong-light preprocessing fallback chain "
             "(raw grayscale only)",
    )
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="Disable the symmetric-jaw constraint: keep only both-tag frames "
             "without cross-validation or single-tag mirror estimates",
    )
    parser.add_argument(
        "--max_interp_gap",
        type=int, default=None,
        help="Interior gaps longer than this many frames stay NaN (classified "
             "as sustained occlusion; default from config or 10). "
             "0 disables the policy (interpolate everything)."
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
        parser.error("--median_filter must be an odd number >= 1")
    return args


def load_config(path):
    """Load and print camera configuration from a JSON file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = json.load(f)
    return cfg


def print_config(cfg, path):
    """Print a summary of the loaded configuration."""
    desc = cfg.get("description", "N/A")
    print(f"Config file : {path}")
    print(f"  description          : {desc}")
    print(f"  gripper_roi          : {cfg['gripper_roi']}")
    print(f"  tag_family           : {cfg['tag_family']}")
    print(f"  left_gripper_tag_id  : {cfg['left_gripper_tag_id']}")
    print(f"  right_gripper_tag_id : {cfg['right_gripper_tag_id']}")
    print(f"  nominal_diag_distance: {cfg.get('nominal_diag_distance', cfg.get('norminal_diag_distance', 'N/A'))}")
    print(f"  threshold            : {cfg['threshold']}")
    manual_min = cfg.get("gripper_min_distance")
    manual_max = cfg.get("gripper_max_distance")
    if manual_min is not None or manual_max is not None:
        print(f"  gripper_min_distance : {manual_min} (manual override)")
        print(f"  gripper_max_distance : {manual_max} (manual override)")


def save_gripper_distance_plot(gripper_distances_normalized, data_folder):
    """Save a time vs normalized gripper distance plot."""
    plt.figure(figsize=(12, 6))
    time_indices = np.arange(len(gripper_distances_normalized))
    plt.plot(time_indices, gripper_distances_normalized, 'b-', linewidth=1.5,
             label='Normalized Gripper Distance')
    plt.xlabel('Frame Index (Time)')
    plt.ylabel('Normalized Gripper Distance')
    plt.title('Time vs Gripper Distance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = data_folder + "gripper_distances.jpg"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved gripper distance plot to", plot_path)


def save_gripper_range_plot(pooled_raw_distances, min_distance, max_distance, output_path):
    """Save a histogram of raw (pre-normalization) gripper distances for the whole
    dataset processed in this run, with the selected min/max marked."""
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


def draw_detections(image, results, left_id, right_id, gripper_roi_top, gripper_roi_bottom):
    """Draw detected tags on image with color-coded outlines and IDs."""
    vis = image.copy()
    # Draw ROI lines
    cv.line(vis, (0, gripper_roi_top), (vis.shape[1], gripper_roi_top), (0, 255, 255), 1)
    cv.line(vis, (0, gripper_roi_bottom), (vis.shape[1], gripper_roi_bottom), (0, 255, 255), 1)
    for tag in results:
        is_left = tag.tag_id == left_id
        is_right = tag.tag_id == right_id
        color = (0, 255, 0) if is_left else (0, 0, 255) if is_right else (128, 128, 128)
        pts = tag.corners.astype(int).reshape(-1, 1, 2)
        cv.polylines(vis, [pts], True, color, 2)
        cx, cy = int(tag.center[0]), int(tag.center[1])
        cv.circle(vis, (cx, cy), 4, color, -1)
        cv.putText(vis, f"id={tag.tag_id}", (cx - 20, cy - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


TAG_FAMILIES = ["tag16h5", "tag25h9", "tag36h11", "tagCircle21h7", "tagCircle49h12",
                "tagCustom48h12", "tagStandard41h12", "tagStandard52h13"]


def find_color_images(data_folder):
    """Sorted color frames of an episode.

    Stereo-only (oak) episodes have no color_* files — their left camera IS
    the color camera — so fall back to left_* frames there.
    """
    images = sorted(glob.glob(os.path.join(data_folder, "color_*.png"))
                    + glob.glob(os.path.join(data_folder, "color_*.jpg")))
    if not images:
        images = sorted(glob.glob(os.path.join(data_folder, "left_*.png"))
                        + glob.glob(os.path.join(data_folder, "left_*.jpg")))
        if images:
            print("No color_* images in", data_folder, "- using left_* frames as color stream")
    return images


def calibrate(data_folder, config_path, output_path):
    """Interactive calibration mode with cv2 GUI."""
    cfg = load_config(config_path)
    print_config(cfg, config_path)

    # Pick a sample image (middle frame for better coverage)
    images = find_color_images(data_folder)
    if not images:
        print("No color images found in", data_folder)
        return
    sample_path = images[len(images) // 2]
    print(f"Using sample image: {sample_path}")

    # Terminal prompts for string/ID values
    current_family = cfg.get("tag_family", "tag16h5")
    print(f"\nAvailable tag families: {TAG_FAMILIES}")
    print(f"Current: {current_family}")
    choice = input("Enter tag family (press Enter to keep current): ").strip()
    tag_family = choice if choice in TAG_FAMILIES else current_family

    current_left = cfg.get("left_gripper_tag_id", 0)
    current_right = cfg.get("right_gripper_tag_id", 15)
    left_id = int(input(f"Left gripper tag ID [{current_left}]: ").strip() or current_left)
    right_id = int(input(f"Right gripper tag ID [{current_right}]: ").strip() or current_right)

    # Read sample image
    image = cv.imread(sample_path)
    h, w = image.shape[:2]

    default_bottom = cfg.get("gripper_roi_bottom", h)

    # Create window and trackbars
    win = "AprilTag Calibration"
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    cv.createTrackbar("top roi", win, cfg["gripper_roi"], h, lambda x: None)
    cv.createTrackbar("bottom roi", win, default_bottom, h, lambda x: None)
    cv.createTrackbar("diag dist", win, cfg.get("nominal_diag_distance", cfg.get("norminal_diag_distance", 50)), 200, lambda x: None)
    cv.createTrackbar("threshold", win, cfg["threshold"], 100, lambda x: None)

    detector = Detector(families=tag_family, nthreads=1, quad_decimate=1.0,
                        quad_sigma=0, refine_edges=1, decode_sharpening=0.25, debug=0)

    print("\nSliders: top roi | bottom roi | diag dist | threshold")
    print("Keys: 's' save config | 'q' quit")
    while True:
        top_roi = cv.getTrackbarPos("top roi", win)
        bottom_roi = cv.getTrackbarPos("bottom roi", win)
        norminal_diag = cv.getTrackbarPos("diag dist", win)
        threshold_val = cv.getTrackbarPos("threshold", win)

        # Detect
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_masked = gray.copy()
        gray_masked[0:top_roi, :] = 0
        gray_masked[bottom_roi:, :] = 0
        results = detector.detect(gray_masked)

        # Draw
        vis = draw_detections(image, results, left_id, right_id, top_roi, bottom_roi)

        # Info overlay
        left_found = any(t.tag_id == left_id for t in results)
        right_found = any(t.tag_id == right_id for t in results)
        ok = left_found and right_found
        y0 = 0
        info_lines = [
            (f"family={tag_family}  left_id={left_id}  right_id={right_id}", (200, 200, 200)),
            (f"top={top_roi}  bottom={bottom_roi}  diag={norminal_diag}  thresh={threshold_val}", (200, 200, 200)),
            (f"Detected: {len(results)} tags", (200, 200, 200)),
            (f"Left(id={left_id}): {'FOUND' if left_found else 'MISSING'}", (0, 255, 0) if left_found else (0, 0, 255)),
            (f"Right(id={right_id}): {'FOUND' if right_found else 'MISSING'}", (0, 255, 0) if right_found else (0, 0, 255)),
        ]
        if ok:
            left_tag = next(t for t in results if t.tag_id == left_id)
            right_tag = next(t for t in results if t.tag_id == right_id)
            dist = abs(right_tag.center[0] - left_tag.center[0])
            info_lines.append((f"Gripper dist: {dist:.1f}px", (255, 255, 0)))

        for text, color in info_lines:
            y0 += 30
            cv.putText(vis, text, (10, y0), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv.imshow(win, vis)
        key = cv.waitKey(100) & 0xFF
        if key == ord('s'):
            new_cfg = {
                "description": cfg.get("description", ""),
                "gripper_roi": top_roi,
                "gripper_roi_bottom": bottom_roi,
                "tag_family": tag_family,
                "left_gripper_tag_id": left_id,
                "right_gripper_tag_id": right_id,
                "nominal_diag_distance": norminal_diag,
                "threshold": threshold_val,
            }
            out = output_path or config_path
            with open(out, 'w') as f:
                json.dump(new_cfg, f, indent=4)
            print(f"Config saved to {out}")
            print(json.dumps(new_cfg, indent=2))
            break
        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv.destroyAllWindows()


def normalize_and_save(
    data_folder, raw_distances, min_distance, max_distance, normalization_manifest,
    detection_stats=None,
):
    """Scale raw distances to [0, 1] and write outputs plus provenance."""
    if not data_folder.endswith('/'):
        data_folder += '/'

    gripper_distances_normalized = (raw_distances - min_distance) / (max_distance - min_distance)
    gripper_distances_normalized[gripper_distances_normalized > 1] = 1
    gripper_distances_normalized[gripper_distances_normalized < 0] = 0

    np.savetxt(data_folder + "gripper_distances.txt", gripper_distances_normalized)
    print("Finished processing gripper distances to file", data_folder + "gripper_distances.txt")
    manifest_path = write_manifest(data_folder, normalization_manifest)
    print("Saved gripper normalization manifest to", manifest_path)
    if detection_stats is not None:
        stats_path = data_folder + DETECTION_STATS_NAME
        with open(stats_path, 'w') as f:
            json.dump(detection_stats, f, indent=2)
        print("Saved detection stats to", stats_path)

    save_gripper_distance_plot(gripper_distances_normalized, data_folder)


def process_episodes(data_folders, config_path, verbose=False, dist_plot_path=None, median_filter_size=1,
                     interp_mode="linear", use_preprocess=True, use_symmetry=True,
                     spike_threshold=DEFAULT_SPIKE_THRESHOLD, max_interp_gap=None):
    """Detect gripper distances for each episode, then normalize all of them
    with one shared min/max.

    Phase 1 runs per-episode detection (gripper_tag_core.detect_episode).
    Phase 2 pools the symmetry-constrained valid samples of every episode,
    resolves the min/max (manual config override or robust clustered
    extremum), then per episode removes spikes, interpolates gaps, applies
    the optional median filter and writes the outputs. If dist_plot_path is
    given, also saves a histogram of the pooled raw distances with the
    selected min/max marked.
    """
    cfg = load_config(config_path)
    print("=" * 50)
    print_config(cfg, config_path)
    print("=" * 50)

    preprocess_chain = tuple(cfg.get("preprocess_chain", DEFAULT_PREPROCESS_CHAIN)) \
        if use_preprocess else ()
    symmetry_tol = float(cfg.get("symmetry_tol_px", DEFAULT_SYMMETRY_TOL_PX))
    prior_center = float(cfg.get("prior_center_px", DEFAULT_PRIOR_CENTER_PX))
    prior_tol = float(cfg.get("prior_center_tol_px", DEFAULT_PRIOR_CENTER_TOL_PX))
    distance_bounds = tuple(cfg.get("tag_distance_bounds", DEFAULT_TAG_DISTANCE_BOUNDS))
    max_interp_gap = cfg.get("max_interp_gap_frames", 10)
    spike_threshold = float(cfg.get("spike_threshold", spike_threshold))
    if max_interp_gap is None:
        max_interp_gap = int(cfg.get("max_interp_gap_frames", 10))

    at_detector = Detector(
        families=cfg["tag_family"],
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    gripper_roi = cfg["gripper_roi"]
    gripper_roi_bottom = cfg.get("gripper_roi_bottom")
    left_id = cfg["left_gripper_tag_id"]
    right_id = cfg["right_gripper_tag_id"]
    nominal_diag = cfg.get("nominal_diag_distance", cfg.get("norminal_diag_distance", 50))
    diag_threshold = cfg["threshold"]

    # Phase 1: detect tag positions per frame for every episode
    detections = {}
    no_image_count = 0
    for data_folder in data_folders:
        folder = str(data_folder)
        images = find_color_images(folder if folder.endswith('/') else folder + '/')
        if not images:
            print(f"\n--- {Path(folder).name}: no color images, skipping")
            no_image_count += 1
            continue
        print(f"\n--- {Path(folder).name}: {len(images)} frames")
        det = detect_episode(
            [Path(p) for p in images], at_detector,
            left_id, right_id, nominal_diag, diag_threshold, gripper_roi,
            gripper_roi_bottom=gripper_roi_bottom,
            left_window=tuple(cfg["left_tag_x_range"]) if cfg.get("left_tag_x_range") else None,
            right_window=tuple(cfg["right_tag_x_range"]) if cfg.get("right_tag_x_range") else None,
            preprocess_chain=preprocess_chain,
            verbose=verbose,
        )
        n_both = sum(1 for l, r in zip(det.left_xs, det.right_xs)
                     if l is not None and r is not None)
        n_left = sum(1 for l in det.left_xs if l is not None)
        n_right = sum(1 for r in det.right_xs if r is not None)
        n_none = sum(1 for l, r in zip(det.left_xs, det.right_xs)
                     if l is None and r is None)
        print(f"    both={n_both}  left_only={n_left - n_both}  right_only={n_right - n_both}"
              f"  none={n_none}"
              f"  preprocess_fallback={det.preprocess_frames}")
        detections[folder] = det

    if not detections:
        print("No episodes with color images; nothing to process.")
        return {"ok": 0, "zero": 0, "no_images": no_image_count}

    # Phase 2a: symmetry centers, pooled valid samples, shared min/max
    run_center = run_center_fallback(list(detections.values()))
    if run_center is not None:
        print(f"\nRun-level symmetry center: {run_center:.2f} px")

    pooled_parts = []
    for folder, det in detections.items():
        center = run_center  # session-level uniformly: one rig per session
        if not use_symmetry:
            center = None
        indices, values, flags = det.series(
            center, prior_center=prior_center, prior_tol=prior_tol,
            distance_bounds=distance_bounds)
        if values:
            pooled_parts.append(np.asarray(values, dtype=float))
        else:
            print(f"Warning: no valid distance samples in {folder}; skipping episode.")

    if not pooled_parts:
        # Whole run has zero valid detections: every episode gets an all-NaN
        # series (verdict left to QC) — no normalization manifest possible.
        print("No episodes with valid AprilTag detections; writing all-NaN series.")
        for folder, det in detections.items():
            n_frames = len(det.left_xs)
            out = folder if folder.endswith('/') else folder + '/'
            np.savetxt(out + "gripper_distances.txt", np.full(n_frames, np.nan))
            with open(out + DETECTION_STATS_NAME, 'w') as f:
                json.dump({"status": "zero_detections", "frame_count": n_frames}, f, indent=2)
        return {"ok": 0, "zero": len(detections), "no_images": no_image_count}
    pooled = np.concatenate(pooled_parts)

    manual_min = cfg.get("gripper_min_distance")
    manual_max = cfg.get("gripper_max_distance")
    if manual_min is not None and manual_max is not None:
        min_distance, max_distance = manual_min, manual_max
        source_type = "manual_config"
        print(f"\nUsing manual gripper range from config: min={min_distance}, max={max_distance}")
    else:
        max_distance, min_distance = find_distances_max_min(
            pooled,
            threshold=ROBUST_EXTREMUM_THRESHOLD,
            threshold_num=ROBUST_EXTREMUM_MIN_SAMPLES,
        )
        fixed_min = float(cfg.get("calibration_min_px", DEFAULT_CALIB_MIN_PX))
        fixed_max = float(cfg.get("calibration_max_px", DEFAULT_CALIB_MAX_PX))
        sanity = float(cfg.get("anchor_sanity_px", DEFAULT_ANCHOR_SANITY_PX))
        dev_min, dev_max = abs(min_distance - fixed_min), abs(max_distance - fixed_max)
        print(
            f"\nSession gripper range from {len(pooled_parts)} episode(s): "
            f"min={min_distance:.2f}, max={max_distance:.2f} "
            f"(deviation from calibration {fixed_min}/{fixed_max}: "
            f"{dev_min:.2f}/{dev_max:.2f} px)"
        )
        if dev_min > sanity or dev_max > sanity:
            print(f"WARNING: session anchors deviate >{sanity} px from the calibration "
                  f"pair -- detection or this session's data is suspect. "
                  f"Falling back to fixed calibration anchors {fixed_min}/{fixed_max}.")
            min_distance, max_distance = fixed_min, fixed_max
            source_type = "fixed_calibration_fallback"
        else:
            source_type = "session_pooled"

    normalization_manifest = build_manifest(
        min_distance=min_distance,
        max_distance=max_distance,
        source_type=source_type,
        config_path=config_path,
        config=cfg,
        median_filter_size=median_filter_size,
        episode_count=len(pooled_parts),
        frame_count=len(pooled),
        interp_mode=interp_mode,
        preprocess_chain=list(preprocess_chain),
        symmetry_enabled=use_symmetry,
        symmetry_tol_px=symmetry_tol,
        spike_threshold=spike_threshold,
    )

    if dist_plot_path is not None:
        save_gripper_range_plot(pooled, min_distance, max_distance, dist_plot_path)
        manifest_path = write_manifest(Path(dist_plot_path).parent, normalization_manifest)
        print("Saved run-level gripper normalization manifest to", manifest_path)

    median_filter_fn = None
    if median_filter_size > 1:
        median_filter_fn = lambda a: medfilt(a, kernel_size=median_filter_size)

    # Phase 2b: per-episode cleanup (spikes, interpolation) and save
    n_ok = n_zero = 0
    for folder, det in detections.items():
        center = run_center  # session-level uniformly: one rig per session
        if not use_symmetry:
            center = None
        raw, stats = finalize_series(
            det, min_distance, max_distance,
            center=center, symmetry_tol=symmetry_tol,
            prior_center=prior_center, prior_tol=prior_tol,
            distance_bounds=distance_bounds,
            interp_mode=interp_mode,
            max_interp_gap=max_interp_gap if max_interp_gap else None,
            spike_threshold=spike_threshold,
            median_filter_fn=median_filter_fn,
        )
        if raw is None:
            # Zero detections: keep processing to the end — write an all-NaN
            # series of the right length and let QC (gripper_missing) decide.
            n_frames = len(det.left_xs)
            out = folder if folder.endswith('/') else folder + '/'
            np.savetxt(out + "gripper_distances.txt", np.full(n_frames, np.nan))
            zero_stats = {
                "status": "zero_detections",
                "frame_count": n_frames,
                "interp_mode": interp_mode,
                "symmetry_enabled": use_symmetry,
            }
            with open(out + DETECTION_STATS_NAME, 'w') as f:
                json.dump(zero_stats, f, indent=2)
            print(f"Warning: no valid AprilTag samples in {Path(folder).name}; "
                  f"wrote all-NaN gripper_distances.txt ({n_frames} frames) — "
                  "QC will flag it (gripper_missing).")
            n_zero += 1
            continue
        stats["interp_mode"] = interp_mode
        normalize_and_save(folder if folder.endswith('/') else folder + '/',
                           raw, min_distance, max_distance, normalization_manifest,
                           detection_stats=stats)
        n_ok += 1
    print(f"\nGripper estimation summary: {n_ok} ok, {n_zero} zero-detection (all-NaN), "
          f"{no_image_count} without color images")
    return {"ok": n_ok, "zero": n_zero, "no_images": no_image_count}


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_folder
    if args.calibrate:
        calibrate(data_path, args.configs, args.output)
        sys.exit(0)
    elif args.recursive:
        dirs = sorted(d for d in Path(data_path).rglob("episode_*")
                      if d.is_dir())
        if not dirs:
            print(f"No episode directories found under {data_path}")
            sys.exit(1)
        print(f"Processing {len(dirs)} episode(s)")
        dist_plot_path = Path(data_path) / "gripper_distance_distribution.png"
        counts = process_episodes([str(d) for d in dirs], args.configs, verbose=args.verbose,
                                   dist_plot_path=dist_plot_path, median_filter_size=args.median_filter,
                                   interp_mode=args.interp, use_preprocess=not args.no_preprocess,
                                   use_symmetry=not args.no_symmetry,
                                   spike_threshold=args.spike_threshold,
                                   max_interp_gap=args.max_interp_gap)
    else:
        dist_plot_path = Path(data_path) / "gripper_distance_distribution.png"
        counts = process_episodes([data_path], args.configs, verbose=args.verbose,
                                   dist_plot_path=dist_plot_path, median_filter_size=args.median_filter,
                                   interp_mode=args.interp, use_preprocess=not args.no_preprocess,
                                   use_symmetry=not args.no_symmetry,
                                   spike_threshold=args.spike_threshold,
                                   max_interp_gap=args.max_interp_gap)
    # Non-zero exit when any episode could not be processed at all (missing
    # color images). Zero-detection episodes are NOT errors: they produced an
    # all-NaN series and the verdict belongs to QC (gripper_missing).
    sys.exit(1 if counts.get("no_images", 0) else 0)
