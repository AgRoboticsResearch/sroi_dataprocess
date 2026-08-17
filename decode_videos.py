#!/usr/bin/env python3
"""
Decode MP4 episode folders back to PNG image sequences.

Recovers frames from left.mp4, right.mp4, color.mp4 in episode directories
that were encoded by record_realsense.py --encode-video. Produces PNG images
and validates frame counts against timestamps.json.

By default, output is placed in a sibling folder with -png postfix.
Use -o to specify a custom output directory.

Metadata files (camera info, times.txt, timestamps.json, ORB-SLAM yaml)
are copied alongside the decoded images.

Usage:
    # Decode a single episode
    python decode_videos.py /path/to/session-mp4/episode_001

    # Recursively decode all episode dirs under a session
    python decode_videos.py /path/to/session-mp4 --recursive

    # Decode to a custom output directory
    python decode_videos.py /path/to/session-mp4 --recursive -o /path/to/output

Dependencies:
    pip install av opencv-python
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2

try:
    import av
except ImportError:
    print("PyAV is required: pip install av")
    raise SystemExit(1)


PNG_COMPRESSION = 3  # Fast compression, still lossless
STREAM_NAMES = ["left", "right", "color"]


def decode_episode(episode_dir: Path, output_dir: Path):
    """Decode MP4 files in episode_dir to PNG images in output_dir."""
    if output_dir.exists() and any(output_dir.glob("left_*.png")):
        print(f"  Already decoded, skipping.")
        return

    # Skip incomplete recordings: an aborted/interrupted episode leaves empty
    # timestamps.json / times.txt (and sometimes 0-byte mp4s). Without timing
    # metadata the episode is not usable downstream (transform_trajectory needs
    # times.txt), so skip it with a warning instead of crashing the whole batch.
    ts_path = episode_dir / "timestamps.json"
    expected_frames = None
    if not ts_path.exists() or ts_path.stat().st_size == 0:
        print(f"  WARNING: missing/empty timestamps.json, skipping incomplete episode.")
        return
    try:
        with open(ts_path) as f:
            ts_data = json.load(f)
        expected_frames = ts_data.get("frame_count")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  WARNING: unreadable timestamps.json ({e}), skipping incomplete episode.")
        return

    # Skip if there are no non-empty mp4 streams to decode.
    if not any((episode_dir / f"{s}.mp4").exists() and (episode_dir / f"{s}.mp4").stat().st_size > 0
               for s in STREAM_NAMES):
        print(f"  WARNING: no non-empty mp4 streams, skipping incomplete episode.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata files
    for name in ["camera_info_left.json", "camera_info_right.json",
                 "camera_info_color.json", "times.txt", "timestamps.json",
                 "imu.txt", "CALIBRATION_NOTE.txt"]:
        src = episode_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    # Copy ORB-SLAM yaml if present
    for yml in episode_dir.glob("*.yaml"):
        shutil.copy2(yml, output_dir / yml.name)

    stream_frame_counts = {}
    for stream_name in STREAM_NAMES:
        mp4_path = episode_dir / f"{stream_name}.mp4"
        if not mp4_path.exists():
            print(f"  {stream_name}.mp4 not found, skipping.")
            continue

        frame_count = 0
        try:
            with av.open(str(mp4_path)) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    img = frame.to_ndarray(format="bgr24")
                    out_path = output_dir / f"{stream_name}_{frame_count:06d}.png"
                    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
                    frame_count += 1
        except av.error.InvalidDataError as e:
            print(f"  WARNING: corrupt {stream_name}.mp4, decoded {frame_count} frames before error: {e}")

        print(f"  {stream_name}: {frame_count} frames decoded")
        stream_frame_counts[stream_name] = frame_count

    # Validate each stream against expected count and each other
    counts = list(stream_frame_counts.values())
    if counts:
        if expected_frames is not None and counts[0] != expected_frames:
            print(f"  WARNING: decoded {counts[0]} frames, expected {expected_frames}")
        if len(set(counts)) > 1:
            print(f"  WARNING: stream frame counts differ: {stream_frame_counts}")


def get_output_dir(episode_dir: Path, output_root: Path = None) -> Path:
    """Derive output directory for decoded episode."""
    session_dir = episode_dir.parent
    session_name = session_dir.name
    # Derive session output name: replace -mp4 with -png
    if session_name.endswith("-mp4"):
        out_session_name = session_name[:-4] + "-png"
    else:
        out_session_name = session_name + "-png"
    if output_root is not None:
        return output_root / out_session_name / episode_dir.name
    # Default: sibling folder
    return session_dir.parent / out_session_name / episode_dir.name


def find_mp4_dirs(path: Path):
    """Find all episode directories containing MP4s under path."""
    dirs = []
    for d in sorted(path.iterdir()):
        if d.is_dir() and d.name.startswith("episode_") and (d / "timestamps.json").exists():
            dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Decode MP4 episode folders back to PNG images.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to an episode dir or session dir (with --recursive)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: sibling folder with -png postfix)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively find and decode all episode dirs under input",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Path not found: {input_path}")
        return

    output_root = Path(args.output) if args.output else None

    if args.recursive:
        dirs = find_mp4_dirs(input_path)
        if not dirs:
            print(f"No episode directories found under {input_path}")
            return
        print(f"Found {len(dirs)} episode(s) to decode")
        for d in dirs:
            out = get_output_dir(d, output_root)
            print(f"Decoding {d.name} -> {out}")
            decode_episode(d, out)
    else:
        out = get_output_dir(input_path, output_root)
        print(f"Decoding {input_path.name} -> {out}")
        decode_episode(input_path, out)

    print("Done.")


if __name__ == "__main__":
    main()
