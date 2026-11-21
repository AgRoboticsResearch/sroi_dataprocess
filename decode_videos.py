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

Power-loss cleanup (--recursive, session flow):
  A Pi power loss can leave an unfinished episode behind and skip the
  session finalize (no meta.json). Normal recording never produces
  incomplete episodes, so:
    - episodes that fail a structural check (missing/inconsistent
      timestamps.json or times.txt, missing streams or required camera
      metadata) or that fail at decode time (truncated/corrupt mp4, decoded
      frame count != frame_count) are deleted from the raw session, with
      the reason logged;
    - a missing or unparseable meta.json is rebuilt from the surviving
      episodes in the recorder's exact format.
  If no episode survives, nothing is deleted and no meta.json is written
  (exit 1) so the session stays intact for human inspection. Downstream
  sees a session indistinguishable from a normally-finalized one.

Exit codes (session flow):
    0  session processed; deleted leftovers are logged, not an error
    1  no surviving episodes (raw session left untouched)
    2  usage or path error

Usage:
    # Decode a single episode
    python decode_videos.py /path/to/session-mp4/episode_001

    # Decode a whole session (with power-loss cleanup)
    python decode_videos.py /path/to/session-mp4 --recursive

    # Decode to a custom output directory
    python decode_videos.py /path/to/session-mp4 --recursive -o /path/to/output

Dependencies:
    pip install av opencv-python
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

try:
    import av
except ImportError:
    print("PyAV is required: pip install av")
    raise SystemExit(1)


PNG_COMPRESSION = 3  # Fast compression, still lossless
STREAM_NAMES = ["left", "right", "color"]

# FFmpeg error base class: av.error.FFmpegError on PyAV >= 15, av.AVError before.
AV_FFMPEG_ERROR = getattr(av.error, "FFmpegError", None) or getattr(av.error, "AVError", None)
DECODE_ERRORS = (AV_FFMPEG_ERROR, EOFError, OSError, ValueError)

# Metadata files copied to the output alongside the decoded images.
COPY_META = ["camera_info_left.json", "camera_info_right.json",
             "camera_info_color.json", "times.txt", "timestamps.json",
             "imu.txt", "CALIBRATION_NOTE.txt"]
# Without these the episode is unusable downstream (transform_trajectory
# needs times.txt, ORB-SLAM needs the yaml) -> structural check fails.
REQUIRED_META = ["camera_info_left.json", "camera_info_right.json",
                 "times.txt"]
YAML_SUFFIX = ".yaml"

# The recorder writes session meta.json only at finalize; rebuild an
# identical one when a power loss skipped it.
RECORDED_CAMERA = "realsense_d405"
RECORDER_META_PATH_PREFIX = "/home/brl/codes/recordings"


def check_episode_structure(episode_dir: Path):
    """Structural check: does the episode dir look like a completed recording?

    Returns (reason, detail). reason is None when the episode may proceed;
    otherwise a short code for the log: missing_timestamps,
    invalid_frame_count, missing_times_txt, times_txt_mismatch,
    missing_stream, missing_required_meta.
    """
    ts_path = episode_dir / "timestamps.json"
    if not ts_path.exists() or ts_path.stat().st_size == 0:
        return "missing_timestamps", ""
    try:
        with open(ts_path) as f:
            ts_data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        return "missing_timestamps", str(e)

    frame_count = ts_data.get("frame_count")
    if not isinstance(frame_count, int) or frame_count <= 0:
        return "invalid_frame_count", f"frame_count={frame_count}"
    if len(ts_data.get("timestamps", [])) != frame_count:
        return "invalid_frame_count", \
            f"frame_count={frame_count} timestamps_len={len(ts_data.get('timestamps', []))}"

    times_path = episode_dir / "times.txt"
    if not times_path.exists():
        return "missing_times_txt", ""
    with open(times_path) as f:
        n_lines = sum(1 for line in f if line.strip())
    if n_lines != frame_count:
        return "times_txt_mismatch", f"lines={n_lines} frame_count={frame_count}"

    missing_streams = [name for name in STREAM_NAMES
                       if not (episode_dir / f"{name}.mp4").exists()
                       or (episode_dir / f"{name}.mp4").stat().st_size == 0]
    if missing_streams:
        return "missing_stream", f"missing={missing_streams}"

    missing_meta = [name for name in REQUIRED_META
                    if not (episode_dir / name).exists()]
    if not list(episode_dir.glob(f"*{YAML_SUFFIX}")):
        missing_meta.append("orb_slam_yaml")
    if missing_meta:
        return "missing_required_meta", f"missing={missing_meta}"

    return None, ""


def decode_episode(episode_dir: Path, output_dir: Path):
    """Decode MP4 files in episode_dir to PNG images in output_dir.

    Returns (status, detail): status is "ok" or the reason the episode is
    incomplete ("decode_error" for a stream that failed mid-decode,
    "frame_count_mismatch" when decoded counts disagree with frame_count
    or across streams). On any failure the derived output_dir is removed;
    the caller decides what happens to the raw episode dir.
    """
    reason, detail = check_episode_structure(episode_dir)
    if reason is not None:
        return reason, detail

    # Idempotent skip: only when all three streams are already fully present.
    expected_frames = json.load(open(episode_dir / "timestamps.json"))["frame_count"]
    existing = {s: count_pngs(output_dir, s) for s in STREAM_NAMES}
    if all(n == expected_frames for n in existing.values()):
        return "ok", "already decoded"
    if any(n > 0 for n in existing.values()):
        # Partial or outdated decode from an earlier run: redo from scratch
        # so a repaired decode never mixes stale with fresh frames.
        print(f"  incomplete existing decode ({existing}), re-decoding")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_META:
        src = episode_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    for yml in episode_dir.glob(f"*{YAML_SUFFIX}"):
        shutil.copy2(yml, output_dir / yml.name)

    stream_frame_counts = {}
    for stream_name in STREAM_NAMES:
        mp4_path = episode_dir / f"{stream_name}.mp4"
        frame_count = 0
        try:
            with av.open(str(mp4_path)) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    img = frame.to_ndarray(format="bgr24")
                    out_path = output_dir / f"{stream_name}_{frame_count:06d}.png"
                    if not cv2.imwrite(str(out_path), img,
                                       [cv2.IMWRITE_PNG_COMPRESSION,
                                        PNG_COMPRESSION]):
                        raise OSError(f"cv2.imwrite failed for {out_path}")
                    frame_count += 1
        except DECODE_ERRORS as e:
            detail = f"{stream_name}.mp4 failed after {frame_count} frames: {e}"
            print(f"  ERROR: {detail}")
            shutil.rmtree(output_dir)
            return "decode_error", detail

        stream_frame_counts[stream_name] = frame_count

    if any(n != expected_frames for n in stream_frame_counts.values()):
        detail = f"decoded {stream_frame_counts}, expected {expected_frames}"
        print(f"  ERROR: {detail}")
        shutil.rmtree(output_dir)
        return "frame_count_mismatch", detail
    if len(set(stream_frame_counts.values())) > 1:
        detail = f"stream frame counts differ: {stream_frame_counts}"
        print(f"  ERROR: {detail}")
        shutil.rmtree(output_dir)
        return "frame_count_mismatch", detail

    print(f"  {stream_frame_counts}")
    return "ok", f"{stream_frame_counts}"


def count_pngs(output_dir: Path, stream_name: str) -> int:
    if not output_dir.is_dir():
        return 0
    return sum(1 for _ in output_dir.glob(f"{stream_name}_*.png"))


def get_session_output_dir(session_dir: Path, output_root: Path = None) -> Path:
    """Derive the output session directory (sibling -png layout)."""
    session_name = session_dir.name
    if session_name.endswith("-mp4"):
        out_session_name = session_name[:-4] + "-png"
    else:
        out_session_name = session_name + "-png"
    if output_root is not None:
        return output_root / out_session_name
    return session_dir.parent / out_session_name


def get_output_dir(episode_dir: Path, output_root: Path = None) -> Path:
    """Derive output directory for decoded episode."""
    return get_session_output_dir(episode_dir.parent, output_root) / episode_dir.name


def rebuild_meta_json(session_dir: Path, episode_dirs):
    """Rebuild the session meta.json the recorder would have written at
    finalize, from the surviving episodes. Field set, key order, rounding
    and indent match the recorded files exactly."""
    total_frames = 0
    total_duration = 0.0
    total_bytes = 0
    for episode_dir in episode_dirs:
        ts_data = json.load(open(episode_dir / "timestamps.json"))
        stamps = ts_data.get("timestamps") or []
        if len(stamps) >= 2:
            total_duration += float(stamps[-1]) - float(stamps[0])
        total_frames += ts_data["frame_count"]
        for f in episode_dir.iterdir():
            if f.is_file() and not f.name.startswith("vive"):
                total_bytes += f.stat().st_size
    n = len(episode_dirs)
    meta = {
        "episodes": n,
        "total_frames": total_frames,
        "total_duration_s": round(total_duration, 2),
        "mean_episode_duration_s": round(total_duration / n, 2) if n else 0.0,
        "total_size_mb": round(total_bytes / (1024 * 1024), 1),
        "camera": RECORDED_CAMERA,
        "path": f"{RECORDER_META_PATH_PREFIX}/{session_dir.name}",
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def process_session(session_dir: Path, output_root: Path = None) -> int:
    """Decode one -mp4 session with power-loss cleanup. Returns exit code."""
    out_session_dir = get_session_output_dir(session_dir, output_root)
    out_session_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(d for d in session_dir.iterdir()
                          if d.is_dir() and d.name.startswith("episode_"))
    if not episode_dirs:
        print(f"  ERROR: no episode_* dirs under {session_dir}")
        return 2

    # 1) Structural check, no deletion yet: a false-negative in the checker
    #    must not be able to wipe a whole session before anything survived.
    structurally_ok, structurally_bad = [], []
    for ep_dir in episode_dirs:
        reason, detail = check_episode_structure(ep_dir)
        if reason is None:
            structurally_ok.append(ep_dir)
        else:
            structurally_bad.append((ep_dir, reason, detail))

    # 2) Decode the episodes that passed; a failure here (truncated mp4,
    #    frame count off) deletes that raw episode dir too.
    decoded = []
    for ep_dir in structurally_ok:
        status, detail = decode_episode(ep_dir, out_session_dir / ep_dir.name)
        if status == "ok":
            decoded.append(ep_dir)
        else:
            shutil.rmtree(ep_dir)
            print(f"  deleted {ep_dir.name}: {status} ({detail})")

    # 3) Nothing survived: leave the raw session exactly as it is.
    if not decoded:
        print(f"  ERROR: no decodable episodes in {session_dir}, "
              f"raw session left untouched")
        return 1

    # 4) Only now drop the structurally-bad leftovers.
    for ep_dir, reason, detail in structurally_bad:
        shutil.rmtree(ep_dir)
        print(f"  deleted {ep_dir.name}: {reason} ({detail})")

    # 5) Backfill the session finalize: meta.json in the recorder's format.
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        rebuild_meta_json(session_dir, decoded)
        print(f"  rebuilt meta.json ({len(decoded)} episodes)")
    else:
        try:
            json.load(open(meta_path))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  meta.json unparseable ({e}), rebuilding")
            rebuild_meta_json(session_dir, decoded)

    return 0


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
        help="Decode all episode dirs under input (session flow, includes "
             "power-loss cleanup)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Path not found: {input_path}")
        return 2

    output_root = Path(args.output) if args.output else None

    if args.recursive:
        return process_session(input_path, output_root)
    else:
        out = get_output_dir(input_path, output_root)
        print(f"Decoding {input_path.name} -> {out}")
        status, detail = decode_episode(input_path, out)
        if status != "ok":
            print(f"ERROR: episode incomplete ({status}: {detail})")
            return 1
        print("Done.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
