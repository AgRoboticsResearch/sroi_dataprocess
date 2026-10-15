#!/usr/bin/env python3
"""
RealSense Stereo IR + Color Recorder

Records synchronized stereo infrared and color frames from a RealSense camera
(D435i or D405) using pyrealsense2. Output is directly compatible with the
ORB-SLAM pipeline (orbslam_batch_local.sh, transform_trajectory.py, sroi_to_lerobot.py).

Controls (in GUI or headless terminal):
    r - Start recording a new episode
    s - Stop recording current episode
    q - Quit

Storage modes (--image-format + --encode-video):
    PNG:   Save lossless PNGs (~169 MB per 300 frames at 640x480).
    JPEG:  Save JPEGs at quality 90 (~30 MB per 300 frames).
    MP4:   Encode to MP4 (H.264 by default) after each episode, then delete
           source images (~5 MB per 300 frames). Requires: pip install av

Folder naming:
    Sessions:   {timestamp}-{format}   e.g. 20260520_143052-png
    Episodes:   episode_NNN             e.g. episode_001
    Where {format} is "png", "jpeg", or "mp4".

Each episode contains:
    left_NNNNNN.{ext}     Left IR frames
    right_NNNNNN.{ext}    Right IR frames
    color_NNNNNN.{ext}    Color frames
    times.txt             Hardware timestamps (one per line, seconds)
    timestamps.json       Timestamps + metadata (from/to, fps, frame_count)
    camera_info_*.json    Camera intrinsics (ROS camera_info format)

Usage examples:
    # Default: PNG with GUI preview
    python record_realsense.py -o /path/to/output --camera realsense_d405

    # JPEG mode (smaller files)
    python record_realsense.py -o /path/to/output --image-format jpeg

    # Headless SSH session (no display needed)
    python record_realsense.py -o /path/to/output --headless

    # Encode to MP4 (smallest, requires: pip install av)
    python record_realsense.py -o /path/to/output --encode-video

    # MP4 with AV1 codec (better compression, needs libsvtav1)
    python record_realsense.py -o /path/to/output --encode-video --vcodec libsvtav1

    # Decode MP4s back to PNGs for downstream pipelines
    python decode_videos.py /path/to/session-mp4 --recursive

Dependencies:
    pip install pyrealsense2 opencv-python numpy
    pip install av  # only needed for --encode-video
"""

import argparse
import glob
import json
import os
import queue
import select
import shutil
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 is required: pip install pyrealsense2")
    raise SystemExit(1)

try:
    import av
except ImportError:
    av = None


# ---------------------------------------------------------------------------
# Camera defaults (IR resolution, color resolution, FPS)
# ---------------------------------------------------------------------------
CAMERA_DEFAULTS = {
    "realsense_d435i": {
        "ir_width": 640,
        "ir_height": 480,
        "ir_fps": 30,
        "color_width": 640,
        "color_height": 480,
        "color_fps": 30,
    },
    "realsense_d405": {
        "ir_width": 640,
        "ir_height": 480,
        "ir_fps": 30,
        "color_width": 640,
        "color_height": 480,
        "color_fps": 30,
        "stereo_mode": "ir1_color",  # D405: right IR (idx=1 y8) + color only
    },
}


# ---------------------------------------------------------------------------
# Camera info helpers
# ---------------------------------------------------------------------------
def get_intrinsics(pipeline_profile, stream_type, stream_index=0):
    """Get intrinsics for a specific stream."""
    for profile in pipeline_profile.get_streams():
        if (
            profile.stream_type() == stream_type
            and profile.stream_index() == stream_index
        ):
            return profile.as_video_stream_profile().get_intrinsics()
    return None


def get_extrinsics(pipeline_profile, from_stream, from_idx, to_stream, to_idx):
    """Get extrinsics (rotation + translation) between two streams."""
    from_profile = None
    to_profile = None
    for profile in pipeline_profile.get_streams():
        if (
            profile.stream_type() == from_stream
            and profile.stream_index() == from_idx
        ):
            from_profile = profile
        if (
            profile.stream_type() == to_stream
            and profile.stream_index() == to_idx
        ):
            to_profile = profile
    if from_profile and to_profile:
        return from_profile.get_extrinsics_to(to_profile)
    return None


def make_camera_info(intrinsics, width, height, baseline_tx=0.0):
    """Convert pyrealsense2 intrinsics to ROS camera_info JSON format.

    This matches the ROS camera_info JSON format consumed by
    create_orb_slam_yaml.py.
    """
    return {
        "height": height,
        "width": width,
        "distortion_model": "plumb_bob",
        "D": list(intrinsics.coeffs[:5]) if hasattr(intrinsics, "coeffs") else [0.0] * 5,
        "K": [
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0,
        ],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [
            intrinsics.fx, 0.0, intrinsics.ppx, baseline_tx,
            0.0, intrinsics.fy, intrinsics.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
    }


def save_orb_slam_yaml(camera_info_left, camera_info_right, output_path, camera_type):
    """Generate ORB-SLAM YAML from camera info dicts, reusing logic from create_orb_slam_yaml.py."""
    if yaml is None:
        print("  Warning: pyyaml not installed, skipping ORB-SLAM YAML generation.")
        return

    template = Path(__file__).parent / "orb_slam_yaml"
    if camera_type == "realsense_d435i":
        tmpl_path = template / "RealSense_D435i.yaml"
    elif camera_type == "realsense_d405":
        tmpl_path = template / "RealSense_D405.yaml"
    else:
        return
    if not tmpl_path.exists():
        print(f"  Warning: ORB-SLAM template not found: {tmpl_path}")
        return

    with open(tmpl_path) as f:
        content = f.read()
    if content.startswith("%YAML:1.0"):
        content = content.replace("%YAML:1.0\n", "", 1)
    try:
        config = yaml.load(content, Loader=yaml.UnsafeLoader)
    except yaml.YAMLError:
        config = yaml.safe_load(content)

    left_K = np.array(camera_info_left["K"]).reshape(3, 3)
    left_P = np.array(camera_info_left["P"]).reshape(3, 4)
    right_P = np.array(camera_info_right["P"]).reshape(3, 4)
    fx, fy = left_K[0, 0], left_K[1, 1]
    cx, cy = left_K[0, 2], left_K[1, 2]
    D = camera_info_left["D"]

    config["Camera1.fx"] = float(fx)
    config["Camera1.fy"] = float(fy)
    config["Camera1.cx"] = float(cx)
    config["Camera1.cy"] = float(cy)
    config["Camera1.k1"] = float(D[0]) if len(D) > 0 else 0.0
    config["Camera1.k2"] = float(D[1]) if len(D) > 1 else 0.0
    config["Camera1.p1"] = float(D[2]) if len(D) > 2 else 0.0
    config["Camera1.p2"] = float(D[3]) if len(D) > 3 else 0.0

    config["Camera2.fx"] = float(fx)
    config["Camera2.fy"] = float(fy)
    config["Camera2.cx"] = float(cx)
    config["Camera2.cy"] = float(cy)
    config["Camera2.k1"] = float(D[0]) if len(D) > 0 else 0.0
    config["Camera2.k2"] = float(D[1]) if len(D) > 1 else 0.0
    config["Camera2.p1"] = float(D[2]) if len(D) > 2 else 0.0
    config["Camera2.p2"] = float(D[3]) if len(D) > 3 else 0.0

    config["Camera.width"] = camera_info_left["width"]
    config["Camera.height"] = camera_info_left["height"]

    baseline = -(right_P[0, 3] - left_P[0, 3]) / left_P[0, 0]
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = baseline
    config["Stereo.T_c1_c2"] = {"rows": 4, "cols": 4, "dt": "f", "data": T.flatten().tolist()}

    with open(output_path, "w") as f:
        f.write("%YAML:1.0\n\n")
        for key, value in config.items():
            if key == "Stereo.T_c1_c2" and isinstance(value, dict):
                f.write(f"{key}: !!opencv-matrix\n")
                f.write(f"  rows: {value['rows']}\n")
                f.write(f"  cols: {value['cols']}\n")
                f.write(f"  dt: {value['dt']}\n")
                f.write(f"  data: {value['data']}\n")
            elif isinstance(value, str):
                f.write(f'{key}: "{value}"\n')
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
    print(f"  ORB-SLAM YAML saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Video encoding (learned from LeRobot's encode_video_frames)
# ---------------------------------------------------------------------------
def encode_episode_to_video(episode_dir, fps, vcodec="libx264", crf=23, g=2,
                            image_format="png", mem_frames=None):
    """Encode per-camera image sequences in an episode directory to MP4 files.

    Produces left.mp4, right.mp4, color.mp4 alongside timestamps.json.
    If mem_frames is provided (dict of {stream_name: [rgb_arrays]}), encodes
    directly from memory without reading from disk.
    Otherwise reads from disk and deletes the original images after encoding.
    """
    if av is None:
        print("PyAV is required for video encoding: pip install av")
        return False

    ext = ".jpg" if image_format == "jpeg" else ".png"
    streams = ["left", "right", "color"]

    for stream_name in streams:
        if mem_frames and stream_name in mem_frames:
            frames_rgb = mem_frames[stream_name]
            if not frames_rgb:
                continue
            height, width = frames_rgb[0].shape[:2]
        else:
            pattern = f"{stream_name}_*{ext}"
            input_list = sorted(glob.glob(str(episode_dir / pattern)))
            if not input_list:
                print(f"  No {stream_name} frames found, skipping.")
                continue
            frames_rgb = None
            first = cv2.imread(input_list[0])
            if first is None:
                print(f"  Warning: cannot read {input_list[0]}, skipping {stream_name}.")
                continue
            height, width = first.shape[:2]

        video_path = episode_dir / f"{stream_name}.mp4"
        video_options = {
            "crf": str(crf),
            "g": str(g),
        }
        n_frames = len(frames_rgb) if frames_rgb else len(input_list)

        print(f"  Encoding {stream_name}: {n_frames} frames "
              f"{width}x{height} -> {video_path.name} ({vcodec}, crf={crf})")

        with av.open(str(video_path), "w") as output:
            out_stream = output.add_stream(vcodec, fps, options=video_options)
            out_stream.pix_fmt = "yuv420p"
            out_stream.width = width
            out_stream.height = height

            if frames_rgb:
                for img_rgb in frames_rgb:
                    frame = av.VideoFrame.from_ndarray(img_rgb, format="rgb24")
                    for packet in out_stream.encode(frame):
                        output.mux(packet)
            else:
                for img_path in input_list:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"  Warning: cannot read {img_path}, skipping frame.")
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frame = av.VideoFrame.from_ndarray(img_rgb, format="rgb24")
                    for packet in out_stream.encode(frame):
                        output.mux(packet)

            for packet in out_stream.encode():
                output.mux(packet)

        if not frames_rgb:
            for img_path in input_list:
                Path(img_path).unlink()

    return True


# ---------------------------------------------------------------------------
# Realtime video encoder (background threads)
# ---------------------------------------------------------------------------
class _RealtimeEncoder:
    """Encodes frames to MP4 in background threads during recording."""

    def __init__(self, episode_dir, fps, vcodec="libx264", crf=23, g=2):
        if av is None:
            raise ImportError("PyAV required: pip install av")
        self._queues = {}
        self._threads = {}
        self._errors = []
        for name in ("left", "right", "color"):
            q = queue.Queue(maxsize=60)
            self._queues[name] = q
            t = threading.Thread(target=self._encode_stream, args=(
                str(episode_dir / f"{name}.mp4"), q, fps, vcodec, crf, g,
            ), daemon=True)
            t.start()
            self._threads[name] = t

    def _encode_stream(self, path, q, fps, vcodec, crf, g):
        try:
            with av.open(path, "w") as out:
                s = out.add_stream(vcodec, fps, options={"crf": str(crf), "g": str(g)})
                s.pix_fmt = "yuv420p"
                s.width = 640
                s.height = 480
                while True:
                    item = q.get()
                    if item is None:
                        break
                    frame = av.VideoFrame.from_ndarray(item, format="rgb24")
                    for pkt in s.encode(frame):
                        out.mux(pkt)
                for pkt in s.encode():
                    out.mux(pkt)
        except Exception as e:
            self._errors.append(e)

    def push(self, stream_name, img_rgb):
        """Push an RGB frame to the named encoder queue."""
        self._queues[stream_name].put(img_rgb, timeout=5.0)

    def flush(self):
        """Signal all encoders to finish and wait."""
        for q in self._queues.values():
            q.put(None)
        for t in self._threads.values():
            t.join(timeout=30)
        if self._errors:
            print(f"  Warning: encoder errors: {self._errors}")


# ---------------------------------------------------------------------------
# Recording session
# ---------------------------------------------------------------------------
class RecordingSession:
    """Manages episode recording state and file I/O."""

    def __init__(self, output_dir: Path, camera_type: str = "realsense_d435i",
                 encode_video: bool = False, vcodec: str = "libx264",
                 image_format: str = "png", fps: int = 30, mem: bool = False,
                 status_file: str = None, realtime_encoding: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_type = camera_type
        self.encode_video = encode_video
        self.vcodec = vcodec
        self.image_format = image_format
        self.fps = fps
        self.mem = mem
        self.realtime_encoding = realtime_encoding
        self.episode_count = 0
        self.is_recording = False
        self.frame_counter = 0
        self.episode_dir = None
        self.times_file = None
        self.timestamps = []
        self._mem_frames = None
        self._rt_encoder = None
        self._status_file = Path(status_file) if status_file else None
        self._status_fps = 0.0
        self._status_last_write = 0.0

    @property
    def ext(self):
        return ".jpg" if self.image_format == "jpeg" else ".png"

    @property
    def imwrite_params(self):
        if self.image_format == "jpeg":
            return [cv2.IMWRITE_JPEG_QUALITY, 90]
        return []

    def write_status(self, state=None, fps=0.0):
        """Write current status to status file (atomic write)."""
        if not self._status_file:
            return
        now = time.monotonic()
        if now - self._status_last_write < 0.9:
            return
        self._status_last_write = now
        self._status_fps = fps
        status = {
            "state": state or ("REC" if self.is_recording else "Idle"),
            "episode": self.episode_count,
            "frame": self.frame_counter,
            "fps": round(fps, 1),
            "format": "MP4" if self.encode_video else self.image_format.upper(),
            "camera": self.camera_type,
            "output": str(self.output_dir),
            "updated": time.time(),
        }
        try:
            tmp = self._status_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(status))
            os.replace(str(tmp), str(self._status_file))
        except OSError:
            pass

    def start_episode(self, camera_info_left, camera_info_right, camera_info_color):
        """Start recording a new episode."""
        if self.is_recording:
            print("Already recording. Press 's' to stop first.")
            return

        self.episode_count += 1
        self.frame_counter = 0
        self.episode_dir = self.output_dir / f"episode_{self.episode_count:03d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        for info, name in [
            (camera_info_left, "camera_info_left.json"),
            (camera_info_right, "camera_info_right.json"),
            (camera_info_color, "camera_info_color.json"),
        ]:
            with open(self.episode_dir / name, "w") as f:
                json.dump(info, f, indent=2)

        yaml_name = f"orb_slam_{self.camera_type}.yaml"
        save_orb_slam_yaml(
            camera_info_left, camera_info_right,
            self.episode_dir / yaml_name, self.camera_type,
        )

        self.times_file = open(self.episode_dir / "times.txt", "w")
        self.is_recording = True
        if self.realtime_encoding:
            self._rt_encoder = _RealtimeEncoder(
                self.episode_dir, self.fps, vcodec=self.vcodec,
            )
            print(f"Recording episode {self.episode_count} -> {self.episode_dir} (realtime encoding)")
        elif self.mem and self.encode_video:
            self._mem_frames = {"left": [], "right": [], "color": []}
            print(f"Recording episode {self.episode_count} -> {self.episode_dir}")
        else:
            print(f"Recording episode {self.episode_count} -> {self.episode_dir}")

    def _mem_usage(self):
        """Return human-readable total memory used by buffered frames."""
        if not self._mem_frames:
            return "0 B"
        total = sum(a.nbytes for frames in self._mem_frames.values() for a in frames)
        for unit in ("B", "KB", "MB", "GB"):
            if total < 1024:
                return f"{total:.1f} {unit}"
            total /= 1024
        return f"{total:.1f} TB"

    def save_frame(self, timestamp, left_img, right_img, color_img):
        """Save a single frame (left IR, right IR, color) and timestamp."""
        if not self.is_recording:
            return

        if self._rt_encoder is not None:
            self._rt_encoder.push("left", cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB))
            self._rt_encoder.push("right", cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB))
            self._rt_encoder.push("color", cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        elif self._mem_frames is not None:
            self._mem_frames["left"].append(cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB))
            self._mem_frames["right"].append(cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB))
            self._mem_frames["color"].append(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        else:
            idx = self.frame_counter
            ext = self.ext
            params = self.imwrite_params
            cv2.imwrite(str(self.episode_dir / f"left_{idx:06d}{ext}"), left_img, params)
            cv2.imwrite(str(self.episode_dir / f"right_{idx:06d}{ext}"), right_img, params)
            cv2.imwrite(str(self.episode_dir / f"color_{idx:06d}{ext}"), color_img, params)

        self.times_file.write(f"{timestamp:.6f}\n")
        self.timestamps.append(timestamp)
        self.frame_counter += 1

        if self._mem_frames is not None and self.frame_counter % 30 == 0:
            print(f"  Buffer: {self._mem_usage()} ({self.frame_counter} frames)", end="\r")

    def stop_episode(self):
        """Stop recording the current episode."""
        if not self.is_recording:
            print("Not currently recording.")
            return

        self.times_file.close()

        ts_data = {
            "timestamps": self.timestamps,
            "from_timestamp": self.timestamps[0] if self.timestamps else None,
            "to_timestamp": self.timestamps[-1] if self.timestamps else None,
            "fps": self.fps,
            "frame_count": self.frame_counter,
        }
        with open(self.episode_dir / "timestamps.json", "w") as f:
            json.dump(ts_data, f, indent=2)

        self.is_recording = False
        print(
            f"Stopped episode {self.episode_count}: "
            f"{self.frame_counter} frames saved to {self.episode_dir}"
        )

        if self._rt_encoder is not None:
            print(f"Flushing realtime encoder for episode {self.episode_count}...")
            self._rt_encoder.flush()
            self._rt_encoder = None
        elif self.encode_video:
            print(f"Encoding episode {self.episode_count} to video...")
            encode_episode_to_video(
                self.episode_dir, self.fps, vcodec=self.vcodec,
                image_format=self.image_format, mem_frames=self._mem_frames,
            )

        self.timestamps = []
        self._mem_frames = None

    def discard_episode(self):
        """Discard the current incomplete episode (e.g. camera disconnected)."""
        if not self.is_recording:
            return

        if self.times_file:
            self.times_file.close()

        if self._rt_encoder is not None:
            self._rt_encoder.flush()
            self._rt_encoder = None

        if self.episode_dir and self.episode_dir.exists():
            import shutil
            shutil.rmtree(self.episode_dir)

        self.is_recording = False
        self.episode_count -= 1
        self.timestamps = []
        self._mem_frames = None


# ---------------------------------------------------------------------------
# OpenCV preview helpers
# ---------------------------------------------------------------------------
def build_preview(color_img, left_img, right_img, session, fps):
    """Build the composite preview image with status overlay."""
    h, w = color_img.shape[:2]

    small_h = h // 2
    left_small = cv2.resize(left_img, (w // 2, small_h))
    right_small = cv2.resize(right_img, (w // 2, small_h))

    if len(left_small.shape) == 2:
        left_small = cv2.cvtColor(left_small, cv2.COLOR_GRAY2BGR)
    if len(right_small.shape) == 2:
        right_small = cv2.cvtColor(right_small, cv2.COLOR_GRAY2BGR)

    ir_panel = np.vstack([left_small, right_small])
    preview = np.hstack([color_img, ir_panel])

    if session.is_recording:
        status = f"RECORDING episode {session.episode_count}"
        color = (0, 0, 255)
    else:
        status = "IDLE"
        color = (0, 200, 0)

    cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(
        preview,
        f"FPS: {fps:.0f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        preview,
        f"Episodes: {session.episode_count}",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )

    if session.is_recording:
        cv2.putText(
            preview,
            f"Frame: {session.frame_counter}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )

    cv2.putText(
        preview,
        "[r] record  [s] stop  [q] quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )

    return preview


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Record stereo IR + color from RealSense camera for ORB-SLAM pipeline",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./recordings",
        help="Output root directory (default: ./recordings)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="realsense_d435i",
        choices=list(CAMERA_DEFAULTS.keys()),
        help="Camera type (default: realsense_d435i)",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpeg"],
        help="Image format for saved frames (default: png)",
    )
    parser.add_argument(
        "--encode-video",
        action="store_true",
        default=True,
        help="Encode images to MP4 after each episode and delete originals (requires: pip install av)",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="libx264",
        choices=["libx264", "libsvtav1", "hevc"],
        help="Video codec for --encode-video (default: libx264)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run without display; use terminal for r/s/q controls",
    )
    parser.add_argument(
        "--mem",
        action="store_true",
        default=True,
        help="Buffer frames in memory with --encode-video instead of writing to disk first",
    )
    parser.add_argument(
        "--realtime-encoding",
        action="store_true",
        default=False,
        help="Encode to MP4 in realtime during recording (no post-encoding wait, overrides --mem)",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        default=None,
        help="Path to write periodic status JSON for external consumers (e.g. e-paper display)",
    )
    args = parser.parse_args()

    defaults = CAMERA_DEFAULTS[args.camera]

    if args.encode_video:
        folder_format = "mp4"
    else:
        folder_format = args.image_format

    session_timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output) / f"{session_timestamp}-{folder_format}"
    session = RecordingSession(
        session_dir,
        camera_type=args.camera,
        encode_video=args.encode_video,
        vcodec=args.vcodec,
        image_format=args.image_format,
        fps=defaults["color_fps"],
        mem=args.mem,
        status_file=args.status_file,
        realtime_encoding=args.realtime_encoding,
    )
    pipeline = rs.pipeline()
    config = rs.config()

    stereo_mode = defaults.get("stereo_mode", "dual_ir")
    if stereo_mode == "ir1_color":
        # D405: only right IR (idx=1, y8) + color
        config.enable_stream(
            rs.stream.infrared, 1,
            defaults["ir_width"], defaults["ir_height"],
            rs.format.y8, defaults["ir_fps"],
        )
    else:
        # D435i: left IR (idx=1) + right IR (idx=2)
        config.enable_stream(
            rs.stream.infrared, 1,
            defaults["ir_width"], defaults["ir_height"],
            rs.format.y8, defaults["ir_fps"],
        )
        config.enable_stream(
            rs.stream.infrared, 2,
            defaults["ir_width"], defaults["ir_height"],
            rs.format.y8, defaults["ir_fps"],
        )
    config.enable_stream(
        rs.stream.color,
        defaults["color_width"], defaults["color_height"],
        rs.format.bgr8, defaults["color_fps"],
    )

    print(f"Starting RealSense pipeline ({args.camera})...")
    print(f"IR:  {defaults['ir_width']}x{defaults['ir_height']} @ {defaults['ir_fps']}fps")
    print(f"Color: {defaults['color_width']}x{defaults['color_height']} @ {defaults['color_fps']}fps")
    print(f"Output: {session_dir}")
    print(f"Image format: {args.image_format}")
    if args.realtime_encoding:
        print(f"Video encoding: REALTIME (codec={args.vcodec})")
    elif args.encode_video:
        print(f"Video encoding: ON (codec={args.vcodec})")
    else:
        print("Video encoding: OFF")
    if args.headless:
        print("Mode: headless (terminal controls)")
    else:
        print("Mode: GUI (OpenCV window controls)")

    pipeline_profile = pipeline.start(config)

    device = pipeline_profile.get_device()
    for sensor in device.query_sensors():
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 0)
            print("IR emitter disabled (no laser dots in IR images)")
            break

    ir_right_intrinsics = get_intrinsics(pipeline_profile, rs.stream.infrared, 1)
    if stereo_mode == "ir1_color":
        ir_left_intrinsics = color_intrinsics = get_intrinsics(pipeline_profile, rs.stream.color, 0)
    else:
        ir_left_intrinsics = get_intrinsics(pipeline_profile, rs.stream.infrared, 1)
        ir_right_intrinsics = get_intrinsics(pipeline_profile, rs.stream.infrared, 2)
    color_intrinsics = get_intrinsics(pipeline_profile, rs.stream.color, 0)

    if stereo_mode == "ir1_color":
        extrinsics = get_extrinsics(
            pipeline_profile,
            rs.stream.infrared, 1,
            rs.stream.color, 0,
        )
    else:
        extrinsics = get_extrinsics(
            pipeline_profile,
            rs.stream.infrared, 2,
            rs.stream.infrared, 1,
        )
    baseline_m = extrinsics.translation[0] if extrinsics else 0.0
    # D405 firmware returns the IR-to-IR translation in millimeters while D435i
    # returns meters, so a sane baseline must be > 1 mm. If we see something
    # physically implausible, multiply by 1000 — otherwise ORB-SLAM3 receives a
    # ~50-micron baseline and the trajectory collapses to identity.
    if 0.0 < abs(baseline_m) < 1e-3:
        print(f"WARNING: pyrealsense2 returned baseline={baseline_m*1000:.4f} mm "
              f"(expected >1 mm); applying 1000x correction for D405.")
        baseline_m *= 1000.0

    camera_info_left = make_camera_info(ir_left_intrinsics, defaults["ir_width"], defaults["ir_height"])
    camera_info_right = make_camera_info(
        ir_right_intrinsics, defaults["ir_width"], defaults["ir_height"],
        baseline_tx=-ir_right_intrinsics.fx * baseline_m,
    )
    camera_info_color = make_camera_info(color_intrinsics, defaults["color_width"], defaults["color_height"])

    print(f"Baseline: {baseline_m * 1000:.2f} mm")
    print(f"IR Left fx={ir_left_intrinsics.fx:.2f} fy={ir_left_intrinsics.fy:.2f}")
    print()
    print("Controls: [r] record  [s] stop  [q] quit")

    print("Warming up camera...")
    for i in range(30):
        try:
            pipeline.wait_for_frames(timeout_ms=10000)
        except RuntimeError:
            if i < 5:
                print(f"  Frame {i}: timeout, retrying...")
                continue
            print("Camera not responding during warm-up. Check USB connection.")
            pipeline.stop()
            return
    print("Camera ready.")

    headless = args.headless
    old_term = None
    if headless and sys.stdin.isatty():
        old_term = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    fps_counter = 0
    fps_timer = time.time()
    display_fps = 0.0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError:
                print("Camera frame timeout — camera may have disconnected.")
                break

            color_frame = frames.get_color_frame()
            if stereo_mode == "ir1_color":
                ir_right_frame = frames.get_infrared_frame(1)
                ir_left_frame = None
            else:
                ir_left_frame = frames.get_infrared_frame(1)
                ir_right_frame = frames.get_infrared_frame(2)

            if not ir_right_frame or not color_frame:
                continue

            if ir_left_frame:
                ir_left_img = np.asanyarray(ir_left_frame.get_data())
            else:
                ir_left_img = np.asanyarray(color_frame.get_data())
                if len(ir_left_img.shape) == 3:
                    import cv2
                    ir_left_img = cv2.cvtColor(ir_left_img, cv2.COLOR_BGR2GRAY)
            ir_right_img = np.asanyarray(ir_right_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())

            if session.is_recording:
                timestamp = frames.get_timestamp() / 1000.0
                session.save_frame(timestamp, ir_left_img, ir_right_img, color_img)

            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()
                if headless and session.is_recording:
                    print(f"\r  FPS: {display_fps:.0f}  Frame: {session.frame_counter}", end="", flush=True)

            session.write_status(fps=display_fps)

            if headless:
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                else:
                    key = ""
            else:
                preview = build_preview(color_img, ir_left_img, ir_right_img, session, display_fps)
                cv2.imshow("RealSense Recorder", preview)
                k = cv2.waitKey(1) & 0xFF
                key = chr(k) if k != 255 else ""

            if key == "r":
                session.start_episode(camera_info_left, camera_info_right, camera_info_color)
            elif key == "s":
                session.stop_episode()
            elif key == "q":
                if session.is_recording:
                    session.stop_episode()
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
        if session.is_recording:
            session.stop_episode()
    finally:
        pipeline.stop()
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print(f"\nSession saved to: {session_dir}")
        print(f"Total episodes: {session.episode_count}")


if __name__ == "__main__":
    main()
