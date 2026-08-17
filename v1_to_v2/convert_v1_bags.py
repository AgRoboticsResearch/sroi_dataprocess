#!/usr/bin/env python3
"""
Convert SROI v1 ROS1 bags into v2-style PNG sessions (optionally also MP4).

Reads a v1 bag (oak_d_sr or realsense_d435i rig), splits it into episodes using
the /upi/status/is_action signal, synchronizes the stereo (+ color, when
present) streams, and writes sessions in the same layout produced by
record_realsense.py:

    <output>/<source-subdir>/<YYYYmmdd_HHMMSS>-png/
        episode_NNN/
            left_000000.png  right_000000.png  [color_000000.png]
            times.txt                     # one timestamp per frame, %.6f
            timestamps.json               # timestamps/from/to/fps/frame_count
            camera_info_{left,right[,color]}.json
            orb_slam_<camera_type>.yaml   # ORB-SLAM3 stereo config
        conversion.json              # provenance: segments, sources, counts

Camera intrinsics are ALWAYS taken from the bag's own camera_info topics.
Bags that carry no camera_info (the 2024-12-30 oak vintage) are only converted
when --fallback-camera-info points at a JSON exported with export_camera_info.py
— nothing is hardcoded here.

OAK notes (both cameras are color; there is no IR stream and no separate color
camera):
  - Episodes contain left_/right_*.png only. Downstream consumers
    (gripper_estimation_april_tag.py, lerobot/sroi_to_lerobot.py,
    visualization/visualize_traj_video.py) fall back to left_* when color_*
    is absent.
  - The OAK driver publishes the stereo baseline offset on the LEFT projection
    matrix (right-reference convention), so the signed baseline comes out
    negative; the magnitude is used for Stereo.T_c1_c2 and a note is printed.

IMU + extrinsics (kept for a future ORB-SLAM3 stereo-inertial pass — not wired
into the pipeline yet):
  - Every v1 bag carries one fused sensor_msgs/Imu stream (/camera/imu on the
    realsense rig ~200 Hz, /oak/imu/data on oak ~400 Hz). Each episode gets an
    EuRoC-style imu.txt: `timestamp_ns gx gy gz ax ay az`, sliced to the episode
    ±2 s. The stamps are the original header stamps, verified to share the image
    header clock on all three vintages, so they line up with times.txt.
  - <session>/imu_info.json holds the realsense IMUInfo messages (frame ids,
    noise/bias variances — zero-filled by the driver on these bags — and its
    `data` extrinsics field).
  - <session>/extrinsics.json holds the /tf_static tree; on the realsense bags
    that is where the real IMU-to-camera transform lives. Oak bags have neither
    file (nothing published).

Usage:
    # One bag
    python v1_to_v2/convert_v1_bags.py /mnt/data1/sroi/sroi_v1/ningbo_sroi_rosbags/rs435_2026-02-04-08-48-37.bag

    # Whole tree (dry run first); oak bags without camera_info need the fallback
    python v1_to_v2/convert_v1_bags.py /mnt/data1/sroi/sroi_v1 --dry-run
    python v1_to_v2/convert_v1_bags.py /mnt/data1/sroi/sroi_v1/shunba-shanjie \
        --fallback-camera-info v1_to_v2/fallback/oak_d_sr_camera_info.json

    # Also produce a parallel <session>-mp4 tree (decode_videos.py compatible)
    python v1_to_v2/convert_v1_bags.py <bag> --encode-mp4

    # Whole corpus as MP4 only: PNG trees would need ~265 GiB, the mp4 store
    # ~18 GiB. Regenerate PNG trees per batch later with decode_videos.py.
    python v1_to_v2/convert_v1_bags.py /mnt/data1/sroi/sroi_v1 \
        --fallback-camera-info v1_to_v2/fallback/oak_d_sr_camera_info.json \
        --mp4-only --skip-existing --jobs 8

Dependencies:
    pip install rosbags opencv-python numpy scipy
    pip install av        # for --encode-mp4 / --mp4-only
"""

import argparse
import json
import re
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.spatial import KDTree

try:
    import av
except ImportError:
    av = None

TYPESTORE = get_typestore(Stores.ROS1_NOETIC)
ACTION_TOPIC = "/upi/status/is_action"
TF_STATIC_TOPIC = "/tf_static"
PNG_COMPRESSION = 3  # match decode_videos.py; fast and lossless
IMU_PAD_S = 2.0      # IMU context kept on each side of an episode window
TEMPLATE_YAML = Path(__file__).resolve().parent.parent / "orb_slam_yaml" / "RealSense_D435i.yaml"

# Logical stream -> candidate base topics (without the /compressed suffix).
# RealSense infra streams are hardware-rectified; OAK v1 bags carry raw streams
# (the image_rect_color topics used by the very old pipeline do not exist here).
PROFILES = {
    "realsense_d435i": {
        "streams": {"left": ["/camera/infra1/image_rect_raw"],
                    "right": ["/camera/infra2/image_rect_raw"],
                    "color": ["/camera/color/image_raw"]},
        "camera_info": {"left": "/camera/infra1/camera_info",
                        "right": "/camera/infra2/camera_info",
                        "color": "/camera/color/camera_info"},
    },
    "oak_d_sr": {
        "streams": {"left": ["/oak/left/image_raw", "/oak/left/image_rect_color"],
                    "right": ["/oak/right/image_raw", "/oak/right/image_rect_color"],
                    "color": []},
        "camera_info": {"left": "/oak/left/camera_info",
                        "right": "/oak/right/camera_info"},
    },
}


def detect_profile(topics):
    """Pick the camera profile from the bag's topic list."""
    if any(t.startswith("/camera/infra") for t in topics):
        return "realsense_d435i"
    if any(t.startswith("/oak/") for t in topics):
        return "oak_d_sr"
    return None


@dataclass
class StreamIndex:
    role: str
    topic: str
    compressed: bool
    mono: bool = False                                  # grayscale payload (realsense infra)
    bag_times: list = field(default_factory=list)       # receive time, seconds
    header_times: list = field(default_factory=list)    # header.stamp, seconds

    def times(self, source):
        return np.asarray(self.header_times if source == "header" else self.bag_times)


@dataclass
class Segment:
    index: int
    start: float   # bag time, seconds (first is_action==True)
    end: float     # bag time, seconds (last is_action==True)


def deserialize(rawdata, msgtype):
    return TYPESTORE.deserialize_ros1(rawdata, msgtype)


_TYPE_LOCK = threading.Lock()


def register_bag_types(reader):
    """Register message types the stock ROS1_NOETIC store doesn't know.

    ROS1 bags embed each connection's message definition, so custom driver
    types (e.g. realsense2_camera/IMUInfo) can be registered from it. Locked
    because --jobs runs several bags in parallel threads.
    """
    unknown = [c for c in reader.connections
               if c.msgdef and c.msgtype not in TYPESTORE.types]
    if not unknown:
        return
    with _TYPE_LOCK:
        for c in unknown:
            if c.msgtype not in TYPESTORE.types:
                msgdef = c.msgdef.data if hasattr(c.msgdef, "data") else c.msgdef
                TYPESTORE.register(get_types_from_msg(msgdef, c.msgtype))


def header_sec(msg):
    stamp = msg.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def header_ns(msg):
    stamp = msg.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def camera_info_dict(msg):
    return {
        "height": int(msg.height),
        "width": int(msg.width),
        "distortion_model": msg.distortion_model,
        "D": [float(x) for x in msg.D],
        "K": [float(x) for x in msg.K],
        "R": [float(x) for x in msg.R],
        "P": [float(x) for x in msg.P],
    }


def imu_info_dict(msg):
    """realsense2_camera/IMUInfo. `data` is the driver's IMU extrinsics
    (column-major 3x3 rotation + translation); on these bags the driver left
    it at identity and the variances at 0 — the usable IMU extrinsics live in
    extrinsics.json (/tf_static) instead."""
    return {
        "frame_id": msg.frame_id,
        "data": [float(x) for x in msg.data],
        "noise_variances": [float(x) for x in msg.noise_variances],
        "bias_variances": [float(x) for x in msg.bias_variances],
    }


def tf_static_dict(msg):
    """/tf_static as a plain list of transform dicts."""
    out = []
    for tr in msg.transforms:
        t, q = tr.transform.translation, tr.transform.rotation
        out.append({
            "parent_frame": tr.header.frame_id,
            "child_frame": tr.child_frame_id,
            "translation_m": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
            "rotation_xyzw": {"x": float(q.x), "y": float(q.y),
                              "z": float(q.z), "w": float(q.w)},
        })
    return out


def resolve_streams(topics, profile):
    """Map logical roles to concrete topics, preferring compressed variants."""
    resolved = {}
    for role, bases in PROFILES[profile]["streams"].items():
        for base in bases:
            if base + "/compressed" in topics:
                resolved[base + "/compressed"] = StreamIndex(role, base + "/compressed", True)
                break
            if base in topics:
                resolved[base] = StreamIndex(role, base, False)
                break
    roles = {s.role for s in resolved.values()}
    if not {"left", "right"} <= roles:
        raise ValueError(f"could not find left/right image topics for {profile}")
    return resolved  # {topic: StreamIndex}


def find_segments(action_msgs, pad):
    """Contiguous True runs of /upi/status/is_action, in bag-receive time."""
    segments = []
    cur = None
    for t, active in action_msgs:
        if active and cur is None:
            cur = [t, t]
        elif active:
            cur[1] = t
        elif cur is not None:
            segments.append(Segment(len(segments) + 1, cur[0], cur[1]))
            cur = None
    if cur is not None:
        segments.append(Segment(len(segments) + 1, cur[0], cur[1]))
    if pad > 0:
        for s in segments:
            s.start -= pad
            s.end += pad
    return segments


def choose_time_source(streams):
    """Prefer header stamps (hardware time); fall back to bag receive time.

    Header stamps are what v2 records (frames.get_timestamp()) and for the
    realsense bags they are identical across the three streams, i.e. true
    hardware sync. Some devices (oak 2025-04 vintage) have a constant clock
    offset vs the bag writer, which is harmless; non-monotonic stamps are not.
    """
    for s in streams.values():
        t = np.asarray(s.header_times)
        if len(t) < 2 or not np.all(np.isfinite(t)):
            return "bag"
        if np.median(np.diff(t)) <= 0:
            return "bag"
        if int(np.sum(np.diff(t) < 0)) > len(t) // 100:
            return "bag"
    return "header"


def sync_streams(streams, source, tol):
    """Nearest-neighbor match of right(/color) frames to each left frame.

    Returns a list of (left_idx, right_idx, color_idx|None) for frames where
    every stream is within `tol` seconds; unmatched left frames are dropped
    (same policy as extract_stereo_rosbags.py).
    """
    by_role = {s.role: s for s in streams.values()}
    left_t = by_role["left"].times(source)
    if tol is None:
        tol = float(np.mean(np.diff(left_t)))  # one average frame period
    others = ["right"] + (["color"] if "color" in by_role else [])
    trees = {role: KDTree(by_role[role].times(source).reshape(-1, 1)) for role in others}

    matches = []
    for i, t in enumerate(left_t):
        picked = [i]
        for role in others:
            dt, idx = trees[role].query([[t]])
            if abs(float(dt[0])) > tol:
                picked = None
                break
            picked.append(int(idx[0]))
        if picked is not None:
            matches.append(tuple(picked))
    return matches, tol


def index_bag(bag_path, profile, streams):
    """Single pass over the relevant topics collecting stamps and metadata.

    Besides the image streams, action signal and camera_info, this also picks
    up everything worth keeping for a later stereo-inertial pass: fused
    sensor_msgs/Imu streams, the realsense IMUInfo messages, and /tf_static.
    """
    camera_infos = {}
    action_msgs = []
    imu_streams = {}   # topic -> [(bag_time_s, header_ns, gx, gy, gz, ax, ay, az)]
    imu_infos = {}     # topic -> dict (realsense2_camera/IMUInfo)
    tf_static = []     # /tf_static transforms as dicts
    with Reader(str(bag_path)) as reader:
        register_bag_types(reader)
        topics = {c.topic for c in reader.connections}
        info_topics = {v: k for k, v in PROFILES[profile]["camera_info"].items()
                       if v in topics}
        imu_topics = {c.topic for c in reader.connections if c.msgtype.endswith("/Imu")}
        imu_info_topics = {c.topic for c in reader.connections
                           if c.msgtype.endswith("/IMUInfo")}
        want_tf = TF_STATIC_TOPIC in topics
        wanted = [c for c in reader.connections
                  if c.topic in streams or c.topic in info_topics or c.topic == ACTION_TOPIC
                  or c.topic in imu_topics or c.topic in imu_info_topics
                  or (want_tf and c.topic == TF_STATIC_TOPIC)]
        mono_probed = set()
        for conn, t_ns, rawdata in reader.messages(wanted):
            topic = conn.topic
            if topic == ACTION_TOPIC:
                action_msgs.append((t_ns / 1e9, bool(deserialize(rawdata, conn.msgtype).data)))
            elif topic in info_topics:
                role = info_topics[topic]
                if role not in camera_infos:
                    camera_infos[role] = camera_info_dict(deserialize(rawdata, conn.msgtype))
            elif topic in imu_topics:
                msg = deserialize(rawdata, conn.msgtype)
                g, a = msg.angular_velocity, msg.linear_acceleration
                imu_streams.setdefault(topic, []).append(
                    (t_ns / 1e9, header_ns(msg), g.x, g.y, g.z, a.x, a.y, a.z))
            elif topic in imu_info_topics:
                if topic not in imu_infos:
                    imu_infos[topic] = imu_info_dict(deserialize(rawdata, conn.msgtype))
            elif want_tf and topic == TF_STATIC_TOPIC:
                if not tf_static:
                    tf_static = tf_static_dict(deserialize(rawdata, conn.msgtype))
            else:
                s = streams[topic]
                msg = deserialize(rawdata, conn.msgtype)
                if s.compressed and topic not in mono_probed:
                    mono_probed.add(topic)
                    s.mono = str(msg.format).startswith("mono")
                s.bag_times.append(t_ns / 1e9)
                s.header_times.append(header_sec(msg))
    return camera_infos, action_msgs, imu_streams, imu_infos, tf_static


def load_fallback_camera_info(path):
    with open(path) as f:
        data = json.load(f)
    for role in ("left", "right"):
        if role not in data:
            raise ValueError(f"fallback camera info {path} is missing the '{role}' entry")
    return data


def write_imu_file(ep_dir, imu_streams, t_first, t_last):
    """Per-episode imu.txt in EuRoC order: `timestamp_ns gx gy gz ax ay az`.

    Rows keep their original header stamps (nanoseconds), which share the image
    header clock on every v1 vintage, so the index lines up with times.txt.
    Sliced to [t_first - IMU_PAD_S, t_last + IMU_PAD_S] for some context on
    either side (initialization margin for a future stereo-inertial run).
    Returns {filename: sample_count}.
    """
    lo = int((t_first - IMU_PAD_S) * 1e9)
    hi = int((t_last + IMU_PAD_S) * 1e9)
    counts = {}
    for i, (topic, rows) in enumerate(sorted(imu_streams.items())):
        name = "imu.txt" if len(imu_streams) == 1 else f"imu_{i + 1}.txt"
        kept = [r for r in rows if lo <= r[1] <= hi]
        with open(ep_dir / name, "w") as f:
            f.write("# timestamp_ns gx gy gz ax ay az (header stamps, EuRoC imu.txt order)\n")
            for _, hns, gx, gy, gz, ax, ay, az in kept:
                f.write(f"{hns} {gx:.9f} {gy:.9f} {gz:.9f} {ax:.9f} {ay:.9f} {az:.9f}\n")
        counts[name] = len(kept)
    return counts


def calibration_note_text(bag_path, roles, fallback_label, fallback_info):
    """Provenance note written as CALIBRATION_NOTE.txt when the bag itself
    carries no camera_info and a fallback file was used instead."""
    src = fallback_info.get("source", {})
    src_bag = src.get("bag", "unknown")
    src_start = src.get("bag_start_utc", "unknown")
    return (
        "CALIBRATION PROVENANCE WARNING\n"
        "==============================\n"
        f"The source rosbag\n\n    {bag_path}\n\n"
        f"contains NO camera_info topics for: {', '.join(roles)}.\n\n"
        "The camera_info_*.json files and the ORB-SLAM yaml in this episode were\n"
        "filled from a fallback calibration file:\n\n"
        f"    {fallback_label}\n\n"
        "which was exported from a different bag of the same rig:\n\n"
        f"    {src_bag}\n"
        f"    (bag start {src_start})\n\n"
        "Intrinsics are therefore NOT from this bag. Treat calibration-sensitive\n"
        "results (ORB-SLAM scale, gripper distance) accordingly.\n\n"
        f"Converted: {datetime.now(tz=timezone.utc).isoformat()}\n"
    )


def session_name(bag_path, start_ns):
    """YYYYmmdd_HHMMSS from the bag filename (v1 recorders stamp UTC start
    times in the name); fall back to the bag start time."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", bag_path.stem)
    if m:
        digits = "".join(m.groups())
        return digits[:8] + "_" + digits[8:]
    return datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def relative_group(bag_path, input_root):
    """Mirror the bag's location under the input root (keeps sites apart)."""
    try:
        rel = bag_path.parent.relative_to(input_root)
        return rel if str(rel) != "." else Path()
    except ValueError:
        return Path(bag_path.parent.name)


def write_orb_slam_yaml(left_info, right_info, fps, out_path):
    """ORB-SLAM3 stereo yaml, same format record_realsense.py writes.

    Camera1 comes from the left camera_info, Camera2 from the right one (the
    OAK pair has slightly different intrinsics per camera). The template only
    supplies the ORB/viewer parameters. Returns (baseline_m, sign_note).
    """
    import yaml
    with open(TEMPLATE_YAML) as f:
        content = f.read()
    if content.startswith("%YAML:1.0"):
        content = content.replace("%YAML:1.0\n", "", 1)
    try:
        config = yaml.load(content, Loader=yaml.UnsafeLoader)
    except yaml.YAMLError:
        config = yaml.safe_load(content)

    lK = np.array(left_info["K"]).reshape(3, 3)
    rK = np.array(right_info["K"]).reshape(3, 3)
    lP = np.array(left_info["P"]).reshape(3, 4)
    rP = np.array(right_info["P"]).reshape(3, 4)

    for prefix, K, info in (("Camera1", lK, left_info), ("Camera2", rK, right_info)):
        D = info["D"]
        config[f"{prefix}.fx"] = float(K[0, 0])
        config[f"{prefix}.fy"] = float(K[1, 1])
        config[f"{prefix}.cx"] = float(K[0, 2])
        config[f"{prefix}.cy"] = float(K[1, 2])
        config[f"{prefix}.k1"] = float(D[0]) if len(D) > 0 else 0.0
        config[f"{prefix}.k2"] = float(D[1]) if len(D) > 1 else 0.0
        config[f"{prefix}.p1"] = float(D[2]) if len(D) > 2 else 0.0
        config[f"{prefix}.p2"] = float(D[3]) if len(D) > 3 else 0.0

    config["Camera.width"] = int(left_info["width"])
    config["Camera.height"] = int(left_info["height"])
    config["Camera.fps"] = int(fps)

    # ROS stereo convention puts -fx*baseline on the RIGHT P; depthai (oak)
    # publishes it on the LEFT P (right-reference), which yields a negative
    # value here. Use the magnitude in both cases.
    baseline = -(rP[0, 3] - lP[0, 3]) / lK[0, 0]
    sign_note = ""
    if baseline < 0:
        sign_note = " (sign flipped: driver puts the offset on the left P matrix)"
        baseline = abs(baseline)
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = baseline
    config["Stereo.T_c1_c2"] = {"rows": 4, "cols": 4, "dt": "f", "data": T.flatten().tolist()}

    with open(out_path, "w") as f:
        f.write("%YAML:1.0\n\n")
        for key, value in config.items():
            if key == "Stereo.T_c1_c2" and isinstance(value, dict):
                f.write(f"{key}: !!opencv-matrix\n")
                f.write(f"  rows: {value['rows']}\n  cols: {value['cols']}\n")
                f.write(f"  dt: {value['dt']}\n  data: {value['data']}\n")
            elif isinstance(value, str):
                f.write(f'{key}: "{value}"\n')
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
    return float(baseline), sign_note


def decode_frame(rawdata, msgtype, stream):
    msg = deserialize(rawdata, msgtype)
    if stream.compressed:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        flags = cv2.IMREAD_GRAYSCALE if stream.mono else cv2.IMREAD_COLOR
        img = cv2.imdecode(buf, flags)
    else:
        if msg.encoding == "mono8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        elif msg.encoding == "bgr8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        elif msg.encoding == "rgb8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"unsupported encoding {msg.encoding} on {stream.topic}")
    if img is None:
        raise ValueError(f"failed to decode frame from {stream.topic}")
    return img


def open_video_encoder(path, width, height, fps, vcodec):
    """Same encoder settings as record_realsense.py / encode_episode_mp4."""
    if av is None:
        raise SystemExit("PyAV is required for mp4 output: pip install av")
    container = av.open(str(path), "w")
    stream = container.add_stream(vcodec, fps, options={"crf": "23", "g": "2"})
    stream.pix_fmt = "yuv420p"
    stream.width = width
    stream.height = height
    return container, stream


def feed_video_encoder(encoder, img_bgr):
    container, stream = encoder
    if img_bgr.ndim == 2:  # grayscale (realsense infra) -> 3 planes for yuv420p
        img = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
        container.mux(packet)


def close_video_encoder(encoder):
    container, stream = encoder
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def encode_episode_mp4(episode_png, episode_mp4, roles, fps, vcodec):
    """left.mp4/right.mp4(/color.mp4) with the same settings as
    record_realsense.py; the -mp4 tree stays decode_videos.py compatible."""
    if av is None:
        raise SystemExit("PyAV is required for --encode-mp4: pip install av")
    episode_mp4.mkdir(parents=True, exist_ok=True)
    for name in ("camera_info_left.json", "camera_info_right.json", "camera_info_color.json",
                 "times.txt", "timestamps.json", "imu.txt", "CALIBRATION_NOTE.txt"):
        src = episode_png / name
        if src.exists():
            shutil.copy2(src, episode_mp4 / name)
    for yml in episode_png.glob("*.yaml"):
        shutil.copy2(yml, episode_mp4 / yml.name)

    for role in roles:
        input_list = sorted(episode_png.glob(f"{role}_*.png"))
        if not input_list:
            continue
        first = cv2.imread(str(input_list[0]))
        h, w = first.shape[:2]
        with av.open(str(episode_mp4 / f"{role}.mp4"), "w") as output:
            out_stream = output.add_stream(vcodec, fps, options={"crf": "23", "g": "2"})
            out_stream.pix_fmt = "yuv420p"
            out_stream.width = w
            out_stream.height = h
            for img_path in input_list:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"cannot read {img_path}")
                frame = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                   format="rgb24")
                for packet in out_stream.encode(frame):
                    output.mux(packet)
            for packet in out_stream.encode():
                output.mux(packet)


def probe_image_sizes(bag_path, streams):
    """Decode the first message of each image stream to get its true (h, w)."""
    sizes = {}
    with Reader(str(bag_path)) as reader:
        for s in streams.values():
            conns = [c for c in reader.connections if c.topic == s.topic]
            for conn, t_ns, rawdata in reader.messages(conns):
                img = decode_frame(rawdata, conn.msgtype, s)
                sizes[s.role] = (int(img.shape[0]), int(img.shape[1]))
                break
    return sizes


def validate_calibration(camera_infos, sizes, fallback_label, fallback_roles=()):
    """Check that every stream's camera_info matches the actual image pixels.

    Returns a list of error strings (empty = OK). Catches both mislabeled bags
    and a fallback calibration that belongs to a different resolution vintage.
    """
    errors = []
    for role, (h, w) in sorted(sizes.items()):
        info = camera_infos.get(role)
        if info is None:
            continue
        if (info["width"], info["height"]) != (w, h):
            src = f" (from fallback {fallback_label})" if role in fallback_roles else ""
            errors.append(f"{role}: images are {w}x{h} but camera_info says "
                          f"{info['width']}x{info['height']}{src}")
            continue
        K = info["K"]
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        if not (fx > 0 and fy > 0 and 0 <= cx <= w and 0 <= cy <= h):
            errors.append(f"{role}: implausible intrinsics fx={fx} fy={fy} "
                          f"cx={cx} cy={cy} for {w}x{h}")
    return errors


def convert_bag(bag_path, input_root, output_root, profile=None, fallback_info=None,
                fallback_label="-", dry_run=False, skip_existing=False, whole_bag=False,
                min_seg_frames=15, pad=0.0, sync_tol=None, encode_mp4=False,
                vcodec="libx264", mp4_only=False):
    bag_path = Path(bag_path)
    result = {"bag": str(bag_path), "status": "ok", "episodes": []}

    with Reader(str(bag_path)) as reader:
        register_bag_types(reader)
        start_ns = reader.start_time
        topics = {c.topic for c in reader.connections}
    result["bag_start_utc"] = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat()

    detected = detect_profile(topics)
    profile = profile or detected
    if profile is None:
        result["status"] = "error"
        result["error"] = "cannot detect camera type from topics"
        return result
    if detected is not None and profile != detected:
        print(f"  NOTE: topics look like {detected}, forcing {profile}")
    result["profile"] = profile

    streams = resolve_streams(topics, profile)
    camera_infos, action_msgs, imu_streams, imu_infos, tf_static = index_bag(
        bag_path, profile, streams)
    result["streams"] = {s.role: s.topic for s in streams.values()}
    if imu_streams:
        result["imu_topics"] = {t: len(rows) for t, rows in imu_streams.items()}
    if tf_static:
        result["tf_static_transforms"] = len(tf_static)

    # Camera info: always from this bag; the fallback only fills gaps.
    missing = [r for r in PROFILES[profile]["camera_info"] if r not in camera_infos]
    fallback_roles_used = ()
    if missing:
        required = [r for r in missing if r in ("left", "right")]
        optional = [r for r in missing if r == "color"]
        if fallback_info is None or not all(r in fallback_info for r in required):
            result["status"] = "error"
            result["error"] = (f"bag has no camera_info for {missing} and no "
                               f"--fallback-camera-info covering {required}; skipping")
            return result
        for role in required + [r for r in optional if r in fallback_info]:
            camera_infos[role] = fallback_info[role]
        fallback_roles_used = tuple(required) + tuple(r for r in optional if r in fallback_info)
        result["camera_info_source"] = f"fallback ({', '.join(required)} from {fallback_label})"
        result["fallback_note"] = calibration_note_text(bag_path, required, fallback_label,
                                                        fallback_info)
        print(f"  WARNING: no camera_info in bag for {missing}; using fallback {fallback_label}")
    else:
        result["camera_info_source"] = "bag"

    # Calibration must match the actual image pixels of this bag.
    sizes = probe_image_sizes(bag_path, streams)
    result["image_sizes"] = {role: f"{w}x{h}" for role, (h, w) in sizes.items()}
    problems = validate_calibration(camera_infos, sizes, fallback_label,
                                    fallback_roles=fallback_roles_used)
    if problems:
        result["status"] = "error"
        result["error"] = "; ".join(problems)
        return result

    # Episodes: is_action segments, or the whole bag when there is no signal.
    left_bt = streams_left(streams).bag_times
    if whole_bag:
        segments = [Segment(1, left_bt[0], left_bt[-1])]
        result["segmentation"] = "whole bag (--whole-bag)"
    else:
        segments = find_segments(action_msgs, pad)
        if segments:
            result["segmentation"] = (f"{len(action_msgs)} is_action msgs -> "
                                      f"{len(segments)} segments")
        elif action_msgs:
            segments = [Segment(1, left_bt[0], left_bt[-1])]
            result["segmentation"] = (f"is_action present but never True "
                                      f"({len(action_msgs)} msgs); whole bag -> 1 episode")
        else:
            segments = [Segment(1, left_bt[0], left_bt[-1])]
            result["segmentation"] = "no is_action topic; whole bag -> 1 episode"

    source = choose_time_source(streams)
    result["time_source"] = source
    by_role = {s.role: s for s in streams.values()}
    matches, used_tol = sync_streams(streams, source, sync_tol)
    result["sync"] = (f"{len(matches)}/{len(by_role['left'].bag_times)} frames synced "
                      f"(tol {used_tol * 1000:.1f} ms)")

    # Assign synced frames to segments by the left frame's bag-receive time
    # (is_action carries no header, so segmentation lives in bag-receive time).
    left_sel = by_role["left"].times(source)
    kept = []
    for seg in segments:
        frames = [m for m in matches if seg.start <= left_bt[m[0]] <= seg.end]
        if len(frames) >= min_seg_frames:
            kept.append((seg, frames))
        elif frames:
            print(f"  dropping segment {seg.index}: {len(frames)} frames < {min_seg_frames}")
    result["dropped_segments"] = len(segments) - len(kept)

    session_dir = output_root / relative_group(bag_path, input_root) / \
        f"{session_name(bag_path, start_ns)}{'-mp4' if mp4_only else '-png'}"
    result["session_dir"] = str(session_dir)

    # Nominal fps for the whole bag (used where an episode has <2 frames).
    dts = np.diff(left_sel[[m[0] for m in matches]])
    fps = int(round(1.0 / float(np.median(dts)))) if len(dts) else 30

    roles = [r for r in ("left", "right", "color") if r in by_role]
    per_episode = []
    for i, (seg, frames) in enumerate(kept):
        ep = {"episode": f"episode_{i + 1:03d}", "segment": seg.index, "frames": len(frames)}
        if len(frames) > 1:
            dt = float(np.median(np.diff(left_sel[[m[0] for m in frames]])))
            ep["duration_s"] = round(dt * len(frames), 2)
            ep["fps"] = int(round(1.0 / dt))
        else:
            ep["fps"] = fps
        per_episode.append(ep)

    if dry_run:
        result["episodes"] = per_episode
        imu_desc = f", imu: {', '.join(result['imu_topics'])}" if result.get("imu_topics") else ""
        print(f"  [dry-run] {session_dir}  ({len(kept)} episodes, {result['sync']}, "
              f"info: {result['camera_info_source']}{imu_desc})")
        for ep in per_episode:
            print(f"    {ep['episode']}: {ep['frames']} frames "
                  f"(~{ep.get('duration_s', '?')}s, segment {ep['segment']})")
        return result

    yaml_name = f"orb_slam_{profile}.yaml"
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"  session: {session_dir.name}  ({len(kept)} episodes, {result['sync']})")

    # keep: (topic, msg_index) -> (episode_dir, out_index)
    keep = {}
    aborted = set()
    for ep, (seg, frames) in zip(per_episode, kept):
        ep_dir = session_dir / ep["episode"]
        ep["dir"] = str(ep_dir)
        if skip_existing and (ep_dir / "timestamps.json").exists():
            ep["skipped"] = True
            # backfill imu.txt for episodes converted before IMU saving existed
            if imu_streams and not (ep_dir / "imu.txt").exists():
                ft = left_sel[[m[0] for m in frames]]
                ep["imu_samples"] = write_imu_file(ep_dir, imu_streams,
                                                   float(ft[0]), float(ft[-1]))
            continue
        for out_idx, m in enumerate(frames):
            for role, msg_idx in zip(roles, m):
                keep[(by_role[role].topic, msg_idx)] = (ep_dir, out_idx)

    # Second pass: decode the frames we keep. PNG mode writes them out; mp4-only
    # feeds them straight into per-episode encoders — no PNG round-trip (the
    # whole corpus is ~7% of the PNG size; decode_videos.py regenerates PNG
    # trees per batch later).
    ep_fps = {session_dir / ep["episode"]: ep["fps"] for ep in per_episode}
    counters = defaultdict(int)
    encoders = {}   # (episode_dir, role) -> (container, stream), opened lazily
    with Reader(str(bag_path)) as reader:
        wanted = [c for c in reader.connections if c.topic in streams]
        for conn, t_ns, rawdata in reader.messages(wanted):
            topic = conn.topic
            idx = counters[topic]
            counters[topic] += 1
            hit = keep.get((topic, idx))
            if hit is None:
                continue
            ep_dir, out_idx = hit
            if ep_dir in aborted:
                continue
            try:
                img = decode_frame(rawdata, conn.msgtype, streams[topic])
                role = streams[topic].role
                if mp4_only:
                    key = (ep_dir, role)
                    if key not in encoders:
                        ep_dir.mkdir(parents=True, exist_ok=True)
                        encoders[key] = open_video_encoder(
                            ep_dir / f"{role}.mp4", img.shape[1], img.shape[0],
                            ep_fps[ep_dir], vcodec)
                    feed_video_encoder(encoders[key], img)
                else:
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(ep_dir / f"{role}_{out_idx:06d}.png"), img,
                                [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            except Exception as e:
                print(f"  ERROR in {ep_dir.name} ({topic} #{idx}): {e}; aborting episode")
                aborted.add(ep_dir)
                if ep_dir.exists():
                    shutil.rmtree(ep_dir)
                for key in [k for k in encoders if k[0] == ep_dir]:
                    try:
                        encoders.pop(key)[0].close()
                    except Exception:
                        pass

    for encoder in encoders.values():
        close_video_encoder(encoder)

    # Per-episode metadata (times.txt, timestamps.json, camera_info, yaml).
    for ep, (seg, frames) in zip(per_episode, kept):
        if ep.get("skipped"):
            ep["status"] = "skipped"
            continue
        ep_dir = Path(ep["dir"])
        if ep_dir in aborted or not ep_dir.exists():
            ep["status"] = "aborted"
            continue
        times = [float(left_sel[m[0]]) for m in frames]
        with open(ep_dir / "times.txt", "w") as f:
            for t in times:
                f.write(f"{t:.6f}\n")
        ep_fps = ep["fps"]
        with open(ep_dir / "timestamps.json", "w") as f:
            json.dump({
                "timestamps": times,
                "from_timestamp": times[0] if times else None,
                "to_timestamp": times[-1] if times else None,
                "fps": ep_fps,
                "frame_count": len(times),
            }, f, indent=2)
        if imu_streams:
            ep["imu_samples"] = write_imu_file(ep_dir, imu_streams, times[0], times[-1])
        for role in ("left", "right", "color"):
            if role in camera_infos:
                with open(ep_dir / f"camera_info_{role}.json", "w") as f:
                    json.dump(camera_infos[role], f, indent=2)
        if result.get("fallback_note"):
            ep_dir.joinpath("CALIBRATION_NOTE.txt").write_text(result["fallback_note"])
        baseline, sign_note = write_orb_slam_yaml(
            camera_infos["left"], camera_infos["right"], ep_fps, ep_dir / yaml_name)
        ep.update({"status": "ok", "fps": ep_fps, "baseline_m": round(baseline, 6)})
        print(f"  {ep['episode']}: {ep['frames']} frames @ {ep_fps} fps, "
              f"baseline {baseline * 1000:.2f} mm{sign_note}")

    result["episodes"] = per_episode
    result["yaml"] = yaml_name
    result["format"] = "mp4-only" if mp4_only else "png"
    if imu_infos:
        with open(session_dir / "imu_info.json", "w") as f:
            json.dump(imu_infos, f, indent=2)
    if tf_static:
        with open(session_dir / "extrinsics.json", "w") as f:
            json.dump({"source": "/tf_static", "transforms": tf_static}, f, indent=2)
    with open(session_dir / "conversion.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    if result.get("fallback_note"):
        (session_dir / "CALIBRATION_NOTE.txt").write_text(result["fallback_note"])

    if encode_mp4:
        mp4_root = session_dir.parent / f"{session_dir.name[:-4]}-mp4"
        for ep in per_episode:
            if ep.get("status") != "ok":
                continue
            ep_png = Path(ep["dir"])
            ep_mp4 = mp4_root / ep_png.name
            if skip_existing and (ep_mp4 / "left.mp4").exists():
                continue
            encode_episode_mp4(ep_png, ep_mp4, roles, ep["fps"], vcodec)
            print(f"  {ep['episode']}: mp4 -> {ep_mp4}")
        result["mp4_dir"] = str(mp4_root)
    return result


def streams_left(streams):
    return next(s for s in streams.values() if s.role == "left")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SROI v1 ROS1 bags to v2-style PNG sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="a .bag file or a directory containing bags")
    parser.add_argument("-o", "--output", default=None,
                        help="output root (default: <input>/v1_to_v2, or <bag parent>/v1_to_v2)")
    parser.add_argument("--camera-type", choices=list(PROFILES), default=None,
                        help="override camera detection (default: auto from topics)")
    parser.add_argument("--fallback-camera-info", default=None,
                        help="JSON with left/right(/color) camera_info, used ONLY for bags that "
                             "carry no camera_info topics (create with export_camera_info.py)")
    parser.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip episodes that already have timestamps.json")
    parser.add_argument("--whole-bag", action="store_true",
                        help="ignore is_action and convert the whole bag as one episode")
    parser.add_argument("--min-seg-frames", type=int, default=15,
                        help="drop action segments shorter than this many frames (default 15)")
    parser.add_argument("--pad-seconds", type=float, default=0.0,
                        help="extend each action segment by this many seconds on both sides")
    parser.add_argument("--sync-tol", type=float, default=None,
                        help="max timestamp mismatch between streams in seconds "
                             "(default: one average frame period)")
    parser.add_argument("--encode-mp4", action="store_true",
                        help="also write a parallel <session>-mp4 tree (decode_videos.py compatible)")
    parser.add_argument("--vcodec", default="libx264", choices=["libx264", "libsvtav1", "hevc"])
    parser.add_argument("--mp4-only", action="store_true",
                        help="write only a <session>-mp4 tree per bag, encoding straight from "
                             "decoded frames (no PNG round-trip; decode_videos.py regenerates "
                             "PNG trees per batch later)")
    parser.add_argument("--jobs", type=int, default=1, help="parallel bags (default 1)")
    args = parser.parse_args()

    if args.mp4_only and args.encode_mp4:
        raise SystemExit("--mp4-only and --encode-mp4 are mutually exclusive: "
                         "mp4-only already writes the mp4 tree")

    input_path = Path(args.input).resolve()
    if input_path.is_dir():
        bags = sorted(p for p in input_path.rglob("*.bag"))
    elif input_path.suffix == ".bag":
        bags = [input_path]
    else:
        raise SystemExit(f"input must be a .bag file or a directory: {input_path}")
    if not bags:
        raise SystemExit(f"no .bag files found under {input_path}")

    input_root = input_path if input_path.is_dir() else input_path.parent
    output_root = Path(args.output).resolve() if args.output else input_root / "v1_to_v2"

    fallback_info = None
    if args.fallback_camera_info:
        fallback_info = load_fallback_camera_info(args.fallback_camera_info)

    print(f"Converting {len(bags)} bag(s) -> {output_root}")
    if args.dry_run:
        print("DRY RUN: nothing will be written")

    def run(bag):
        t0 = time.time()
        try:
            rel = bag.relative_to(input_root) if bag.is_relative_to(input_root) else bag
        except ValueError:
            rel = bag
        print(f"\n=== {rel} ===")
        try:
            res = convert_bag(
                bag, input_root, output_root, profile=args.camera_type,
                fallback_info=fallback_info,
                fallback_label=args.fallback_camera_info or "-",
                dry_run=args.dry_run, skip_existing=args.skip_existing,
                whole_bag=args.whole_bag, min_seg_frames=args.min_seg_frames,
                pad=args.pad_seconds, sync_tol=args.sync_tol,
                encode_mp4=args.encode_mp4, vcodec=args.vcodec, mp4_only=args.mp4_only)
        except Exception as e:
            res = {"bag": str(bag), "status": "error",
                   "error": f"{type(e).__name__}: {e}"}
        res["elapsed_s"] = round(time.time() - t0, 1)
        return res

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run, bags))
    else:
        results = [run(b) for b in bags]

    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    total_eps = sum(len(r.get("episodes", [])) for r in ok)
    print(f"\n{'=' * 60}")
    print(f"Done: {len(ok)}/{len(results)} bags ok, {total_eps} episodes")
    for r in err:
        print(f"  ERROR {Path(r['bag']).name}: {r['error']}")
    return 0 if not err else 1


if __name__ == "__main__":
    sys.exit(main())
