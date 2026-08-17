#!/usr/bin/env python3
"""
Export camera_info from a v1 rosbag to a JSON file.

The output is the file consumed by convert_v1_bags.py --fallback-camera-info
for bags that carry no camera_info topics of their own (the 2024-12-30 oak
vintage). Export from a bag of the SAME physical rig so the fallback is a real
calibration, not an invented one:

    python v1_to_v2/export_camera_info.py \
        /mnt/data1/sroi/sroi_v1/shunba-shanjie/oak_d_sr_2025-04-14-08-21-26.bag \
        -o v1_to_v2/fallback/oak_d_sr_camera_info.json

Output format:
    {
      "source": {...provenance...},
      "left":  {height,width,distortion_model,D,K,R,P},
      "right": {...},
      "color": {...}     # only when the bag has a color stream
    }
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

TYPESTORE = get_typestore(Stores.ROS1_NOETIC)

INFO_TOPICS = {
    "left": ["/oak/left/camera_info", "/camera/infra1/camera_info"],
    "right": ["/oak/right/camera_info", "/camera/infra2/camera_info"],
    "color": ["/camera/color/camera_info"],
}


def first_camera_info(reader, topic):
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        return None
    for conn, t_ns, rawdata in reader.messages(conns):
        msg = TYPESTORE.deserialize_ros1(rawdata, conn.msgtype)
        return {
            "height": int(msg.height),
            "width": int(msg.width),
            "distortion_model": msg.distortion_model,
            "D": [float(x) for x in msg.D],
            "K": [float(x) for x in msg.K],
            "R": [float(x) for x in msg.R],
            "P": [float(x) for x in msg.P],
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Export camera_info from a v1 rosbag.")
    parser.add_argument("bag", help="path to the source .bag")
    parser.add_argument("-o", "--output", required=True, help="output JSON path")
    args = parser.parse_args()

    out = {"source": {
        "bag": str(Path(args.bag).resolve()),
        "bag_start_utc": None,
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
    }}
    with Reader(args.bag) as reader:
        out["source"]["bag_start_utc"] = datetime.fromtimestamp(
            reader.start_time / 1e9, tz=timezone.utc).isoformat()
        topics = {c.topic for c in reader.connections}
        for role, candidates in INFO_TOPICS.items():
            for topic in candidates:
                if topic in topics:
                    info = first_camera_info(reader, topic)
                    if info:
                        out[role] = info
                        print(f"{role}: {topic} ({info['width']}x{info['height']}, "
                              f"{info['distortion_model']})")
                    break

    for role in ("left", "right"):
        if role not in out:
            raise SystemExit(f"no left/right camera_info found in {args.bag}")

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
