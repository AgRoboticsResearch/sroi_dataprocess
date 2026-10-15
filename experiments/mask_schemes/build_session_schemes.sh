#!/bin/bash
# Build a full schemes tree for one session and run ORB for every scheme.
# Intermediate -png and -png-mask292 directories are removed after copying.
# Usage: build_session_schemes.sh <session-basename>  Example: 20260709_144719
set -e
B="$1"
BASE="${MASK_DATA:-/home/ss/data/1000_onesb_labpicking}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VIZ="$(cd "$(dirname "$0")" && pwd)"
MP4="$BASE/$B-mp4"; PNG="$BASE/$B-png"; MASK="$BASE/$B-png-mask292"; SCH="$BASE/$B-schemes"

echo "### 1) decode $B"
conda run -n lerobot --no-capture-output python "$REPO/decode_videos.py" "$MP4" --recursive

echo "### 2) mask half (cutoff=292)"
python3 "$VIZ/mask_session.py" "$PNG" 292

echo "### 3) schemes init (raw + maskhalf)"
python3 "$VIZ/schemes_init.py" "$PNG" "$MASK" "$SCH"

echo "### 4) remove copied -png / -png-mask292 intermediates"
rm -rf "$PNG" "$MASK"

echo "### 5) apply maskgripper trapezoid mask"
python3 "$VIZ/mask_gripper_trapezoid.py" "$SCH"

echo "### 6) ORB: raw / maskhalf / maskgripper"
for sch in raw maskhalf maskgripper; do
  python3 "$VIZ/run_orb_scheme.py" "$SCH" "$sch"
done

echo "=== $B schemes complete -> $SCH ==="
