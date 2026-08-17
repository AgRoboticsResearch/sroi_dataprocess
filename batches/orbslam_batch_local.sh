#!/bin/bash
# Batch process ORB_SLAM3 for episode folders (host install).
#
# Usage:
#   ./orbslam_batch_local.sh <session_dir> <orb_slam_dir> [skip_existing] [visualization] \
#       [--mask-config PATH] [--jobs N]
#
# Flags may appear in any order. --jobs N runs N stereo_kitti episodes concurrently
# (default 4; 1 = sequential, the original behavior). Each episode runs in its own
# subshell with its own masked temp dir, so concurrent runs are fully isolated.
#
# Examples:
#   ./orbslam_batch_local.sh /data/20260521_153031-png ~/code/ORB_SLAM3
#   ./orbslam_batch_local.sh /data/20260521_153031-png ~/code/ORB_SLAM3 true false
#   ./orbslam_batch_local.sh /data/...-png ~/code/ORB_SLAM3 true false --mask-config configs/gripper_mask_sroi_v2_d405.json
#   ./orbslam_batch_local.sh /data/...-png ~/code/ORB_SLAM3 false false --mask-config configs/gripper_mask_sroi_v2_d405.json --jobs 8

SESSION_DIR=""
ORB_SLAM_DIR=""
SKIP_EXIST="true"
VISUALIZATION="false"
MASK_CONFIG=""
JOBS="4"

_positionals=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mask-config)
            [ -z "${2:-}" ] && { echo "Error: --mask-config needs a path" >&2; exit 1; }
            MASK_CONFIG="$2"; shift 2;;
        --mask-config=*)
            MASK_CONFIG="${1#--mask-config=}"; shift;;
        --jobs)
            [ -z "${2:-}" ] && { echo "Error: --jobs needs a positive integer" >&2; exit 1; }
            JOBS="$2"; shift 2;;
        --jobs=*)
            JOBS="${1#--jobs=}"; shift;;
        --)
            shift; while [ $# -gt 0 ]; do _positionals+=("$1"); shift; done;;
        *)
            _positionals+=("$1"); shift;;
    esac
done
SESSION_DIR="${_positionals[0]:-}"
ORB_SLAM_DIR="${_positionals[1]:-}"
SKIP_EXIST="${_positionals[2]:-true}"
VISUALIZATION="${_positionals[3]:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/orbslam_mask_helpers.sh"

if [ -z "$SESSION_DIR" ] || [ -z "$ORB_SLAM_DIR" ]; then
    echo "Usage: $0 <session_dir> <orb_slam_dir> [skip_existing] [visualization] [--mask-config PATH] [--jobs N]"
    echo "  session_dir:    path containing episode_* folders"
    echo "  orb_slam_dir:   path to ORB_SLAM3 installation"
    echo "  skip_existing:  \"true\" to skip episodes with CameraTrajectory.txt (default: true)"
    echo "  visualization:  \"true\" to enable GUI (default: false)"
    echo "  --mask-config:  optional gripper-mask JSON; omitted means no masking"
    echo "  --jobs N:       concurrent stereo_kitti episodes (default: 4; 1 = sequential)"
    exit 1
fi

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
    echo "Error: --jobs must be a positive integer (got '$JOBS')" >&2
    exit 1
fi

if ! sroi_configure_mask "$MASK_CONFIG"; then
    exit 1
fi

STEREO_KITTI="$ORB_SLAM_DIR/Examples/Stereo/stereo_kitti"
VOCAB="$ORB_SLAM_DIR/Vocabulary/ORBvoc.txt"

if [ ! -x "$STEREO_KITTI" ]; then
    echo "Error: stereo_kitti not found or not executable: $STEREO_KITTI"
    exit 1
fi

if [ ! -f "$VOCAB" ]; then
    echo "Error: Vocabulary file not found: $VOCAB"
    exit 1
fi

SESSION_DIR="${SESSION_DIR%/}"
ORB_SLAM_DIR="${ORB_SLAM_DIR%/}"

echo "============================================"
echo "ORB_SLAM3 Batch Processor (D405)"
echo "============================================"
echo "Session dir:   $SESSION_DIR"
echo "ORB_SLAM3 dir: $ORB_SLAM_DIR"
echo "Skip existing: $SKIP_EXIST"
echo "Visualization: $VISUALIZATION"
echo "Mask config:   ${SROI_MASK_CONFIG:-none}"
echo "Jobs:          $JOBS"
echo ""

EPISODE_FOLDERS=()
for dir in "$SESSION_DIR"/episode_*/; do
    [ -d "$dir" ] || continue
    dir="${dir%/}"
    if (ls "$dir"/left_*.png 1>/dev/null 2>&1 || ls "$dir"/left_*.jpg 1>/dev/null 2>&1) && \
       (ls "$dir"/right_*.png 1>/dev/null 2>&1 || ls "$dir"/right_*.jpg 1>/dev/null 2>&1); then
        EPISODE_FOLDERS+=("$dir")
    fi
done

TOTAL=${#EPISODE_FOLDERS[@]}
echo "Found $TOTAL episode folder(s)"

if [ "$TOTAL" -eq 0 ]; then
    echo "No episode folders found with left/right images"
    exit 1
fi

# Per-episode outcomes are tallied through a temp file with atomic appends (flock). Each
# episode runs in its own subshell so the mask helper's globals (SROI_MASK_TEMP_DIR,
# SROI_ORB_INPUT_DIR) stay isolated across concurrent episodes.
RESULTS_FILE="$(mktemp)"
LOCK_FILE="$(mktemp)"
trap 'sroi_cleanup_mask_input; rm -f "$RESULTS_FILE" "$LOCK_FILE"' EXIT
record_result() { { flock 9; echo "$1" >> "$RESULTS_FILE"; } 9>"$LOCK_FILE"; }

run_episode() {
    local folder="$1" idx="$2"
    local name cfg
    name="$(basename "$folder")"
    # Prefer any per-episode ORB-SLAM config the session carries (d405, d435i,
    # oak, ...); fall back to the historical d405 name, then the install default.
    cfg="$(ls "$folder"/orb_slam_*.yaml 2>/dev/null | head -1)"
    if [ -z "$cfg" ]; then
        cfg="$folder/orb_slam_realsense_d405.yaml"
    fi
    if [ ! -f "$cfg" ]; then
        echo "  Warning ($name): $cfg not found, using default"
        cfg="$ORB_SLAM_DIR/Examples/Stereo/RealSense_D405.yaml"
    fi
    echo "[$idx/$TOTAL] $name (started)"
    if ! sroi_prepare_orb_input "$folder"; then
        echo "  Error ($name): failed to prepare masked ORB input"
        record_result "error"
        return
    fi
    "$STEREO_KITTI" "$VOCAB" "$cfg" "$SROI_ORB_INPUT_DIR" "$VISUALIZATION"
    if sroi_collect_trajectory "$folder"; then
        echo "  Done ($name)"
        record_result "ok"
    else
        echo "  Error ($name): no CameraTrajectory.txt generated"
        record_result "error"
    fi
    sroi_cleanup_mask_input
}

running=0
idx=0
for folder in "${EPISODE_FOLDERS[@]}"; do
    name="$(basename "$folder")"
    if [ "$SKIP_EXIST" = "true" ] && [ -f "$folder/CameraTrajectory.txt" ]; then
        idx=$((idx + 1))
        echo "[$idx/$TOTAL] $name (skipped: exists)"
        record_result "skip"
        continue
    fi
    idx=$((idx + 1))
    ( run_episode "$folder" "$idx" ) &
    running=$((running + 1))
    if (( running >= JOBS )); then
        wait -n
        running=$((running - 1))
    fi
done
wait

SUCCESS=$(grep -c '^ok$' "$RESULTS_FILE")
SKIP=$(grep -c '^skip$' "$RESULTS_FILE")
ERROR=$(grep -c '^error$' "$RESULTS_FILE")

echo ""
echo "============================================"
echo "Summary: $TOTAL total, $SUCCESS ok, $SKIP skipped, $ERROR errors"
echo "============================================"
