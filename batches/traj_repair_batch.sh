#!/bin/bash
# Robust outer layer for repair_trajectory.py (kernel — do not call directly).
#
# Aligns SLAM poses to frames through the index ledger, fills short holes
# (interp) / leaves long holes as NaN (invalid), applies segment-wise
# Whittaker smoothing, and atomically rewrites CameraTrajectory.txt (N x 12)
# + CameraTrajectory_idx.txt (frame state seg_len).
#
# Usage:
#   bash batches/traj_repair_batch.sh <episode | session-png | data root> [--force] [--no-smooth]
#
#   Idempotent: episodes already repaired (N rows + 3-col ledger) are skipped.
#   Exit code non-zero if ANY episode fails (kernel aggregates and reports).
#   After this, run batches/transform_batch.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPAIR_PY="$REPO_DIR/repair_trajectory.py"
CONFIG="$REPO_DIR/configs/traj_repair.json"

if [ $# -lt 1 ]; then
    echo "Usage: bash batches/traj_repair_batch.sh <episode | session-png | data root> [--force] [--no-smooth]" >&2
    exit 1
fi
ROOT="$1"; shift || true

# Interpreter policy: repair needs numpy + scipy(Slerp) — system python3 has both.
PY_SYS="/usr/bin/python3"
if ! "$PY_SYS" -c "import numpy, scipy" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1 && python3 -c "import numpy, scipy" >/dev/null 2>&1; then
        PY_SYS="python3"
    else
        echo "Error: no python with numpy+scipy found." >&2
        exit 1
    fi
fi

[ -f "$CONFIG" ] || { echo "Error: $CONFIG not found" >&2; exit 1; }

exec "$PY_SYS" "$REPAIR_PY" "$ROOT" --recursive --config "$CONFIG" "$@"
