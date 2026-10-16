#!/bin/bash
# Robust outer layer for repair_trajectory.py (kernel — do not call directly).
#
# Aligns SLAM poses to frames through the index ledger and marks missing
# frames NaN + "invalid" (NO interpolation, NO smoothing — raw values stay
# pristine so dropping non-raw rows reconstructs the exact K-row SLAM output).
# Atomically rewrites CameraTrajectory.txt (N x 12) + CameraTrajectory_idx.txt
# (frame state seg_len). Smoothing lives in transform_batch.sh.
#
# Usage:
#   bash batches/traj_repair_batch.sh <episode | session-png | data root> [--force]
#
#   Idempotent: episodes already repaired (N rows + 3-col ledger) are skipped.
#   Exit code non-zero if ANY episode fails (kernel aggregates and reports).
#   After this, run batches/transform_batch.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPAIR_PY="$REPO_DIR/repair_trajectory.py"

if [ $# -lt 1 ]; then
    echo "Usage: bash batches/traj_repair_batch.sh <episode | session-png | data root> [--force] [--no-smooth]" >&2
    exit 1
fi
ROOT="$1"; shift || true

# Interpreter policy: repair needs numpy only — system python3 has it.
PY_SYS="/usr/bin/python3"
if ! "$PY_SYS" -c "import numpy, scipy" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1 && python3 -c "import numpy" >/dev/null 2>&1; then
        PY_SYS="python3"
    else
        echo "Error: no python with numpy found." >&2
        exit 1
    fi
fi

exec "$PY_SYS" "$REPAIR_PY" "$ROOT" --recursive "$@"
