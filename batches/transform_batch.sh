#!/bin/bash
# Robust outer layer for transform_trajectory.py (kernel — do not call directly).
#
# Smooths (Whittaker, lambda from configs/transform_trajectory.json; --no-smooth
# to skip) then converts to the camera_link frame (X-forward, Y-left, Z-up) as
# CameraTrajectoryTransformed.txt (13 cols: timestamp + 3x4), taking timestamps
# through the ledger's frame ids. Hard-fails (non-zero exit) if repair has not
# run: rows must equal frames and NaN rows must match the ledger's invalid rows.
#
# Usage:
#   bash batches/transform_batch.sh <episode | session-png | data root>
#
#   Requires batches/traj_repair_batch.sh to have run on the same tree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRANSFORM_PY="$REPO_DIR/transform_trajectory.py"

if [ $# -lt 1 ]; then
    echo "Usage: bash batches/transform_batch.sh <episode | session-png | data root>" >&2
    exit 1
fi
ROOT="$1"; shift || true

# Interpreter policy: transform needs numpy only — system python3.
PY_SYS="/usr/bin/python3"
if ! "$PY_SYS" -c "import numpy" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1 && python3 -c "import numpy" >/dev/null 2>&1; then
        PY_SYS="python3"
    else
        echo "Error: no python with numpy found." >&2
        exit 1
    fi
fi

exec "$PY_SYS" "$TRANSFORM_PY" "$ROOT" --recursive "$@"
