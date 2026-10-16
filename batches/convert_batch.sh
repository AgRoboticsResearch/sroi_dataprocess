#!/bin/bash
# Robust outer layer for lerobot/sroi_to_lerobot.py (kernel — do not call directly).
#
# Converts QC-passed episodes into ONE LeRobot dataset. Executes the QC verdict
# by default: --qc-csv (from visualization/qc.py) + --qc-categories ok.
# Hard-fails (non-zero exit) if any episode has mismatched stream counts
# (trajectory / gripper / image rows must all equal the frame count — the
# upstream batches guarantee it; a mismatch means an upstream step is stale).
#
# Usage:
#   bash batches/convert_batch.sh <data-root> <repo_id> [out-root] [extra args...]
#   bash batches/convert_batch.sh ~/data/1000_onesb_labpicking sroi/lab_picking \
#            ~/data/lerobot_out --qc-csv /path/qc.csv
#   Add --no-qc to convert everything without a QC filter (not recommended).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONVERT_PY="$REPO_DIR/lerobot/sroi_to_lerobot.py"

if [ $# -lt 2 ]; then
    echo "Usage: bash batches/convert_batch.sh <data-root> <repo_id> [out-root] [extra args...]" >&2
    exit 1
fi
DATA_ROOT="$1"; REPO_ID="$2"; OUT_ROOT="${3:-}"; shift 3 || shift $#

# Interpreter policy: needs the lerobot package — conda `lerobot` env.
PY_CONV="$HOME/miniconda3/envs/lerobot/bin/python"
if [ ! -x "$PY_CONV" ]; then
    if command -v python3 >/dev/null 2>&1 && python3 -c "import lerobot" >/dev/null 2>&1; then
        PY_CONV="python3"
    else
        echo "Error: no python with the lerobot package found (looked for $HOME/miniconda3/envs/lerobot/bin/python)." >&2
        exit 1
    fi
fi

[ -d "$DATA_ROOT" ] || { echo "Error: data root '$DATA_ROOT' not found" >&2; exit 1; }

# QC filtering is per-episode: each episode dir carries its own qc.json
# (verdict + start_frame), written by visualization/qc.py and updated by
# human review. No central csv needed here.
exec "$PY_CONV" "$CONVERT_PY" --data_path "$DATA_ROOT" --repo_id "$REPO_ID" \
    --fps 30 ${OUT_ROOT:+--root "$OUT_ROOT"} "$@"
