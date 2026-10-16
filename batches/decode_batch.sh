#!/bin/bash
# Batch-decode *-mp4 sessions into a SEPARATE PNG root, so the raw MP4 store stays
# pristine and every derived PNG lives under one trivially-wipeable tree.
# (For MP4 recordings, PNGs only appear at decode time. This keeps them off the
# raw -mp4 folders instead of creating -png siblings beside them.)
#
# Usage:
#   ./decode_batch.sh <input> [png_root] [--jobs N]
#
#   <input>:    a day-folder (containing *-mp4 sessions) or a single *-mp4 session.
#   [png_root]: root dir for the decoded -png sessions (default: alongside the
#               -mp4 folders — sibling layout, same as the existing corpus).
#               Each session decodes to <png_root>/<ts>-png/episode_*/.
#   --jobs N:   concurrent session decodes (default 4).
#
# Examples:
#   ./decode_batch.sh /data/20260728-occluded-cases
#       -> /data/20260728-occluded-cases/<ts>-png/... (siblings of the -mp4 dirs)
#   ./decode_batch.sh /data/20260728-occluded-cases /tmp/pngout --jobs 8
#   ./decode_batch.sh /data/20260728-occluded-cases/20260728_200203-mp4
#
# Idempotent: sessions whose <png_root>/<ts>-png target already has decoded frames
# are skipped. Raw *-mp4 folders are never modified. Downstream steps (ORB-SLAM3,
# transform_trajectory.py, gripper_estimation_april_tag.py, visualization/qc.py)
# then run against <png_root>.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DECODE_PY="$REPO_DIR/decode_videos.py"

# Interpreter policy: decode needs PyAV, which only the conda `lerobot` env has
# on this machine (system python3 has no `av`). Pick it explicitly so this
# script works from any shell, activated or not.
PY_DECODE="$HOME/miniconda3/envs/lerobot/bin/python"
if [ ! -x "$PY_DECODE" ]; then
    # Fallback: any python on PATH that can import av.
    if command -v python3 >/dev/null 2>&1 && python3 -c "import av" >/dev/null 2>&1; then
        PY_DECODE="python3"
    else
        echo "Error: no python with PyAV found (looked for $HOME/miniconda3/envs/lerobot/bin/python, then python3 with 'av')." >&2
        echo "Fix: create the conda env (see README) or 'pip install av' into the active python." >&2
        exit 1
    fi
fi

INPUT=""
PNG_ROOT=""
JOBS="4"

_positionals=()
while [ $# -gt 0 ]; do
    case "$1" in
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
INPUT="${_positionals[0]:-}"
PNG_ROOT="${_positionals[1]:-}"

if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input> [png_root] [--jobs N]"
    echo "  input:    day-folder (with *-mp4 sessions) or a single *-mp4 session"
    echo "  png_root: where -png sessions go (default: alongside the -mp4 folders)"
    echo "  --jobs N: concurrent session decodes (default 4)"
    exit 1
fi

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
    echo "Error: --jobs must be a positive integer (got '$JOBS')" >&2
    exit 1
fi

if [ ! -d "$INPUT" ]; then
    echo "Error: input not found or not a directory: $INPUT" >&2
    exit 1
fi
if [ ! -f "$DECODE_PY" ]; then
    echo "Error: decode_videos.py not found: $DECODE_PY" >&2
    exit 1
fi

INPUT="${INPUT%/}"

# Decide day-folder vs single *-mp4 session, and the default PNG root.
if [[ "$(basename "$INPUT")" == *-mp4 ]]; then
    DAY_DIR="$(cd "$INPUT/.." && pwd)"
    SESSIONS=("$INPUT")
else
    DAY_DIR="$INPUT"
    mapfile -t SESSIONS < <(find "$DAY_DIR" -maxdepth 1 -type d -name '*-mp4' | sort)
    if [ "${#SESSIONS[@]}" -eq 0 ]; then
        echo "Error: no *-mp4 sessions found under $DAY_DIR" >&2
        exit 1
    fi
fi

# Default PNG root: the mp4's own parent dir, i.e. -png sessions are created
# as SIBLINGS of the -mp4 folders (matches the existing corpus layout; the
# episode dirs under the -png tree are the live working tree, not scratch).
PNG_ROOT="${PNG_ROOT:-$DAY_DIR}"
PNG_ROOT="${PNG_ROOT%/}"
mkdir -p "$PNG_ROOT"
LOG_DIR="$PNG_ROOT/.decode_logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Decode to PNG root"
echo "============================================"
echo "Input:     $INPUT"
echo "PNG root:  $PNG_ROOT"
echo "Sessions:  ${#SESSIONS[@]}    Jobs: $JOBS"
echo ""

RESULTS_FILE="$(mktemp)"
LOCK_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE" "$LOCK_FILE"' EXIT
record_result() { { flock 9; echo "$1" >> "$RESULTS_FILE"; } 9>"$LOCK_FILE"; }

session_complete() {
    local session="$1" target="$2"
    local source_ep target_ep expected stream actual found=0
    for source_ep in "$session"/episode_*/; do
        [ -d "$source_ep" ] || continue
        [ -f "$source_ep/timestamps.json" ] || continue
        found=1
        target_ep="$target/$(basename "${source_ep%/}")"
        [ -f "$target_ep/timestamps.json" ] || return 1
        expected="$(python3 -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["frame_count"]))' \
            "$source_ep/timestamps.json" 2>/dev/null)" || return 1
        [ "$expected" -gt 0 ] 2>/dev/null || return 1
        for stream in left right color; do
            actual="$(find "$target_ep" -maxdepth 1 -type f -name "${stream}_*.png" | wc -l)"
            [ "$actual" -eq "$expected" ] || return 1
        done
    done
    [ "$found" -eq 1 ]
}

decode_one() {
    local session="$1" idx="$2" total="$3"
    local name target log stage staged_target
    name="$(basename "$session")"
    name="${name%-mp4}-png"
    target="$PNG_ROOT/$name"
    log="$LOG_DIR/$name.log"
    if session_complete "$session" "$target"; then
        echo "[$idx/$total] $name (skipped: already decoded)"
        record_result "skip"
        return
    fi

    # Decode through staging because decode_videos.py skips an episode as soon as
    # any left PNG exists. Merging a fresh staged decode repairs partial targets
    # while preserving downstream trajectory/gripper files already in them.
    stage="$(mktemp -d "$PNG_ROOT/.decode-stage.XXXXXX")" || {
        echo "  ERROR: could not create staging directory for $name"
        record_result "error"
        return
    }
    staged_target="$stage/$name"
    echo "[$idx/$total] decoding $name"
    if ! "$PY_DECODE" "$DECODE_PY" "$session" --recursive -o "$stage" >"$log" 2>&1; then
        rm -rf -- "$stage"
        echo "  ERROR: $name (log: $log)"
        record_result "error"
        return
    fi
    if [ ! -d "$staged_target" ]; then
        rm -rf -- "$stage"
        echo "  ERROR: $name produced no decoded session (log: $log)"
        record_result "error"
        return
    fi
    mkdir -p "$target"
    if ! cp -a "$staged_target"/. "$target"/; then
        rm -rf -- "$stage"
        echo "  ERROR: could not copy staged decode for $name"
        record_result "error"
        return
    fi
    rm -rf -- "$stage"
    if session_complete "$session" "$target"; then
        echo "  done: $name"
        record_result "ok"
    else
        echo "  ERROR: incomplete decode for $name (log: $log)"
        record_result "error"
    fi
}

TOTAL=${#SESSIONS[@]}
running=0
idx=0
for session in "${SESSIONS[@]}"; do
    idx=$((idx + 1))
    ( decode_one "$session" "$idx" "$TOTAL" ) &
    running=$((running + 1))
    if (( running >= JOBS )); then
        wait -n
        running=$((running - 1))
    fi
done
wait

SUCCESS=$(grep -c '^ok$' "$RESULTS_FILE" || true)
SKIP=$(grep -c '^skip$' "$RESULTS_FILE" || true)
ERROR=$(grep -c '^error$' "$RESULTS_FILE" || true)

echo ""
echo "============================================"
echo "Summary: $TOTAL total, $SUCCESS decoded, $SKIP skipped, $ERROR errors"
echo "PNG root: $PNG_ROOT"
echo "Next: run ORB-SLAM3 / transform / gripper / QC against: $PNG_ROOT"
echo "============================================"

[ "$ERROR" -eq 0 ]
