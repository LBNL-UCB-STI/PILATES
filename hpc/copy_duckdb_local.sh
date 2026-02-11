#!/usr/bin/env bash
#
# Safely copy a DuckDB database to node-local storage for fast analysis.
#
# Default destination is stable so notebooks can auto-detect it:
#   /local/job$SLURM_JOB_ID/db/pilates-analysis.duckdb
#
# Examples:
#   hpc/copy_duckdb_local.sh --src /global/.../provenance.duckdb
#   hpc/copy_duckdb_local.sh --src /global/.../foo.duckdb --dst /local/job$SLURM_JOB_ID/db/seattle.duckdb
#

set -euo pipefail

SRC=""
DST=""
SLEEP_SECONDS=2
VALIDATE=1

usage() {
    cat <<'EOF'
Usage:
  copy_duckdb_local.sh --src <path> [--dst <path>] [--sleep-seconds <n>] [--no-validate]

Options:
  --src <path>         Source .duckdb file (required)
  --dst <path>         Destination file path (default: /local/job$SLURM_JOB_ID/db/pilates-analysis.duckdb)
  --sleep-seconds <n>  Seconds between source-size checks (default: 2)
  --no-validate        Skip duckdb CLI validation
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --src)
        SRC="$2"
        shift 2
        ;;
    --dst)
        DST="$2"
        shift 2
        ;;
    --sleep-seconds)
        SLEEP_SECONDS="$2"
        shift 2
        ;;
    --no-validate)
        VALIDATE=0
        shift
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown option: $1" >&2
        usage
        exit 2
        ;;
    esac
done

if [[ -z "$SRC" ]]; then
    echo "Missing required --src argument." >&2
    usage
    exit 2
fi

if [[ ! -f "$SRC" ]]; then
    echo "Source DB not found: $SRC" >&2
    exit 1
fi

if [[ -z "$DST" ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "SLURM_JOB_ID is not set. Pass --dst explicitly." >&2
        exit 2
    fi
    DST="/local/job${SLURM_JOB_ID}/db/pilates-analysis.duckdb"
fi

DST_DIR="$(dirname "$DST")"
mkdir -p "$DST_DIR"

if stat -c%s "$SRC" >/dev/null 2>&1; then
    s1="$(stat -c%s "$SRC")"
else
    s1="$(stat -f%z "$SRC")"
fi

sleep "$SLEEP_SECONDS"

if stat -c%s "$SRC" >/dev/null 2>&1; then
    s2="$(stat -c%s "$SRC")"
else
    s2="$(stat -f%z "$SRC")"
fi

if [[ "$s1" != "$s2" ]]; then
    echo "Source DB appears to be changing; aborting copy." >&2
    exit 1
fi

tmp="${DST}.tmp.$$"
cp "$SRC" "$tmp"

if [[ -f "${SRC}.wal" ]]; then
    cp "${SRC}.wal" "${tmp}.wal"
fi

mv "$tmp" "$DST"
if [[ -f "${tmp}.wal" ]]; then
    mv "${tmp}.wal" "${DST}.wal"
fi

if [[ "$VALIDATE" -eq 1 ]]; then
    if command -v duckdb >/dev/null 2>&1; then
        duckdb "$DST" "PRAGMA database_list; SELECT 1;" >/dev/null
    else
        echo "duckdb CLI not found; skipped validation."
    fi
fi

echo "Copied safely: $DST"
echo "Notebook env override (bash): export PILATES_DB_PATH=\"$DST\""

