#!/usr/bin/env fish
#
# Safely copy a DuckDB database to node-local storage for fast analysis.
#
# Default destination is stable so notebooks can auto-detect it:
#   /local/job$SLURM_JOB_ID/db/pilates-analysis.duckdb
#
# Examples:
#   hpc/copy_duckdb_local.fish --src /global/.../provenance.duckdb
#   hpc/copy_duckdb_local.fish --src /global/.../foo.duckdb --dst /local/job$SLURM_JOB_ID/db/seattle.duckdb
#

function usage
    echo "Usage:"
    echo "  copy_duckdb_local.fish --src <path> [--dst <path>] [--sleep-seconds <n>] [--no-validate]"
    echo ""
    echo "Options:"
    echo "  --src <path>         Source .duckdb file (required)"
    echo "  --dst <path>         Destination file path (default: /local/job\$SLURM_JOB_ID/db/pilates-analysis.duckdb)"
    echo "  --sleep-seconds <n>  Seconds between source-size checks (default: 2)"
    echo "  --no-validate        Skip duckdb CLI validation"
    echo "  -h, --help           Show this help"
end

set SRC ""
set DST ""
set SLEEP_SECONDS 2
set VALIDATE 1

while test (count $argv) -gt 0
    switch $argv[1]
        case --src
            if test (count $argv) -lt 2
                echo "Missing value for --src" >&2
                exit 2
            end
            set SRC $argv[2]
            set -e argv[1..2]
        case --dst
            if test (count $argv) -lt 2
                echo "Missing value for --dst" >&2
                exit 2
            end
            set DST $argv[2]
            set -e argv[1..2]
        case --sleep-seconds
            if test (count $argv) -lt 2
                echo "Missing value for --sleep-seconds" >&2
                exit 2
            end
            set SLEEP_SECONDS $argv[2]
            set -e argv[1..2]
        case --no-validate
            set VALIDATE 0
            set -e argv[1]
        case -h --help
            usage
            exit 0
        case '*'
            echo "Unknown option: $argv[1]" >&2
            usage
            exit 2
    end
end

if test -z "$SRC"
    echo "Missing required --src argument." >&2
    usage
    exit 2
end

if not test -f "$SRC"
    echo "Source DB not found: $SRC" >&2
    exit 1
end

if test -z "$DST"
    if test -z "$SLURM_JOB_ID"
        echo "SLURM_JOB_ID is not set. Pass --dst explicitly." >&2
        exit 2
    end
    set DST "/local/job$SLURM_JOB_ID/db/pilates-analysis.duckdb"
end

set DST_DIR (dirname "$DST")
mkdir -p "$DST_DIR"; or begin
    echo "mkdir failed: $DST_DIR" >&2
    exit 1
end

set s1 (stat -c%s "$SRC" 2>/dev/null)
if test -z "$s1"
    set s1 (stat -f%z "$SRC" 2>/dev/null)
end
if test -z "$s1"
    echo "Cannot stat source: $SRC" >&2
    exit 1
end

sleep "$SLEEP_SECONDS"

set s2 (stat -c%s "$SRC" 2>/dev/null)
if test -z "$s2"
    set s2 (stat -f%z "$SRC" 2>/dev/null)
end
if test -z "$s2"
    echo "Cannot stat source on second check: $SRC" >&2
    exit 1
end

if test "$s1" != "$s2"
    echo "Source DB appears to be changing; aborting copy." >&2
    exit 1
end

set tmp "$DST.tmp.$fish_pid"
cp "$SRC" "$tmp"; or begin
    echo "Copy failed: $SRC -> $tmp" >&2
    exit 1
end

if test -f "$SRC.wal"
    cp "$SRC.wal" "$tmp.wal"; or begin
        echo "WAL copy failed, cleaning temp." >&2
        rm -f "$tmp"
        exit 1
    end
end

mv "$tmp" "$DST"; or begin
    echo "Move failed: $tmp -> $DST" >&2
    rm -f "$tmp" "$tmp.wal"
    exit 1
end

if test -f "$tmp.wal"
    mv "$tmp.wal" "$DST.wal"; or begin
        echo "WAL move failed." >&2
        exit 1
    end
end

if test "$VALIDATE" -eq 1
    if type -q duckdb
        duckdb "$DST" "PRAGMA database_list; SELECT 1;" >/dev/null; or begin
            echo "DuckDB validation failed for: $DST" >&2
            exit 1
        end
    else
        echo "duckdb CLI not found; skipped validation."
    end
end

echo "Copied safely: $DST"
echo "Notebook env override (fish): set -gx PILATES_DB_PATH \"$DST\""

