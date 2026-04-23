#!/bin/bash

# Slurm submission wrapper for PILATES HPC jobs.
# Full usage and design notes: hpc/README.md

set -euo pipefail

# Comprehensive error reporting
error_exit() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    echo "ERROR: job_runner.sh failed!" >&2
    echo "  Line: $1" >&2
    echo "  Command: $2" >&2
    echo "  Exit code: $3" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    exit "$3"
}

trap 'error_exit ${LINENO} "$BASH_COMMAND" $?' ERR

# Generate random ID (avoid SIGPIPE from urandom pipe)
RANDOM_PART="$(head -c 32 /dev/urandom | base64 | tr -dc A-Z0-9 | head -c 8)"
DATETIME="$(date "+%Y.%m.%d-%H.%M.%S")"
JOB_NAME="$RANDOM_PART.$DATETIME"

PILATES_DIR="${PILATES_DIR:-/global/scratch/users/$USER/sources/PILATES}"
settings_file="settings.yaml"  # May be a template with ${BEAM_MEMORY}
generated_settings_file="settings_${JOB_NAME}.yaml"
stage_file=""
partition_arg="lr7"
account_arg=""
high_mem=false
beam_profile=false
archive_mode="scratch"
hours_arg=""

while [ $# -gt 0 ]; do
    case "$1" in
    -c)
        settings_file="${2:-}"
        shift 2
        ;;
    -s)
        stage_file="${2:-}"
        shift 2
        ;;
    -p)
        partition_arg="${2:-}"
        shift 2
        ;;
    -a|--account)
        account_arg="${2:-}"
        shift 2
        ;;
    --high-mem|-H)
        high_mem=true
        shift
        ;;
    --beam-profile)
        beam_profile=true
        shift
        ;;
    --archive)
        archive_mode="${2:-}"
        shift 2
        ;;
    -t|--hours)
        hours_arg="${2:-}"
        shift 2
        ;;
    -h|--help)
        echo "Usage: $0 [-c settings file] [-s stage file] [-p partition] [-a account] [--high-mem|-H] [--beam-profile] [--archive shared|scratch] [-t hours]"
        echo "  -a, --account: Slurm account name (required)"
        echo "  --high-mem: for lr7 only, request 480G instead of default 240G."
        echo "  --beam-profile: enable BEAM Java Flight Recorder output in /app/output."
        echo "  --archive: set archive target to shared or scratch."
        echo "  -t, --hours: job time limit in hours (e.g. -t 12); default is 72 (3 days)."
        exit 0
        ;;
    *)
        printf "Usage: %s [-c settings file] [-s stage file] [-p partition] [-a account] [--high-mem|-H] [--beam-profile] [--archive shared|scratch] [-t hours]\n" "$0"
        exit 2
        ;;
    esac
done

case "$archive_mode" in
    shared|scratch)
        ;;
    *)
        echo "ERROR: unsupported --archive mode '$archive_mode' (expected shared or scratch)"
        exit 2
        ;;
esac

if [ "$beam_profile" = true ] && [ -z "${BEAM_EXTRA_JVM_ARGS:-}" ]; then
    BEAM_EXTRA_JVM_ARGS="-XX:StartFlightRecording=delay=5s,duration=30m,filename=/app/output/recording_${JOB_NAME}.jfr,dumponexit=true,settings=default"
fi

case "$partition_arg" in
    lr8)
        PARTITION="lr8"
        QOS="lr8_normal"
        NUM_CPUS=128
        MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-700}"
        BEAM_MEMORY="${BEAM_MEMORY:-600g}"
        if [ "$high_mem" = true ]; then
            echo "NOTE: --high-mem is only applied for lr7; ignoring for lr8."
        fi
        ;;
    lr7)
        PARTITION="lr7"
        QOS="lr_normal"
        NUM_CPUS=56
        if [ "$high_mem" = true ]; then
            MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-480}"
            BEAM_MEMORY="${BEAM_MEMORY:-300g}"
        else
            MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-240}"
            BEAM_MEMORY="${BEAM_MEMORY:-180g}"
        fi
        ;;
    *)
        echo "ERROR: unsupported partition '$partition_arg' (expected lr7 or lr8)"
        exit 2
        ;;
esac

if [ -z "$account_arg" ]; then
    echo "ERROR: Slurm account is required." >&2
    echo "Provide it explicitly on the command line with:" >&2
    echo "  ./hpc/job_runner.sh -c <settings.yaml> -a <slurm_account>" >&2
    echo "Example:" >&2
    echo "  ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -a my_account" >&2
    exit 2
fi
ACCOUNT="$account_arg"
if [ -n "$hours_arg" ]; then
    days=$(( hours_arg / 24 ))
    remaining_hours=$(( hours_arg % 24 ))
    EXPECTED_EXECUTION_DURATION="$(printf '%d-%02d:00:00' "$days" "$remaining_hours")"
else
    EXPECTED_EXECUTION_DURATION="${EXPECTED_EXECUTION_DURATION:-3-00:00:00}"
fi
JOB_LOG_FILE_PATH="/global/scratch/users/$USER/pilates_logs/log_${DATETIME}_${RANDOM_PART}.log"

resolve_path() {
    local path_arg="$1"
    if [ "${path_arg#/}" != "$path_arg" ]; then
        echo "$path_arg"
    else
        echo "$PILATES_DIR/$path_arg"
    fi
}

settings_template_path="$(resolve_path "$settings_file")"
generated_settings_path="$PILATES_DIR/$generated_settings_file"
stage_path=""
if [ -n "$stage_file" ]; then
    stage_path="$(resolve_path "$stage_file")"
fi

if [ ! -f "$settings_template_path" ]; then
    echo "ERROR: settings file not found: $settings_template_path"
    exit 1
fi

mkdir -p "/global/scratch/users/$USER/pilates_logs"

if command -v envsubst >/dev/null 2>&1; then
    BEAM_MEMORY="$BEAM_MEMORY" BEAM_EXTRA_JVM_ARGS="${BEAM_EXTRA_JVM_ARGS:-}" \
        envsubst '$BEAM_MEMORY $BEAM_EXTRA_JVM_ARGS' < "$settings_template_path" > "$generated_settings_path"
else
    sed "s|\${BEAM_MEMORY}|$BEAM_MEMORY|g" "$settings_template_path" > "$generated_settings_path"
    if [ -n "${BEAM_EXTRA_JVM_ARGS:-}" ]; then
        sed -i "s|\${BEAM_EXTRA_JVM_ARGS}|$BEAM_EXTRA_JVM_ARGS|g" "$generated_settings_path"
    else
        sed -i "s|\${BEAM_EXTRA_JVM_ARGS}||g" "$generated_settings_path"
    fi
fi

SCRATCH_OUTPUT_ROOT="/global/scratch/users/$USER/pilates-outputs"
SHARED_OUTPUT_ROOT="/clusterfs/beem-core-data-nfs/pilates-outputs"
LOCAL_WORKSPACE_ROOT="/local/job\${SLURM_JOB_ID}/pilates-workspace"

apply_archive_mode_override() {
    local config_path="$1"
    local mode="$2"
    python3 - "$config_path" "$mode" "$SCRATCH_OUTPUT_ROOT" "$SHARED_OUTPUT_ROOT" "$LOCAL_WORKSPACE_ROOT" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
mode = sys.argv[2]
scratch_root = sys.argv[3]
shared_root = sys.argv[4]
local_root = sys.argv[5]

with config_path.open("r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

run_cfg = data.setdefault("run", {})
if mode == "shared":
    run_cfg["output_directory"] = shared_root
    run_cfg["local_workspace_root"] = local_root
    run_cfg["enable_archive_copy"] = True
elif mode == "scratch":
    run_cfg["output_directory"] = scratch_root
    run_cfg["local_workspace_root"] = local_root
    run_cfg["enable_archive_copy"] = True
else:
    raise SystemExit(f"unsupported archive mode: {mode}")

with config_path.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(data, fh, sort_keys=False)
PY
}

apply_archive_mode_override "$generated_settings_path" "$archive_mode"

export BEAM_MEMORY

sbatch_args=(
    --partition="$PARTITION"
    --exclusive
    --cpus-per-task="$NUM_CPUS"
    --mem="${MEMORY_LIMIT_GB}G"
    --qos="$QOS"
    --account="$ACCOUNT"
    --job-name="$JOB_NAME"
    --output="$JOB_LOG_FILE_PATH"
    --time="$EXPECTED_EXECUTION_DURATION"
    --export="ALL,JOB_LOG_FILE_PATH=$JOB_LOG_FILE_PATH"
    "$PILATES_DIR/hpc/job.sh"
    "$generated_settings_path"
)
if [ -n "$stage_path" ]; then
    sbatch_args+=("$stage_path")
fi

sbatch "${sbatch_args[@]}"

echo "$JOB_LOG_FILE_PATH"
