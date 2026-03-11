#!/bin/bash

# Slurm submission wrapper for PILATES HPC jobs.
# Full usage and design notes: hpc/README.md

set -euo pipefail

RANDOM_PART="$(printf '%04X%04X' "$RANDOM" "$RANDOM")"
DATETIME="$(date "+%Y.%m.%d-%H.%M.%S")"
JOB_NAME="$RANDOM_PART.$DATETIME"

PILATES_DIR="${PILATES_DIR:-/global/scratch/users/$USER/sources/PILATES}"
settings_file="settings.yaml"  # May be a template with ${BEAM_MEMORY}
generated_settings_file="settings_${JOB_NAME}.yaml"
stage_file=""
partition_arg="lr7"
high_mem=false

usage() {
    echo "Usage: $0 [-c settings file] [-s stage file] [-p partition] [--high-mem|-H]"
    echo "  --high-mem: for lr7 only, request 480G instead of the default node memory."
}

while [ $# -gt 0 ]; do
    case "$1" in
    -c)
        if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
            echo "ERROR: -c requires a settings file path"
            usage
            exit 2
        fi
        settings_file="${2:-}"
        shift 2
        ;;
    -s)
        if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
            echo "ERROR: -s requires a stage file path"
            usage
            exit 2
        fi
        stage_file="${2:-}"
        shift 2
        ;;
    -p)
        if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
            echo "ERROR: -p requires a partition name"
            usage
            exit 2
        fi
        partition_arg="${2:-}"
        shift 2
        ;;
    --high-mem|-H)
        high_mem=true
        shift
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage
        exit 2
        ;;
    esac
done

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
            BEAM_MEMORY="${BEAM_MEMORY:-400g}"
        else
            MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-230}"
            BEAM_MEMORY="${BEAM_MEMORY:-180g}"
        fi
        ;;
    *)
        echo "ERROR: unsupported partition '$partition_arg' (expected lr7 or lr8)"
        exit 2
        ;;
esac

ACCOUNT="${ACCOUNT:-pc_beamcore}"
EXPECTED_EXECUTION_DURATION="${EXPECTED_EXECUTION_DURATION:-3-00:00:00}"
PILATES_LOG_DIR="${PILATES_LOG_DIR:-/global/scratch/users/$USER/pilates_logs}"
JOB_LOG_FILE_PATH="$PILATES_LOG_DIR/log_${DATETIME}_${RANDOM_PART}.log"

resolve_path() {
    local path_arg="$1"
    if [ -z "$path_arg" ]; then
        return
    fi
    if [ "${path_arg#/}" != "$path_arg" ]; then
        printf '%s\n' "$path_arg"
    else
        printf '%s\n' "$PILATES_DIR/$path_arg"
    fi
}

settings_template_path="$(resolve_path "$settings_file")"
generated_settings_path="$PILATES_DIR/$generated_settings_file"
stage_path=""

if [ ! -f "$settings_template_path" ]; then
    echo "ERROR: settings file not found: $settings_template_path"
    exit 1
fi

if [ -n "$stage_file" ]; then
    stage_path="$(resolve_path "$stage_file")"
    if [ ! -f "$stage_path" ]; then
        echo "ERROR: stage file not found: $stage_path"
        exit 1
    fi
fi

mkdir -p "$PILATES_LOG_DIR"

if command -v envsubst >/dev/null 2>&1; then
    BEAM_MEMORY="$BEAM_MEMORY" envsubst '$BEAM_MEMORY' < "$settings_template_path" > "$generated_settings_path"
else
    sed "s|\${BEAM_MEMORY}|$BEAM_MEMORY|g" "$settings_template_path" > "$generated_settings_path"
fi

export BEAM_MEMORY

SBATCH_ARGS=(
    --partition="$PARTITION"
    --exclusive
    --cpus-per-task="$NUM_CPUS"
    --mem="${MEMORY_LIMIT_GB}G"
    --qos="$QOS"
    --account="$ACCOUNT"
    --job-name="$JOB_NAME"
    --output="$JOB_LOG_FILE_PATH"
    --time="$EXPECTED_EXECUTION_DURATION"
    "$PILATES_DIR/hpc/job.sh"
    "$generated_settings_path"
)

if [ -n "$stage_path" ]; then
    SBATCH_ARGS+=("$stage_path")
fi

sbatch "${SBATCH_ARGS[@]}"

echo "$JOB_LOG_FILE_PATH"
