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
RUN_NOTIFICATIONS_ENV="${PILATES_RUN_NOTIFICATIONS_ENV:-$PILATES_DIR/hpc/run-notifications.env}"
settings_file="settings.yaml"  # May be a template with ${BEAM_MEMORY}
generated_settings_file="settings_${JOB_NAME}.yaml"
stage_file=""
partition_arg="lr7"
account_arg=""
high_mem=false

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
    -h|--help)
        echo "Usage: $0 [-c settings file] [-s stage file] [-p partition] [-a account] [--high-mem|-H]"
        echo "  -a, --account: Slurm account name (required)"
        echo "  --high-mem: for lr7 only, request 480G instead of default 240G."
        exit 0
        ;;
    *)
        printf "Usage: %s [-c settings file] [-s stage file] [-p partition] [-a account] [--high-mem|-H]\n" "$0"
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
    echo "  ./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a my_account" >&2
    exit 2
fi
ACCOUNT="$account_arg"
EXPECTED_EXECUTION_DURATION="${EXPECTED_EXECUTION_DURATION:-3-00:00:00}"
JOB_LOG_FILE_PATH="/global/scratch/users/$USER/pilates_logs/log_${DATETIME}_${RANDOM_PART}.log"

print_notification_status() {
    local gchat_enabled="${PILATES_GCHAT_NOTIFICATIONS:-0}"
    local slack_enabled="${PILATES_SLACK_NOTIFICATIONS:-0}"
    local gsheet_enabled="${PILATES_GSHEET_PUBLISH:-0}"
    local gchat_webhook_status="missing"
    local slack_webhook_status="missing"
    local gsheet_webhook_status="missing"
    local event_log_enabled="${PILATES_RUN_EVENT_LOG:-1}"
    local summary_html_enabled="${PILATES_RUN_SUMMARY_HTML:-1}"

    if [ -n "${PILATES_GCHAT_WEBHOOK_URL:-}" ]; then
        gchat_webhook_status="set"
    fi
    if [ -n "${PILATES_SLACK_WEBHOOK_URL:-}" ]; then
        slack_webhook_status="set"
    fi
    if [ -n "${PILATES_GSHEET_WEBHOOK_URL:-}" ]; then
        gsheet_webhook_status="set"
    fi

    echo "Run notifications: google_chat enabled=$gchat_enabled webhook=$gchat_webhook_status; slack enabled=$slack_enabled webhook=$slack_webhook_status"
    echo "Run publishing: archive_jsonl enabled=$event_log_enabled; summary_html enabled=$summary_html_enabled; google_sheet enabled=$gsheet_enabled webhook=$gsheet_webhook_status"
}

if [ -f "$RUN_NOTIFICATIONS_ENV" ]; then
    # Export assignments from the env file so sbatch --export=ALL passes them
    # through to job.sh and run.py.
    set -a
    # shellcheck disable=SC1090
    source "$RUN_NOTIFICATIONS_ENV"
    set +a
    echo "Loaded run notification environment: $RUN_NOTIFICATIONS_ENV"
elif [ "$RUN_NOTIFICATIONS_ENV" = "$PILATES_DIR/hpc/run-notifications.env" ] \
    && [ -f "$PILATES_DIR/hpc/run-notifications.env.template" ]; then
    echo "Run notifications disabled: copy hpc/run-notifications.env.template to hpc/run-notifications.env to enable."
elif [ -n "${PILATES_RUN_NOTIFICATIONS_ENV:-}" ]; then
    echo "Run notifications disabled: PILATES_RUN_NOTIFICATIONS_ENV points to a missing file: $RUN_NOTIFICATIONS_ENV"
fi
print_notification_status

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
    BEAM_MEMORY="$BEAM_MEMORY" envsubst '$BEAM_MEMORY' < "$settings_template_path" > "$generated_settings_path"
else
    sed "s|\${BEAM_MEMORY}|$BEAM_MEMORY|g" "$settings_template_path" > "$generated_settings_path"
fi

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
