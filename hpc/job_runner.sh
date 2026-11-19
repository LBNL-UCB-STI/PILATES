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

_pilates_watch() {
    local job_id="$1"
    local log_file="$2"
    local partition="$3"

    printf "\n  Job %s queued on %s.\n" "$job_id" "$partition"
    printf "  Ctrl+C to detach at any time — job will keep running.\n\n"
    printf "  Waiting for log"

    while [ ! -f "$log_file" ]; do
        sleep 2
        printf "."
    done
    printf "\n\n"

    _pilates_detach() {
        printf "\n"
        printf "── Detached ──────────────────────────────────────────\n"
        printf "Re-attach : tail -f %s\n" "$log_file"
        printf "Status    : squeue -j %s\n" "$job_id"
        printf "Cancel    : scancel %s\n" "$job_id"
        printf "──────────────────────────────────────────────────────\n"
    }

    trap '_pilates_detach; exit 0' INT

    tail -f "$log_file" &
    local tail_pid=$!
    wait "$tail_pid" || true

    trap - INT
}

_print_submission_summary() {
    local config_file="$1"
    local partition="$2"
    local memory_gb="$3"
    local beam_memory="$4"
    local duration="$5"
    local archive="$6"
    local stage="$7"

    python3 - "$config_file" "$partition" "$memory_gb" "$beam_memory" "$duration" "$archive" "$stage" <<'PY'
import sys, yaml, re

config_file, partition, memory_gb, beam_memory, duration, archive, stage = sys.argv[1:]

def fmt_duration(d):
    m = re.match(r'(\d+)-(\d+):(\d+):(\d+)', d)
    if m:
        days, hours = int(m.group(1)), int(m.group(2))
        total_hours = days * 24 + hours
        if total_hours % 24 == 0:
            return f"{total_hours // 24}d"
        return f"{total_hours}h"
    return d

duration = fmt_duration(duration)

with open(config_file) as f:
    cfg = yaml.safe_load(f)

run = cfg.get("run", {})
models_cfg = run.get("models", {})

run_name    = run.get("output_run_name", "?")
region      = run.get("region", "?")
scenario    = run.get("scenario", "?")
start_year  = run.get("start_year", "?")
end_year    = run.get("end_year", "?")
sd_iters    = run.get("supply_demand_iters", "?")
output_dir  = run.get("output_directory", "?")

active = [v for v in models_cfg.values() if v]
off    = [k for k, v in models_cfg.items() if not v]
model_parts = active[:]
if off:
    model_parts.append("(" + ", ".join(off) + ": off)")
models_str = "  ".join(model_parts) if model_parts else "(none)"

print()
print(f"  Run      : {run_name}")
print(f"  Region   : {region} / {scenario}  ({start_year} → {end_year}, {sd_iters} SD iters)")
print(f"  Models   : {models_str}")
print(f"  Output   : {output_dir}")
print(f"  Resources: {partition}  {memory_gb}G  {duration}  {archive}")
print(f"  BEAM mem : {beam_memory}")
PY
}

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
beam_profile=false
archive_mode=""
hours_arg=""
watch_flag=false
editable_consist=""

while [ $# -gt 0 ]; do
    case "$1" in
    -c)
        if [ -z "${2:-}" ]; then
            echo "ERROR: -c requires a settings file path." >&2
            exit 2
        fi
        settings_file="$2"
        shift 2
        ;;
    -s)
        if [ -z "${2:-}" ]; then
            echo "ERROR: -s requires a stage file path." >&2
            exit 2
        fi
        stage_file="$2"
        shift 2
        ;;
    -p)
        if [ -z "${2:-}" ]; then
            echo "ERROR: -p requires a partition name." >&2
            exit 2
        fi
        partition_arg="$2"
        shift 2
        ;;
    -a|--account)
        if [ -z "${2:-}" ]; then
            echo "ERROR: -a/--account requires an account name." >&2
            exit 2
        fi
        account_arg="$2"
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
        if [ -z "${2:-}" ]; then
            echo "ERROR: --archive requires shared or scratch." >&2
            exit 2
        fi
        archive_mode="${2:-}"
        shift 2
        ;;
    -t|--hours)
        if [ -z "${2:-}" ]; then
            echo "ERROR: -t/--hours requires a number of hours." >&2
            exit 2
        fi
        hours_arg="${2:-}"
        shift 2
        ;;
    -w|--watch)
        watch_flag=true
        shift
        ;;
    --editable-consist)
        if [ -z "${2:-}" ]; then
            echo "ERROR: --editable-consist requires an absolute Consist checkout path." >&2
            exit 2
        fi
        editable_consist="$2"
        shift 2
        ;;
    -h|--help)
        echo "Usage: $0 [-c settings file] [-s stage file] [-p partition] [-a account] [--high-mem|-H] [--beam-profile] [--archive shared|scratch] [-t hours] [-w|--watch] [--editable-consist /absolute/consist]"
        echo "  -a, --account: Slurm account name (required)"
        echo "  --high-mem: for lr6 or lr7, request high memory."
        echo "  --beam-profile: enable BEAM Java Flight Recorder output in /app/output."
        echo "  --archive: set live archive target to shared or scratch."
        echo "  -t, --hours: job time limit in hours (e.g. -t 12); default is 72 (3 days)."
        echo "  -w, --watch: stream the job log after submission; Ctrl+C detaches (job keeps running)."
        echo "  --editable-consist: explicit absolute Consist checkout path exported as CONSIST_SRC_DIR."
        exit 0
        ;;
    *)
        printf "Usage: %s [-c settings file] [-s stage file] [-p partition] [-a account] [-H] [--beam-profile] [--archive shared|scratch] [-t hours] [-w|--watch] [--editable-consist /absolute/consist]\n" "$0"
        exit 2
        ;;
    esac
done

case "$archive_mode" in
    "")
        ;;
    shared|scratch)
        ;;
    *)
        echo "ERROR: unsupported --archive mode '$archive_mode' (expected shared or scratch)" >&2
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
            echo "NOTE: --high-mem is only applied for lr6 and lr7; ignoring for lr8."
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
    lr6|lr6_m192)
        PARTITION="lr6"
        QOS="lr_normal"
        NUM_CPUS="${NUM_CPUS:-32}"
        if [ "$high_mem" = true ] || [ "$partition_arg" = "lr6_m192" ]; then
            MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-180}"
            BEAM_MEMORY="${BEAM_MEMORY:-150g}"
        else
            MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-90}"
            BEAM_MEMORY="${BEAM_MEMORY:-70g}"
        fi
        ;;
    *)
        echo "ERROR: unsupported partition '$partition_arg' (expected lr6, lr7, or lr8)"
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
if [ -n "$hours_arg" ]; then
    days=$(( hours_arg / 24 ))
    remaining_hours=$(( hours_arg % 24 ))
    EXPECTED_EXECUTION_DURATION="$(printf '%d-%02d:00:00' "$days" "$remaining_hours")"
else
    EXPECTED_EXECUTION_DURATION="${EXPECTED_EXECUTION_DURATION:-3-00:00:00}"
fi
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
if [ -n "$editable_consist" ] \
    && { [ "${editable_consist#/}" = "$editable_consist" ] || [ ! -d "$editable_consist" ]; }; then
    echo "ERROR: --editable-consist must name an existing absolute Consist checkout." >&2
    exit 2
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
    run_cfg.pop("recovery_archive_roots", None)
elif mode == "scratch":
    run_cfg["output_directory"] = scratch_root
    run_cfg["local_workspace_root"] = local_root
    run_cfg["enable_archive_copy"] = True
    run_cfg.pop("recovery_archive_roots", None)
else:
    raise SystemExit(f"unsupported archive mode: {mode}")

with config_path.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(data, fh)
PY
}

if [ -n "$archive_mode" ]; then
    apply_archive_mode_override "$generated_settings_path" "$archive_mode"
fi

export BEAM_MEMORY
if [ -n "$editable_consist" ]; then
    export CONSIST_SRC_DIR="$editable_consist"
fi

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
    --signal="${PILATES_SLURM_SIGNAL:-B:SIGTERM@120}"
    --export="ALL,JOB_LOG_FILE_PATH=$JOB_LOG_FILE_PATH"
    "$PILATES_DIR/hpc/job.sh"
    "$generated_settings_path"
)
if [ -n "$stage_path" ]; then
    sbatch_args+=("$stage_path")
fi

SBATCH_OUTPUT="$(sbatch "${sbatch_args[@]}")"
echo "$SBATCH_OUTPUT"
JOB_ID="$(printf '%s\n' "$SBATCH_OUTPUT" | awk '/Submitted batch job/{print $NF}')"

echo "Job submitted. Log: $JOB_LOG_FILE_PATH"

_print_submission_summary \
    "$generated_settings_path" \
    "$partition_arg" \
    "$MEMORY_LIMIT_GB" \
    "$BEAM_MEMORY" \
    "$EXPECTED_EXECUTION_DURATION" \
    "${archive_mode:-configured}" \
    "$stage_path"

if [ "$watch_flag" = true ]; then
    _pilates_watch "$JOB_ID" "$JOB_LOG_FILE_PATH" "$partition_arg"
fi
