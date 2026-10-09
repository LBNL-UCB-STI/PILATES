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
native_structural_canary_seed=""
beam_preprocess_acceptance_manifest=""
activitysim_run_acceptance_manifest=""
activitysim_run_acceptance_editable_consist=""
activitysim_run_acceptance_image=""
urbansim_h5_snapshot_acceptance_manifest=""

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
    --native-structural-canary)
        if [ -z "${2:-}" ]; then
            echo "ERROR: --native-structural-canary requires a seed manifest path." >&2
            exit 2
        fi
        native_structural_canary_seed="$2"
        shift 2
        ;;
    --beam-preprocess-acceptance)
        if [ -z "${2:-}" ]; then
            echo "ERROR: --beam-preprocess-acceptance requires an input manifest path." >&2
            exit 2
        fi
        beam_preprocess_acceptance_manifest="$2"
        shift 2
        ;;
    --activitysim-run-acceptance)
        if [ -z "${2:-}" ]; then
            echo "ERROR: --activitysim-run-acceptance requires an input manifest path." >&2
            exit 2
        fi
        activitysim_run_acceptance_manifest="$2"
        shift 2
        ;;
    --editable-consist)
        if [ -z "${2:-}" ]; then
            echo "ERROR: --editable-consist requires an absolute Consist checkout path." >&2
            exit 2
        fi
        activitysim_run_acceptance_editable_consist="$2"
        shift 2
        ;;
    --activitysim-image)
        if [ -z "${2:-}" ]; then
            echo "ERROR: --activitysim-image requires an immutable docker:// image reference." >&2
            exit 2
        fi
        activitysim_run_acceptance_image="$2"
        shift 2
        ;;
    --urbansim-h5-snapshot-acceptance)
        if [ -z "${2:-}" ]; then
            echo "ERROR: --urbansim-h5-snapshot-acceptance requires an input manifest path." >&2
            exit 2
        fi
        urbansim_h5_snapshot_acceptance_manifest="$2"
        shift 2
        ;;
    -h|--help)
        echo "Usage: $0 [-c settings file] [-s stage file] [-p partition] [-a account] [--high-mem|-H] [--native-structural-canary seed-manifest.json] [--beam-preprocess-acceptance input-manifest.json] [--activitysim-run-acceptance input-manifest.json [--editable-consist /absolute/consist] [--activitysim-image docker://image@sha256:digest]] [--urbansim-h5-snapshot-acceptance input-manifest.json]"
        echo "  -a, --account: Slurm account name (required)"
        echo "  --high-mem: for lr7 only, request 480G instead of default 240G."
        echo "  --native-structural-canary: copy a reviewed seed manifest into per-job canary evidence."
        echo "  --beam-preprocess-acceptance: run the isolated cold-to-fresh BEAM preprocess acceptance harness."
        echo "  --activitysim-run-acceptance: run the isolated cold-to-fresh ActivitySim run acceptance harness."
        echo "  --editable-consist: explicit source for an editable ActivitySim pre-merge acceptance manifest."
        echo "  --activitysim-image: immutable singularity image override for ActivitySim acceptance only."
        echo "  --urbansim-h5-snapshot-acceptance: capture and reconcile the UrbanSim HDF5 snapshot acceptance harness."
        exit 0
        ;;
    *)
        printf "Usage: %s [-c settings file] [-s stage file] [-p partition] [-a account] [--high-mem|-H] [--native-structural-canary seed-manifest.json] [--beam-preprocess-acceptance input-manifest.json] [--activitysim-run-acceptance input-manifest.json [--editable-consist /absolute/consist] [--activitysim-image docker://image@sha256:digest]] [--urbansim-h5-snapshot-acceptance input-manifest.json]\n" "$0"
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
native_structural_canary_seed_path=""
if [ -n "$native_structural_canary_seed" ]; then
    native_structural_canary_seed_path="$(resolve_path "$native_structural_canary_seed")"
fi
beam_preprocess_acceptance_manifest_path=""
if [ -n "$beam_preprocess_acceptance_manifest" ]; then
    beam_preprocess_acceptance_manifest_path="$(resolve_path "$beam_preprocess_acceptance_manifest")"
fi
activitysim_run_acceptance_manifest_path=""
if [ -n "$activitysim_run_acceptance_manifest" ]; then
    activitysim_run_acceptance_manifest_path="$(resolve_path "$activitysim_run_acceptance_manifest")"
fi
urbansim_h5_snapshot_acceptance_manifest_path=""
if [ -n "$urbansim_h5_snapshot_acceptance_manifest" ]; then
    urbansim_h5_snapshot_acceptance_manifest_path="$(resolve_path "$urbansim_h5_snapshot_acceptance_manifest")"
fi

if [ ! -f "$settings_template_path" ]; then
    echo "ERROR: settings file not found: $settings_template_path"
    exit 1
fi
if [ -n "$beam_preprocess_acceptance_manifest" ] \
    && [ ! -f "$beam_preprocess_acceptance_manifest_path" ]; then
    echo "ERROR: beam preprocess acceptance input manifest not found: $beam_preprocess_acceptance_manifest_path" >&2
    exit 1
fi
if [ -n "$activitysim_run_acceptance_manifest" ] \
    && [ ! -f "$activitysim_run_acceptance_manifest_path" ]; then
    echo "ERROR: ActivitySim run acceptance input manifest not found: $activitysim_run_acceptance_manifest_path" >&2
    exit 1
fi
if [ -n "$activitysim_run_acceptance_editable_consist" ] \
    && [ -z "$activitysim_run_acceptance_manifest" ]; then
    echo "ERROR: --editable-consist is only valid with --activitysim-run-acceptance." >&2
    exit 2
fi
if [ -n "$activitysim_run_acceptance_image" ] \
    && [ -z "$activitysim_run_acceptance_manifest" ]; then
    echo "ERROR: --activitysim-image is only valid with --activitysim-run-acceptance." >&2
    exit 2
fi
if [ -n "$activitysim_run_acceptance_image" ] \
    && ! printf '%s\n' "$activitysim_run_acceptance_image" | grep -Eq '^docker://[A-Za-z0-9._/@:+-]+@sha256:[0-9a-f]{64}$'; then
    echo "ERROR: --activitysim-image must be an immutable docker:// image digest." >&2
    exit 2
fi
if [ -n "$activitysim_run_acceptance_editable_consist" ] \
    && { [ "${activitysim_run_acceptance_editable_consist#/}" = "$activitysim_run_acceptance_editable_consist" ] || [ ! -d "$activitysim_run_acceptance_editable_consist" ]; }; then
    echo "ERROR: --editable-consist must name an existing absolute Consist checkout." >&2
    exit 2
fi
if [ -n "$urbansim_h5_snapshot_acceptance_manifest" ] \
    && [ ! -f "$urbansim_h5_snapshot_acceptance_manifest_path" ]; then
    echo "ERROR: UrbanSim HDF5 snapshot acceptance input manifest not found: $urbansim_h5_snapshot_acceptance_manifest_path" >&2
    exit 1
fi
if [ -n "$beam_preprocess_acceptance_manifest" ] \
    && [ -n "$native_structural_canary_seed" ]; then
    echo "ERROR: --beam-preprocess-acceptance cannot be combined with --native-structural-canary." >&2
    exit 2
fi
if [ -n "$urbansim_h5_snapshot_acceptance_manifest" ] \
    && [ -n "$beam_preprocess_acceptance_manifest" ]; then
    echo "ERROR: --urbansim-h5-snapshot-acceptance cannot be combined with --beam-preprocess-acceptance." >&2
    exit 2
fi
if [ -n "$activitysim_run_acceptance_manifest" ] \
    && [ -n "$beam_preprocess_acceptance_manifest" ]; then
    echo "ERROR: --activitysim-run-acceptance cannot be combined with --beam-preprocess-acceptance." >&2
    exit 2
fi
if [ -n "$activitysim_run_acceptance_manifest" ] \
    && [ -n "$urbansim_h5_snapshot_acceptance_manifest" ]; then
    echo "ERROR: --activitysim-run-acceptance cannot be combined with --urbansim-h5-snapshot-acceptance." >&2
    exit 2
fi
if [ -n "$urbansim_h5_snapshot_acceptance_manifest" ] \
    && [ -n "$native_structural_canary_seed" ]; then
    echo "ERROR: --urbansim-h5-snapshot-acceptance cannot be combined with --native-structural-canary." >&2
    exit 2
fi
if [ -n "$activitysim_run_acceptance_manifest" ] \
    && [ -n "$native_structural_canary_seed" ]; then
    echo "ERROR: --activitysim-run-acceptance cannot be combined with --native-structural-canary." >&2
    exit 2
fi
if [ -n "$beam_preprocess_acceptance_manifest" ] && [ -n "$stage_file" ]; then
    echo "ERROR: --beam-preprocess-acceptance cannot be combined with -s/--stage." >&2
    exit 2
fi
if [ -n "$activitysim_run_acceptance_manifest" ] && [ -n "$stage_file" ]; then
    echo "ERROR: --activitysim-run-acceptance cannot be combined with -s/--stage." >&2
    exit 2
fi
if [ -n "$urbansim_h5_snapshot_acceptance_manifest" ] && [ -n "$stage_file" ]; then
    echo "ERROR: --urbansim-h5-snapshot-acceptance cannot be combined with -s/--stage." >&2
    exit 2
fi
if [ -n "$native_structural_canary_seed" ] && [ -n "$stage_file" ]; then
    echo "ERROR: --native-structural-canary cannot be combined with -s/--stage." >&2
    exit 2
fi
if [ -n "$native_structural_canary_seed" ] \
    && [ ! -f "$native_structural_canary_seed_path" ]; then
    echo "ERROR: native structural canary seed manifest not found: $native_structural_canary_seed_path"
    exit 1
fi

mkdir -p "/global/scratch/users/$USER/pilates_logs"

if command -v envsubst >/dev/null 2>&1; then
    BEAM_MEMORY="$BEAM_MEMORY" envsubst '$BEAM_MEMORY' < "$settings_template_path" > "$generated_settings_path"
else
    sed "s|\${BEAM_MEMORY}|$BEAM_MEMORY|g" "$settings_template_path" > "$generated_settings_path"
fi

if [ -n "$activitysim_run_acceptance_image" ]; then
    image_override_path="${generated_settings_path}.activitysim-image-override"
    if ! awk -v image="$activitysim_run_acceptance_image" '
        /^  singularity_images:[[:space:]]*$/ {
            in_singularity_images = 1
            print
            next
        }
        in_singularity_images && /^  [[:alnum:]_]+:[[:space:]]*$/ { in_singularity_images = 0 }
        in_singularity_images && /^    activitysim:[[:space:]]*/ {
            print "    activitysim: " image
            replaced = 1
            next
        }
        { print }
        END {
            if (!replaced) {
                exit 1
            }
        }
    ' "$generated_settings_path" > "$image_override_path"; then
        echo "ERROR: --activitysim-image could not replace infrastructure.singularity_images.activitysim in $settings_template_path." >&2
        exit 2
    fi
    mv "$image_override_path" "$generated_settings_path"
fi

export BEAM_MEMORY

if [ -n "$native_structural_canary_seed_path" ]; then
    canary_root="${PILATES_NATIVE_STRUCTURAL_CANARY_ROOT:-/global/scratch/users/$USER/pilates-canaries}"
    canary_evidence_dir="${canary_root%/}/$JOB_NAME"
    canary_manifest_path="$canary_evidence_dir/canary.json"
    mkdir -p "$canary_evidence_dir/generated"
    cp "$native_structural_canary_seed_path" "$canary_manifest_path"
    cp "$generated_settings_path" "$canary_evidence_dir/generated/settings.yaml"
    export PILATES_NATIVE_STRUCTURAL_CANARY_MANIFEST="$canary_manifest_path"
    echo "Native structural canary: $canary_evidence_dir"
fi

beam_preprocess_acceptance_evidence_dir=""
beam_preprocess_acceptance_evidence_manifest=""
if [ -n "$beam_preprocess_acceptance_manifest_path" ]; then
    acceptance_root="${PILATES_BEAM_PREPROCESS_ACCEPTANCE_ROOT:-/global/scratch/users/$USER/pilates-boundary-promotions}"
    beam_preprocess_acceptance_evidence_dir="${acceptance_root%/}/$JOB_NAME"
    beam_preprocess_acceptance_evidence_manifest="$beam_preprocess_acceptance_evidence_dir/submitted-input-manifest.json"
    mkdir -p "$beam_preprocess_acceptance_evidence_dir"
    cp "$beam_preprocess_acceptance_manifest_path" "$beam_preprocess_acceptance_evidence_manifest"
    cp "$generated_settings_path" "$beam_preprocess_acceptance_evidence_dir/generated-settings.yaml"
    echo "BEAM preprocess acceptance evidence: $beam_preprocess_acceptance_evidence_dir"
fi

activitysim_run_acceptance_evidence_dir=""
activitysim_run_acceptance_evidence_manifest=""
if [ -n "$activitysim_run_acceptance_manifest_path" ]; then
    acceptance_root="${PILATES_ACTIVITYSIM_RUN_ACCEPTANCE_ROOT:-/global/scratch/users/$USER/pilates-boundary-promotions}"
    activitysim_run_acceptance_evidence_dir="${acceptance_root%/}/$JOB_NAME"
    activitysim_run_acceptance_evidence_manifest="$activitysim_run_acceptance_evidence_dir/submitted-input-manifest.json"
    mkdir -p "$activitysim_run_acceptance_evidence_dir"
    cp "$activitysim_run_acceptance_manifest_path" "$activitysim_run_acceptance_evidence_manifest"
    cp "$generated_settings_path" "$activitysim_run_acceptance_evidence_dir/generated-settings.yaml"
    if [ -n "$activitysim_run_acceptance_image" ]; then
        printf '%s\n' "$activitysim_run_acceptance_image" \
            > "$activitysim_run_acceptance_evidence_dir/activitysim-image-override.txt"
    fi
    echo "ActivitySim run acceptance evidence: $activitysim_run_acceptance_evidence_dir"
fi

urbansim_h5_snapshot_acceptance_evidence_dir=""
if [ -n "$urbansim_h5_snapshot_acceptance_manifest_path" ]; then
    acceptance_root="${PILATES_URBANSIM_H5_SNAPSHOT_ACCEPTANCE_ROOT:-/global/scratch/users/$USER/pilates-boundary-promotions}"
    urbansim_h5_snapshot_acceptance_evidence_dir="${acceptance_root%/}/$JOB_NAME"
    mkdir -p "$urbansim_h5_snapshot_acceptance_evidence_dir"
    cp "$urbansim_h5_snapshot_acceptance_manifest_path" "$urbansim_h5_snapshot_acceptance_evidence_dir/submitted-input-manifest.json"
    cp "$generated_settings_path" "$urbansim_h5_snapshot_acceptance_evidence_dir/generated-settings.yaml"
    echo "UrbanSim HDF5 snapshot acceptance evidence: $urbansim_h5_snapshot_acceptance_evidence_dir"
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
    --export="ALL,JOB_LOG_FILE_PATH=$JOB_LOG_FILE_PATH"
    "$PILATES_DIR/hpc/job.sh"
)
if [ -n "$activitysim_run_acceptance_evidence_dir" ]; then
    sbatch_args+=(
        --activitysim-run-acceptance
        "$generated_settings_path"
        "$activitysim_run_acceptance_evidence_manifest"
        "$activitysim_run_acceptance_evidence_dir"
    )
    if [ -n "$activitysim_run_acceptance_editable_consist" ]; then
        sbatch_args+=(
            --editable-consist
            "$activitysim_run_acceptance_editable_consist"
        )
    fi
elif [ -n "$urbansim_h5_snapshot_acceptance_evidence_dir" ]; then
    sbatch_args+=(
        --urbansim-h5-snapshot-acceptance
        "$generated_settings_path"
        "$urbansim_h5_snapshot_acceptance_manifest_path"
        "$urbansim_h5_snapshot_acceptance_evidence_dir"
    )
elif [ -n "$beam_preprocess_acceptance_evidence_dir" ]; then
    sbatch_args+=(
        --beam-preprocess-acceptance
        "$generated_settings_path"
        "$beam_preprocess_acceptance_manifest_path"
        "$beam_preprocess_acceptance_evidence_dir"
    )
else
    sbatch_args+=("$generated_settings_path")
fi
if [ -n "$stage_path" ]; then
    sbatch_args+=("$stage_path")
fi

sbatch "${sbatch_args[@]}"

echo "$JOB_LOG_FILE_PATH"
