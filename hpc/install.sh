#!/bin/bash

# Install and verify PILATES HPC runtime prerequisites without launching a job.

set -euo pipefail

error_exit() {
    echo "ERROR: install.sh failed at line $1: $2" >&2
    exit "$3"
}

trap 'error_exit ${LINENO} "$BASH_COMMAND" $?' ERR

PILATES_DIR="${PILATES_DIR:-/global/scratch/users/$USER/sources/PILATES}"
VENV_PATH="${PILATES_VENV_PATH:-$PILATES_DIR/PILATES-env}"
REQUIREMENTS_FILE="${PILATES_REQUIREMENTS_FILE:-$PILATES_DIR/hpc/requirements-hpc.txt}"
CONSIST_SRC_DIR="${CONSIST_SRC_DIR:-$PILATES_DIR/../consist}"
CONSIST_PYPI_PACKAGE="${CONSIST_PYPI_PACKAGE:-}"
DEFAULT_CONSIST_PYPI_PACKAGE="consist==0.1.4"
RUN_NOTIFICATIONS_ENV="${PILATES_RUN_NOTIFICATIONS_ENV:-$PILATES_DIR/hpc/run-notifications.env}"
LOCAL_PREFIX="${PILATES_LOCAL_PREFIX:-/global/home/users/$USER/.local}"
GIT_LFS_VERSION="${PILATES_GIT_LFS_VERSION:-2.3.4}"
GIT_LFS_TARBALL="git-lfs-linux-amd64-${GIT_LFS_VERSION}.tar.gz"
GIT_LFS_URL="https://github.com/git-lfs/git-lfs/releases/download/v${GIT_LFS_VERSION}/${GIT_LFS_TARBALL}"

CHECK_ONLY=0
SKIP_NOTIFICATIONS_ENV=0
STUDY_AREA=""
ASIM_CONFIGS_BRANCH=""
BEAM_DATA_BRANCH=""
ENABLE_GCHAT_NOTIFICATIONS=0
ENABLE_SLACK_NOTIFICATIONS=0
ENABLE_GSHEET_PUBLISH=0

# Editable study-area manifest. Add new areas to STUDY_AREAS and the helper
# functions below.
STUDY_AREAS=("sfbay" "seattle" "austin")

join_by() {
    local delimiter="$1"
    shift
    local first=1
    local item
    for item in "$@"; do
        if [ "$first" -eq 1 ]; then
            printf "%s" "$item"
            first=0
        else
            printf "%s%s" "$delimiter" "$item"
        fi
    done
}

study_area_value() {
    local area="$1"
    local field="$2"

    case "$area:$field" in
        sfbay:urbansim_region_id) echo "06197001" ;;
        seattle:urbansim_region_id) echo "53199100" ;;
        austin:urbansim_region_id) echo "48197301" ;;

        sfbay:urbansim_input_url) echo "https://storage.googleapis.com/beam-core-outputs/urbansim-inputs/custom_mpo_06197001_model_data_2017.h5" ;;
        seattle:urbansim_input_url) echo "" ;;
        austin:urbansim_input_url) echo "" ;;

        sfbay:activitysim_repo_url) echo "https://github.com/LBNL-UCB-STI/activitysim-configs-sfbay.git" ;;
        seattle:activitysim_repo_url) echo "" ;;
        austin:activitysim_repo_url) echo "" ;;

        sfbay:activitysim_repo_branch) echo "main" ;;
        seattle:activitysim_repo_branch) echo "" ;;
        austin:activitysim_repo_branch) echo "" ;;

        sfbay:activitysim_repo_dir) echo "sfbay" ;;
        seattle:activitysim_repo_dir) echo "seattle" ;;
        austin:activitysim_repo_dir) echo "austin" ;;

        sfbay:beam_repo_url) echo "https://github.com/LBNL-UCB-STI/beam-data-sfbay.git" ;;
        seattle:beam_repo_url) echo "" ;;
        austin:beam_repo_url) echo "" ;;

        sfbay:beam_repo_branch) echo "develop" ;;
        seattle:beam_repo_branch) echo "" ;;
        austin:beam_repo_branch) echo "" ;;

        sfbay:beam_repo_dir) echo "sfbay" ;;
        seattle:beam_repo_dir) echo "seattle" ;;
        austin:beam_repo_dir) echo "austin" ;;

        *)
            echo "ERROR: unknown study area mapping '$area:$field'" >&2
            exit 1
            ;;
    esac
}

usage() {
    cat <<EOF
Usage: $0 --study-area <$(join_by '|' "${STUDY_AREAS[@]}")> [options]

Options:
  --study-area AREA          Study area to install data for.
  --asim-configs-branch BR   ActivitySim configs branch to use.
  --beam-data-branch BR      BEAM production data branch to use.
  --enable-gchat             Enable Google Chat notifications in run-notifications.env.
  --enable-slack             Enable Slack notifications in run-notifications.env.
  --enable-gsheet-publish    Enable Google Sheet publishing in run-notifications.env.
  --check-only               Verify prerequisites without downloading/installing.
  --skip-notifications-env   Do not create or refresh hpc/run-notifications.env.
  -h, --help                 Show this help.

Environment overrides:
  PILATES_DIR
  PILATES_VENV_PATH
  PILATES_REQUIREMENTS_FILE
  CONSIST_SRC_DIR
  CONSIST_PYPI_PACKAGE
  PILATES_RUN_NOTIFICATIONS_ENV
  PILATES_LOCAL_PREFIX
  PILATES_GIT_LFS_VERSION
  PILATES_GCHAT_WEBHOOK_URL
  PILATES_SLACK_WEBHOOK_URL
  PILATES_GSHEET_WEBHOOK_URL
  PILATES_GSHEET_SECRET

Editable study-area resources live near the top of this script:
  STUDY_AREAS
  study_area_value()
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --study-area)
            STUDY_AREA="${2:-}"
            shift 2
            ;;
        --asim-configs-branch)
            ASIM_CONFIGS_BRANCH="${2:-}"
            shift 2
            ;;
        --beam-data-branch)
            BEAM_DATA_BRANCH="${2:-}"
            shift 2
            ;;
        --enable-gchat)
            ENABLE_GCHAT_NOTIFICATIONS=1
            shift
            ;;
        --enable-slack)
            ENABLE_SLACK_NOTIFICATIONS=1
            shift
            ;;
        --enable-gsheet-publish)
            ENABLE_GSHEET_PUBLISH=1
            shift
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --skip-notifications-env)
            SKIP_NOTIFICATIONS_ENV=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$STUDY_AREA" ]; then
    echo "ERROR: --study-area is required." >&2
    usage >&2
    exit 2
fi

is_supported_area=0
for area in "${STUDY_AREAS[@]}"; do
    if [ "$area" = "$STUDY_AREA" ]; then
        is_supported_area=1
        break
    fi
done
if [ "$is_supported_area" -ne 1 ]; then
    echo "ERROR: unsupported study area '$STUDY_AREA'" >&2
    usage >&2
    exit 2
fi

log() {
    echo "[install-runtime] $*"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: $1" >&2
        exit 1
    }
}

load_hpc_modules() {
    if command -v module >/dev/null 2>&1; then
        set +u
        module load gcc/11.4.0
        module load proj/9.2.1
        module load python/3.11.6
        set -u
    else
        log "Environment modules are unavailable; assuming compilers/Python are already configured."
    fi
}

resolve_requirements_file() {
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "$REQUIREMENTS_FILE"
    else
        echo "ERROR: requirements file not found: '$REQUIREMENTS_FILE'" >&2
        exit 1
    fi
}

install_python_deps() {
    local req_file="$1"
    local marker="$VENV_PATH/.last_requirements_hash"
    local filtered_req
    local current_hash
    filtered_req="$(mktemp)"
    grep -Ev '^[[:space:]]*consist([[:space:]]|[<>=!~].*)?$' "$req_file" > "$filtered_req" || true
    current_hash="$(sha256sum "$filtered_req" | awk '{print $1}')"

    if [ ! -f "$marker" ] || [ "$current_hash" != "$(cat "$marker")" ]; then
        log "Installing/updating Python dependencies from $req_file"
        python3 -m pip install --upgrade pip setuptools wheel
        python3 -m pip install -r "$filtered_req"
        printf "%s\n" "$current_hash" > "$marker"
    else
        log "Python dependencies are current"
    fi

    rm -f "$filtered_req"
}

resolve_consist_package_spec() {
    local package_spec
    local req_file

    if [ -n "$CONSIST_PYPI_PACKAGE" ]; then
        echo "$CONSIST_PYPI_PACKAGE"
        return
    fi

    for req_file in "$@"; do
        if [ -f "$req_file" ]; then
            package_spec="$(grep -E '^[[:space:]]*consist([<>=!~].*)?$' "$req_file" | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            if [ -n "$package_spec" ]; then
                echo "$package_spec"
                return
            fi
        fi
    done

    echo "$DEFAULT_CONSIST_PYPI_PACKAGE"
}

install_consist() {
    local req_file="$1"
    local package_spec
    package_spec="$(resolve_consist_package_spec "$req_file")"

    if [ -d "$CONSIST_SRC_DIR" ]; then
        log "Installing editable consist from $CONSIST_SRC_DIR"
        if ! python3 -m pip install -e "$CONSIST_SRC_DIR"; then
            log "Editable consist install failed; falling back to $package_spec"
            python3 -m pip install "$package_spec"
        fi
    else
        log "Local consist source not found; installing $package_spec"
        python3 -m pip install "$package_spec"
    fi
}

ensure_venv() {
    local req_file="$1"
    if [ ! -x "$VENV_PATH/bin/python3" ]; then
        log "Creating virtual environment at $VENV_PATH"
        python3 -m venv "$VENV_PATH"
    fi
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    install_python_deps "$req_file"
    install_consist "$req_file"
}

ensure_git_lfs() {
    local install_root="$PILATES_DIR/.downloads/git-lfs"
    local tarball_path="$install_root/$GIT_LFS_TARBALL"
    local tar_path="${tarball_path%.gz}"
    local extracted_dir="$install_root/git-lfs-$GIT_LFS_VERSION"
    local install_script="$extracted_dir/install.sh"
    local git_lfs_bin="$LOCAL_PREFIX/bin/git-lfs"

    mkdir -p "$install_root"
    mkdir -p "$LOCAL_PREFIX/bin"

    if [ -x "$git_lfs_bin" ]; then
        log "git-lfs already installed at $git_lfs_bin"
        return
    fi

    require_cmd wget
    require_cmd gunzip
    require_cmd tar

    log "Downloading git-lfs $GIT_LFS_VERSION"
    wget -O "$tarball_path" "$GIT_LFS_URL"
    rm -f "$tar_path"
    gunzip -f "$tarball_path"
    tar -xf "$tar_path" -C "$install_root"

    if [ ! -f "$install_script" ]; then
        echo "ERROR: expected git-lfs installer not found: $install_script" >&2
        exit 1
    fi

    sed -i "s|^prefix=.*$|prefix=\"$LOCAL_PREFIX\"|" "$install_script"
    sed -i 's|git lfs install$|git lfs install --skip-repo 2>/dev/null || true|' "$install_script"

    log "Installing git-lfs into $LOCAL_PREFIX"
    (
        cd "$extracted_dir"
        ./install.sh
    )
}

ensure_directory() {
    mkdir -p "$1"
}

download_if_missing() {
    local url="$1"
    local output_path="$2"

    if [ -z "$url" ]; then
        log "No download URL configured for $output_path; skipping"
        return
    fi

    ensure_directory "$(dirname "$output_path")"
    if [ -f "$output_path" ]; then
        log "Already present: $output_path"
        return
    fi

    require_cmd wget
    log "Downloading $(basename "$output_path")"
    wget -O "$output_path" "$url"
}

clone_or_verify_repo() {
    local repo_url="$1"
    local branch="$2"
    local target_dir="$3"

    if [ -z "$repo_url" ]; then
        log "No repo configured for $target_dir; skipping"
        return
    fi

    ensure_directory "$(dirname "$target_dir")"
    if [ -d "$target_dir/.git" ]; then
        log "Repo already exists: $target_dir"
        (
            cd "$target_dir"
            git fetch origin "$branch"
            git checkout "$branch"
        )
        log "Checked out existing repo at branch $branch: $target_dir"
        return
    fi

    require_cmd git
    log "Cloning $repo_url into $target_dir (branch=$branch)"
    git clone --branch "$branch" "$repo_url" "$target_dir"
}

init_notifications_env() {
    local gchat_notifications="${PILATES_GCHAT_NOTIFICATIONS:-$ENABLE_GCHAT_NOTIFICATIONS}"
    local slack_notifications="${PILATES_SLACK_NOTIFICATIONS:-$ENABLE_SLACK_NOTIFICATIONS}"
    local gsheet_publish="${PILATES_GSHEET_PUBLISH:-$ENABLE_GSHEET_PUBLISH}"
    local gchat_webhook="${PILATES_GCHAT_WEBHOOK_URL:-}"
    local slack_webhook="${PILATES_SLACK_WEBHOOK_URL:-}"
    local gsheet_webhook="${PILATES_GSHEET_WEBHOOK_URL:-}"
    local gsheet_secret="${PILATES_GSHEET_SECRET:-}"

    ensure_directory "$(dirname "$RUN_NOTIFICATIONS_ENV")"
    cat > "$RUN_NOTIFICATIONS_ENV" <<EOF
# PILATES run notifications loaded by hpc/job_runner.sh.
# This file is generated by hpc/install.sh.
# Secret values should be supplied via environment variables when invoking the
# installer, not hardcoded in the script.

# Google Chat notifications
export PILATES_GCHAT_NOTIFICATIONS=${gchat_notifications}
export PILATES_GCHAT_WEBHOOK_URL="${gchat_webhook}"
export PILATES_GCHAT_TIMEOUT_SECONDS=5

# Optional Slack notifications
export PILATES_SLACK_NOTIFICATIONS=${slack_notifications}
export PILATES_SLACK_WEBHOOK_URL="${slack_webhook}"
export PILATES_SLACK_TIMEOUT_SECONDS=5

# Include internal bootstrap/setup traces if desired.
export PILATES_RUN_NOTIFICATIONS_INCLUDE_INTERNAL=0

# Local run artifact publishers
export PILATES_RUN_EVENT_LOG=1
export PILATES_RUN_SUMMARY_HTML=1

# Optional Google Sheet publishing
export PILATES_GSHEET_PUBLISH=${gsheet_publish}
export PILATES_GSHEET_WEBHOOK_URL="${gsheet_webhook}"
export PILATES_GSHEET_SECRET="${gsheet_secret}"
export PILATES_GSHEET_TIMEOUT_SECONDS=5
EOF
    log "Wrote notifications env: $RUN_NOTIFICATIONS_ENV"
    log "Use installer environment variables for webhook URLs and secrets to avoid hardcoding them in the script."
}

ensure_notifications_env() {
    if [ "$SKIP_NOTIFICATIONS_ENV" -eq 1 ]; then
        log "Skipping notifications env setup by request"
        return
    fi

    if [ ! -f "$RUN_NOTIFICATIONS_ENV" ]; then
        init_notifications_env
    else
        log "Notifications env already exists: $RUN_NOTIFICATIONS_ENV"
    fi
}

check_file() {
    local path="$1"
    local label="$2"
    if [ -f "$path" ]; then
        echo "[ok] $label -> $path"
    else
        echo "[missing] $label -> $path"
        return 1
    fi
}

check_dir() {
    local path="$1"
    local label="$2"
    if [ -d "$path" ]; then
        echo "[ok] $label -> $path"
    else
        echo "[missing] $label -> $path"
        return 1
    fi
}

check_repo() {
    local path="$1"
    local label="$2"
    if [ -d "$path/.git" ]; then
        echo "[ok] $label repo -> $path"
    else
        echo "[missing] $label repo -> $path"
        return 1
    fi
}

check_notifications_env() {
    if [ ! -f "$RUN_NOTIFICATIONS_ENV" ]; then
        echo "[missing] notifications env -> $RUN_NOTIFICATIONS_ENV"
        return 1
    fi
    echo "[ok] notifications env -> $RUN_NOTIFICATIONS_ENV"
}

verify_prereqs() {
    local req_file="$1"
    local status=0
    local region_id
    local urbansim_url
    local asim_repo_url
    local asim_repo_dir_name
    local beam_repo_url
    local beam_repo_dir_name
    local urbansim_input
    local asim_repo_dir
    local beam_repo_dir
    local git_lfs_bin="$LOCAL_PREFIX/bin/git-lfs"

    region_id="$(study_area_value "$STUDY_AREA" "urbansim_region_id")"
    urbansim_url="$(study_area_value "$STUDY_AREA" "urbansim_input_url")"
    asim_repo_url="$(study_area_value "$STUDY_AREA" "activitysim_repo_url")"
    asim_repo_dir_name="$(study_area_value "$STUDY_AREA" "activitysim_repo_dir")"
    beam_repo_url="$(study_area_value "$STUDY_AREA" "beam_repo_url")"
    beam_repo_dir_name="$(study_area_value "$STUDY_AREA" "beam_repo_dir")"
    urbansim_input="$PILATES_DIR/pilates/urbansim/data/custom_mpo_${region_id}_model_data.h5"
    asim_repo_dir="$PILATES_DIR/pilates/activitysim/configs/${asim_repo_dir_name}"
    beam_repo_dir="$PILATES_DIR/pilates/beam/production/${beam_repo_dir_name}"

    check_dir "$PILATES_DIR" "PILATES root" || status=1
    check_file "$req_file" "requirements file" || status=1
    check_file "$VENV_PATH/bin/python3" "virtualenv python" || status=1
    check_file "$git_lfs_bin" "git-lfs binary" || status=1
    if [ -n "$urbansim_url" ]; then
        check_file "$urbansim_input" "UrbanSim input" || status=1
    fi
    if [ -n "$asim_repo_url" ]; then
        check_repo "$asim_repo_dir" "ActivitySim configs" || status=1
    fi
    if [ -n "$beam_repo_url" ]; then
        check_repo "$beam_repo_dir" "BEAM production data" || status=1
    fi
    if [ "$SKIP_NOTIFICATIONS_ENV" -eq 0 ]; then
        check_notifications_env || status=1
    fi

    if [ -x "$VENV_PATH/bin/python3" ]; then
        "$VENV_PATH/bin/python3" - <<'PY' || status=1
import importlib
modules = ["yaml", "pandas", "numpy", "consist"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python modules: {', '.join(missing)}")
print("[ok] Python imports -> yaml, pandas, numpy, consist")
PY
    fi

    if [ "$SKIP_NOTIFICATIONS_ENV" -eq 0 ] && [ -f "$RUN_NOTIFICATIONS_ENV" ]; then
        echo "[info] Notification status snapshot"
        # shellcheck disable=SC1090
        (
            set -a
            source "$RUN_NOTIFICATIONS_ENV"
            set +a
            echo "  gchat enabled=${PILATES_GCHAT_NOTIFICATIONS:-0}"
            if [ -n "${PILATES_GCHAT_WEBHOOK_URL:-}" ]; then
                echo "  gchat webhook=set"
            else
                echo "  gchat webhook=missing"
            fi
            echo "  gsheet enabled=${PILATES_GSHEET_PUBLISH:-0}"
            if [ -n "${PILATES_GSHEET_WEBHOOK_URL:-}" ]; then
                echo "  gsheet webhook=set"
            else
                echo "  gsheet webhook=missing"
            fi
        )
    fi

    return "$status"
}

install_study_area_assets() {
    local region_id
    local urbansim_url
    local asim_repo_url
    local asim_repo_branch
    local asim_repo_dir_name
    local beam_repo_url
    local beam_repo_branch
    local beam_repo_dir_name
    local urbansim_input
    local asim_repo_dir
    local beam_repo_dir

    region_id="$(study_area_value "$STUDY_AREA" "urbansim_region_id")"
    urbansim_url="$(study_area_value "$STUDY_AREA" "urbansim_input_url")"
    asim_repo_url="$(study_area_value "$STUDY_AREA" "activitysim_repo_url")"
    asim_repo_branch="$(study_area_value "$STUDY_AREA" "activitysim_repo_branch")"
    asim_repo_dir_name="$(study_area_value "$STUDY_AREA" "activitysim_repo_dir")"
    beam_repo_url="$(study_area_value "$STUDY_AREA" "beam_repo_url")"
    beam_repo_branch="$(study_area_value "$STUDY_AREA" "beam_repo_branch")"
    beam_repo_dir_name="$(study_area_value "$STUDY_AREA" "beam_repo_dir")"
    asim_repo_branch="${ASIM_CONFIGS_BRANCH:-$asim_repo_branch}"
    beam_repo_branch="${BEAM_DATA_BRANCH:-$beam_repo_branch}"
    urbansim_input="$PILATES_DIR/pilates/urbansim/data/custom_mpo_${region_id}_model_data.h5"
    asim_repo_dir="$PILATES_DIR/pilates/activitysim/configs/${asim_repo_dir_name}"
    beam_repo_dir="$PILATES_DIR/pilates/beam/production/${beam_repo_dir_name}"

    ensure_directory "$PILATES_DIR/pilates/urbansim/data"
    ensure_directory "$PILATES_DIR/pilates/activitysim/configs"
    ensure_directory "$PILATES_DIR/pilates/beam/production"

    download_if_missing "$urbansim_url" "$urbansim_input"
    clone_or_verify_repo "$asim_repo_url" "$asim_repo_branch" "$asim_repo_dir"
    clone_or_verify_repo "$beam_repo_url" "$beam_repo_branch" "$beam_repo_dir"
}

main() {
    local req_file

    require_cmd python3
    require_cmd sha256sum
    require_cmd grep
    require_cmd sed
    require_cmd mkdir

    load_hpc_modules
    req_file="$(resolve_requirements_file)"

    if [ "$CHECK_ONLY" -eq 0 ]; then
        ensure_directory "$PILATES_DIR"
        cd "$PILATES_DIR"
        ensure_venv "$req_file"
        ensure_git_lfs
        install_study_area_assets
        ensure_notifications_env
    fi

    log "Verifying runtime prerequisites for study area: $STUDY_AREA"
    verify_prereqs "$req_file"
}

main "$@"
