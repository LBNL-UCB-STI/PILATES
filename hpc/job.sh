#!/bin/bash

# HPC execution entrypoint for a scheduled Slurm job.
# Full usage and design notes: hpc/README.md

set -euo pipefail

PILATES_DIR="${PILATES_DIR:-/global/scratch/users/$USER/sources/PILATES}"
VENV_PATH="${PILATES_VENV_PATH:-$PILATES_DIR/PILATES-env}"
REQUIREMENTS_FILE="${PILATES_REQUIREMENTS_FILE:-$PILATES_DIR/hpc/requirements-hpc.txt}"
FALLBACK_REQUIREMENTS_FILE="$PILATES_DIR/requirements.txt"
CONSIST_SRC_DIR="${CONSIST_SRC_DIR:-$PILATES_DIR/consist}"
CONSIST_PYPI_PACKAGE="${CONSIST_PYPI_PACKAGE:-consist==0.1.0}"
MIGRATED_CONFIG_PATH=""

show_system_info() {
    echo "=== MEMORY INFORMATION ==="
    free -h
    grep MemTotal /proc/meminfo || true
    grep -i numa /proc/cpuinfo | head -n 8 || true
    echo "=========================="

    echo "=== NODE USAGE INFORMATION ==="
    squeue -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R" | grep "$(hostname)" || echo "Node info not found in squeue"
    echo "=========================="
}

hash_file() {
    local file_path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file_path" | awk '{print $1}'
    else
        shasum -a 256 "$file_path" | awk '{print $1}'
    fi
}

install_python_deps() {
    local req_file="$1"
    local marker="$VENV_PATH/.last_requirements_hash"
    local current_hash

    current_hash="$(hash_file "$req_file")"
    if [ ! -f "$marker" ] || [ "$current_hash" != "$(cat "$marker")" ]; then
        echo "Installing/updating Python dependencies from $req_file ..."
        python3 -m pip install --upgrade pip setuptools wheel
        python3 -m pip install -r "$req_file"
        printf '%s\n' "$current_hash" > "$marker"
    else
        echo "Python dependencies are up to date; skipping pip install."
    fi
}

install_consist() {
    local marker="$VENV_PATH/.last_consist_install_spec"
    local install_needed=false

    if ! python3 -c "from consist import create_tracker" >/dev/null 2>&1; then
        install_needed=true
    fi

    if [ -d "$CONSIST_SRC_DIR" ]; then
        local consist_spec="editable:$CONSIST_SRC_DIR"
        if command -v git >/dev/null 2>&1 && [ -d "$CONSIST_SRC_DIR/.git" ]; then
            local consist_rev
            consist_rev="$(git -C "$CONSIST_SRC_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")"
            consist_spec="${consist_spec}@${consist_rev}"
        fi

        if [ ! -f "$marker" ] || [ "$consist_spec" != "$(cat "$marker")" ]; then
            install_needed=true
        fi

        if [ "$install_needed" = true ]; then
            echo "Installing local editable consist from $CONSIST_SRC_DIR ..."
            python3 -m pip install -e "$CONSIST_SRC_DIR"
            printf '%s\n' "$consist_spec" > "$marker"
        else
            echo "Editable consist is up to date; skipping reinstall."
        fi
    else
        local consist_spec="pypi:$CONSIST_PYPI_PACKAGE"
        if [ ! -f "$marker" ] || [ "$consist_spec" != "$(cat "$marker")" ]; then
            install_needed=true
        fi

        if [ "$install_needed" = true ]; then
            echo "Installing consist from PyPI package '$CONSIST_PYPI_PACKAGE' ..."
            python3 -m pip install "$CONSIST_PYPI_PACKAGE"
            printf '%s\n' "$consist_spec" > "$marker"
        else
            echo "PyPI consist install is up to date; skipping reinstall."
        fi
    fi

    python3 -c "from consist import create_tracker" >/dev/null
    echo "consist import check passed."
}

normalize_path() {
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

is_new_format() {
    grep -q '^run:' "$1" && grep -q '^shared:' "$1" && grep -q '^infrastructure:' "$1"
}

migrate_legacy_config_if_needed() {
    local config_path="$1"

    if is_new_format "$config_path"; then
        printf '%s\n' "$config_path"
        return
    fi

    MIGRATED_CONFIG_PATH="$(mktemp "$PILATES_DIR/hpc_migrated_config.XXXXXX.yaml")"
    echo "Detected legacy settings format; migrating to $MIGRATED_CONFIG_PATH ..."
    if python3 scripts/migrate_config.py "$config_path" "$MIGRATED_CONFIG_PATH" --no-validate; then
        printf '%s\n' "$MIGRATED_CONFIG_PATH"
        return
    fi

    echo "WARNING: migration failed, continuing with original config"
    rm -f "$MIGRATED_CONFIG_PATH"
    MIGRATED_CONFIG_PATH=""
    printf '%s\n' "$config_path"
}

cleanup() {
    if [ -n "$MIGRATED_CONFIG_PATH" ] && [ -f "$MIGRATED_CONFIG_PATH" ]; then
        rm -f "$MIGRATED_CONFIG_PATH"
    fi
}

trap cleanup EXIT

if [ "${1:-}" = "" ]; then
    echo "Usage: $0 <settings_file> [stage_file]"
    exit 2
fi

CONFIG_FILE="$(normalize_path "$1")"
STAGE_FILE=""
if [ "${2:-}" != "" ]; then
    STAGE_FILE="$(normalize_path "$2")"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found: $CONFIG_FILE"
    exit 1
fi
if [ -n "$STAGE_FILE" ] && [ ! -f "$STAGE_FILE" ]; then
    echo "ERROR: stage file not found: $STAGE_FILE"
    exit 1
fi

echo "Setting up HPC runtime environment..."
module load gcc/11.4.0
module load proj/9.2.1
module load python/3.11.6

export LD_LIBRARY_PATH=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/gcc-11.4.0-nfcdl6bpyabpnhhasfzu6y4ge4kfskvl/lib64:${LD_LIBRARY_PATH:-}
echo "Using LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

cd "$PILATES_DIR"

if [ ! -x "$VENV_PATH/bin/python3" ]; then
    echo "Creating virtual environment at $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    REQUIREMENTS_FILE="$FALLBACK_REQUIREMENTS_FILE"
fi
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: requirements file not found at '$REQUIREMENTS_FILE'"
    exit 1
fi

install_python_deps "$REQUIREMENTS_FILE"
install_consist

CONFIG_FILE="$(migrate_legacy_config_if_needed "$CONFIG_FILE")"

echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"
echo "Config: $CONFIG_FILE"
echo "Stage: ${STAGE_FILE:-<none>}"

show_system_info

# dlt 1.19.x reads the runtime section directly from RUNTIME__* env vars.
# Keep the older DLT__* form as well for compatibility with older installs.
export RUNTIME__DLTHUB_TELEMETRY=false
export DLT__RUNTIME__DLTHUB_TELEMETRY=false

# Cap implicit native thread pools used by NumPy/BLAS/OpenMP-backed libraries.
# This avoids oversubscription/lock contention while still requesting full node
# resources for containerized model runs. Override per job with PILATES_THREADS.
THREADS="${PILATES_THREADS:-8}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export BLIS_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
echo "Thread caps: PILATES_THREADS=$THREADS (OMP/MKL/OPENBLAS/NUMEXPR/BLIS/VECLIB)"

RUN_ARGS=(-c "$CONFIG_FILE")
if [ -n "$STAGE_FILE" ]; then
    RUN_ARGS+=(-S "$STAGE_FILE")
fi

python3 run.py "${RUN_ARGS[@]}"
