#!/bin/bash

# HPC execution entrypoint for a scheduled Slurm job.
# Full usage and design notes: hpc/README.md

set -euo pipefail

# Comprehensive error reporting
error_exit() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    echo "ERROR: job.sh failed!" >&2
    echo "  Line: $1" >&2
    echo "  Command: $2" >&2
    echo "  Exit code: $3" >&2
    echo "  Job log: ${JOB_LOG_FILE_PATH:-<not set>}" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    exit "$3"
}

trap 'error_exit ${LINENO} "$BASH_COMMAND" $?' ERR

PILATES_DIR="${PILATES_DIR:-/global/scratch/users/$USER/sources/PILATES}"
VENV_PATH="${PILATES_VENV_PATH:-$PILATES_DIR/PILATES-env}"
REQUIREMENTS_FILE="${PILATES_REQUIREMENTS_FILE:-$PILATES_DIR/hpc/requirements-hpc.txt}"
FALLBACK_REQUIREMENTS_FILE="$PILATES_DIR/requirements.txt"

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

install_python_deps() {
    local req_file="$1"
    local marker="$VENV_PATH/.last_requirements_hash"
    local current_hash
    current_hash="$(sha256sum "$req_file" | awk '{print $1}')"

    if [ ! -f "$marker" ] || [ "$current_hash" != "$(cat "$marker")" ]; then
        echo "Installing/updating Python dependencies from $req_file ..."
        python3 -m pip install --upgrade pip setuptools wheel
        python3 -m pip install -r "$req_file"
        printf "%s\n" "$current_hash" > "$marker"
    else
        echo "Python dependencies are up to date; skipping pip install."
    fi
}


normalize_path() {
    local path_arg="$1"
    if [ -z "$path_arg" ]; then
        echo "$path_arg"
        return
    fi
    if [ "${path_arg#/}" != "$path_arg" ]; then
        echo "$path_arg"
    else
        echo "$PILATES_DIR/$path_arg"
    fi
}

if [ "${1:-}" = "" ]; then
    echo "Usage: $0 <settings_file> [stage_file]"
    exit 2
fi

CONFIG_FILE="$(normalize_path "$1")"
STAGE_FILE="$(normalize_path "${2:-}")"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Setting up HPC runtime environment..."
set +u
module load gcc/11.4.0
module load proj/9.2.1
module load python/3.11.6
set -u

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

echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"
echo "Config: $CONFIG_FILE"
if [ -n "$STAGE_FILE" ]; then
    echo "Stage: $STAGE_FILE"
else
    echo "Stage: <fresh run>"
fi

show_system_info

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

# ActivitySim Zarr write debugging. Defaults are enabled here so queued HPC runs
# probe the skim cache write path and log the last successfully written variable
# before any native abort. The probe directory must be visible inside the
# ActivitySim container, so keep it under the /tmp bind mount.
ASIM_DEBUG_PROBE_BASE_DEFAULT="/tmp/asim_zarr_probe"
export ASIM_DEBUG_ZARR_WRITE="${ASIM_DEBUG_ZARR_WRITE:-1}"
export ASIM_DEBUG_ZARR_PROBE="${ASIM_DEBUG_ZARR_PROBE:-1}"
export ASIM_DEBUG_ZARR_PROBE_ONLY="${ASIM_DEBUG_ZARR_PROBE_ONLY:-1}"
export ASIM_DEBUG_ZARR_PROBE_DIR="${ASIM_DEBUG_ZARR_PROBE_DIR:-$ASIM_DEBUG_PROBE_BASE_DEFAULT}"
export ASIM_DEBUG_ZARR_PROBE_LIMIT="${ASIM_DEBUG_ZARR_PROBE_LIMIT:-20}"
mkdir -p "$ASIM_DEBUG_ZARR_PROBE_DIR"
echo "ActivitySim Zarr debug: WRITE=$ASIM_DEBUG_ZARR_WRITE PROBE=$ASIM_DEBUG_ZARR_PROBE PROBE_ONLY=$ASIM_DEBUG_ZARR_PROBE_ONLY LIMIT=$ASIM_DEBUG_ZARR_PROBE_LIMIT DIR=$ASIM_DEBUG_ZARR_PROBE_DIR"

is_new_format() {
    grep -q "^run:" "$1" && grep -q "^shared:" "$1" && grep -q "^infrastructure:" "$1"
}

if ! is_new_format "$CONFIG_FILE"; then
    echo "Detected legacy settings format; migrating..."
    MIGRATED_CONFIG="${CONFIG_FILE%.yaml}_migrated.yaml"
    if python3 scripts/migrate_config.py "$CONFIG_FILE" "$MIGRATED_CONFIG" --no-validate; then
        CONFIG_FILE="$MIGRATED_CONFIG"
        echo "Using migrated config: $CONFIG_FILE"
    else
        echo "WARNING: migration failed, continuing with original config"
    fi
fi

if [ -n "$STAGE_FILE" ]; then
    python3 run.py -c "$CONFIG_FILE" -S "$STAGE_FILE"
else
    python3 run.py -c "$CONFIG_FILE"
fi
