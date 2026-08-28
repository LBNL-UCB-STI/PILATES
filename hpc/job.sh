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
CONSIST_SRC_DIR_SUPPLIED="${CONSIST_SRC_DIR+x}"
CONSIST_SRC_DIR="${CONSIST_SRC_DIR:-$PILATES_DIR/../consist}"
CONSIST_PYPI_PACKAGE="${CONSIST_PYPI_PACKAGE:-}"
DEFAULT_CONSIST_PYPI_PACKAGE="consist==0.2.0"

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
    local filtered_req
    local current_hash
    filtered_req="$(mktemp)"
    grep -Ev '^[[:space:]]*consist([[:space:]]|[<>=!~].*)?$' "$req_file" > "$filtered_req" || true
    current_hash="$(sha256sum "$filtered_req" | awk '{print $1}')"

    if [ ! -f "$marker" ] || [ "$current_hash" != "$(cat "$marker")" ]; then
        echo "Installing/updating Python dependencies from $req_file ..."
        python3 -m pip install --upgrade pip setuptools wheel
        python3 -m pip install -r "$filtered_req"
        printf "%s\n" "$current_hash" > "$marker"
    else
        echo "Python dependencies are up to date; skipping pip install."
    fi

    rm -f "$filtered_req"
}


resolve_consist_package_spec() {
    if [ -n "$CONSIST_PYPI_PACKAGE" ]; then
        echo "$CONSIST_PYPI_PACKAGE"
        return
    fi

    local package_spec
    local req_file
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
        echo "Attempting editable consist install from $CONSIST_SRC_DIR ..."
        if python3 -m pip install -e "$CONSIST_SRC_DIR"; then
            echo "Installed local editable consist from $CONSIST_SRC_DIR"
        else
            echo "WARNING: editable consist install failed; falling back to $package_spec" >&2
            python3 -m pip install "$package_spec"
        fi
    else
        echo "Local consist source not found at $CONSIST_SRC_DIR; installing $package_spec"
        python3 -m pip install "$package_spec"
    fi

    python3 - <<'PY'
import consist
print(f"consist import path: {consist.__file__}")
PY
}


install_hdf5_acceptance_consist() {
    local evidence_root="$1"
    local consist_revision
    local pilates_revision
    local runtime_record="$evidence_root/runtime-environment.json"

    if [ "$CONSIST_SRC_DIR_SUPPLIED" != "x" ] || [ ! -d "$CONSIST_SRC_DIR" ]; then
        echo "ERROR: UrbanSim HDF5 snapshot acceptance requires an existing editable Consist checkout supplied through CONSIST_SRC_DIR." >&2
        return 1
    fi
    if ! consist_revision="$(git -C "$CONSIST_SRC_DIR" rev-parse HEAD)"; then
        echo "ERROR: UrbanSim HDF5 snapshot acceptance requires CONSIST_SRC_DIR to be a Git checkout." >&2
        return 1
    fi
    if ! pilates_revision="$(git -C "$PILATES_DIR" rev-parse HEAD)"; then
        echo "ERROR: UrbanSim HDF5 snapshot acceptance requires PILATES_DIR to be a Git checkout." >&2
        return 1
    fi

    echo "Installing required editable Consist checkout from $CONSIST_SRC_DIR ..."
    if ! python3 -m pip install -e "$CONSIST_SRC_DIR"; then
        echo "ERROR: failed to install required editable Consist checkout from $CONSIST_SRC_DIR." >&2
        return 1
    fi

    CONSIST_ACCEPTANCE_RUNTIME_RECORD="$runtime_record" \
    CONSIST_ACCEPTANCE_CONSIST_REVISION="$consist_revision" \
    CONSIST_ACCEPTANCE_PILATES_REVISION="$pilates_revision" \
    python3 - <<'PY'
import json
import os
from pathlib import Path
import platform
import sys

import consist

source = Path(os.environ["CONSIST_SRC_DIR"]).resolve()
module = Path(consist.__file__).resolve()
try:
    module.relative_to(source)
except ValueError as error:
    raise SystemExit(
        "ERROR: Consist import is not provided by the required editable checkout: "
        f"{module} is outside {source}"
    ) from error

record = {
    "consist": {
        "editable_source": str(source),
        "import_path": str(module),
        "revision": os.environ["CONSIST_ACCEPTANCE_CONSIST_REVISION"],
    },
    "pilates": {
        "source_path": os.environ["PILATES_DIR"],
        "revision": os.environ["CONSIST_ACCEPTANCE_PILATES_REVISION"],
    },
    "python": {
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "version": sys.version,
    },
}
destination = Path(os.environ["CONSIST_ACCEPTANCE_RUNTIME_RECORD"])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"consist import path: {module}")
PY
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

acceptance_mode=""
if [ "${1:-}" = "--urbansim-h5-snapshot-acceptance" ]; then
    acceptance_mode="urbansim-h5-snapshot"
    shift
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 --urbansim-h5-snapshot-acceptance <settings_file> <input_manifest> <evidence_root>" >&2
        exit 2
    fi
elif [ "${1:-}" = "--beam-preprocess-acceptance" ]; then
    acceptance_mode="beam-preprocess"
    shift
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 --beam-preprocess-acceptance <settings_file> <input_manifest> <evidence_root>" >&2
        exit 2
    fi
elif [ "$#" -gt 2 ]; then
    echo "Usage: $0 <settings_file> [stage_file]" >&2
    exit 2
fi

if [ "${1:-}" = "" ]; then
    echo "Usage: $0 [--beam-preprocess-acceptance] <settings_file> [stage_file]"
    exit 2
fi

CONFIG_FILE="$(normalize_path "$1")"
STAGE_FILE="$(normalize_path "${2:-}")"
ACCEPTANCE_EVIDENCE_ROOT="$(normalize_path "${3:-}")"
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
if [ "$acceptance_mode" = "urbansim-h5-snapshot" ]; then
    install_hdf5_acceptance_consist "$ACCEPTANCE_EVIDENCE_ROOT"
else
    install_consist "$REQUIREMENTS_FILE"
fi

echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"
echo "Config: $CONFIG_FILE"
if [ "$acceptance_mode" = "urbansim-h5-snapshot" ]; then
    if [ -z "$STAGE_FILE" ] || [ -z "$ACCEPTANCE_EVIDENCE_ROOT" ]; then
        echo "ERROR: UrbanSim HDF5 snapshot acceptance mode requires settings, input manifest, and evidence root." >&2
        exit 2
    fi
    echo "Launching UrbanSim HDF5 snapshot capture driver with unbuffered output..."
    PYTHONUNBUFFERED=1 python3 -u -m pilates.runtime.urbansim_h5_snapshot_acceptance capture \
        --settings "$CONFIG_FILE" \
        --manifest "$STAGE_FILE" \
        --evidence-root "$ACCEPTANCE_EVIDENCE_ROOT"
    echo "Launching UrbanSim HDF5 snapshot reconciliation driver with unbuffered output..."
    PYTHONUNBUFFERED=1 python3 -u -m pilates.runtime.urbansim_h5_snapshot_acceptance reconcile \
        --evidence-root "$ACCEPTANCE_EVIDENCE_ROOT"
    exit 0
elif [ "$acceptance_mode" = "beam-preprocess" ]; then
    if [ -z "$STAGE_FILE" ] || [ -z "$ACCEPTANCE_EVIDENCE_ROOT" ]; then
        echo "ERROR: acceptance mode requires settings, input manifest, and evidence root." >&2
        exit 2
    fi
    echo "Launching BEAM-preprocess acceptance driver with unbuffered output..."
    PYTHONUNBUFFERED=1 python3 -u -m pilates.runtime.beam_preprocess_acceptance \
        --settings "$CONFIG_FILE" \
        --manifest "$STAGE_FILE" \
        --evidence-root "$ACCEPTANCE_EVIDENCE_ROOT"
    exit 0
elif [ -n "$STAGE_FILE" ]; then
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

# ActivitySim Zarr write debugging. Keep the dataset summary logging enabled by
# default, but leave the probe writes opt-in so the real compile cache write
# path is exercised.
export ASIM_DEBUG_ZARR_WRITE="${ASIM_DEBUG_ZARR_WRITE:-1}"
unset ASIM_DEBUG_ZARR_PROBE
unset ASIM_DEBUG_ZARR_PROBE_ONLY
unset ASIM_DEBUG_ZARR_PROBE_DIR
unset ASIM_DEBUG_ZARR_PROBE_LIMIT
echo "ActivitySim Zarr debug: WRITE=$ASIM_DEBUG_ZARR_WRITE"

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
