#!/bin/bash

# ==============================================================================
# SETUP INSTRUCTIONS FOR NEW USERS
# ==============================================================================
# Before running this script for the first time, you must set up the environment
# on a LOGIN NODE (do not run these setup steps inside a job).
#
# 1. Load Modules:
#    module load python/3.11.6 gcc/11.4.0
#
# 2. Navigate to your source directory:
#    cd /global/scratch/users/$USER/sources/PILATES
#
# 3. Create the Virtual Environment (named PILATES-env):
#    python -m venv PILATES-env
#
# 4. Activate and Upgrade Pip:
#    source PILATES-env/bin/activate
#    python -m pip install --upgrade pip
#
# 5. Install Dependencies (Crucial: Pin Numpy < 2.0 for compatibility):
#    python -m pip install "numpy<2.0"
#    python -m pip install shapely openmatrix pygeos geopandas PyYAML dlt duckdb pyarrow pydantic rich sqlalchemy sqlmodel tables xarray zarr typer tqdm scipy h5py blosc matplotlib openlineage-python joblib
#
# 6. Install Custom 'consist' Library (Editable mode):
#    # Assumes 'consist' folder is next to PILATES-env
#    python -m pip install -e ./consist "numpy<2.0"
#
# ==============================================================================

# --- 1. Load System Modules ---
module load python/3.11.6
module load gcc/11.4.0

# --- 2. Configure System Libraries ---
# Set LD_LIBRARY_PATH to use the GCC 11.4.0 libraries first
export LD_LIBRARY_PATH=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/gcc-11.4.0-nfcdl6bpyabpnhhasfzu6y4ge4kfskvl/lib64:$LD_LIBRARY_PATH
echo "Using LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# --- 3. Activate Virtual Environment ---
# Path to the virtual environment created in the setup steps
ENV_PATH="/global/scratch/users/$USER/sources/PILATES/PILATES-env"

if [ -f "$ENV_PATH/bin/activate" ]; then
    source "$ENV_PATH/bin/activate"
    echo "Activated virtual environment at: $ENV_PATH"
else
    echo "ERROR: Virtual environment not found at $ENV_PATH"
    echo "Please follow the setup instructions at the top of this script."
    exit 1
fi

# Note: We do NOT set PYTHONPATH here.
# The venv handles imports automatically. Setting PYTHONPATH can
# accidentally leak system packages into your environment.

# --- 4. Move to Project Directory ---
PROJECT_DIR="/global/scratch/users/$USER/sources/PILATES"
cd "$PROJECT_DIR" || { echo "Directory not found: $PROJECT_DIR"; exit 1; }

# --- 5. Debugging / Node Info ---
echo "Script Arguments: $1 ${2:-<none>}"

echo "=== MEMORY INFORMATION ==="
free -h
grep MemTotal /proc/meminfo
grep -i numa /proc/cpuinfo
echo "=========================="

echo "=== NODE USAGE INFORMATION ==="
# Check if hostname exists in queue (prevents grep error if empty)
squeue -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R" | grep "$(hostname)" || echo "Node info not found in squeue"
echo "=========================="

# --- 6. Run Python Script ---
# 'python' now refers to the binary inside PILATES-env
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

if [ -n "${2:-}" ]; then
    python run.py -c "$1" -S "$2"
else
    python run.py -c "$1"
fi
