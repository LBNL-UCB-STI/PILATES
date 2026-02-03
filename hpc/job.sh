#!/bin/bash
#
# PILATES HPC Job Script
# Handles environment setup, dependency installation, and execution
#
# Usage: sbatch job.sh <config_file> <scenario>
#
# Note: Do NOT set PYTHONPATH manually. The environment handles imports
# automatically. Setting PYTHONPATH can leak system packages into your environment.
#

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PILATES_DIR="/global/scratch/users/$USER/sources/PILATES"
VENV_PATH="$PILATES_DIR/PILATES-env"
CONDA_ENV_DIR="$HOME/.conda/envs/pilates"

# Minimum conda version that has fast libmamba solver built-in
MIN_FAST_CONDA_VERSION="23.10.0"

# ==============================================================================
# FUNCTION DEFINITIONS
# ==============================================================================

show_system_info() {
    echo "=== MEMORY INFORMATION ==="
    free -h
    grep MemTotal /proc/meminfo || true
    echo "=========================="

    echo "=== NODE USAGE INFORMATION ==="
    squeue -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R" | grep "$(hostname)" || echo "Node info not found in squeue"
    echo "=========================="
}

# Compare version strings: returns 0 if $1 >= $2
version_gte() {
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

# Setup using venv + pip (fast, always works)
setup_venv() {
    echo "Using venv + pip approach..."

    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating Python virtual environment..."
        python -m venv "$VENV_PATH"
    fi

    echo "Activating virtual environment at $VENV_PATH..."
    source "$VENV_PATH/bin/activate"

    if [ "$VIRTUAL_ENV" != "$VENV_PATH" ]; then
        echo "ERROR: Failed to activate virtual environment"
        exit 1
    fi

    UPDATE_MARKER="$VENV_PATH/.last_update"
    if [ ! -f "$UPDATE_MARKER" ] || [ requirements.txt -nt "$UPDATE_MARKER" ]; then
        echo "Installing/updating dependencies from requirements.txt..."
        pip install --upgrade pip
        if ! pip install -r requirements.txt; then
            echo "ERROR: Failed to install dependencies"
            exit 1
        fi
        touch "$UPDATE_MARKER"
    else
        echo "Dependencies up to date, skipping install"
    fi
}

# Setup using conda with fast libmamba solver
setup_conda_fast() {
    echo "Using conda with libmamba solver (fast)..."

    CONDA_BASE=$(conda info --base 2>/dev/null)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    CONDA_EXE="$CONDA_BASE/bin/conda"

    if [ ! -d "$CONDA_ENV_DIR" ]; then
        echo "Creating conda environment from environment.yml..."
        "$CONDA_EXE" env create -f environment.yml --prefix "$CONDA_ENV_DIR"
    fi

    UPDATE_MARKER="$CONDA_ENV_DIR/.last_update"
    if [ ! -f "$UPDATE_MARKER" ] || [ environment.yml -nt "$UPDATE_MARKER" ]; then
        echo "Updating conda environment..."
        if ! "$CONDA_EXE" env update -f environment.yml --prefix "$CONDA_ENV_DIR" --prune; then
            echo "ERROR: Failed to update conda environment"
            exit 1
        fi
        touch "$UPDATE_MARKER"
    else
        echo "Environment up to date, skipping update"
    fi

    echo "Activating conda environment at $CONDA_ENV_DIR..."
    conda activate "$CONDA_ENV_DIR"

    if [ "$CONDA_PREFIX" != "$CONDA_ENV_DIR" ]; then
        echo "ERROR: Failed to activate conda environment"
        exit 1
    fi
}

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

echo "Setting up environment..."

# Load required modules
module load python/3.11.6
module load gcc/11.4.0
module load proj/9.2.1

# Try to load conda module (may or may not be available)
module load miniconda3/22.11.1-gcc-11.4.0 2>/dev/null || true

# Configure system libraries
export LD_LIBRARY_PATH=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/gcc-11.4.0-nfcdl6bpyabpnhhasfzu6y4ge4kfskvl/lib64:$LD_LIBRARY_PATH
echo "Using LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

cd "$PILATES_DIR" || exit 1

# ==============================================================================
# CHOOSE ENVIRONMENT STRATEGY
# ==============================================================================

USE_CONDA=false

# Check if conda is available and has fast solver
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version 2>/dev/null | awk '{print $2}')
    echo "Detected conda version: $CONDA_VERSION"

    if version_gte "$CONDA_VERSION" "$MIN_FAST_CONDA_VERSION"; then
        echo "Conda $CONDA_VERSION >= $MIN_FAST_CONDA_VERSION, using fast conda solver"
        USE_CONDA=true
    else
        echo "Conda $CONDA_VERSION < $MIN_FAST_CONDA_VERSION, using venv+pip instead"
    fi
else
    echo "Conda not available, using venv+pip"
fi

# Setup environment based on strategy
if [ "$USE_CONDA" = true ]; then
    setup_conda_fast
else
    setup_venv
fi

echo "Environment setup complete!"

# ==============================================================================
# CONSIST LIBRARY SETUP
# ==============================================================================

CONSIST_DIR="$PILATES_DIR/consist"
if [ ! -d "$CONSIST_DIR" ]; then
    echo "Cloning consist library..."
    git clone https://github.com/LBNL-UCB-STI/consist.git "$CONSIST_DIR"
fi

if ! python -c "import consist" 2>/dev/null; then
    echo "Installing consist library in editable mode..."
    pip install -e "$CONSIST_DIR"
else
    echo "consist library already installed"
fi

# Verify environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# ==============================================================================
# EXECUTION
# ==============================================================================

echo "Starting PILATES execution..."

cd "$PILATES_DIR" || exit 1

echo "Config: $1"
echo "Scenario: $2"
echo ""

show_system_info

echo ""
echo "Running PILATES model..."

python run.py -c "$1" -S "$2"