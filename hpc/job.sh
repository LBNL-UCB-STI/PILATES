#!/bin/bash
#
# PILATES HPC Job Script
# Handles environment setup, dependency installation, and execution
#
# Usage: sbatch job.sh <config_file> <scenario>
#
# Note: Do NOT set PYTHONPATH manually. The conda environment handles imports
# automatically. Setting PYTHONPATH can leak system packages into your environment.
#

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PILATES_DIR="/global/scratch/users/$USER/sources/PILATES"

# Uncomment to force clean singularity cache (not recommended for routine use)
#singularity cache clean --force

# ==============================================================================
# FUNCTION DEFINITIONS
# ==============================================================================

# Display system information
show_system_info() {
    echo "=== MEMORY INFORMATION ==="
    free -h
    grep MemTotal /proc/meminfo || true
    grep -i numa /proc/cpuinfo || true
    echo "=========================="

    echo "=== NODE USAGE INFORMATION ==="
    squeue -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R" | grep "$(hostname)" || echo "Node info not found in squeue"
    echo "=========================="
}

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

echo "Setting up environment..."

# Load required modules
module load python/3.11.6
module load gcc/11.4.0
module load miniconda3/22.11.1-gcc-11.4.0  # Load conda for environment management
module load proj/9.2.1

# Configure system libraries - use GCC 11.4.0 libraries first
export LD_LIBRARY_PATH=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/gcc-11.4.0-nfcdl6bpyabpnhhasfzu6y4ge4kfskvl/lib64:$LD_LIBRARY_PATH
echo "Using LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# ==============================================================================
# CONDA ENVIRONMENT SETUP
# ==============================================================================

echo "Setting up conda environment..."

# Initialize conda for non-interactive shell
# Find and source conda.sh to enable conda activate
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh at $CONDA_BASE/etc/profile.d/conda.sh"
    exit 1
fi

# Change to PILATES directory
cd "$PILATES_DIR" || exit 1

# Define conda environment path
CONDA_ENV_DIR="$HOME/.conda/envs/pilates"

# Create or update conda environment
if [ ! -d "$CONDA_ENV_DIR" ]; then
    echo "Creating new conda environment from environment.yml..."
    conda env create -f environment.yml --prefix "$CONDA_ENV_DIR"
fi

# Activate the environment
echo "Activating conda environment at $CONDA_ENV_DIR..."
conda activate "$CONDA_ENV_DIR"

# Verify activation worked
if [ "$CONDA_PREFIX" != "$CONDA_ENV_DIR" ]; then
    echo "ERROR: Failed to activate conda environment"
    echo "CONDA_PREFIX: $CONDA_PREFIX"
    echo "Expected: $CONDA_ENV_DIR"
    exit 1
fi

# Only update if environment.yml has changed since last update
UPDATE_MARKER="$CONDA_ENV_DIR/.last_update"
if [ ! -f "$UPDATE_MARKER" ] || [ environment.yml -nt "$UPDATE_MARKER" ]; then
    echo "Updating environment from environment.yml..."
    conda env update -f environment.yml --prefix "$CONDA_ENV_DIR" --prune
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to update conda environment"
        exit 1
    fi
    touch "$UPDATE_MARKER"
else
    echo "Environment is up to date, skipping update (environment.yml unchanged)"
fi

echo "Environment setup complete!"

# Install consist library in editable mode if not already installed
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
echo "Conda environment: $CONDA_DEFAULT_ENV"

# ==============================================================================
# EXECUTION
# ==============================================================================

echo "Starting PILATES execution..."

# Change to working directory
cd "$PILATES_DIR" || exit 1

# Display job parameters
echo "Config: $1"
echo "Scenario: $2"
echo ""

# Show system information
show_system_info

echo ""
echo "Running PILATES model..."

# Execute the model
python run.py -c "$1" -S "$2"