#!/bin/bash
#
# PILATES HPC Job Script
# Handles environment setup, dependency installation, and execution
#
# Usage: sbatch job.sh <config_file> <scenario>
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
    squeue -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R" | grep "$(hostname)" || true
    echo "=========================="
}

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

echo "Setting up environment..."

# Load required modules
module load gcc/11.4.0
module load miniconda3/22.11.1-gcc-11.4.0  # Load conda for environment management
module load proj/9.2.1

# ==============================================================================
# CONDA ENVIRONMENT SETUP
# ==============================================================================

echo "Setting up conda environment..."

# Change to PILATES directory
cd "$PILATES_DIR" || exit 1

# Define conda environment path
CONDA_ENV_DIR="$HOME/.conda/envs/pilates"

# Create or update conda environment
if [ ! -d "$CONDA_ENV_DIR" ]; then
    echo "Creating new conda environment from environment.yml..."
    conda env create -f environment.yml --prefix "$CONDA_ENV_DIR"

    # Activate environment and install mamba for faster future updates
    conda activate "$CONDA_ENV_DIR"
    echo "Installing mamba in user environment for faster future updates..."
    conda install mamba -c conda-forge -y
else
    # Activate environment first
    conda activate "$CONDA_ENV_DIR"

    # Install mamba if not in this environment
    if ! command -v mamba &> /dev/null; then
        echo "Installing mamba for faster dependency resolution..."
        conda install mamba -c conda-forge -y
    fi

    # Only update if environment.yml has changed since last update
    UPDATE_MARKER="$CONDA_ENV_DIR/.last_update"
    if [ ! -f "$UPDATE_MARKER" ] || [ environment.yml -nt "$UPDATE_MARKER" ]; then
        echo "Updating environment from environment.yml (using mamba)..."
        mamba env update -f environment.yml --prefix "$CONDA_ENV_DIR" --prune
        touch "$UPDATE_MARKER"
    else
        echo "Environment is up to date, skipping update (environment.yml unchanged)"
    fi
fi

echo "Environment setup complete!"

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
