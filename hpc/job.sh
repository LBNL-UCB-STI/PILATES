#!/bin/bash
#
# PILATES HPC Job Script
# Handles environment setup, dependency installation, and execution
#
# Usage: sbatch job.sh <config_file> <scenario>
#
# Note: Do NOT set PYTHONPATH manually. The venv handles imports
# automatically. Setting PYTHONPATH can leak system packages into your environment.
#

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PILATES_DIR="/global/scratch/users/$USER/sources/PILATES"
ENV_PATH="$PILATES_DIR/PILATES-env"

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

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

echo "Setting up environment..."

# Load required modules
module load python/3.11.6
module load gcc/11.4.0
module load proj/9.2.1

# Configure system libraries
export LD_LIBRARY_PATH=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/gcc-11.4.0-nfcdl6bpyabpnhhasfzu6y4ge4kfskvl/lib64:$LD_LIBRARY_PATH
echo "Using LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Change to PILATES directory
cd "$PILATES_DIR" || exit 1

# ==============================================================================
# PYTHON VENV SETUP (faster than conda)
# ==============================================================================

# Create venv if it doesn't exist
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating Python virtual environment..."
    python -m venv "$ENV_PATH"
fi

# Activate the environment
echo "Activating virtual environment at $ENV_PATH..."
source "$ENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "$ENV_PATH" ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Install/update dependencies if requirements changed
UPDATE_MARKER="$ENV_PATH/.last_update"
if [ ! -f "$UPDATE_MARKER" ] || [ requirements.txt -nt "$UPDATE_MARKER" ]; then
    echo "Installing/updating dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
    touch "$UPDATE_MARKER"
else
    echo "Dependencies up to date, skipping install"
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
echo "Virtual env: $VIRTUAL_ENV"

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