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

GEOS_VERSION="3.12.0"
GEOS_INSTALL_DIR="$HOME/.local/geos"
GEOS_SRC_DIR="$HOME/.local/src"
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
module load python/3.10.12-gcc-11.4.0
module load proj/9.2.1

# ==============================================================================
# SYSTEM DEPENDENCIES
# ==============================================================================

echo "Checking system dependencies..."

# Install GEOS if not already present
if [ ! -f "$GEOS_INSTALL_DIR/bin/geos-config" ]; then
    echo "GEOS not found, building from source..."

    mkdir -p "$GEOS_SRC_DIR"
    mkdir -p "$GEOS_INSTALL_DIR"

    cd "$GEOS_SRC_DIR" || exit 1

    # Download GEOS if not already downloaded
    if [ ! -f "geos-${GEOS_VERSION}.tar.bz2" ]; then
        wget "https://download.osgeo.org/geos/geos-${GEOS_VERSION}.tar.bz2"
    fi

    tar xjf "geos-${GEOS_VERSION}.tar.bz2"
    cd "geos-${GEOS_VERSION}" || exit 1

    ./configure --prefix="$GEOS_INSTALL_DIR"
    make -j4
    make install

    echo "GEOS installation complete!"
else
    echo "GEOS already installed at $GEOS_INSTALL_DIR, skipping build..."
fi

# Set up environment variables for GEOS
export LD_LIBRARY_PATH="$GEOS_INSTALL_DIR/lib64:$LD_LIBRARY_PATH"
export PATH="$GEOS_INSTALL_DIR/bin:$PATH"
export GEOS_CONFIG="$GEOS_INSTALL_DIR/bin/geos-config"

# ==============================================================================
# PYTHON DEPENDENCIES
# ==============================================================================

echo "Checking Python dependencies..."

# Change to PILATES directory to access requirements.txt
cd "$PILATES_DIR" || exit 1

echo "Installing packages from requirements.txt..."
# Use --constraint to ensure versions but allow pip to resolve dependencies
pip install --user -r requirements.txt


# Set Python path
export PYTHONPATH
PYTHONPATH="$(python -m site --user-site):$PYTHONPATH"

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
