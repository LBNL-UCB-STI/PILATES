#!/bin/bash
#
# PILATES HPC Job Script
# Handles environment setup, dependency installation, and execution
#
# Usage: sbatch job.sh <config_file> <scenario>
#

set -e  # Exit on error

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

# Check and install Python package if missing
install_if_missing() {
    local package=$1
    local import_name=$2
    local pip_flags=$3

    # Extract import name from package if not provided (remove version specifiers)
    if [ -z "$import_name" ]; then
        import_name=$(echo "$package" | sed 's/[=<>].*//' | sed 's/-/_/g')
    fi

    if ! python -c "import $import_name" 2>/dev/null; then
        echo "Installing $package..."
        pip install --user $pip_flags "$package"
    else
        echo "$package already installed, skipping..."
    fi
}

# Install GEOS-dependent package with GEOS_CONFIG
install_geos_package() {
    local package=$1
    local import_name=$2
    local version=$3

    if ! python -c "import $import_name" 2>/dev/null; then
        echo "Installing $package with GEOS support..."
        GEOS_CONFIG="$GEOS_INSTALL_DIR/bin/geos-config" \
            pip install --user --no-binary "$package" "$package==$version"
    else
        echo "$package already installed, skipping..."
    fi
}

# Display system information
show_system_info() {
    echo "=== MEMORY INFORMATION ==="
    free -h
    grep MemTotal /proc/meminfo
    grep -i numa /proc/cpuinfo
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

# Core numerical/data packages (must be installed in order with --no-deps)
install_if_missing "numpy==1.23.5" "numpy" "--no-deps"
install_if_missing "six==1.17.0" "six" "--no-deps"
install_if_missing "python-dateutil==2.9.0.post0" "dateutil" "--no-deps"
install_if_missing "pytz==2025.1" "pytz" "--no-deps"
install_if_missing "pandas==1.5.3" "pandas" "--no-deps"

# Geospatial packages (require GEOS)
install_geos_package "shapely" "shapely" "1.8.5"
install_if_missing "pyproj==3.6.1" "pyproj" "--no-deps"
install_geos_package "pygeos" "pygeos" "0.14"
install_if_missing "geopandas==0.11.1" "geopandas" "--no-deps"

# Scientific/data packages
install_if_missing "tables==3.8.0" "tables" "--no-deps"
install_if_missing "matplotlib==3.7.1" "matplotlib" "--no-deps"
install_if_missing "numexpr==2.10.2" "numexpr" "--no-deps"
install_if_missing "bottleneck==1.4.2" "bottleneck" "--no-deps"

# Utility packages
install_if_missing "requests"
install_if_missing "urllib3<2.0.0" "urllib3"
install_if_missing "certifi"
install_if_missing "charset-normalizer" "charset_normalizer"
install_if_missing "idna"
install_if_missing "tqdm"
install_if_missing "pyyaml" "yaml"
install_if_missing "h5py"
install_if_missing "psutil"
install_if_missing "joblib"
install_if_missing "docker"
install_if_missing "sortedcontainers"

# Data format packages
install_if_missing "pyarrow==10.0.1" "pyarrow"
install_if_missing "fastparquet==0.8.3" "fastparquet"

# Provenance tracking
install_if_missing "openlineage-python" "openlineage"

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
