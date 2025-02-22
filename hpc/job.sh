#!/bin/bash

# Load required modules
module load gcc/11.4.0
module load python/3.10.12-gcc-11.4.0

# Set up environment variables for GEOS
export LD_LIBRARY_PATH=$HOME/.local/geos/lib64:$LD_LIBRARY_PATH
export PATH=$HOME/.local/geos/bin:$PATH
export GEOS_CONFIG=$HOME/.local/geos/bin/geos-config

# Clean previous installations
rm -rf ~/.local/lib/python3.10/site-packages/numpy*
rm -rf ~/.local/lib/python3.10/site-packages/pandas*
rm -rf ~/.local/lib/python3.10/site-packages/geopandas*
rm -rf ~/.local/lib/python3.10/site-packages/shapely*
rm -rf ~/.local/lib/python3.10/site-packages/pyproj*
rm -rf ~/.local/lib/python3.10/site-packages/pygeos*
rm -rf ~/.local/lib/python3.10/site-packages/tables*

# Force reinstall all packages
pip install --user --force-reinstall --no-deps numpy==1.23.5
pip install --user --force-reinstall --no-deps pandas==1.5.3
pip install --user --force-reinstall --no-binary shapely shapely==1.8.5
pip install --user --force-reinstall --no-deps pyproj==3.6.1
pip install --user --force-reinstall --no-deps pygeos==0.14
pip install --user --force-reinstall --no-deps geopandas==0.11.1
pip install --user --force-reinstall --no-deps tables==3.8.0
pip install --user --force-reinstall --no-deps matplotlib==3.7.1

# Install dependencies
pip install --user --force-reinstall python-dateutil pytz numexpr bottleneck

# Verify GEOS installation
echo "Checking GEOS installation:"
ls -l $HOME/.local/geos/lib64/libgeos*
echo "GEOS_CONFIG location: $GEOS_CONFIG"
$GEOS_CONFIG --version

# Set PYTHONPATH
export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH

# Change to your working directory and run
cd /global/scratch/users/$USER/sources/PILATES
echo "$1"
python run.py -c "$1" -S "$2"

