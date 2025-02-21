#!/bin/bash

# Load required modules
module load gcc/11.4.0
module load geos/3.12.0
module load python/3.10.12

# Find and export GEOS library path
GEOS_PREFIX=$(dirname $(dirname $(which geos-config)))
export LD_LIBRARY_PATH=$GEOS_PREFIX/lib:$LD_LIBRARY_PATH

# Let's also try to find the PROJ data directory
export PROJ_LIB=$GEOS_PREFIX/share/proj

# First, remove all relevant packages from user space
rm -rf ~/.local/lib/python3.10/site-packages/numpy*
rm -rf ~/.local/lib/python3.10/site-packages/pandas*
rm -rf ~/.local/lib/python3.10/site-packages/geopandas*
rm -rf ~/.local/lib/python3.10/site-packages/shapely*
rm -rf ~/.local/lib/python3.10/site-packages/pyproj*
rm -rf ~/.local/lib/python3.10/site-packages/pygeos*
rm -rf ~/.local/lib/python3.10/site-packages/tables*

# Install exact versions that work together
pip install --user --no-deps numpy==1.23.5
pip install --user --no-deps pandas==1.5.3
pip install --user --no-deps shapely==1.8.5
pip install --user --no-deps pyproj==3.6.1
pip install --user --no-deps pygeos==0.14
pip install --user --no-deps geopandas==0.11.1
pip install --user --no-deps tables==3.8.0
pip install --user --no-deps matplotlib==3.7.1

# Install dependencies
pip install --user python-dateutil pytz numexpr bottleneck

# Print the library paths to verify
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PROJ_LIB: $PROJ_LIB"
ls -l $GEOS_PREFIX/lib/libgeos*

export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH
cd /global/scratch/users/$USER/sources/PILATES
echo "$1"
python run.py -c "$1" -S "$2"

