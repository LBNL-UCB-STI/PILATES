#!/bin/bash

# Load required modules
module load gcc/11.4.0
module load python/3.10.12-gcc-11.4.0

# Create directories for local installation
mkdir -p $HOME/.local/src
mkdir -p $HOME/.local/geos

# Download and build GEOS
cd $HOME/.local/src
wget https://download.osgeo.org/geos/geos-3.12.0.tar.bz2
tar xjf geos-3.12.0.tar.bz2
cd geos-3.12.0
./configure --prefix=$HOME/.local/geos
make -j4
make install

# Set up environment variables
export LD_LIBRARY_PATH=$HOME/.local/geos/lib:$LD_LIBRARY_PATH
export PATH=$HOME/.local/geos/bin:$PATH

# Clean previous installations
rm -rf ~/.local/lib/python3.10/site-packages/numpy*
rm -rf ~/.local/lib/python3.10/site-packages/pandas*
rm -rf ~/.local/lib/python3.10/site-packages/geopandas*
rm -rf ~/.local/lib/python3.10/site-packages/shapely*
rm -rf ~/.local/lib/python3.10/site-packages/pyproj*
rm -rf ~/.local/lib/python3.10/site-packages/pygeos*
rm -rf ~/.local/lib/python3.10/site-packages/tables*

# Install Python packages
pip install --user --no-deps numpy==1.23.5
pip install --user --no-deps pandas==1.5.3
pip install --user --no-binary shapely shapely==1.8.5
pip install --user --no-deps pyproj==3.6.1
pip install --user --no-deps pygeos==0.14
pip install --user --no-deps geopandas==0.11.1
pip install --user --no-deps tables==3.8.0
pip install --user --no-deps matplotlib==3.7.1

# Install dependencies
pip install --user python-dateutil pytz numexpr bottleneck

# Set PYTHONPATH
export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH

# Change to your working directory and run
cd /global/scratch/users/$USER/sources/PILATES
echo "$1"
python run.py -c "$1" -S "$2"

