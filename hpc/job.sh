#!/bin/bash

# Load required modules
module load gcc/11.4.0
module load python/3.10.12-gcc-11.4.0

# Install GEOS and its dependencies in your user space
# We'll use pip to install the compiled wheels which include the necessary libraries
pip install --user GEOS

# Install exact versions that work together
pip install --user --no-deps numpy==1.23.5
pip install --user --no-deps pandas==1.5.3
pip install --user --no-deps shapely==1.8.5 --no-binary shapely  # Force compilation against system GEOS
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

