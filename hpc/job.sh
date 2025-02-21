#!/bin/bash

# Load required modules
module load gcc/11.4.0
module load geos/3.12.0
module load python/3.10.12

# First uninstall everything
pip uninstall -y numpy pandas geopandas shapely pyproj pygeos tables numexpr bottleneck matplotlib

# Install specific version of numpy first
pip install --user numpy==1.24.4

# Then install other packages specifying versions that work with numpy 1.24
pip install --user pandas==1.5.3
pip install --user shapely==1.8.5
pip install --user pyproj==3.6.1
pip install --user pygeos==0.14
pip install --user geopandas==0.11.1
pip install --user tables==3.8.0
pip install --user matplotlib==3.7.1

# If you need tables package:
pip install --user --force-reinstall tables

export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH
cd /global/scratch/users/$USER/sources/PILATES
echo "$1"
python run.py -c "$1" -S "$2"

