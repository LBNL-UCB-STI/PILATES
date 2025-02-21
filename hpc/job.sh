#!/bin/bash

# Load required modules
module load gcc/11.4.0
module load geos/3.12.0
module load python/3.10.12

# First install numpy with a compatible version
pip install --user --force-reinstall "numpy<2.0.0"

# Then install other packages
pip install --user --force-reinstall shapely==1.8.5
pip install --user --force-reinstall pyproj==3.6.1
pip install --user --force-reinstall pygeos==0.14
pip install --user --force-reinstall geopandas==0.11.1
pip install --user --force-reinstall pandas<2.0.0  # Use older pandas to ensure compatibility

# If you need tables package:
pip install --user --force-reinstall tables

export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH
cd /global/scratch/users/$USER/sources/PILATES
echo "$1"
python run.py -c "$1" -S "$2"

