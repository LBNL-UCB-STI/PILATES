#!/bin/bash

module load python/3.10.12
#python -m pip uninstall --user shapely
#python -m pip install --user shapely
#python -m pip install --user openmatrix
#python -m pip install --user pygeos
#python -m pip install --user geopandas
#python -m pip install --user table
#python -m pip install --user PyYAML
export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH
cd /global/scratch/users/$USER/sources/PILATES
echo "$1"



python run.py -c "$1" -S "$2"

