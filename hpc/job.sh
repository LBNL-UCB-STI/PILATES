#!/bin/bash

module load python/3.8.8
#pip install --user openmatrix
#pip install --user geopandas
export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH
cd /global/scratch/users/$USER/sources/PILATES
python run.py -v -c="$1" -s="$2"

