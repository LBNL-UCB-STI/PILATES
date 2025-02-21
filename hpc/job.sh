#!/bin/bash

#module load anaconda3
#conda init
#conda activate PILATES
#conda install -c conda-forge geopandas h5py openmatrix pytables numpy docker-py tqdm pyyaml psutil cloudpickle
module load python/3.10.12
pip uninstall shapely
pip uninstall geopandas
pip install --user shapely
pip install --user openmatrix
pip install --user pygeos
pip install --user geopandas
pip install --user table
pip install --user PyYAML
export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH
cd /global/scratch/users/$USER/sources/PILATES
echo "$1"



python run.py -c "$1" -S "$2"

