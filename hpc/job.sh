#!/bin/bash

#module load anaconda3
#conda init
#conda activate PILATES
#conda install -c conda-forge geopandas h5py openmatrix pytables numpy docker-py tqdm pyyaml psutil cloudpickle
module load python/3.10.12
export PYTHONPATH=/global/home/groups-sw/pc_beamcore/python_packages:$PYTHONPATH
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

