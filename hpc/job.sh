#!/bin/bash

#module load anaconda3
#conda init
#conda activate PILATES
#conda install -c conda-forge geopandas h5py openmatrix pytables numpy docker-py tqdm pyyaml psutil cloudpickle
# Load required modules
module load gcc/11.4.0
module load geos/3.12.0
module load python/3.10.12
# Install packages at user level, force reinstall to ensure clean state
pip install --user --force-reinstall shapely==1.8.5
pip install --user --force-reinstall pyproj==3.6.1
pip install --user --force-reinstall pygeos==0.14
pip install --user --force-reinstall geopandas==0.11.1
conda create -n geo_env
conda activate geo_env
conda install -c conda-forge geos
#pip uninstall shapely
#pip uninstall geopandas
#pip install --user shapely
#pip install --user openmatrix
#pip install --user pygeos
#pip install --user geopandas
#pip install --user table
#pip install --user PyYAML
export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH
cd /global/scratch/users/$USER/sources/PILATES
echo "$1"



python run.py -c "$1" -S "$2"

