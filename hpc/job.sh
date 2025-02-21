#!/bin/bash

#module load anaconda3
#conda init
#conda activate PILATES
#conda install -c conda-forge geopandas h5py openmatrix pytables numpy docker-py tqdm pyyaml psutil cloudpickle
module load python/3.10.12
pip uninstall -y shapely geopandas pygeos pyproj
pip install --user shapely==1.8.5  # Match the version that was previously installed
pip install --user pyproj==3.6.1   # Match the version from the log
pip install --user pygeos==0.14    # Match the version from the log
pip install --user geopandas==0.11.1  # Match the version from the log
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

