#!/bin/bash

singularity cache clean --force

# Load required modules
module load gcc/11.4.0
module load python/3.10.12-gcc-11.4.0
module load proj/9.2.1

## Create directories for local installation
#mkdir -p $HOME/.local/src
#mkdir -p $HOME/.local/geos
#
## Download and build GEOS
#cd $HOME/.local/src
#wget https://download.osgeo.org/geos/geos-3.12.0.tar.bz2
#tar xjf geos-3.12.0.tar.bz2
#cd geos-3.12.0
#./configure --prefix=$HOME/.local/geos
#make -j4
#make install


# Set up environment variables for GEOS
export LD_LIBRARY_PATH=$HOME/.local/geos/lib64:$LD_LIBRARY_PATH
export PATH=$HOME/.local/geos/bin:$PATH
export GEOS_CONFIG=$HOME/.local/geos/bin/geos-config

## Clean previous installations more thoroughly
#pip uninstall -y numpy pandas geopandas shapely pyproj pygeos tables matplotlib python-dateutil pytz numexpr bottleneck
#
## Force reinstall core packages in correct order
#pip install --user --no-deps numpy==1.23.5
#pip install --user --no-deps six==1.17.0
#pip install --user --no-deps python-dateutil==2.9.0.post0
#pip install --user --no-deps pytz==2025.1
#pip install --user --no-deps pandas==1.5.3
# GEOS_CONFIG=$HOME/.local/geos/bin/geos-config pip install --user --no-binary shapely shapely==1.8.5
# pip install --user --no-deps pyproj==3.6.1
# GEOS_CONFIG=$HOME/.local/geos/bin/geos-config pip install --user --no-binary pygeos pygeos==0.14
#pip install --user --no-deps geopandas==0.11.1
#pip install --user --no-deps tables==3.8.0
#pip install --user --no-deps matplotlib==3.7.1
#pip install --user --no-deps numexpr==2.10.2
#pip install --user --no-deps bottleneck==1.4.2
#
## Install additional required packages
#pip install --user requests
#pip install --user 'urllib3<2.0.0'
#pip install --user certifi
#pip install --user charset-normalizer
#pip install --user idna
#pip install --user tqdm
#pip install --user pyyaml
#pip install --user h5py
#pip install --user psutil
#pip install --user joblib  # Adding joblib for parallel processing
#pip install --user docker
#pip install --user sortedcontainers
#pip install --user pyarrow==10.0.1
#pip install --user fastparquet==0.8.3

# Set PYTHONPATH
export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH

# Change to your working directory and run
cd /global/scratch/users/$USER/sources/PILATES
echo "$1"

echo "=== MEMORY INFORMATION ==="
free -h
grep MemTotal /proc/meminfo
grep -i numa /proc/cpuinfo
echo "=========================="

echo "=== NODE USAGE INFORMATION ==="
squeue -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R" | grep $(hostname)
echo "=========================="


python run.py -c "$1" -S "$2"

