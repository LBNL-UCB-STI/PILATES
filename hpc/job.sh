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

echo "=== MEMORY INFORMATION ==="
free -h
grep MemTotal /proc/meminfo
grep -i numa /proc/cpuinfo
echo "=========================="

echo "=== NODE USAGE INFORMATION ==="
squeue -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R" | grep $(hostname)
echo "=========================="


python run.py -c "$1" -S "$2"

