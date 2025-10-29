#!/bin/bash
module load python/3.10.12
#python -m pip uninstall --user shapely
module load gcc/11.4.0

# Set LD_LIBRARY_PATH to use the GCC 11.4.0 libraries first
export LD_LIBRARY_PATH=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/gcc-11.4.0-nfcdl6bpyabpnhhasfzu6y4ge4kfskvl/lib64:$LD_LIBRARY_PATH

# Print for debugging
echo "Using LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

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

