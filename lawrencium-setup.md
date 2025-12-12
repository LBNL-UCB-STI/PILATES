## Logging into Lawrencium

Make sure you have an account set up at MyLRC: https://mylrc.lbl.gov

Either ssh into the login node:

```commandline
ssh [your_username]@lrc-login.lbl.gov
```

Or start up an interactive Jupyter session on a [small] node via LRC-OnDemand: https://lrc-ondemand.lbl.gov/

> [!TIP]
> It's useful to set up a symlink from the home directory (where you're logged in) to the scratch directory (where we want to store big files like the PILATES data)
> ```commandline
> ln -s /global/scratch/users/[username] ./scratch
> ```

## Installing PILATES

```commandline
cd /global/scratch/users/$USER
mkdir sources
cd sources
git clone --branch copy-working-directory https://github.com/LBNL-UCB-STI/PILATES.git
cd PILATES
```

## Setup Python

```commandline
module load python/3.10.12
python -m pip uninstall --user shapely
python -m pip install --user shapely
python -m pip install --user openmatrix
python -m pip install --user pygeos
python -m pip install --user geopandas
python -m pip install --user table
python -m pip install --user PyYAML
export PYTHONPATH=`python -m site --user-site`:$PYTHONPATH
```

## Setup Git-lfs

```commandline
wget https://github.com/git-lfs/git-lfs/releases/download/v2.3.4/git-lfs-linux-amd64-2.3.4.tar.gz
gunzip git-lfs-linux-amd64-2.3.4.tar.gz 
tar -xf git-lfs-linux-amd64-2.3.4.tar 
cd git-lfs-2.3.4/
```

modify the relevant line in `install.sh` (e.g. with vim) to say

```text
prefix="/global/home/users/your_user_name/.local"
```

then run

```commandline
./install.sh
```

## Download Data
### Download UrbanSim data

```commandline
cd /global/scratch/users/$USER/sources/PILATES/pilates/urbansim/data/

# For SFBay (normal)
wget -O custom_mpo_06197001_model_data.h5 https://www.dropbox.com/scl/fi/l8396ztutpbcoucytywpz/custom_mpo_06197001_model_data.h5?rlkey=xyon6ck73deced7hoqlqtdass&dl=1
# For SFBay (2017 warm start)
wget -O custom_mpo_06197001_model_data.h5 https://demos-data-output.s3.us-east-2.amazonaws.com/simulation_2050/custom_mpo_06197001_model_data_2017.h5
# For Seattle
wget -O custom_mpo_53199100_model_data.h5 https://storage.googleapis.com/beam-core-outputs/urbansim/custom_mpo_53199100_model_data.h5

cd ../../../
```

### Checkout ActivitySim data

```commandline
cd /global/scratch/users/$USER/sources/PILATES/pilates/activitysim/configs/

# For SFBay
git clone --branch demos https://github.com/LBNL-UCB-STI/activitysim-configs-sfbay.git sfbay
# For Seattle
git clone --branch configs-first-calibration https://github.com/LBNL-UCB-STI/activitysim-configs-seattle.git seattle

cd ../../../
```

### Checkout BEAM data

```commandline
cd /global/scratch/users/$USER/sources/PILATES/pilates/beam/production/

# For SFBay
git clone --branch zn/PILATES https://github.com/LBNL-UCB-STI/beam-data-sfbay.git sfbay
# For Seattle
git clone --branch pilates https://github.com/LBNL-UCB-STI/beam-data-seattle.git seattle

cd ../../../
```

## Run PILATES

```commandline
cd hpc
./job_runner.sh
```
#### Optional tags:
`-c`: Define a new settings file, defaults to `settings.yaml`

`-s`: Current stage file name of run to restart in the middle of. If none, starts a new run with a new current stage file. Defaults to none.
