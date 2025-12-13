# PILATES Dependency Management

## Overview

PILATES uses **conda** for dependency management via `environment.yml`. This provides robust dependency resolution, handles complex binary packages (GEOS, GDAL), and works seamlessly with HPC environments.

## File Structure

- **`environment.yml`** - All dependencies with version specifications (canonical source)

## Quick Reference

### Add a new dependency

```bash
# 1. Add to environment.yml under the appropriate section
vim environment.yml

# Add under relevant section, e.g.:
# - pip:
#   - new-package==1.2.3

# 2. Update your local environment
conda env update -f environment.yml --prune

# 3. Test locally
python -c "import new_package; print('OK')"

# 4. Test on HPC
ssh hpc
cd /global/scratch/users/hmlaarabi/sources/PILATES
sbatch hpc/job.sh config.yaml scenario.yaml

# 5. If it works, commit
git add environment.yml
git commit -m "Add new-package dependency"
```

### Update a dependency

```bash
# 1. Edit environment.yml
vim environment.yml
# Change: old-package=1.0.0
# To:     old-package=2.0.0

# 2. Update local environment
conda env update -f environment.yml --prune

# 3. Test locally
python -m pytest tests/

# 4. Test on HPC
ssh hpc
cd /global/scratch/users/hmlaarabi/sources/PILATES
sbatch hpc/job.sh config.yaml scenario.yaml

# 5. If it works, commit
git add environment.yml
git commit -m "Update old-package to 2.0.0"
```

### Recreate environment from scratch

```bash
# Local development
conda env remove -n pilates
conda env create -f environment.yml
conda activate pilates

# HPC (automatic via job.sh)
# The job script automatically creates/updates the environment
sbatch hpc/job.sh config.yaml scenario.yaml
```

## Critical Version Constraints

**DO NOT change these without testing on HPC:**

```yaml
# Core stack (binary compatibility)
numpy=1.23.5          # Binary packages compiled against this
pandas=1.5.3          # Compatible with numpy 1.23.5
pyarrow=10.0.1        # Compatible with numpy 1.23.5
scipy=1.11.3          # Compatible with numpy 1.23.5
numba=0.57.0          # Compatible with numpy 1.23.5

# Geospatial stack
geos>=3.9             # Conda manages GEOS automatically
shapely=1.8.5         # Compatible with GEOS 3.9+
pygeos=0.14           # Compatible with GEOS 3.9+
geopandas=0.11.1      # Compatible with shapely 1.8.5

# Data formats
xarray=2023.12.0      # Compatible with numpy 1.23.5
zarr=2.16.1           # Compatible with numpy 1.23.5
pytables=3.8.0        # Compatible with numpy 1.23.5
```

### Why these versions?

1. **numpy==1.23.5**: Binary packages (tables, pyarrow, scipy) are compiled against this specific version. Upgrading numpy requires ensuring all binary dependencies are compatible.

2. **GEOS packages**: Conda automatically manages GEOS and ensures shapely/pygeos compatibility. No manual GEOS compilation needed.

3. **pandas==1.5.3**: Later versions require numpy>=1.24, which would break binary compatibility with current stack.

## Workflow Best Practices

### Before Making Changes

1. **Test locally** with `conda env update`
2. **Document why** you're changing versions (in commit message)
3. **Keep organized** - add packages to the appropriate section in environment.yml

### Making Changes

```bash
# Good commit message
git commit -m "Update requests from 2.28.0 to 2.31.0

- Security fix for CVE-2023-xxxxx
- Tested locally and on HPC
- No breaking changes"
```

### After Changes

1. **Test locally** - ensure your code still works
2. **Test on HPC** - submit a test job
3. **Document** - update this file if adding new critical constraints

## HPC Deployment

The `hpc/job.sh` script automatically handles conda environment setup:

```bash
# On HPC, job.sh will:
# 1. Load anaconda3 module
# 2. Create conda environment (first run)
# 3. Update environment from environment.yml (subsequent runs)
# 4. Activate environment
# 5. Run your simulation

# Simply submit your job:
sbatch hpc/job.sh scenarios/config.yaml scenario.yaml
```

### First-time HPC setup

```bash
# The first job submission will create the environment
# This takes ~10-15 minutes as conda installs all packages

ssh hpc
cd /global/scratch/users/$USER/sources/PILATES
sbatch hpc/job.sh scenarios/test-config.yaml test-scenario.yaml

# Check job output for environment creation status
tail -f /global/scratch/users/$USER/pilates_logs/log_*.log
```

### Manual environment management on HPC

```bash
# If needed, you can manually manage the environment

ssh hpc
module load anaconda3

# Create environment
conda env create -f environment.yml --prefix $HOME/.conda/envs/pilates

# Update environment
conda env update -f environment.yml --prefix $HOME/.conda/envs/pilates --prune

# Activate environment
source activate $HOME/.conda/envs/pilates

# Verify
python -c "import numpy, pandas, geopandas; print('OK')"
```

## File Organization

The `environment.yml` is organized into sections:

```yaml
dependencies:
  # ============================================================================
  # Core numerical/data packages
  # ============================================================================
  - numpy=1.23.5
  - pandas=1.5.3

  # ============================================================================
  # Geospatial packages
  # ============================================================================
  - geos>=3.9
  - shapely=1.8.5

  # ============================================================================
  # Packages only available via pip
  # ============================================================================
  - pip:
    - h5py
    - openmatrix
```

**Keep this structure when adding packages.** Add packages to the appropriate section, or create a new section if needed.

## Troubleshooting

### "Solving environment" takes forever

```bash
# Conda's dependency solver can be slow. Use mamba as a faster alternative:
conda install mamba -n base -c conda-forge
mamba env update -f environment.yml --prune
```

### Environment broken after update

```bash
# Recreate from scratch
conda env remove -n pilates
conda env create -f environment.yml
conda activate pilates
```

### Package not available in conda

```bash
# Add to pip section in environment.yml
- pip:
  - package-not-in-conda==1.0.0
```

### Version conflict

```bash
# Check what's causing the conflict
conda env create -f environment.yml --dry-run

# Adjust version constraints in environment.yml
# Use >= instead of == for more flexibility where appropriate
```

### HPC job fails with import error

```bash
# Check the job log for details
tail -100 /global/scratch/users/$USER/pilates_logs/log_*.log

# Common issues:
# 1. Environment creation failed - check log for conda errors
# 2. Module not loaded - ensure anaconda3 module is available
# 3. Package missing - add to environment.yml
```

## Advantages of Conda

- **Automatic dependency resolution**: Conda handles complex dependency trees
- **Binary package management**: No compilation needed for GEOS, GDAL, etc.
- **Reproducible environments**: Same versions across local dev and HPC
- **Environment isolation**: Clean separation from system packages
- **Easy updates**: Single command to update all packages

## Quick Commands

```bash
# Local development
conda env create -f environment.yml              # First time
conda activate pilates                           # Activate
conda env update -f environment.yml --prune      # Update
conda deactivate                                 # Deactivate

# HPC deployment
scp environment.yml hpc:/global/scratch/users/$USER/sources/PILATES/
ssh hpc
sbatch hpc/job.sh config.yaml scenario.yaml

# Verify installation (local or HPC)
python -c "import numpy, pandas, geopandas, shapely, xarray; print('All imports OK')"

# Check package versions
conda list | grep numpy
conda list | grep pandas
```

## Summary

**Conda with environment.yml** provides:
1. Robust dependency management
2. Automatic handling of binary packages (GEOS, GDAL)
3. Reproducible environments across platforms
4. Simple workflow for adding/updating dependencies
5. Seamless HPC integration via job.sh

Keep dependencies organized in sections, test on HPC before committing, and document critical version constraints. 🎯