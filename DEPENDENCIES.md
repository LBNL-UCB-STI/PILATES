# PILATES Dependency Management

## Overview

PILATES uses a **manual requirements.txt** approach for dependency management. This ensures compatibility with the HPC environment and provides clear, transparent dependency tracking.

## File Structure

- **`requirements.txt`** - All dependencies with pinned versions (manually maintained)
- **`requirements.in`** - High-level dependencies for documentation (optional reference)

## Quick Reference

### Add a new dependency

```bash
# 1. Add to requirements.txt with version
echo "new-package==1.2.3" >> requirements.txt

# 2. Test on HPC
ssh hpc
cd /global/scratch/users/hmlaarabi/sources/PILATES
pip install --user -r requirements.txt

# 3. If it works, commit
git add requirements.txt
git commit -m "Add new-package"
```

### Update a dependency

```bash
# 1. Edit requirements.txt
vim requirements.txt
# Change: old-package==1.0.0
# To:     old-package==2.0.0

# 2. Test on HPC
ssh hpc
cd /global/scratch/users/hmlaarabi/sources/PILATES
rm -rf ~/.local/lib/python3.10/site-packages/*
pip install --user -r requirements.txt

# 3. If it works, commit
git add requirements.txt
git commit -m "Update old-package to 2.0.0"
```

### Capture current working state

```bash
# On HPC, after everything works
ssh hpc
module load python/3.10.12-gcc-11.4.0
pip list --format=freeze > requirements.txt

# Review and clean up
vim requirements.txt  # Remove system packages, add comments
git add requirements.txt
git commit -m "Update requirements from working HPC environment"
```

## Why Manual?

### ✅ Advantages
- **Simple**: Just edit a text file
- **HPC-compatible**: No special tools needed on HPC
- **Transparent**: Easy to see what changed in git diff
- **Cross-platform friendly**: No Mac ARM64 vs HPC x86_64 issues
- **No dependencies**: Standard pip only
- **Fast**: No compilation or resolution step

### ⚠️ Tradeoffs
- Manual dependency resolution
- Must track transitive dependencies yourself
- Requires testing on HPC

## Critical Version Constraints

**DO NOT change these without testing on HPC:**

```
# Core stack (binary compatibility)
numpy==1.23.5          # tables, pyarrow, scipy compiled against this
pandas==1.5.3          # Compatible with numpy 1.23.5
pyarrow==10.0.1        # Compatible with numpy 1.23.5
scipy==1.11.3          # Compatible with numpy 1.23.5
numba==0.57.0          # Compatible with numpy 1.23.5

# GEOS-dependent (compiled against custom GEOS 3.12.0)
shapely==1.8.5         # Must match HPC GEOS version
pygeos==0.14           # Must match HPC GEOS version

# Data formats (numpy compatibility)
xarray==2023.12.0      # Compatible with numpy 1.23.5
zarr==2.16.1           # Compatible with numpy 1.23.5
```

### Why these versions?

1. **numpy==1.23.5**: Binary packages (tables, pyarrow, scipy) are compiled against this specific version. Upgrading numpy requires recompiling all binary dependencies.

2. **GEOS packages**: shapely and pygeos must be compiled against the GEOS library built on HPC (`~/.local/geos`). Version mismatch causes runtime errors.

3. **pandas==1.5.3**: Later versions require numpy>=1.24, which breaks binary compatibility.

## Workflow Best Practices

### Before Making Changes

1. **Test locally** (if possible) on similar Python version
2. **Document why** you're changing versions (in commit message)
3. **Keep comments** in requirements.txt explaining constraints

### Making Changes

```bash
# Good commit message
git commit -m "Update requests from 2.28.0 to 2.31.0

- Security fix for CVE-2023-xxxxx
- Tested on HPC: works with existing dependencies
- No breaking changes"
```

### After Changes

1. **Test on HPC immediately**
2. **Clean install** to catch missing dependencies
3. **Run a test job** before deploying to production

## File Organization

The `requirements.txt` is organized into sections:

```txt
# ============================================================================
# Core numerical/data packages
# ============================================================================
numpy==1.23.5
pandas==1.5.3
...

# ============================================================================
# Geospatial packages
# ============================================================================
shapely==1.8.5
...
```

Keep this structure when adding packages.

## Troubleshooting

### "Package conflict" on HPC

```bash
# Clean install
ssh hpc
module load python/3.10.12-gcc-11.4.0
rm -rf ~/.local/lib/python3.10/site-packages/*
pip install --user -r requirements.txt
```

### "Missing dependency"

```bash
# Find what's missing
pip install --user --dry-run -r requirements.txt

# Add to requirements.txt
echo "missing-package==1.0.0" >> requirements.txt
```

### "Version conflict"

```bash
# Check what requires what
pip show package-name

# Adjust versions in requirements.txt to satisfy all constraints
```

## Alternative Approaches (Not Used)

We considered but **chose not to use**:

### pip-tools
- **Issue**: Cross-platform problems (Mac ARM64 ≠ HPC x86_64)
- **Workaround**: Would need to run on HPC
- **Decision**: Manual is simpler

### Poetry
- **Issue**: Same cross-platform problems + tool dependency
- **Workaround**: Would need Poetry on HPC
- **Decision**: Too complex for HPC deployment

### Conda
- **Issue**: Requires conda on HPC, slower, different ecosystem
- **Workaround**: Export to requirements.txt anyway
- **Decision**: pip is standard on HPC

## When to Reconsider

Consider automation (pip-tools/Poetry) if:
- ✅ You can run it on HPC directly
- ✅ You have >100 dependencies to manage
- ✅ Multiple people are updating dependencies frequently
- ✅ You need to support multiple Python versions

For now, manual works great for PILATES.

## Quick Commands

```bash
# Deploy to HPC
scp requirements.txt hpc:/global/scratch/users/hmlaarabi/sources/PILATES/

# Fresh install on HPC
ssh hpc
module load python/3.10.12-gcc-11.4.0
rm -rf ~/.local/lib/python3.10/site-packages/*
cd /global/scratch/users/hmlaarabi/sources/PILATES
pip install --user -r requirements.txt

# Verify installation
python -c "import numpy, pandas, tables; print('OK')"

# Run test job
sbatch hpc/job.sh scenarios/your-config.yaml your-scenario.yaml
```

## Summary

**Manual requirements.txt** is the right choice for PILATES because:
1. Simple and transparent
2. No cross-platform issues
3. Works perfectly with HPC
4. Easy for collaborators
5. Standard pip workflow

Keep dependencies pinned, test on HPC, and document changes. That's it! 🎯