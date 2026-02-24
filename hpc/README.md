# HPC Job Scripts

This directory contains the Slurm submission scripts used to run PILATES on Lawrencium-style HPC environments.

## What This Setup Does

The current HPC workflow is based on `venv + pip` rather than conda solving at job runtime.

Why:

- Faster startup in queued jobs.
- More predictable installs for shared clusters.
- Easy support for local editable `consist` today, and PyPI install later.
- Explicit version gating in one place (`hpc/requirements-hpc.txt`), including `zarr==3.0.6`.

## Files

- `job_runner.sh`: submit wrapper that chooses partition resources and submits `job.sh`.
- `job.sh`: runs inside the allocated node; bootstraps Python env and executes `run.py`.
- `requirements-hpc.txt`: HPC-focused pinned/guarded dependency list.

## Quick Start

1. Clone PILATES into your scratch/source location (default expected path is `/global/scratch/users/$USER/sources/PILATES`).
2. Ensure modules are available (at least Python/GCC/PROJ as used in `job.sh`).
3. Submit from repo root:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml
```

The command prints the Slurm log path.

## Common Submit Patterns

Default `lr7`:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml
```

Use `lr8`:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -p lr8
```

Restart from an existing stage file:

```bash
./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -s current_stage_restart.yaml
```

## BEAM Memory Templating

`job_runner.sh` supports settings files that contain `${BEAM_MEMORY}`.

- For `lr7`, default is `400g`.
- For `lr8`, default is `600g`.

Override explicitly:

```bash
BEAM_MEMORY=450g ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -p lr7
```

`job_runner.sh` writes a per-job generated settings file (`settings_<jobid>.yaml`) and submits that file to `job.sh`.

## Python Environment Behavior

`job.sh` uses:

- `PILATES_DIR` (default `/global/scratch/users/$USER/sources/PILATES`)
- `PILATES_VENV_PATH` (default `$PILATES_DIR/PILATES-env`)
- `PILATES_REQUIREMENTS_FILE` (default `$PILATES_DIR/hpc/requirements-hpc.txt`, fallback to `$PILATES_DIR/requirements.txt`)

Bootstrapping logic:

1. Create venv if missing.
2. Install dependencies from requirements file when needed.
3. Cache requirements hash in `$VENV_PATH/.last_requirements_hash` to skip unnecessary reinstall.

To force dependency reinstall:

```bash
rm -f /global/scratch/users/$USER/sources/PILATES/PILATES-env/.last_requirements_hash
```

## `consist` Install Behavior

`job.sh` installs `consist` as follows:

1. If local source exists at `CONSIST_SRC_DIR` (default `$PILATES_DIR/consist`), install editable (`pip install -e`).
2. Otherwise, install from PyPI package name in `CONSIST_PYPI_PACKAGE` (default `consist`) if not already importable.
3. Validate with `from consist import create_tracker`.

Override examples:

```bash
CONSIST_SRC_DIR=/global/scratch/users/$USER/sources/consist ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml
```

```bash
CONSIST_PYPI_PACKAGE=consist==0.5.2 ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml
```

## Dependency Gates

Version-sensitive dependencies should be managed in `hpc/requirements-hpc.txt`.

Current important gates include:

- `numpy<2.0`
- `zarr==3.0.6`
- `tables>=3.9.0,<4.0`

When changing these, keep compatibility with the active Python module version and rerun a small smoke job before broad rollout.

## Useful Runtime Overrides

Examples:

```bash
ACCOUNT=pc_beamcore EXPECTED_EXECUTION_DURATION=2-00:00:00 ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml
```

```bash
MEMORY_LIMIT_GB=550 BEAM_MEMORY=420g ./hpc/job_runner.sh -c settings-seattle-newconfig-hpc.yaml -p lr7
```

`job.sh` also sets thread caps (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.) from `PILATES_THREADS` (default `8`).

## Legacy Config Migration

If the passed settings file is in legacy format (does not contain `run:`, `shared:`, and `infrastructure:`), `job.sh` runs:

```bash
python3 scripts/migrate_config.py <old> <old>_migrated.yaml --no-validate
```

and uses the migrated file if migration succeeds.

