# Getting Started

This guide is the practical "first run" walkthrough for PILATES.

It is written for a new contributor who has cloned the repo and wants to run
`python run.py` locally using the current CLI and scenario configs that exist
in this repository.

## Goals

- Install the project locally.
- Prepare the minimum required data inputs (ActivitySim configs + BEAM data).
- Run a minimal scenario successfully.
- Know what "success" looks like on disk so you can debug failures quickly.

## Prerequisites

Required:

- `conda` (or `mamba`) to create the Python environment from `environment.yml`.
- Python is pinned via `environment.yml` (currently `python=3.11.6`).
- `consist` must be importable at runtime. PILATES uses Consist for provenance and caching.

Optional (depends on config):

- Docker or Singularity. Only needed when you run real model containers; not strictly
  required for initial "stub" smoke runs if `run.use_stubs: true`.

## Installation

From the repo root:

```bash
conda env create -f environment.yml
conda activate pilates
```

Confirm PILATES imports:

```bash
python -c "import pilates; print('PILATES import ok')"
```

Confirm Consist imports:

```bash
python -c "import consist; print('Consist import ok')"
```

If Consist is missing, install it (either from PyPI or from a local checkout if
you have one). The exact method depends on your environment and workflow.

## Required Inputs (Data Repos And Large Files)

PILATES expects certain region-specific inputs to exist on disk. The scenario
config files under `scenarios/` assume you have:

- ActivitySim configs checked out under `pilates/activitysim/configs/<region>/`
- BEAM "production" inputs checked out under `pilates/beam/production/<region>/`

On some developer machines these are symlinks to external clones. That is fine
as long as the paths resolve correctly.

For a concrete source of truth on where these repos come from, see:

- `lawrencium-setup.md` (includes commands to fetch UrbanSim data, ActivitySim configs, and BEAM data)

### Quick Path Check (Seattle Example)

If you plan to run the Seattle local scenario config:

```bash
python - <<'PY'
from pathlib import Path

paths = [
    "scenarios/seattle/settings-seattle-newconfig-local.yaml",
    "pilates/activitysim/configs/seattle",
    "pilates/beam/production/seattle",
]
missing = [p for p in paths if not Path(p).exists()]
print("missing:", missing)
PY
```

If `pilates/activitysim/configs/seattle` or `pilates/beam/production/seattle` is
missing, you need to clone (or symlink) the appropriate data repos before the
run will work.

Tip: if your BEAM production inputs are checked out as git repos under
`pilates/beam/production/*`, you can update them with `./pull_latest_data.sh`.

## Choose A First Config

The current CLI supports:

- `-c/--config <yaml>` to select a config file
- `-S/--stage <run_state.yaml>` to resume/restart from a prior run state file
- `--allow-rewind-resume` to allow resuming from an earlier year than the archived run state

For a first local run, start from one of the "newconfig local" scenario configs:

- Seattle: `scenarios/seattle/settings-seattle-newconfig-local.yaml`
- SFBay: `scenarios/sfbay/settings-sfbay-newconfig-local.yaml`

These are full nested configs (the newer `run:` / `shared:` / `infrastructure:` layout).

### Create A Local Working Copy

Do not edit the scenario files in place. Copy one to a local config file and
edit the run output location:

```bash
cp scenarios/seattle/settings-seattle-newconfig-local.yaml settings.local.yaml
```

In `settings.local.yaml`, set:

- `run.output_directory`: a directory you can write to
- `run.output_run_name`: a short run label (used as the run folder prefix)

The shipped scenario configs may contain developer-specific absolute paths; you
should treat them as templates.

If your config includes `${BEAM_MEMORY}`, either export it or replace it with a
literal value (it is used for BEAM container configuration in non-stub runs):

```bash
export BEAM_MEMORY=8g
```

## First Run (Local)

From the repo root:

```bash
python run.py -c settings.local.yaml
```

Notes:

- Logging is configured in the runtime launcher and is typically very verbose
  (DEBUG to stdout).
- If `run.use_stubs: true`, PILATES will skip container execution. This is
  useful for smoke tests, but it does not eliminate the need for the input data
  directories described above, because preprocess/postprocess stages still run.

## What Success Looks Like

PILATES creates a run folder under `run.output_directory` with a timestamped
name derived from `run.output_run_name`.

You should see:

- `<output_directory>/<output_run_name>-YYYYMMDD-HHMMSS/`
- `run_state.yaml` inside that run folder (used for restart/resume)

Within the run folder, you should also see the mutable workspace structure
described by your config (common examples include `activitysim/`, `beam/`,
`urbansim/`, and `pilates/postprocessing/`).

If Consist DB logging is enabled (typical default), you may also see a run-local
DuckDB file such as `provenance.duckdb` (the exact filename is controlled by
config).

## Restarting / Resuming A Run

To resume from an existing run directory, pass the archived `run_state.yaml`:

```bash
python run.py -c settings.local.yaml -S /path/to/output/<run_name>/run_state.yaml
```

If you intentionally need to resume from an earlier year than the archived state
file indicates, add:

```bash
python run.py -c settings.local.yaml -S /path/to/output/<run_name>/run_state.yaml --allow-rewind-resume
```

## Common First-Run Problems

### Missing ActivitySim Configs Or BEAM Production Data

Symptom:
- The run fails early with file-not-found errors under `pilates/activitysim/configs/...`
  or `pilates/beam/production/...`.

Likely cause:
- You cloned PILATES but did not fetch the region-specific input repos, or your
  symlinks point to non-existent locations.

How to confirm:
- Check whether the paths referenced by your config exist on disk.

Fix:
- Follow `lawrencium-setup.md` to fetch the missing repos, or create correct
  symlinks to existing local clones.

### Consist Not Installed

Symptom:
- Import errors for `consist`, or runtime failure during tracker initialization.

Likely cause:
- The conda environment was created, but Consist was not installed into it.

How to confirm:
- `python -c "import consist"`

Fix:
- Install Consist into the active environment (from PyPI or editable local source).

### Output Directory Permission Errors

Symptom:
- The run fails when creating the run directory.

Likely cause:
- `run.output_directory` is not writable, or points to a path that does not
  exist and cannot be created.

How to confirm:
- Verify `run.output_directory` in your config and test you can create files there.

Fix:
- Change `run.output_directory` to a writable path (for example `./tmp` or an
  explicit location under your home directory).

## Related Docs

- `docs/architecture.md`
- `docs/workflow_primer.md`
- `docs/configuration_reference.md`
- `docs/troubleshooting.md`
