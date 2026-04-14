---
title: HPC Overview
summary: Generic HPC execution posture for PILATES and how it differs from local runs.
---

# HPC Overview

## What Stays The Same

- PILATES still loads the same YAML settings file.
- The launcher still resolves the same run roots, workflow state, and restart behavior.
- The workflow still runs through the same scenario lifecycle after the environment is ready.

## What Changes On HPC

- The run is submitted through `hpc/job_runner.sh` instead of starting directly in your shell.
- `job_runner.sh` creates a per-job settings file when `${BEAM_MEMORY}` templating is present.
- `job.sh` bootstraps a Python virtual environment inside the job.
- `job.sh` installs PILATES dependencies from `hpc/requirements-hpc.txt` when that file exists, or from `requirements.txt` otherwise.
- `job.sh` installs `consist` from local source when it can, and otherwise falls back to the configured PyPI package or the default `consist==0.1.1`.

## Most Important Difference From Local Runs

Local and HPC runs go through the same launcher and workflow logic. The main difference is operational:

- local runs start directly from your shell
- HPC runs start through the Slurm wrapper, generated settings file, and job-side environment bootstrap

## Storage Posture

- `run.output_directory` remains the archive run root.
- `run.local_workspace_root` can point at a node-local mutable workspace.
- The launcher exports `PILATES_LOCAL_RUN_DIR`, `PILATES_ARCHIVE_RUN_DIR`, and `PILATES_ENABLE_ARCHIVE_COPY` so archive-copy helpers know where to write.

## Practical Environment Variables

The current wrapper scripts recognize:

- `PILATES_DIR`
- `PILATES_VENV_PATH`
- `PILATES_REQUIREMENTS_FILE`
- `CONSIST_SRC_DIR`
- `CONSIST_PYPI_PACKAGE`
- `EXPECTED_EXECUTION_DURATION`
- `MEMORY_LIMIT_GB`
- `BEAM_MEMORY`
- `PILATES_THREADS`

## Adjacent Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) first.
- Use [Lawrencium](lawrencium.md) for the concrete LBNL path.
- Use [Restart and Resume](restart_and_resume.md) for replay-first recovery behavior.
