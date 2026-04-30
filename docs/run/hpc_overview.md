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
- `job.sh` installs `consist` from local source when it can, and otherwise falls back to the configured PyPI package or the default `consist==0.1.3`.

## Most Important Difference From Local Runs

Local and HPC runs go through the same launcher and workflow logic. The main difference is operational:

- local runs start directly from your shell
- HPC runs start through the Slurm wrapper, generated settings file, and job-side environment bootstrap

## Storage Posture

- `run.output_directory` is the active durable archive root for the run. On HPC this should normally live on shared scratch, not on node-local storage.
- `run.local_workspace_root` can point at a node-local mutable workspace for the actual model execution.
- `run.recovery_archive_roots` is an optional list of colder post-run promotion roots, such as project NFS storage.
- The launcher exports `PILATES_LOCAL_RUN_DIR`, `PILATES_ARCHIVE_RUN_DIR`, and `PILATES_ENABLE_ARCHIVE_COPY` so archive-copy helpers know where to write.

## Recommended HPC Storage Split

For the replay-first archive model, treat the three tiers differently:

- shared scratch: active archive root during the run via `run.output_directory`
- node-local scratch: mutable workspace via `run.local_workspace_root`
- colder shared storage: post-run promotion destination via `run.recovery_archive_roots`

This keeps slower archival storage out of the hot path while preserving a complete run archive after promotion.

## Post-Run Promotion

PILATES now supports a manual promotion helper for copying a completed archive run into one or more recovery roots and recording those promoted roots on the run's logged artifacts:

```bash
python -m pilates.runtime.promote_run_archive -c path/to/settings.yaml
```

Use `--run-dir` when you want to promote a specific completed archive directory, and `--root` to add or override recovery roots for a one-off promotion. The helper copies the full archive run directory, updates artifact `recovery_roots` in the run-local Consist DB, syncs the updated `.consist` state into the promoted copy, and writes a promotion marker at `.consist/recovery_promotion.json`.

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
