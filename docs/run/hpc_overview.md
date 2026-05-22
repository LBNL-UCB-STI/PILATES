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
- `job.sh` installs `consist` from local source when it can, and otherwise falls back to the configured PyPI package or the default `consist==0.1.5`.

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

PILATES supports a manual promotion helper for copying a completed archive run into one or more recovery roots and recording those promoted roots on the run's logged artifacts:

```bash
python -m pilates.runtime.promote_run_archive -c path/to/settings.yaml
```

Use `--run-dir` when you want to promote a specific completed archive directory, and `--root` to add or override recovery roots for a one-off promotion. The helper copies the full archive run directory, updates artifact `recovery_roots` in the run-local Consist DB, syncs the updated `.consist` state into the promoted copy, and writes a promotion marker at `.consist/recovery_promotion.json`.

Read [Run Promotion](run_promotion.md) for the full post-run workflow,
including NFS promotion, verification, and merging the completed run-local DB
into a central main DB.

## Practical Environment Variables

The current wrapper scripts recognize the following controls. The Slurm account
is required on the command line; the environment variables are optional unless
your cluster layout differs from the defaults.

| Setting | Required? | Default or source                                                             | What it controls |
| --- | --- |-------------------------------------------------------------------------------| --- |
| `-a`, `--account` | Yes | none                                                                          | Slurm account passed to `sbatch --account`. |
| `-p` | No | `lr7`                                                                         | Partition preset. Current scripts support `lr7` and `lr8`. |
| `PILATES_DIR` | No | `/global/scratch/users/$USER/sources/PILATES`                                 | Checkout used by `job_runner.sh` and `job.sh`. Override this first if the repo lives elsewhere. |
| `PILATES_VENV_PATH` | No | `$PILATES_DIR/PILATES-env`                                                    | Job-side Python virtual environment. |
| `PILATES_REQUIREMENTS_FILE` | No | `$PILATES_DIR/hpc/requirements-hpc.txt`, then `$PILATES_DIR/requirements.txt` | Requirements file installed inside the job. |
| `CONSIST_SRC_DIR` | No | `$PILATES_DIR/../consist`                                                     | Editable Consist checkout used when present. |
| `CONSIST_PYPI_PACKAGE` | No | requirement pin when present, otherwise `consist==0.1.5`                      | Package spec used when editable Consist install is not available. |
| `EXPECTED_EXECUTION_DURATION` | No | `3-00:00:00`                                                                  | Slurm wall time. |
| `MEMORY_LIMIT_GB` | No | partition preset                                                              | Slurm memory request. Explicit env value overrides partition and `--high-mem` defaults. |
| `BEAM_MEMORY` | No | partition preset                                                              | Value substituted into settings templates containing `${BEAM_MEMORY}`. Explicit env value overrides partition and `--high-mem` defaults. |
| `PILATES_THREADS` | No | `8`                                                                           | Caps native Python/BLAS/OpenMP thread pools inside `job.sh`. |

Memory precedence is: explicit `MEMORY_LIMIT_GB` / `BEAM_MEMORY` environment
values first, then partition presets. On `lr7`, `--high-mem` changes only the
default preset from `240G` / `180g` to `480G` / `400g`; explicit environment
values still win. On `lr8`, the current defaults are `700G` / `600g`.

## Adapting To Other Slurm Clusters

The Lawrencium paths are examples, not a portability requirement. On another
Slurm cluster, set `PILATES_DIR`, choose the equivalent partition/account
arguments, and update `run.output_directory`, `run.local_workspace_root`, and
`run.recovery_archive_roots` to match that cluster's scratch, node-local, and
archive filesystems. If the module stack differs, edit `hpc/job.sh` in the
small setup block that loads compiler, PROJ, and Python modules.

## Adjacent Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) first.
- Use [Lawrencium](lawrencium.md) for the concrete LBNL path.
- Use [Restart and Resume](restart_and_resume.md) for replay-first recovery behavior.
