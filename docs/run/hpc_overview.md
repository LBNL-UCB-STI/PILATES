---
title: HPC Overview
summary: Facility-agnostic Slurm execution workflow, 3-tier storage split, and environment controls for PILATES on HPC clusters.
---

# HPC Overview

PILATES runs the same launcher, scenario lifecycle, and Consist workflow contracts on an HPC cluster as on a workstation. Cluster execution adds a two-script Slurm wrapper (`hpc/job_runner.sh` and `hpc/job.sh`), Apptainer/Singularity container execution, and a 3-tier storage layout that separates mutable node-local scratch from durable shared storage.

## Slurm Wrapper Flow

Cluster runs use two scripts in `hpc/`:

1. **`hpc/job_runner.sh` (login node)**: Parses CLI options (`-c`, `-s`, `-p`, `-a/--account`, `--high-mem`, `--editable-consist`), selects CPU and memory presets for the target partition, substitutes `${BEAM_MEMORY}` into a per-job `settings_<jobid>.yaml` when the scenario template contains that variable, and submits `hpc/job.sh` via `sbatch`.
2. **`hpc/job.sh` (compute node)**: Loads cluster compiler, PROJ, and Python modules, creates or reuses the Python virtual environment at `PILATES_VENV_PATH`, installs dependencies from `hpc/requirements-hpc.txt` (or `requirements.txt`), installs `consist` (from `CONSIST_SRC_DIR`, `CONSIST_PYPI_PACKAGE`, or `consist>=0.3.3`), and runs `python run.py`.

## Three-Tier HPC Storage Split

Configure three separate storage tiers in your HPC scenario YAML to keep high-churn I/O off shared filesystems while preserving durable archives:

| Storage Tier | YAML Field | Cluster Filesystem Role | Lifecycle Purpose |
| --- | --- | --- | --- |
| Tier 1: Fast mutable workspace | `run.local_workspace_root` | Node-local NVMe/SSD (`/local/job${SLURM_JOB_ID}/...`) or fast scratch | Holds staged model inputs and high-churn intermediate files during container execution. Can be wiped when the Slurm job ends. |
| Tier 2: Active durable archive | `run.output_directory` | Shared parallel scratch | Receives mirrored step outputs (`run.enable_archive_copy: true`), `run_state.yaml` checkpoints, and `.consist/provenance.duckdb` snapshots during the run. Survives job termination so `-S` restarts work across compute nodes. |
| Tier 3: Cold recovery archive | `run.recovery_archive_roots` | Project NFS or long-term storage | Receives completed run archives and provenance metadata after the job finishes via `python -m pilates.runtime.promote_run_archive`. |

During startup, the launcher exports `PILATES_LOCAL_RUN_DIR`, `PILATES_ARCHIVE_RUN_DIR`, and `PILATES_ENABLE_ARCHIVE_COPY` so step wrappers know both the mutable workspace and the active archive target.

## Wrapper Flags and Environment Variables

Pass `-a/--account` and `-c` to `hpc/job_runner.sh`; override environment variables when your cluster paths or resource requirements differ from the defaults:

| Control | Required | Default / Resolution Order | Description |
| --- | --- | --- | --- |
| `-a`, `--account <acct>` | Yes | None | Slurm billing account passed to `sbatch --account`. |
| `-c <settings.yaml>` | No | `settings.yaml` | Scenario YAML file (or `base:` overlay) to execute. |
| `-s <run_state.yaml>` | No | None | Existing `run_state.yaml` path forwarded to `python run.py -S` for restart. |
| `-p <partition>` | No | `lr7` | Partition preset (`lr6`, `lr6_m192`, `lr7`, `lr8`). |
| `-H`, `--high-mem` | No | Disabled | Selects high-memory partition presets on `lr6` (`180G` / `150g`) or `lr7` (`480G` / `400g`). |
| `--editable-consist <dir>` | No | None | Sets `CONSIST_SRC_DIR` to an absolute checkout path for `pip install --no-deps -e`. |
| `PILATES_DIR` | No | `/global/scratch/users/$USER/sources/PILATES` | Repository checkout root on the cluster. |
| `PILATES_VENV_PATH` | No | `$PILATES_DIR/PILATES-env` | Python virtual environment path created or reused by `hpc/job.sh`. |
| `PILATES_REQUIREMENTS_FILE` | No | `$PILATES_DIR/hpc/requirements-hpc.txt`, fallback `$PILATES_DIR/requirements.txt` | Dependency requirements file installed inside the job environment. |
| `CONSIST_SRC_DIR` | No | `$PILATES_DIR/../consist` | Editable Consist checkout path tried first before wheel installation. |
| `CONSIST_PYPI_PACKAGE` | No | Explicit env, then pin in requirements file, then `consist>=0.3.3` | Package specification installed when no valid `CONSIST_SRC_DIR` is present. |
| `EXPECTED_EXECUTION_DURATION` | No | `3-00:00:00` | Slurm walltime request (`sbatch --time`). |
| `MEMORY_LIMIT_GB` | No | Partition preset | Slurm memory limit (`sbatch --mem`); overrides partition and `--high-mem` defaults. |
| `BEAM_MEMORY` | No | Partition preset | JVM heap limit substituted into `${BEAM_MEMORY}` in the scenario YAML; overrides partition and `--high-mem` defaults. |
| `PILATES_THREADS` | No | `8` | Thread cap exported to `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`. |

## Adapting to Another Slurm Cluster

To run PILATES on a Slurm cluster outside Lawrencium:

1. Set `PILATES_DIR` to your repository checkout and configure `run.output_directory`, `run.local_workspace_root`, and `run.recovery_archive_roots` for your cluster's parallel scratch, node-local scratch, and project filesystems.
2. Set `infrastructure.container_manager: singularity` and point `infrastructure.singularity_images` at your local `.sif` image files or registry URIs.
3. Update the partition case block in `hpc/job_runner.sh` and the `module load` block in `hpc/job.sh` to match your facility's partition names and Python/PROJ module names.

## Post-Run Promotion

Once a cluster job finishes on shared scratch, promote the completed archive run directory to cold storage and merge its run-local provenance DB into your shared catalog:

```bash
python -m pilates.runtime.promote_run_archive -c path/to/settings.yaml
```

See [Run Promotion](run_promotion.md) for preview, verification, and database merge commands.

## Related Pages

- Read [Lawrencium](lawrencium.md) for LBNL Lawrencium partition presets, filesystem paths, and submission commands.
- Follow [Run Promotion](run_promotion.md) for post-run archive promotion and DuckDB merging.
- Consult [Scenario Lifecycle](scenario_lifecycle.md) for restart and checkpoint behavior across Slurm jobs.
