---
title: Lawrencium
summary: LBNL Lawrencium partition presets, storage paths, module setup, and Slurm submission workflow for PILATES.
---

# Lawrencium

Use this page for LBNL Lawrencium specifics. Read [HPC Overview](hpc_overview.md) first for the generic Slurm wrapper flow, 3-tier storage architecture, and environment variable reference.

## Lawrencium Filesystem Conventions

Map the 3-tier PILATES storage model onto Lawrencium filesystems as follows:

- **Source checkout and regional input data**: `/global/scratch/users/$USER/sources/PILATES` (default `PILATES_DIR` in `hpc/job_runner.sh` and `hpc/job.sh`). Avoid running active simulation I/O out of `/global/home/users/$USER`, which has small quotas and lower throughput.
- **Active archive directory (`run.output_directory`)**: `/global/scratch/users/$USER/pilates-outputs` on shared parallel scratch.
- **Mutable node-local workspace (`run.local_workspace_root`)**: `/local/job${SLURM_JOB_ID}/pilates-workspace` on the compute node's local disk.
- **Cold recovery archive (`run.recovery_archive_roots`)**: `/clusterfs/<project-or-nfs-root>/$USER/pilates-outputs` on project NFS storage.
- **Slurm job logs**: `/global/scratch/users/$USER/pilates_logs/pilates_<jobid>.out` and `.err`.

Example Lawrencium `run` configuration block:

```yaml
run:
  output_directory: /global/scratch/users/${USER}/pilates-outputs
  output_run_name: consist-sfbay-usim-base
  local_workspace_root: /local/job${SLURM_JOB_ID}/pilates-workspace
  recovery_archive_roots:
    - /clusterfs/<project-or-nfs-root>/${USER}/pilates-outputs
  enable_archive_copy: true
```

Keep `enable_archive_copy: true` so the launcher mirrors completed step outputs from `/local/job${SLURM_JOB_ID}` into `/global/scratch/users/$USER/pilates-outputs` before the Slurm allocation ends.

## Partition Presets (`lr6`, `lr7`, `lr8`)

`hpc/job_runner.sh` configures CPU count, QoS, Slurm memory (`MEMORY_LIMIT_GB`), and BEAM JVM heap (`BEAM_MEMORY`) based on `-p <partition>` and `--high-mem`:

| Partition Flag | Slurm Partition | QoS | CPUs | Standard Memory (`--mem` / `BEAM_MEMORY`) | `--high-mem` (`-H`) Memory (`--mem` / `BEAM_MEMORY`) |
| --- | --- | --- | --- | --- | --- |
| `-p lr6` | `lr6` | `lr_normal` | 32 | `90G` / `70g` | `180G` / `150g` |
| `-p lr6_m192` | `lr6` | `lr_normal` | 32 | `180G` / `150g` (`--constraint=lr6_m192`) | `180G` / `150g` (`--constraint=lr6_m192`) |
| `-p lr7` (default) | `lr7` | `lr_normal` | 56 | `240G` / `180g` | `480G` / `400g` |
| `-p lr8` | `lr8` | `lr8_normal` | 128 | `700G` / `600g` | `700G` / `600g` |

Explicit `MEMORY_LIMIT_GB` or `BEAM_MEMORY` environment variables override the partition and `--high-mem` presets.

## Module and Apptainer Setup

Inside the allocated compute node, `hpc/job.sh` loads the Lawrencium software environment:

```bash
module purge
module load gcc/11.4.0
module load proj/9.2.1
module load python/3.11.6
```

Lawrencium scenario configs (`scenarios/sfbay/settings-sfbay-newconfig-hpc.yaml`, `scenarios/seattle/settings-seattle-consist-hpc.yaml`) set `infrastructure.container_manager: singularity` so PILATES invokes Apptainer/Singularity against the configured `.sif` images or Docker URIs.

## Submitting and Resuming Jobs

Submit a job from a Lawrencium login node with `hpc/job_runner.sh`:

```bash
# Submit a standard lr7 run
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml -a <slurm_account> -p lr7

# Submit a high-memory lr7 run
./hpc/job_runner.sh -c scenarios/sfbay/settings-sfbay-newconfig-hpc.yaml -a <slurm_account> -p lr7 --high-mem

# Submit an lr8 run using an editable Consist checkout
./hpc/job_runner.sh -c scenarios/sfbay/settings-sfbay-newconfig-hpc.yaml \
  -a <slurm_account> -p lr8 \
  --editable-consist /global/scratch/users/$USER/sources/consist

# Resume an interrupted run from its scratch archive state file
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  -s /global/scratch/users/$USER/pilates-outputs/pilates-run--.../run_state.yaml \
  -a <slurm_account> -p lr7
```

If the template contains `${BEAM_MEMORY}`, `job_runner.sh` writes `settings_<jobid>.yaml` in the repository root and submits `hpc/job.sh` with that generated file.

## Promoting Completed Scratch Runs to Project NFS

Because `/global/scratch/users/$USER` is subject to cluster purge policies, promote completed runs to `/clusterfs/...` and merge their provenance shards into your project main database after the Slurm job finishes:

```bash
python -m pilates.runtime.promote_run_archive \
  -c scenarios/seattle/settings-seattle-consist-hpc.yaml \
  --merge-main-db /clusterfs/<project-or-nfs-root>/pilates-main/provenance.duckdb
```

Follow [Run Promotion](run_promotion.md) for the complete promotion, `--dry-run` preview, `--verify-only` check, and main-database merge workflow.

## Related Pages

- Read [HPC Overview](hpc_overview.md) for the generic 3-tier storage model and full environment variable table.
- Follow [Run Promotion](run_promotion.md) to copy scratch archives to NFS and merge provenance databases.
- Consult [CLI](cli.md) and [Configuration Reference](configuration_reference.md) for flag and YAML field details.
