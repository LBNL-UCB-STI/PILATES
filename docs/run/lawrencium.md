---
title: Lawrencium
summary: Lawrencium-specific setup, data layout, and Slurm submission path for PILATES.
---

# Lawrencium

## Storage Model

For the current replay-first PILATES runtime, the safest Lawrencium layout is:

- input trees and source checkout: backed-up home or scratch-based source tree, depending on your workflow
- active archive root during execution: shared scratch via `run.output_directory`
- mutable run workspace: node-local storage via `run.local_workspace_root`
- long-term cold archive: project NFS or other paid shared storage via `run.recovery_archive_roots`

The public LBNL docs describe `/global/home/users/$USER` as a small backed-up home tier and `/global/scratch/users/$USER` as the high-performance scratch filesystem. PILATES should use scratch for the active archive root and keep slower archival storage out of the hot path. If your project also has paid NFS storage, treat that as a promotion target rather than as the live archive root.

## Recommended Config Shape

For Lawrencium, the `run` section should usually follow this pattern:

```yaml
run:
  output_directory: /global/scratch/users/${USER}/pilates-outputs
  output_run_name: consist-sfbay-usim-base
  local_workspace_root: /local/job${SLURM_JOB_ID}/pilates-workspace
  recovery_archive_roots:
    - /clusterfs/<project-or-nfs-root>/$USER/pilates-outputs
  enable_archive_copy: true
```

Notes:

- `output_directory` is the durable archive root during the run, so put it on shared scratch.
- `local_workspace_root` is the mutable workspace for BEAM, ActivitySim, UrbanSim, and ATLAS. The active templates use `/local/job${SLURM_JOB_ID}` because it expands safely in the config and maps cleanly to node-local storage on the current cluster path.
- `recovery_archive_roots` is optional but recommended when you want a post-run cold archive on NFS.
- `enable_archive_copy: true` should remain enabled so logged outputs are mirrored from the node-local workspace into the active scratch archive during the run.

## Submission And Runtime Flow

## What To Set Up

- Clone PILATES into the scratch/source location expected by the wrapper scripts unless you override `PILATES_DIR`.
- Make sure the region data trees referenced by your active settings file are present.
- Make sure the modules used by `hpc/job.sh` are available on the cluster node.
- Provide a Slurm account on the submission command line.

## Current Submission Flow

The normal path is:

```bash
./hpc/job_runner.sh -c scenarios/seattle/settings-seattle-newconfig-hpc.yaml -a <slurm_account>
```

`job_runner.sh`:

- resolves the settings file path
- replaces `${BEAM_MEMORY}` when the template contains it
- chooses `lr7` or `lr8`
- fills in the partition, QoS, CPU, memory, and time request
- submits `hpc/job.sh`

`job.sh` then:

- loads the current modules used by the cluster runtime
- creates or reuses the job virtual environment
- installs Python dependencies
- installs `consist`
- runs `python run.py -c <config>` or `python run.py -c <config> -S <stage>`

## Path Conventions

- The wrapper scripts default `PILATES_DIR` to `/global/scratch/users/$USER/sources/PILATES`.
- The job log is written under `/global/scratch/users/$USER/pilates_logs/`.
- The job wrapper writes a generated per-job settings file named `settings_<jobid>.yaml`.

## Promotion To NFS After The Run

The recommended pattern is:

1. run on node-local workspace plus shared-scratch archive root
2. let PILATES finish and leave behind a complete archive run directory on scratch
3. promote that completed archive to NFS

Use:

```bash
python -m pilates.runtime.promote_run_archive -c scenarios/sfbay/settings-sfbay-consist-usim-hpc.yaml
```

When the helper runs successfully it:

- copies the full archive run directory to each configured recovery root
- verifies the promoted copy contains the expected `.consist`, workflow metadata, and run-state files
- records the promoted run directory on the run's logged artifacts as a Consist `recovery_root`
- syncs the updated `.consist` DB state into the promoted copy
- writes `.consist/recovery_promotion.json` as an operator-visible marker

Use `--run-dir` when you want to promote a specific completed archive run directory instead of the default latest match, and `--root` when you want to add a one-off destination without changing the YAML file.

## What This Buys You

This split keeps the active run on the fast tiers while still preserving a complete historical run on NFS:

- scratch remains the durable active archive during execution
- node-local storage handles high-churn mutable files
- NFS stores the cold promoted archive and can satisfy historical artifact hydration later through Consist `recovery_roots`

This v1 promotion flow preserves the run-local shard DB inside the promoted archive. A later shard-to-master merge can be layered on top of that, but the promoted run should remain a self-contained historical archive.

## Region Data

The active cluster settings templates point at the region-specific UrbanSim, ActivitySim, and BEAM trees already used by the active scenario configs. The archived Lawrencium note is only useful here as a migration source for missing concrete path examples.

## Adjacent Pages

- Read [HPC Overview](hpc_overview.md) first for the generic model.
- Pair this with [CLI](cli.md) and [Configuration Reference](configuration_reference.md).
- Use [Restart and Resume](restart_and_resume.md) for restart behavior after interruption.
