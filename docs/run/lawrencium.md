---
title: Lawrencium
summary: Lawrencium-specific setup, data layout, and Slurm submission path for PILATES.
---

# Lawrencium

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

## Region Data

The active cluster settings templates point at the region-specific UrbanSim, ActivitySim, and BEAM trees already used by the active scenario configs. The archived Lawrencium note is only useful here as a migration source for missing concrete path examples.

## Adjacent Pages

- Read [HPC Overview](hpc_overview.md) first for the generic model.
- Pair this with [CLI](cli.md) and [Configuration Reference](configuration_reference.md).
- Use [Restart and Resume](restart_and_resume.md) for restart behavior after interruption.
