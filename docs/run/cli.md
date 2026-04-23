---
title: CLI
summary: Public command-line entrypoints for local runs, resume, and HPC submission.
---

# CLI

## What `python run.py` Accepts

The runtime loader currently parses:

- `-c` or `--config` to select the YAML settings file
- `-S` or `--stage` to resume from an existing stage or state file
- `--allow-rewind-resume` to permit rewind-style resume when the restart guardrail would otherwise stop it

`run.py` itself is a thin entrypoint. It calls the launcher, and the launcher owns the runtime assembly and restart logging.

## Most Common Usage

For most readers, there are only two normal paths:

```bash
python run.py -c path/to/settings.yaml
python run.py -c path/to/settings.yaml -S /path/to/state-file.yaml
```

Use the first for a fresh run and the second when you are resuming or replaying from an existing state file.

## CLI Versus Config

- Use the CLI to choose which file to load and whether to restart from a state file.
- Use the YAML file to choose the region, years, enabled models, workspace roots, archive behavior, and Consist settings.
- `-S/--stage` is a file path, not a workflow flag. The loader checks that it exists before it loads the config.
- The loader stores the chosen settings file and stage file on the runtime object for restart logging.

## Practical Invocation

```bash
python run.py -c scenarios/seattle/settings-seattle-newconfig-local.yaml
python run.py -c scenarios/seattle/settings-seattle-newconfig-local.yaml -S /path/to/run_state.yaml
python run.py -c scenarios/seattle/settings-seattle-newconfig-local.yaml --allow-rewind-resume
```

## HPC Submission

The Slurm wrapper is separate from the Python CLI:

- `./hpc/job_runner.sh` selects the partition/account and submits the job
- `./hpc/job.sh` runs inside the allocated node and invokes `python run.py`

## Adjacent Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) first for the runtime sequence.
- Pair this with [Configuration Reference](configuration_reference.md).
- For cluster execution, use [HPC Overview](hpc_overview.md) and [Lawrencium](lawrencium.md).
