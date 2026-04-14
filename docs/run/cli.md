---
title: CLI
summary: Public command-line entrypoints for local runs, resume, and HPC submission.
---

# CLI

## Purpose

Define the current user-facing command surface for running and resuming PILATES.

## Who This Is For

- Users running scenarios from the repo root.
- Operators who need the thin CLI surface before switching to config details or HPC wrappers.

## This Page Answers

- Which arguments does `python run.py` currently support?
- What belongs in the CLI versus in config?
- Which wrapper scripts are still current and which ones are only partial helpers?

## Adjacent Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) first for the mental model.
- Pair this with [Configuration Reference](configuration_reference.md).
- For Slurm submission, go to [HPC Overview](hpc_overview.md) and [Lawrencium](lawrencium.md).

## Source Material To Mine

- Runtime CLI parsing in `pilates/utils/io.py`.
- Thin entrypoint behavior in `run.py`.
- HPC wrappers in `hpc/job_runner.sh` and `hpc/job.sh`.
