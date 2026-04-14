---
title: Lawrencium
summary: Lawrencium-specific setup, data layout, and Slurm submission path for PILATES.
---

# Lawrencium

## Purpose

Provide the concrete LBNL cluster path for setting up data, bootstrapping the environment, and submitting jobs.

## Who This Is For

- PILATES users running on Lawrencium.
- Contributors updating the cluster-specific workflow docs after wrapper or environment changes.

## This Page Answers

- Where should PILATES live on Lawrencium and how is the Python environment bootstrapped?
- Which data repos and large files need to be present before submission?
- What is the normal `job_runner.sh` submission flow and what should users customize first?

## Adjacent Pages

- Read [HPC Overview](hpc_overview.md) first for the general model.
- Pair this with [CLI](cli.md) and [Configuration Reference](configuration_reference.md).
- Use [Restart and Resume](restart_and_resume.md) for restart behavior after preemption or interruption.

## Source Material To Mine

- Archived `lawrencium-setup.md`.
- Current `hpc/README.md`, `hpc/job_runner.sh`, and `hpc/job.sh`.
- Active Lawrencium-oriented settings templates such as `settings-sfbay-consist-hpc.yaml`.
