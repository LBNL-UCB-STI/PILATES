---
title: HPC Overview
summary: Generic HPC execution posture for PILATES and how it differs from local runs.
---

# HPC Overview

## Purpose

Explain the current HPC operating model without tying the whole page to one cluster.

## Who This Is For

- Users who already understand local runs and need the HPC mental model.
- Operators deciding how archive roots, local workspace, Slurm wrappers, and restart fit together.

## This Page Answers

- What stays the same between local and HPC execution?
- What changes operationally on HPC around submission, environment bootstrap, and storage?
- Which parts of the current public HPC story are generic versus Lawrencium-specific?

## Adjacent Pages

- Read [Scenario Lifecycle](scenario_lifecycle.md) first.
- Use [Lawrencium](lawrencium.md) for the concrete LBNL path.
- Use [Restart and Resume](restart_and_resume.md) for replay-first recovery behavior.

## Source Material To Mine

- Existing `hpc/README.md` and `hpc/job_runner.sh` behavior.
- Current `hpc/job.sh` environment bootstrap logic.
- Replay-first storage notes from the restart/archive integration refactor.
