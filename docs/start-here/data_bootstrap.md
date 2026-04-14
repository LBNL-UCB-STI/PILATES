---
title: Data Bootstrap
summary: First-run data expectations and input layout for local and HPC setups.
---

# Data Bootstrap

## Purpose

Describe the minimum input data layout PILATES expects before the first run.

## Who This Is For

- Users who have the code checked out but not the region-specific data repos.
- Operators validating whether a config points at a plausible local or shared-data layout.

## This Page Answers

- Which input trees are expected to exist before a run starts?
- Which parts of bootstrap are repo layout versus runtime initialization?
- How do Seattle and SF Bay input setups differ at a high level?

## Adjacent Pages

- Read [Getting Started](getting_started.md) first.
- Use [First Run Walkthrough](first_run_walkthrough.md) for a concrete local path.
- Use [Lawrencium](../run/lawrencium.md) for cluster-specific setup.

## Source Material To Mine

- Existing bootstrap layout notes from the old page.
- Region-specific assumptions in `scenarios/` and active settings files.
- Archived Lawrencium setup notes for concrete repo/data checkout commands.
