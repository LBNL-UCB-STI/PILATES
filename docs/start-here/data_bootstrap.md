---
title: Data Bootstrap
summary: First-run data expectations and input layout for local and HPC setups.
---

# Data Bootstrap

## What PILATES Reads During Bootstrap

- UrbanSim mutable input data under the configured `urbansim.local_data_input_folder`.
- ActivitySim mutable data under `activitysim.local_mutable_data_folder`.
- ActivitySim mutable config trees under `activitysim.local_mutable_configs_folder`, including `configs`, `configs_extended`, `configs_mp`, and `configs_sh_compile`.
- BEAM mutable input under `beam.local_mutable_data_folder`, including the region subdirectory and the model config file.
- ATLAS mutable input under `atlas.host_mutable_input_folder` when vehicle ownership is enabled.

## What The Active Templates Already Assume

The active Seattle, SF Bay, and Breathe templates already point at region-specific input trees under `pilates/urbansim/data`, `pilates/activitysim/data`, and `pilates/beam/production`. The test suite also locks in the expected region-to-region-id mapping, zone sources, router directories, and warmstart linkstats paths for those templates.

## Bootstrap Versus Runtime

Bootstrap is the part of the run that stages or validates the local workspace before the yearly loop starts. It is not the same thing as the model runtime itself. The launcher uses the workspace layout from the settings file, then checks for the files that later restart paths expect to find locally.

## When Bootstrap Is Thin

If a workflow is restarted instead of started fresh, PILATES may hydrate some of the workspace from archive material before the scenario loop begins. The docs here should stay practical and not assume more than the current loader, bootstrap, and restart checks prove.

## Adjacent Pages

- Read [Getting Started](getting_started.md) first.
- Use [First Run Walkthrough](first_run_walkthrough.md) for a concrete local path.
- Use [Lawrencium](../run/lawrencium.md) for the cluster-specific path.
