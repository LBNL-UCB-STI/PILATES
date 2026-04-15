---
title: Configuration Basics
summary: Mental model for how PILATES config is organized before the full reference.
---

# Configuration Basics

## What Belongs Where

- `run` holds the simulation run identity and runtime posture.
- `shared` holds inputs shared across models, such as geography, skims, and the database path.
- `infrastructure` holds the container/runtime selection.
- Model sections such as `urbansim`, `atlas`, `activitysim`, and `beam` hold model-specific paths and command templates.
- `postprocessing` holds downstream output and validation settings.

## Files Versus CLI

- The YAML file selects the region, years, enabled models, output roots, and runtime flags.
- The command line selects which YAML file to load and whether to resume from a stage file.
- `python run.py -c ...` is the normal invocation for choosing a settings file.
- `-S/--stage` points at an existing state file for restart or resume.

## The Practical Config Split

- `run.output_directory` sets the parent directory where the launcher creates the archive run directory for each run.
- `run.local_workspace_root` sets the parent directory for the mutable workspace when you want it to differ from the archive side.
- `run.output_run_name` is part of the generated run-directory name.
- `run.enable_archive_copy`, `run.bootstrap_cache_enabled`, `run.restart_strict`, and the Consist DB snapshot fields change runtime behavior, not model science.
- `run.models` decides which model sections must be present and which stages the launcher enables.

## What Most Users Edit First

Most first-run edits are in the `run` section:

- where the archive run directory should live
- whether the mutable workspace should live somewhere else
- which models are enabled
- the run name prefix

Most readers do not need to change deeper model sections until the active template already loads and starts.

## Validation And Normalization

PILATES loads config with Pydantic validation. The current loader expands environment variables in `run.output_directory`, `run.local_workspace_root`, and the database path. It also enforces basic guardrails such as `end_year >= start_year`, a basename-only `consist_db_filename`, and the `atlas.beamac` restriction that appears in the loader.

## Adjacent Pages

- Read [First Run Walkthrough](first_run_walkthrough.md) for the concrete path.
- Use [Configuration Reference](../run/configuration_reference.md) for the complete field list.
- Use [Scenario Lifecycle](../run/scenario_lifecycle.md) to connect config to runtime behavior.
