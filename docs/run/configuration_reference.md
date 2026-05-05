---
title: Configuration Reference
summary: Reference for the current nested PILATES settings model and its workflow effects.
---

# Configuration Reference

## Top-Level Shape

PILATES expects a YAML file with these top-level sections:

- `run`
- `shared`
- `infrastructure`
- optional model sections such as `urbansim`, `atlas`, `activitysim`, `beam`, and `postprocessing`

The active templates under `scenarios/` already use this shape. The config loader validates the file with Pydantic and rejects missing model sections when the selected models require them.

## `run`

`run` holds the settings that shape the workflow and the runtime posture:

- `region`, `scenario`, `start_year`, `end_year`
- model frequencies such as `land_use_freq`, `travel_model_freq`, `vehicle_ownership_freq`, and `supply_demand_iters`
- `models`, which selects the enabled model names
- `output_directory` and `output_run_name`
- `local_workspace_root`, which can point at a node-local mutable workspace
- `recovery_archive_roots`, which can point at colder long-term archive destinations used after the run completes
- archive and restart controls such as `enable_archive_copy` and `restart_strict`
- bootstrap and Consist DB controls such as `bootstrap_cache_enabled`, `consist_db_local_run`, `consist_db_snapshot_*`, `consist_db_restore_*`, and `consist_db_seed_*`
- Consist cache controls such as `consist_code_identity` and `consist_hashing_strategy`

The loader expands environment variables in `output_directory`, `local_workspace_root`, and `recovery_archive_roots`. It also enforces `end_year >= start_year` and a basename-only `consist_db_filename`.

## `shared`

`shared` contains inputs that are shared across multiple models:

- `geography`, including `FIPS`, `local_crs`, `zones`, and `alternative_zones`
- `skims`, including skim filenames, highway path names, transit paths, and periods
- `database`, which selects the shared database file and whether it is enabled

The config and test suite pin the active Seattle and SF Bay zone sources, CRS values, and warmstart linkstats paths.

## `infrastructure`

`infrastructure` selects the container manager and the image references:

- `container_manager` is `docker` or `singularity`
- `singularity_images` maps model names to image URIs
- `docker_images` maps model names to image tags
- `docker_config` controls Docker logging and image-pull behavior

## Model Sections

- `urbansim` describes the land-use container paths and templates.
- `atlas` describes the vehicle-ownership container paths and command template.
- `activitysim` describes the demand model folders, sampling, replanning, and output tables.
- `beam` describes the travel-model config, output directories, memory, and warmstart inputs.
- `postprocessing` describes downstream validation and output folders.

## Consist-Relevant Fields

These settings affect cache identity or run persistence, not the model science itself:

- `run.consist_code_identity`
- `run.consist_hashing_strategy`
- `run.bootstrap_cache_enabled`
- `run.enable_archive_copy`
- `run.recovery_archive_roots`
- the Consist DB snapshot/restore/seed fields

## Adjacent Pages

- Start with [Configuration Basics](../start-here/configuration_basics.md) if you are new.
- Pair this with [CLI](cli.md) for invocation behavior.
- Use [Restart and Resume](restart_and_resume.md) for replay-specific behavior.
