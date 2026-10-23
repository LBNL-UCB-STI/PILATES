---
title: Configuration Basics
summary: Conceptual structure of PILATES YAML settings, base inheritance, CLI versus YAML controls, and workspace paths.
---

# Configuration Basics

PILATES reads a single YAML configuration file (or a child overlay inheriting via `base:`), validates it with Pydantic, and builds an immutable runtime plan before any model stage executes.

## Top-Level Sections

Each scenario file organizes settings into five areas:

- `base` (optional): Path to a parent YAML scenario file whose values provide the baseline for the current file.
- `run`: Scenario identity, simulation years, model frequencies, enabled models, storage directories, restart behavior, and Consist database settings.
- `shared`: Inputs shared across models, including regional `geography` (`FIPS`, `local_crs`, `zones`), `skims` definitions, and the shared `database` path.
- `infrastructure`: Container runtime (`docker` or `singularity`) and per-model image tags or `.sif` paths.
- Model and pipeline sections (`urbansim`, `atlas`, `activitysim`, `beam`, `postprocessing`): Model-specific input/output paths, container mounts, memory limits, and command parameters.

## Scenario Inheritance (`base:`)

Scenario files in `scenarios/` use `base: <parent.yaml>` inheritance so a variant specifies only the keys that differ from its regional baseline:

```yaml
base: settings-sfbay-newconfig-local.yaml

run:
  scenario: sfbay-high-density
  output_run_name: sfbay-high-density
  end_year: 2020
```

During config loading, `resolve_config_dict` merges nested dictionaries from the parent and child files, while child scalars, lists, and `null` values replace parent values. Inspect and validate the merged configuration for any overlay from the command line:

```bash
python -m pilates.config.models path/to/scenario.yaml --validate
```

## YAML Settings Versus CLI Flags

Keep scenario parameters in YAML and execution selection on the command line:

- **YAML file**: Defines region, start and end years, enabled models, workspace and archive paths, container images, and provenance database options.
- **CLI flags (`python run.py`)**: Selects the YAML file to run (`-c/--config`), passes an existing `run_state.yaml` path for restart (`-S/--stage`), or permits resuming from an earlier simulation year (`--allow-rewind-resume`).

The launcher initializes runtime flags once at startup and passes the resulting workflow plan to every stage.

## Archive Directory Versus Mutable Workspace

PILATES separates durable run records from high-churn working files across two directory trees, plus an optional post-run recovery tier:

- **Archive run directory (`run.output_directory`)**: Parent directory where the launcher creates `<output_directory>/pilates-run--<region>--<name>--<timestamp>/`. This directory stores `run_state.yaml` (`WorkflowState`), `.consist/provenance.duckdb`, configuration snapshots, and archived step outputs.
- **Mutable workspace (`run.local_workspace_root`)**: Parent directory where models read and write staged inputs and live intermediate files during execution. Leave `local_workspace_root` unset on a local workstation to colocate the workspace inside the archive directory; set it to node-local scratch on HPC clusters.
- **Cold recovery roots (`run.recovery_archive_roots`)**: Optional list of long-term storage paths (such as project NFS) used by `pilates.runtime.promote_run_archive` after a cluster run finishes.

## Fields You Edit First

To set up a new run or adapt a template to your machine, start with `run`:

- `run.output_directory` and `run.local_workspace_root` for storage paths.
- `run.scenario` and `run.output_run_name` for run naming.
- `run.start_year`, `run.end_year`, and `run.supply_demand_iters` for simulation horizon and iteration count.
- `run.models` (`land_use`, `vehicle_ownership`, `activity_demand`, `travel`) to enable or disable individual models.

The loader expands environment variables (such as `${USER}` and `${SLURM_JOB_ID}`) in `run.output_directory`, `run.local_workspace_root`, `run.recovery_archive_roots`, and `shared.database.path`. It also enforces `end_year >= start_year`, requires `consist_db_filename` to be a basename without path separators, and restricts `atlas.beamac > 0` to `sfbay` runs with `skims_zone_type: taz`.

## Next Steps

- Follow [Getting Started](getting_started.md) for the step-by-step first local run walkthrough.
- Consult [Configuration Reference](../run/configuration_reference.md) for the complete field-by-field reference.
- Read [Scenario Lifecycle](../run/scenario_lifecycle.md) to see how configuration drives startup, bootstrap, and year-loop execution.
