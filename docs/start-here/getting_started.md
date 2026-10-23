---
title: Getting Started
summary: Quickstart from repository clone and regional input staging to a first local PILATES run.
---

# Getting Started

Use this guide to configure a scenario file, stage regional input data, and run PILATES on a local workstation.

## Prerequisites and Regional Input Directories

Before launching a run, verify your local software environment and regional data trees:

- Python 3.11+ with repository dependencies installed (`pip install -r requirements.txt`) and an installed `consist` package.
- Docker (`infrastructure.container_manager: docker`) for local execution, or Apptainer/Singularity (`singularity`) on HPC clusters.
- Regional input data placed in the model directories referenced by your scenario YAML.

Active scenario templates under `scenarios/` (such as `scenarios/sfbay/` and `scenarios/seattle/`) read source data from four repository-relative trees and stage working copies into the mutable run workspace during bootstrap:

| Model Stage | Source Input Directory | Bootstrap Staged Workspace Target | Staged Contents |
| --- | --- | --- | --- |
| UrbanSim (`land_use`) | `pilates/urbansim/data/` | `<workspace>/urbansim/data/` | Base regional HDF5 datastore (`custom_mpo_<region_id>_model_data.h5` or year-scoped `.h5`). When UrbanSim is disabled, PILATES still stages this datastore here if ActivitySim or ATLAS is enabled. |
| ActivitySim (`activity_demand`) | `pilates/activitysim/data/` and `pilates/activitysim/configs/` | `<workspace>/activitysim/data/` and `<workspace>/activitysim/configs/` | Zone and household inputs, initial OMX skims, canonical zone coordinates (`canonical_zones.parquet`), and config trees (`configs`, `configs_extended`, `configs_mp`, `configs_sh_compile`). |
| BEAM (`travel`) | `pilates/beam/production/` | `<workspace>/beam/input/<region>/` | Primary BEAM `.conf` file, `r5/` network router directory, and warmstart linkstats. |
| ATLAS (`vehicle_ownership`) | `pilates/atlas/atlas_input/` | `<workspace>/atlas/atlas_input/` | Vehicle ownership input tables and warmstart data when `vehicle_ownership` is enabled. |

## First Local Run Walkthrough

Tracked files under `scenarios/` serve as baseline templates that contain machine-specific paths and model selections. Create a local scenario overlay or copy before running:

1. Create a scenario YAML file that inherits from a tracked regional config via `base:`, or copy a template such as `scenarios/sfbay/settings-sfbay-newconfig-local.yaml`:

    ```yaml
    base: scenarios/sfbay/settings-sfbay-newconfig-local.yaml

    run:
      scenario: sfbay-local-first-run
      output_run_name: sfbay-local-first-run
      output_directory: /abs/path/to/pilates-outputs
      local_workspace_root: /abs/path/to/pilates-workspace
      start_year: 2010
      end_year: 2011
      supply_demand_iters: 1
    ```

2. Edit `run.output_directory` to point to the parent folder where PILATES stores durable run archives. Set `run.local_workspace_root` to a separate scratch folder if you want mutable model execution files isolated from the archive directory, or omit it to colocate both in `run.output_directory`.
3. Check `run.models` and any model input paths in the parent template to verify that every enabled model has its source data tree in place.
4. Validate the merged configuration before starting the workflow:

    ```bash
    python -m pilates.config.models path/to/your-settings.yaml --validate
    ```

5. Launch the simulation:

    ```bash
    python run.py -c path/to/your-settings.yaml
    ```

## Run Startup and Outputs

During launch, `run.py` validates the YAML settings, initializes runtime flags, builds the `WorkflowPlan`, and creates or restores `WorkflowState`. Before entering the year loop, the bootstrap phase stages the regional input trees into the mutable workspace and publishes starting artifact handles to the Consist coupler.

A completed or in-progress run produces two directories and two state records:

- **Archive run directory** (`<run.output_directory>/pilates-run--<region>--<name>--<timestamp>/`): Durable record of the run containing exported outputs, configuration snapshots, state checkpoints, and provenance metadata.
- **Mutable workspace** (`<run.local_workspace_root>/pilates-run--.../` or colocated in the archive directory): Live working directory where containers read and write high-churn intermediate files.
- **`run_state.yaml`** (managed by `WorkflowState`): YAML checkpoint written inside both the archive directory and mutable workspace that tracks `data_initialized`, completed stages per year, and iteration frontiers for restart and resume.
- **`.consist/provenance.duckdb`** (or `run.consist_db_filename`): Run-local DuckDB provenance database recording execution lineage, artifact digests, and cache metadata, snapshotted to `.consist/snapshots/latest/` at stage boundaries and shutdown.

If a run stops early, the launcher logs the exact `python run.py -c <config> -S <archive_run_dir>/run_state.yaml` command required to resume from the last recorded frontier.

## Next Steps

| Goal | Page |
| --- | --- |
| Understand YAML layout, `base:` inheritance, and workspace paths | [Configuration Basics](configuration_basics.md) |
| Look up every configuration section and field | [Configuration Reference](../run/configuration_reference.md) |
| Check CLI flags for local execution, resume, and Slurm wrappers | [CLI](../run/cli.md) |
| Trace startup, bootstrap, year-loop execution, and restart behavior | [Scenario Lifecycle](../run/scenario_lifecycle.md) |
| Configure 3-tier cluster storage and Slurm jobs | [HPC Overview](../run/hpc_overview.md) and [Lawrencium](../run/lawrencium.md) |
| Promote scratch archives to cold storage and merge provenance DBs | [Run Promotion](../run/run_promotion.md) |
| Diagnose startup, bootstrap, restart, or cluster failures | [Troubleshooting](../run/troubleshooting.md) |
