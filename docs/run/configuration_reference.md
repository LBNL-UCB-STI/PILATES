---
title: Configuration Reference
summary: Field reference for PILATES YAML configuration sections, base inheritance, and model settings.
---

# Configuration Reference

PILATES loads scenario configuration through `pilates.config.models.PilatesConfig`. The loader resolves `base:` inheritance, expands environment variables on path fields, validates types with Pydantic, and verifies that each model enabled in `run.models` has a corresponding configuration section.

## Top-Level Structure

A scenario YAML file supports these top-level keys:

- `base` (or `extends`): Optional path to a parent YAML scenario file.
- `run`: Scenario identity, simulation horizon, frequencies, storage paths, restart policy, and Consist provenance controls.
- `write_skims_to_omx`: Optional boolean (default `false`) enabling OMX skim export alongside Zarr skims.
- `shared`: Regional geography, skims parameters, and shared DuckDB database settings.
- `infrastructure`: Container manager (`docker` or `singularity`) and per-model image mappings.
- `urbansim`, `atlas`, `activitysim`, `beam`, `postprocessing`: Model-specific sections required when the corresponding stage is enabled in `run.models`.
- `workflow`: Optional workflow contract validation flags (`contract_validation: true`).

## Scenario Inheritance (`base:`)

Child scenario files inherit from a parent YAML file via `base:` (resolved relative to the child YAML directory or repository root):

```yaml
base: settings-sfbay-consist-usim-hpc-2019.yaml

run:
  scenario: sfbay-2019-canary
  output_run_name: sfbay-2019-canary
```

`resolve_config_dict` applies three merge rules:

- Nested dictionaries merge key by key so child files specify only overridden fields.
- Lists, scalars, and `null` values in the child file replace the parent value.
- Circular `base:` references raise a `ValueError` during load.

Validate and print the merged configuration from the command line:

```bash
python -m pilates.config.models path/to/scenario.yaml --validate
```

## `run` Section

### Scenario Horizon and Model Selection

| Field | Type / Default | Purpose |
| --- | --- | --- |
| `region` | `str` (required) | Regional identifier (`sfbay`, `seattle`, `austin`, `breathe`). |
| `scenario` | `str` (required) | Scenario label recorded in run metadata and output naming. |
| `start_year`, `end_year` | `int` (required) | Simulation start and end years (`end_year >= start_year`). |
| `land_use_freq` | `int` (`1`) | Year step interval between land-use stages. |
| `travel_model_freq` | `int` (`1`) | Year step interval between travel stages. |
| `vehicle_ownership_freq` | `int \| null` | Year step interval for ATLAS vehicle ownership (`null` defaults to `travel_model_freq`). |
| `supply_demand_iters` | `int` (`1`) | Number of ActivitySim/BEAM feedback iterations per travel year (`0` skips the loop). |
| `max_year_intervals` | `int \| null` (`null`) | Optional cap on the number of outer year intervals executed in a single run. |
| `use_stubs` | `bool` (`false`) | Replaces container execution with stubbed steps for testing. |
| `models` | `RunModelsConfig` | Selects active model backends: `land_use` (`urbansim`), `vehicle_ownership` (`atlas`), `activity_demand` (`activitysim`), and `travel` (`beam` or `polaris`). |

### Storage Paths and Restart Policy

| Field | Type / Default | Purpose |
| --- | --- | --- |
| `output_directory` | `str` (`output/runs`) | Parent directory for durable archive run directories. Supports environment variable expansion. |
| `output_run_name` | `str \| null` (`null`) | Custom prefix included in the generated `pilates-run--<region>--<name>--<timestamp>` directory. |
| `local_workspace_root` | `str \| null` (`null`) | Parent directory for the mutable run workspace. When `null`, the workspace lives inside the archive run directory. Supports environment variable expansion. |
| `recovery_archive_roots` | `list[str]` (`[]`) | Cold storage destinations (such as project NFS) for post-run promotion via `promote_run_archive`. |
| `enable_archive_copy` | `bool` (`true`) | Copies logged artifacts from the mutable workspace into the archive run directory during execution. |
| `restart_strict` | `bool` (`false`) | Fails fast after bootstrap on restart if required local workspace artifacts remain missing. |

### Consist Provenance and Caching Controls

| Field | Type / Default | Purpose |
| --- | --- | --- |
| `bootstrap_cache_enabled` | `bool` (`false`) | Enables Consist cache lookups during the pre-scenario bootstrap staging steps. |
| `consist_db_local_run` | `bool` (`true`) | Stores the active DuckDB provenance database inside the run directory (`.consist/<consist_db_filename>`) instead of writing to `shared.database.path`. |
| `consist_db_filename` | `str` (`provenance.duckdb`) | Basename for the run-local DuckDB file (must not contain path separators). |
| `consist_db_snapshot_enabled` | `bool` (`true`) | Snapshots the active provenance DB to `.consist/snapshots/` during execution and shutdown. |
| `consist_db_snapshot_interval_seconds` | `int` (`3600`) | Minimum elapsed seconds between periodic DB snapshots (`0` snapshots at every check). |
| `consist_db_snapshot_on_outer_iteration` | `bool` (`true`) | Writes a DB snapshot at the end of each outer simulation year. |
| `consist_db_snapshot_keep_last` | `int` (`5`) | Number of timestamped snapshots retained under `.consist/snapshots/history/`. |
| `consist_db_restore_on_start` | `bool` (`true`) | Restores the run-local DB from the latest archived snapshot when restarting an existing run. |
| `consist_db_restore_strict` | `bool` (`false`) | Raises an error if DB snapshot restore fails during restart startup. |
| `consist_db_seed_from_shared_on_start` | `bool` (`false`) | Seeds a new run-local DB from `shared.database.path` at startup so historical runs are available for cache hits and `prior_run` admission checks. |
| `consist_db_seed_strict` | `bool` (`false`) | Raises an error if seeding from `shared.database.path` fails. |
| `consist_code_identity` | `str` (`repo_git`) | Consist code-versioning mode (`repo_git`, `callable_module`, or `callable_source`). |
| `consist_hashing_strategy` | `str` (`fast`) | Artifact digest strategy (`fast` metadata-aware hashing or `full` byte hashing). |

## `shared` Section

- `geography`: Regional `FIPS` map, `local_crs` projection string, primary `zones` definition (`zone_type`, `source`), and optional `alternative_zones` map.
- `skims`: Skim filenames (`skims_fname`, `taz_skims_fname`, `beam_Wait_skims_fname`), `skims_zone_type`, highway and transit path labels, and time `periods`.
- `database`: Shared provenance catalog configuration (`enabled: bool`, `path: str`, `strict_Read: bool`). The loader expands environment variables in `database.path`.

## `infrastructure` Section

- `container_manager`: Container runtime (`docker` or `singularity`).
- `docker_images`: Map of model names (`urbansim`, `atlas`, `activitysim`, `beam`, `polaris`) to Docker image tags.
- `singularity_images`: Map of model names to Singularity/Apptainer `.sif` paths or URIs.
- `docker_config`: Controls `enable_logging` (`true`) and `pull_latest` (`false`).

## Model Sections (`models`)

- `urbansim`: Land-use container settings, data directories (`local_data_input_folder`, `local_mutable_data_folder`, `local_output_folder`), `form_to_btype`, and optional `admission.initial_datastore` validation policy.
- `atlas`: Vehicle-ownership container settings (`host_input_folder`, `host_mutable_input_folder`, `host_output_folder`, `beamac`, `num_processes`, `nsample`). Setting `beamac > 0` requires `run.region: sfbay` and `shared.skims.skims_zone_type: taz`.
- `activitysim`: Demand model folders (`local_input_folder`, `local_configs_folder`, `local_mutable_data_folder`, `local_mutable_configs_folder`, `local_output_folder`), `households_sample_size`, `chunk_size`, `num_processes`, `replan_hh_samp_size`, `canonical_zones_source`, and optional `admission.initial_skims` validation policy.
- `beam`: Travel simulation configuration (`local_input_folder`, `local_mutable_data_folder`, `local_output_folder`, `memory`, `warmstart_linkstats`, `able_to_replan`, `max_iters`), optional `admission.linkstats` policy, and `artifact_formats` for derived BEAM outputs (defaults: `activitysim_skims: zarr`, `exchange: parquet`, `events: parquet`, `linkstats: parquet`). When UrbanSim or ATLAS requires OMX skims, PILATES derives an OMX skim file alongside the Zarr skims store.
- `postprocessing`: Output validation settings (`enabled`, `fail_on_error`, `local_output_folder`, `validation_output_subfolder`).

## Related Pages

- Read [Configuration Basics](../start-here/configuration_basics.md) for the high-level layout and workspace split.
- Check [CLI](cli.md) for command-line flags and config validation commands.
- See [Scenario Lifecycle](scenario_lifecycle.md) for startup, bootstrap, and restart mechanics.
