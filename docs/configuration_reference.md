# Configuration Reference

PILATES configuration is a YAML file loaded and validated as a Pydantic model
(`pilates.config.models.PilatesConfig`).

This doc describes the **current canonical hierarchical config schema**
(`run`, `shared`, `infrastructure`, plus model sections like `beam`), and how
config choices affect:

- what the workflow runs (enabled models and schedule)
- container execution (docker vs singularity + images)
- caching/provenance identity (Consist step identity config + hashed inputs)

## Quick Pointers

- Canonical (hierarchical) configs live under `scenarios/<region>/...yaml`.
  Example: `scenarios/seattle/settings-seattle-newconfig-local.yaml`.
- A legacy flat schema exists in some files (e.g., `settings.yaml`) as migration
  residue. New runs should prefer the hierarchical schema.
  Some runtime code paths still consult legacy keys as a compatibility layer,
  but `load_config(...)` validates the hierarchical schema.

## File Format And Validation

- Configs are YAML. The loader is `pilates.config.models.load_config(...)`.
- Enabled models are validated:
  - if `run.models.travel` is set, a `beam:` section must exist
  - if `run.models.activity_demand` is set, an `activitysim:` section must exist
  - similarly for `urbansim:` and `atlas:`
- Environment variable expansion:
  - `run.output_directory` expands env vars (e.g., `$USER`)
  - `run.local_workspace_root` expands env vars
  - `shared.database.path` expands env vars
- Basic sanity checks:
  - `run.end_year` must be `>= run.start_year`
  - `run.consist_db_filename` must be a basename (not a path)

## Config Shape At A Glance (Hierarchical)

PILATES expects a hierarchical YAML with top-level sections:

- `run`
- `shared`
- `infrastructure`
- optional model sections: `urbansim`, `atlas`, `activitysim`, `beam`, `postprocessing`

In practice, the best starting point is to copy an existing scenario config
under `scenarios/<region>/...yaml` and edit it.

```yaml
run:
  region: seattle
  scenario: base
  start_year: 2018
  end_year: 2019
  land_use_freq: 1
  travel_model_freq: 1
  vehicle_ownership_freq: 1
  supply_demand_iters: 1
  output_directory: /path/to/output
  output_run_name: my-run
  use_stubs: false
  models:
    land_use: null
    travel: beam
    activity_demand: activitysim
    vehicle_ownership: null

shared:
  geography:
    FIPS: { state: "53", counties: ["033"] }
    local_crs: EPSG:32048
    zones:
      zone_type: block_group
      source_file: pilates/activitysim/data/seattle/block_groups_seattle_4326.geojson
      canonical_id_col: OBJECTID
      activitysim_index_col: TAZ
      source_crs: EPSG:4326
  skims:
    fname: as-base-skims-seattle-bg.omx
    origin_fname: as-origin-skims-seattle-bg.csv.gz
    hwy_paths: [SOV]
    periods: [AM]
    transit_paths: {}
  database:
    enabled: false
    type: duckdb
    path: pilates/database/pilates_data.duckdb

infrastructure:
  container_manager: docker
  docker_images: {}
  singularity_images: {}
  docker_config: { stdout: false, pull_latest: false }
```

Notes:
- If a model is enabled in `run.models`, its model section is required and must
  include its required fields.
- Example configs for Seattle/SF Bay are in:
  - `scenarios/seattle/settings-seattle-newconfig-local.yaml`
  - `scenarios/sfbay/settings-sfbay-newconfig-local.yaml`

### Minimal Valid BEAM/ActivitySim Sections (Example)

Below is a representative minimal set of required fields for BEAM and ActivitySim.
Treat this as an example; actual required values are region-specific.

```yaml
activitysim:
  household_sample_size: 0
  chunk_size: 12000000000
  num_processes: 25
  file_format: parquet
  local_input_folder: pilates/activitysim/data/
  local_mutable_data_folder: activitysim/data/
  local_output_folder: activitysim/output/
  local_configs_folder: pilates/activitysim/configs/
  local_mutable_configs_folder: activitysim/configs/
  validation_folder: pilates/activitysim/validation
  command_template: --households_sample_size {0} -m {1} -g {2}
  final_plans_folder: pilates/activitysim/output/final_plans

beam:
  config: seattle-pilates.conf
  sample: 1.0
  replanning_portion: 0.4
  memory: ${BEAM_MEMORY}
  local_input_folder: pilates/beam/production/
  local_mutable_data_folder: beam/input/
  local_output_folder: beam/beam_output/
  scenario_folder: urbansim/
  router_directory: r5/<region-router-dir>
  skims_shapefile: shape/<zones>.shp
  skim_zone_source_id_col: <colname>
  skim_zone_geoid_col: <colname>
```

## `run`: Run Configuration

`run` governs scenario identity, scheduling, output locations, restart behavior,
and Consist DB behavior.

Required:

- `run.region`: region identifier (e.g., `seattle`, `sfbay`)
- `run.scenario`: scenario name
- `run.start_year`, `run.end_year`: inclusive year range
- `run.output_directory`: where the run workspace is written
- `run.output_run_name`: human-readable run label
- `run.models`: which models are active (see below)

Scheduling and orchestration:

- `run.supply_demand_iters`: outer iterations for the travel-demand feedback loop
- `run.land_use_freq`: how often land use runs (years)
- `run.travel_model_freq`: how often travel runs (years)
- `run.vehicle_ownership_freq`: how often vehicle ownership runs (years)
- `run.use_stubs`: use stub model implementations (primarily for tests)

Output/workspace behavior:

- `run.local_workspace_root` (optional): node-local scratch workspace root
  - when unset, defaults to `run.output_directory`
- `run.enable_archive_copy`: mirror logged outputs into an archive root during execution

Restart and hydration behavior:

- `run.bootstrap_cache_enabled`: allow cache probing for the pre-scenario bootstrap phase
- `run.restart_strict`: fail startup if required restart artifacts are missing after hydration/preflight

Consist DB behavior (run-local DB mirroring and snapshots):

- `run.consist_db_local_run`: store the Consist DB in the node-local run dir and mirror to archive at shutdown
- `run.consist_db_filename`: basename for the run-local Consist DB (default `provenance.duckdb`)
- `run.consist_db_snapshot_enabled`: enable periodic snapshots
- `run.consist_db_snapshot_interval_seconds`: minimum seconds between snapshots
- `run.consist_db_snapshot_on_outer_iteration`: snapshot at supply-demand iteration boundaries
- `run.consist_db_snapshot_keep_last`: number of snapshots to keep
- `run.consist_db_restore_on_start`: restore local Consist DB from latest archived snapshot if missing
- `run.consist_db_restore_strict`: fail if restore fails when restore is enabled
- `run.consist_db_seed_from_shared_on_start`: seed run-local DB from `shared.database.path` when no snapshot restore is available
- `run.consist_db_seed_strict`: fail if seed-from-shared fails when enabled

Consist identity and hashing controls:

- `run.consist_code_identity` (optional): override Consist code identity mode
- `run.consist_hashing_strategy`: `fast` (mtime/size) or `full` (content hashing)

### `run.models`: Model Selection

`run.models` selects which model adapters are active. When you enable a model
here, you must also provide the matching top-level config section.

- `run.models.land_use`: `urbansim` or `null`
  - requires `urbansim:`
- `run.models.vehicle_ownership`: `atlas` or `null`
  - requires `atlas:`
- `run.models.activity_demand`: `activitysim` or `null`
  - requires `activitysim:`
- `run.models.travel`: `beam` (or `polaris` in some deployments) or `null`
  - requires `beam:` for BEAM

## `shared`: Shared Configuration

`shared` defines cross-cutting inputs used by multiple steps, and it is a
critical part of run identity.

### `shared.geography`

Shared geography configuration.

- `shared.geography.FIPS`: study area FIPS codes (state + counties)
- `shared.geography.local_crs`: local projected CRS used for geo operations
- `shared.geography.zones` (optional in some tests): canonical zone system definition
- `shared.geography.alternative_zones` (optional): fallback zone source metadata

Zones config (`shared.geography.zones`):

- `zone_type`: `taz`, `block_group`, etc.
- `source_file`: canonical zone geometry file (GeoJSON/shapefile)
- `canonical_id_col`: column containing the canonical zone id
- `activitysim_index_col`: column used for ActivitySim's internal 0-based index
- `source_crs` (optional): CRS hint/override for the source file

Alternative zones (`shared.geography.alternative_zones`) uses the same structure
as zones, and is intended as a portability/fallback source when the primary
zone source is unavailable (e.g., different mount layouts).

### `shared.skims`

Shared skim configuration. These values define baseline impedances and drive
both ActivitySim and BEAM integration.

- `fname`: base skim file name (OMX or other supported format)
- `origin_fname` (optional): origin skim file name
- `hwy_paths`: list of highway path types (e.g., `SOV`, `HOV2TOLL`)
- `periods`: list of time periods (e.g., `EA`, `AM`, `MD`, `PM`, `EV`)
- `transit_paths` (optional): transit path definitions and measures to extract

### `shared.database`

Database configuration for the optional database-backed workflow and for
inspection/documentation outputs.

- `enabled`: turn database mode on/off
- `type`: database type (currently `duckdb`)
- `path`: database file path
- `snapshot_path` / `shapshot_path`: legacy spelling compatibility for snapshot path

## `infrastructure`: Containers And Images

`infrastructure` controls which container runtime is used and which images to run.

- `container_manager`: `docker` or `singularity`
- `singularity_images`: mapping `{ model_name: image_uri }`
- `docker_images`: mapping `{ model_name: image_tag }`
- `docker_config.stdout`: show container stdout
- `docker_config.pull_latest`: pull latest images before running

Model keys in the image maps are typically: `urbansim`, `atlas`, `activitysim`, `beam`.

## Model-Specific Sections

These sections contain per-model adapter configuration. They are required only
when the corresponding model is enabled in `run.models`.

### `urbansim`

UrbanSim configuration is largely about:

- where input data lives (host-side paths)
- where mutable work-in-progress state is written
- how the container command is constructed

Key fields:

- `region_id`: region identifier used in input templates
- `local_data_input_folder`: baseline input folder on the host
- `local_mutable_data_folder`: mutable data folder on the host
- `client_base_folder`, `client_data_folder`: container-side folders
- `input_file_template`, `input_file_template_year`, `output_file_template`
- `command_template`: how PILATES invokes UrbanSim inside the container
- `region_mappings`: region-specific mappings (commonly region -> region_id)

### `atlas`

ATLAS configuration includes explicit host vs container paths and scenario knobs.

Key fields:

- host-side paths:
  - `host_input_folder`, `warmstart_input_folder`, `host_mutable_input_folder`, `host_output_folder`
- container-side paths:
  - `container_input_folder`, `container_output_folder`
- execution/control:
  - `max_retries`, `sample_size`, `num_processes`
  - `beamac`, `mod`, `scenario`, `adscen`, `rebfactor`, `taxfactor`, `discIncent`
  - `command_template`

Operational constraint enforced by startup validation:
- If `atlas.beamac > 0`, PILATES currently requires `run.region == "sfbay"` and
  `shared.geography.zones.zone_type == "taz"`.

### `activitysim`

ActivitySim configuration covers sampling, parallelism, data/config locations,
and optional database input mode.

Key fields:

- performance/sampling:
  - `household_sample_size` (0 = full population)
  - `chunk_size`, `num_processes`
  - `file_format`: `parquet` or `csv`
  - `persist_sharrow_cache` (optional)
- host-side paths:
  - `local_input_folder`, `local_mutable_data_folder`, `local_output_folder`
  - `local_configs_folder`, `local_mutable_configs_folder`
  - `validation_folder`
- integration/settings:
  - `command_template`
  - `from_urbansim_col_maps`, `to_urbansim_col_maps`
  - `output_tables` (what to publish/track from ActivitySim outputs)
  - `clipped_geoms_path` (optional): constrain activity locations to BEAM geoms
- replanning:
  - `warm_start_activities`
  - `replan_iters`, `replan_hh_samp_size`, `replan_after`
  - `random_seed` (optional)
- database input mode:
  - `database.enabled`
  - `database.use_processed_data`
  - `database.year` (optional)
- output:
  - `final_plans_folder`

### `beam`

BEAM configuration covers the BEAM config file selection, sampling, JVM memory,
and a set of paths and integration mapping fields.

Key fields:

- `config`: main BEAM `.conf` file name
- `sample`: population fraction (`0.0` to `1.0`)
- `replanning_portion`: fraction replanned each iteration
- `memory`: JVM memory string (e.g., `180g`)
- `local_input_folder`, `local_mutable_data_folder`, `local_output_folder`
- `scenario_folder`, `router_directory`
- `warmstart_linkstats_path` (optional)
- zone/skim metadata:
  - `skims_shapefile`
  - `skim_zone_source_id_col`
  - `skim_zone_geoid_col`
- plan/state behavior:
  - `discard_plans_every_year`
  - `max_plans_memory`
- skim merging behavior:
  - `skim_previous_weight`
- optional full-skim mode:
  - `full_skim` (run schedule, router type, modes, etc.)

### `postprocessing`

Postprocessing configuration is used for downstream validation/summary steps.

Key fields:

- `output_folder`
- `mep_output_folder`
- `scenario_definitions` (optional)
- `validation_metrics` (optional)

## How Config Affects Workflow Behavior

At a high level:

- **What runs** is decided by `run.models` plus model frequencies (`*_freq`) and
  `run.supply_demand_iters`.
- **Where it runs** is decided by `infrastructure.container_manager` and the
  image mappings.
- **What gets (re)used on restart** is influenced by tracker-based completed-run
  reconstruction, bootstrap behavior, and whether required artifacts are present.

See `docs/workflow_primer.md` for the step/stage lifecycle.

## How Config Affects Hashing/Provenance (Consist)

PILATES uses Consist for provenance and caching. Each scenario and step has:

- an **identity config** (`config`): affects cache identity/hits
- a **facet** (`facet`): queryable metadata, intended not to affect cache identity
- optional **identity inputs** (`identity_inputs`): on-disk files/dirs hashed as part of identity

### Scenario And Initialization Identity

Scenario-level identity config comes from `PilatesConfig.get_initialization_signature()`.
It intentionally focuses on the "logical world" (space/time/topology), not
output locations.

It includes:

- run context: `run.region`, `run.scenario`, `run.start_year`
- shared geography: `shared.geography`
- shared initial conditions pointer: `shared.skims`
- orchestration topology: enabled models, frequencies, supply-demand iters, `use_stubs`

### Step Identity (What Changes Cache Identity)

Step identity config is built per model. The current mappings are:

- ActivitySim:
  - identity config includes sampling/parallelism/replanning knobs and ActivitySim DB input mode
  - identity inputs include the mutable ActivitySim configs directory inside the workspace
- BEAM:
  - identity config includes `config`, `sample`, `memory`, `router_directory`, `scenario_folder`, etc.
  - identity inputs include BEAM `.conf` files under the mutable BEAM input root (when present)
- UrbanSim:
  - identity config includes templates/command shape and region id
  - facet stores the full UrbanSim config for query/debug
- ATLAS:
  - identity config includes key scenario knobs and the command template
  - facet stores the full ATLAS config for query/debug

Hashing strategy:

- `run.consist_hashing_strategy = fast` uses file metadata (mtime/size)
- `run.consist_hashing_strategy = full` hashes file contents

## Common Override Recipes

Switch container runtime:

```yaml
infrastructure:
  container_manager: docker  # or singularity
```

Change run output location:

```yaml
run:
  output_directory: /some/path
  output_run_name: my-run
```

Disable a model:

```yaml
run:
  models:
    land_use: null
```

Turn on stub models for tests or fast smoke runs:

```yaml
run:
  use_stubs: true
```

Enable database mode:

```yaml
shared:
  database:
    enabled: true
    type: duckdb
    path: pilates/database/pilates_data.duckdb
```

## Related Docs

- `docs/getting_started.md`
- `docs/workflow_primer.md`
- `docs/model_integration_guide.md`
- `docs/zone_id_management.md`
