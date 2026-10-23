# Artifact Flow, Roles & Facets

PILATES passes data between models through `scenario.coupler`, a key-value registry on the active Consist `ScenarioContext`. Rather than passing ad-hoc file paths between stages, steps publish and consume named artifacts defined in `pilates/workflows/artifact_keys.py`. This page documents the artifact flow across stages, the semantic roles of canonical coupler keys, and the run and artifact facet schemas stored in DuckDB.

## How Artifacts Move Through `scenario.coupler`

Each artifact in PILATES follows a four-step lifecycle:

1. **Bootstrap registration**: Before the simulation year loop starts, `run_bootstrap_phase()` registers initial seed artifacts in `scenario.coupler` (including `usim_datastore_h5`, `usim_datastore_base_h5`, `omx_skims`, `asim_mutable_data_dir`, `beam_mutable_data_dir`, and optional warm-start files).
2. **Role-based binding**: When `execute_step()` starts a step, `resolve_inputs()` evaluates the step's `ArtifactBindingRule` list against `scenario.coupler` and materializes the selected artifacts into the workspace (`resolved_inputs.logical_destinations`).
3. **Immutable archiving**: After `scenario.run()` completes, `archive_completed_run()` copies the step's declared outputs from the working directory into the immutable year/iteration archive under `<output_folder>/<year>/` (or `<output_folder>/<year>/iter_<iteration>/`).
4. **Coupler publication**: `execute_step()` binds each archived `Artifact` into `scenario.coupler` via `set_coupler_from_artifact()`, overwriting the key with the produced artifact for downstream steps in the same year or subsequent years.

For step-by-step file paths and producer-consumer tables, see the [Lineage Map](lineage_map.md).

## Canonical Coupler Keys and Semantic Roles

All canonical coupler keys live in `pilates/workflows/artifact_keys.py` (`ArtifactKeys`). Several keys point to files with similar formats (such as HDF5 datastores, OMX skims, or BEAM plan XML files) that serve distinct workflow roles.

### UrbanSim Datastores and Population Tables

| Coupler Key | Producer | Primary Consumers | Semantic Role |
| :--- | :--- | :--- | :--- |
| `usim_datastore_h5` (`USIM_DATASTORE_CURRENT_H5`) | `initialize_data`, `postprocessing` | `urbansim_run`, `atlas_preprocess`, `activitysim_preprocess` | Active UrbanSim HDF5 datastore for the start of the current simulation year. `USIM_DATASTORE_H5` is a compatibility alias for this key. |
| `usim_datastore_base_h5` | `initialize_data` | `urbansim_postprocess`, `atlas_preprocess`, `postprocessing` | Immutable base-year UrbanSim HDF5 datastore registered at bootstrap. Used as a fallback source for static geometry or base-year tables absent from forecast outputs. |
| `usim_forecast_output` | `urbansim_run` | `urbansim_postprocess` | Raw `model_data_{forecast_year}.h5` file emitted by the UrbanSim container before table extraction. |
| `usim_population_source_h5` | `urbansim_postprocess` | `atlas_preprocess`, `activitysim_preprocess`, `postprocessing` | Postprocessed UrbanSim HDF5 datastore for the forecast year prior to ATLAS vehicle updates. |
| `usim_h5_updated` | `atlas_postprocess` | `atlas_preprocess` (subsequent sub-years), `activitysim_preprocess`, `postprocessing` | UrbanSim HDF5 datastore enriched with ATLAS household vehicle counts (`cars`, `vehicl`) and the `atlas_vehicles` table. |
| `usim_population_households_table`, `usim_population_persons_table`, `usim_population_jobs_table`, `usim_population_blocks_table` | `urbansim_postprocess` | Analytics and downstream table inspection | Extracted population and land-use tables (`households`, `persons`, `jobs`, `blocks`) logged as individual artifacts. |
| `usim_input_next` / `usim_input_archive_{forecast_year}` | `postprocessing` | Next simulation year (`usim_datastore_h5`) | Assembled next-year UrbanSim HDF5 input datastore archived at the end of year $t + \Delta t$. |

### ATLAS Vehicle Ownership Artifacts

| Coupler Key | Producer | Primary Consumers | Semantic Role |
| :--- | :--- | :--- | :--- |
| `atlas_output_dir` | `atlas_run` | `atlas_postprocess` | Working directory containing raw ATLAS sub-year outputs. |
| `atlas_vehicles2_output` | `atlas_run` | `atlas_postprocess`, `beam_preprocess` | Detailed household vehicle fleet table (`vehicles_{atlas_year}.csv`) with body type, fuel/powertrain, and vintage attributes. |
| `atlas_vehicles2_input` | `atlas_postprocess` | `beam_preprocess` | Archived vehicle fleet table staged for downstream travel simulation. |

### ActivitySim Demand Artifacts

| Coupler Key | Producer | Primary Consumers | Semantic Role |
| :--- | :--- | :--- | :--- |
| `asim_mutable_data_dir` | `initialize_data` | `activitysim_preprocess`, `activitysim_run` | Mutable ActivitySim configuration directory copied into the workspace at bootstrap. |
| `land_use_asim_in`, `households_asim_in`, `persons_asim_in` | `activitysim_preprocess` | `activitysim_run` | Preprocessed CSV input tables (`land_use.csv`, `households.csv`, `persons.csv`) formatted for ActivitySim. |
| `asim_output_dir` | `activitysim_run`, `activitysim_postprocess` | `activitysim_postprocess`, `beam_preprocess` | Directory containing ActivitySim output tables (`plans.csv`, `households.csv`, `persons.csv`, `tours.csv`, `trips.csv`). |
| `asim_sharrow_cache_dir` | `activitysim_run` | Subsequent `activitysim_run` calls | Optional compiled Sharrow expression cache directory when Sharrow acceleration is enabled. |

### BEAM Traffic Assignment and Warm-Start Artifacts

| Coupler Key | Producer | Primary Consumers | Semantic Role |
| :--- | :--- | :--- | :--- |
| `beam_mutable_data_dir` | `initialize_data` | `beam_preprocess`, `beam_run`, `beam_full_skim` | Mutable BEAM scenario and configuration directory in the workspace. |
| `plans_beam_in`, `households_beam_in`, `persons_beam_in`, `vehicles_beam_in` | `beam_preprocess` | `beam_run` | Active BEAM scenario input CSVs staged for the current year and inner iteration. |
| `beam_input_plans_archived`, `beam_input_households_archived`, `beam_input_persons_archived`, `beam_input_vehicles_archived`, `beam_input_config_archived`, `beam_input_config_references_archived` | `beam_preprocess` | Provenance archive | Immutable archived snapshots of the exact inputs and HOCON config bundle prepared by `beam_preprocess`. |
| `linkstats_warmstart` | `initialize_data`, `beam_postprocess` | `beam_preprocess`, `beam_run` | Prior-year or bootstrap link travel-time statistics (`*.linkstats.csv.gz`) used to warm-start BEAM network assignment. |
| `beam_output_dir` | `beam_run` | `beam_postprocess` | Raw BEAM iteration output directory containing sub-iteration subfolders (`ITERS/it.<n>/`) and top-level summaries. |
| `linkstats` | `beam_postprocess` | `beam_postprocess` (warmstart update), analysis | Final iteration link statistics (`{last_iter}.linkstats.csv.gz`) promoted from `beam_run`. |
| `beam_network_final` | `beam_postprocess` | Analysis and network inspection | Final road and transit network table (`network.csv.gz`) emitted by BEAM. |
| `beam_plans_out` / `beam_output_plans_xml` | `beam_postprocess` | `beam_preprocess` (next iteration/year) | Merged agent plans (`plans.csv.gz` / `output_plans.xml.gz`) carried forward across replanning iterations. |
| `beam_experienced_plans_xml` / `beam_output_experienced_plans_xml` | `beam_postprocess` | `beam_preprocess` (next iteration/year) | Experienced plans with simulated travel times and scores (`experiencedPlans.xml.gz` / `output_experienced_plans.xml.gz`) used by plan-memory selection in `beam_preprocess`. |

### Cross-Model Skim Keys

| Coupler Key | Producer | Primary Consumers | Semantic Role |
| :--- | :--- | :--- | :--- |
| `omx_skims` (`OMX_SKIMS` / `ASIM_OMX_SKIMS`) | `initialize_data`, `beam_postprocess` | `urbansim_run`, `atlas_preprocess`, ` atlas_run`, `activitysim_preprocess`, `activitysim_run` | Active cross-model zone-to-zone OMX travel skim matrix (`skims.omx`). Initialized at bootstrap and updated after each `beam_postprocess` execution. |
| `final_skims_omx` | `beam_postprocess` | Coupler (`omx_skims`), analysis | Year-and-iteration-stamped OMX skim archive produced by converting BEAM skims at the end of a BEAM pass. |
| `zarr_skims` | `beam_postprocess` | Analysis | Multi-dimensional Zarr skim store (`skims_{year}_{iteration}.zarr`) preserving fine-grained BEAM skim dimensions. |
| `beam_full_skims` | `beam_full_skim` | Analysis | Optional full-origin-destination OMX skim matrix (`full_skims_{year}.omx`) produced when `beam.full_skim.enabled` is true. |

## Run and Step Facet Catalog

PILATES attaches structured **facets** (`facet`, `facet_schema_version`, and `facet_index=True`) to both scenario headers and individual step runs via `pilates/utils/consist_config.py` and `pilates/workflows/step_consist_meta.py`. Facets are indexed in DuckDB (`provenance.duckdb`) for fast SQL and Python filtering without altering cache hashes.

### Scenario and Step Facet Schema Versions

| Scope / Step Family | `facet_schema_version` | Key Facet Fields |
| :--- | :--- | :--- |
| Scenario header (`tracker.scenario`) | `pilates_scenario_v1` | `run` (`PilatesRunConfig.to_consist_facet()`, including `scenario_id`, `region`, `start_year`, `end_year`, `travel_model_only`, stage enablement flags). |
| Bootstrap (`initialization`) | `initialization_v1` | Same `run` facet payload as the parent scenario header. |
| UrbanSim (`urbansim_run`, `urbansim_postprocess`) | `urbansim_run_v1`, `urbansim_postprocess_v1` | Full `UrbanSimConfig` dump (`region_id`, templates, admission config) plus `native_input_contract`. |
| ATLAS (`atlas_preprocess`, `atlas_run`, `atlas_postprocess`) | `atlas_preprocess_v1`, `atlas_run_v1`, `atlas_postprocess_v1` | Full `AtlasConfig` dump plus runtime identity fields: `atlas_subyear`, `main_forecast_year`, `atlas_usim_households_sha256`, `atlas_usim_households_table_path`, `atlas_usim_households_row_count`, `atlas_usim_households_column_count`. |
| ActivitySim (`activitysim_preprocess`, `activitysim_run`, `activitysim_postprocess`) | `activitysim_preprocess_v1`, `activitysim_run_v1`, `activitysim_postprocess_v1` | `ActivitySimConfig.to_consist_facet()` (`household_sample_size`, `chunk_size`, `num_processes`, `replan_iters`, `replan_hh_samp_size`, `replan_after`, `random_seed`). |
| BEAM (`beam_preprocess`, `beam_run`, `beam_postprocess`, `beam_full_skim`) | `beam_preprocess_v1`, `beam_run_v1`, `beam_postprocess_v1`, `beam_full_skim_v1` | `BeamConfig.to_consist_facet()` (`config`, `sample`, `replanning_portion`, `memory`, plan memory parameters, `write_skims_to_omx`, `full_skim`). |
| Postprocessing (`postprocessing`) | `postprocessing_v1` | `PostprocessingConfig` dump. |

In addition to the JSON `facet` dictionary, Consist stores first-class run columns on every `Run` row in DuckDB, including `id`, `model_name`, `year`, `iteration`, `parent_run_id` (the scenario ID), and `status`.

## Artifact Facet Catalog

PILATES also attaches artifact-level facets (`metadata={"facet": ..., "facet_schema_version": ..., "facet_index": True}`) to specific multi-iteration output artifacts so analysts can filter artifacts with `consist.find_artifacts(tracker=..., facet={...})` instead of parsing filenames.

### `linkstats_unmodified_phys_sim_iter_parquet`

`pilates/beam/runner.py` logs unmodified physical-simulation linkstats Parquet files produced inside BEAM sub-iterations with `facet_schema_version="v1"` and `facet_index=True`.

#### Artifact Key Conventions

- **Promoted final BEAM sub-iteration (`it == last_iter`)**:
  `linkstats_unmodified_parquet__y{year}__i{iteration}__phys_sim_iter{N}`
- **Intermediate BEAM sub-iteration (`it < last_iter`)**:
  `linkstats_unmodified_parquet__y{year}__i{iteration}__phys_sim_iter{N}__beam_sub_iter{M}`

#### Facet Fields

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `artifact_family` | `str` | Constant `"linkstats_unmodified_phys_sim_iter_parquet"`. |
| `year` | `int` | Simulation forecast year (for example, `2018`). |
| `iteration` | `int` | PILATES supply-demand loop iteration (`WorkflowState.current_inner_iter`). |
| `phys_sim_iteration` | `int` | Physical simulation suffix `N` parsed from `*.physSimIter-{N}.linkstats.unmodified.parquet`. |
| `beam_sub_iteration` | `int` | BEAM internal `ITERS/it.{M}` sub-iteration index `M` (recorded on both intermediate and promoted final artifacts). |

#### Query Examples

```python
import consist

# Fetch all unmodified physSimIter linkstats Parquet artifacts for year 2018, iteration 0
artifacts = consist.find_artifacts(
    tracker=tracker,
    facet={
        "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet",
        "year": 2018,
        "iteration": 0,
    },
)

# Fetch a specific BEAM sub-iteration and physical simulation iteration
artifact = consist.find_artifact(
    tracker=tracker,
    facet={
        "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet",
        "year": 2018,
        "iteration": 0,
        "beam_sub_iteration": 1,
        "phys_sim_iteration": 2,
    },
)
```

## Related Documentation

- [Workflow Architecture](architecture.md): Control flow, stages, and simulation logic.
- [Step Contracts & Execution](step_contracts.md): Input binding rules, `execute_step()`, and Consist configuration channels.
- [Lineage Map](lineage_map.md): Complete step-by-step artifact producers, consumers, and file paths.
- [SQL & DuckDB Recipes](../analysis/sql_and_duckdb.md): Querying run and artifact facets in `provenance.duckdb`.
