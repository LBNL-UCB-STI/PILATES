# Glossary

## Workflow and Provenance Terms

| Term | Definition |
| --- | --- |
| **Scenario** | A complete multi-year or single-year simulation run managed by a Consist `ScenarioContext`, combining configuration, step execution history, and the artifact coupler. |
| **Stage** | A major phase of the simulation year loop (`land_use`, `vehicle_ownership_model`, `supply_demand_loop`, `postprocessing`) implemented in `pilates/workflows/stages/`. |
| **Step** | A single tracked unit of execution inside a stage (such as `urbansim_run`, `atlas_preprocess`, `activitysim_run`, or `beam_postprocess`), defined by a `StepDefinition` in `pilates/workflows/steps/`. |
| **Coupler** | The typed key-value registry (`scenario.coupler`) that passes validated `Artifact` references from upstream steps to downstream steps. See [Artifact Flow](../workflow/artifact_flow.md). |
| **Artifact** | A versioned file, directory, HDF5 store, Zarr group, or DuckDB table logged with Consist and addressed by a semantic coupler key and portable container URI. |
| **Facet** | A structured dictionary of configuration parameters or artifact metadata indexed in `provenance.duckdb` (`facet_index=True`) for cross-run queries. |
| **Execution Workspace** | The mutable scratch directory (`run.local_workspace_root`) where steps stage mutable configs and containers mount input and output folders during execution. See [Workspace Layout](workspace_layout.md). |
| **Archive Root** | The durable output directory (`run.output_directory`) containing per-year/iteration outputs and optional `consist-recovery/` step mirrors (`RunCompletionArchivePolicy`). |
| **Recovery Root** | A read-only historical archive directory listed in `run.recovery_archive_roots` that supplies cached outputs for Consist hydration (`cache_hydration="outputs-requested"`). |
| **Epoch** | A `(year, iteration)` coordinate within a simulation archive representing either a forecast year's land-use step or a demand-supply replanning pass. |
| **AnalysisSession** | The multi-root discovery interface in `pilates_consist_analysis` that scans directories for `provenance.duckdb` files and filters runs into a `RunSet`. See [Analysis Overview](../analysis/overview.md). |
| **RunSet** | A filtered collection of comparable runs supporting aligned epoch panels, configuration diffs, and baseline delta calculations. |

## Transportation Co-Simulation Terms

| Term | Definition |
| --- | --- |
| **TAZ / MAZ** | Traffic Analysis Zone / Micro Analysis Zone. Geographic polygons used to aggregate land-use attributes and index origin-destination skim matrices. See [Zone and Skim Alignment](zone_and_skim_alignment.md). |
| **Skims** | Origin-destination matrices of network impedance (travel time, distance, toll cost, transit wait time, and accessibility) segmented by mode and time of day. |
| **OMX** | OpenMatrix, an HDF5-based file format (`*.omx`) used to store 2D origin-destination skim matrices. |
| **Zarr Skims** | Multi-dimensional chunked array representation (`*.zarr`) of origin-destination skims with 0-based canonical zone coordinates used for fast slicing in ActivitySim and UrbanSim. |
| **Linkstats** | Per-hour network link volume, speed, capacity, and travel time summaries (`beam_linkstats`) emitted by BEAM at the end of each supply iteration. |
| **Replanning Iteration** | A demand-supply feedback cycle within a single forecast year where ActivitySim re-runs travel demand using updated BEAM skims and BEAM re-simulates network supply. |
| **Warm Start** | Initializing a simulation year or stage from pre-computed skims, linkstats, or population stores hydrated from a prior run or recovery archive. See [Step Contracts](../workflow/step_contracts.md). |
