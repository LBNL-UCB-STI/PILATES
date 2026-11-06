---
title: Scenario Lifecycle
summary: End-to-end runtime lifecycle from startup and coupler seeding through the year loop, restart and resume, and shutdown.
---

# Scenario Lifecycle

A PILATES run moves through five phases: startup preparation, Consist scenario context & coupler seeding, step-owned workspace preparation & the simulation year loop, restart & resume, and shutdown snapshotting.

## 1. Startup

At startup (`python run.py`), `pilates.runtime.launcher.main()` executes these steps via `_prepare_run_context()`:

1. Parses CLI flags (`-c/--config`, `-S/--stage`, `--allow-rewind-resume`) and loads the YAML configuration with Pydantic validation.
2. Initializes runtime flags (`ensure_runtime_flags_initialized`), constructs or restores `WorkflowState` from the `run_state.yaml` path passed via `-S/--stage`, and builds the `EnabledWorkflowSurface`, which determines the enabled stages, step contracts, and expected coupler outputs for the run.
3. Resolves the durable archive directory (`<run.output_directory>/pilates-run--<region>--<name>--<timestamp>/`) and the mutable execution directory (`<run.local_workspace_root>/pilates-run--.../` when set, or colocated inside the archive directory).
4. Initializes the Consist `Tracker` against the archive directory (`.consist/provenance.duckdb`), mounts repository inputs (`inputs`), the mutable workspace (`workspace` and `beam_input`), and shared scratch (`scratch`), and restores or seeds the run-local DuckDB when configured.
5. Creates the `Workspace` (`pilates/workspace.py`) inside the mutable run directory, sets up model working subdirectories, enforces restart guardrails (`_enforce_resume_rewind_guardrail` and `_rewind_uncommitted_activitysim_restart_to_provider_boundary`), and binds `run_state.yaml` paths in both the archive directory and the local workspace.

## 2. Scenario Context, Preflight Admission & Initial Coupler Seeding

After startup preparation, the launcher opens the Consist scenario context (`with cr.scenario(run_name, tracker=tracker, ...) as scenario:`), declares the coupler output schema (`coupler.declare_outputs(...)`), and prepares initial workflow artifacts:

- **Preflight inbound admission**: At the initial workflow boundary, evaluates optional `admission` policies (`preflight_bootstrap_urbansim_datastore_admission` for `urbansim.admission.initial_datastore` and `preflight_activitysim_initial_skims_admission` for `activitysim.admission.initial_skims`) before any model step executes.
- **Restart coupler restoration**: On mid-workflow restarts, `restore_restart_coupler_snapshot()` restores persisted coupler artifacts from `.consist/provenance.duckdb`, and `hydrate_restart_atlas_continuation_inputs()` restores prior-subyear ATLAS continuation inputs when needed.
- **Initial coupler seeding (`seed_bootstrap_artifacts_to_coupler`)**: Seeds initial static/staged artifacts into `scenario.coupler` via `bootstrap_stage_boundary_durability_policy()` in `pilates/workflows/binding.py`:
  - `urbansim_initial_datastore`: Publishes `usim_datastore_base_h5` and `usim_datastore_h5` (plus `usim_population_source_h5` when `land_use` is disabled) from the configured initial HDF5 datastore (`resolve_urbansim_initial_datastore_source_path`).
  - `initial_skims`: Publishes `omx_skims` (and `final_skims_omx` when BEAM is disabled) so cold-scratch and non-BEAM workflows resolve initial OMX skims through the coupler.
  - `beam_exchange_inputs`: For BEAM-only runs without ActivitySim, prepares the mutable BEAM workspace (`ensure_beam_mutable_workspace`) and seeds `plans_beam_in`, `households_beam_in`, and `persons_beam_in`.
  - `beam_warmstart`: Publishes `linkstats_warmstart` when configured and no prior BEAM linkstats handoff is required.
- **Initial coupler snapshot**: On fresh runs, captures `scenario.capture_coupler_snapshot(phase="init")` in Consist.

## 3. Step-Owned Workspace Preparation and the Year Loop

There is no separate pre-scenario `bootstrap` staging phase. Instead, each model owns its own idempotent mutable-workspace, configuration, and static-input preparation directly inside its step resolver or execution step:

- **UrbanSim (`land_use`)**: `stage_urbansim_run_workspace()` (`pilates/urbansim/preprocessor.py`) stages the bound `usim_datastore_h5`, static CSV inputs (`selected_urbansim_static_input_sources`), `geoid_to_zone.csv`, and skims into `<workspace>/urbansim/data/` immediately before `urbansim_run`, and `run_land_use_stage()` preserves `usim_datastore_base_h5` and `usim_population_source_h5` handoffs across `urbansim_run` and `urbansim_postprocess`.
- **ATLAS (`vehicle_ownership_model`)**: `AtlasPreprocessor` and `AtlasRunner` idempotently stage selected static ATLAS lookup files (`stage_selected_atlas_static_inputs` in `pilates/atlas/preprocessor.py`) into `<workspace>/atlas/atlas_input/` and export year-scoped CSV tables from the bound UrbanSim HDF5 datastore.
- **ActivitySim (`activity_demand`)**: Step resolvers in `pilates/workflows/steps/activitysim.py` call `ensure_activitysim_mutable_configs(settings, workspace)` idempotently before `activitysim_preprocess`, `activitysim_run`, and `activitysim_postprocess`.
- **BEAM (`traffic_assignment`)**: Step resolvers in `pilates/workflows/steps/beam.py` and `_compile_beam_launch_config()` call `ensure_beam_mutable_workspace(settings, workspace)` idempotently before `beam_preprocess`, `beam_run`, `beam_postprocess`, and `beam_full_skim`.

For each simulation year from `run.start_year` to `run.end_year`, the launcher executes the enabled stages in fixed order:

1. **Land Use (`land_use` / UrbanSim)**: Runs `urbansim_run` and `urbansim_postprocess` for each forecast interval, projecting households, persons, jobs, and blocks for `state.forecast_year` and creating the immutable `usim_population_source_h5` snapshot.
2. **Vehicle Ownership (`vehicle_ownership_model` / ATLAS)**: Runs `atlas_preprocess`, `atlas_run`, and `atlas_postprocess` across biannual `AtlasSubState` sub-years up to `state.forecast_year`, updating household vehicle ownership and fleet tables.
3. **Supply and Demand Loop (`supply_demand_loop`: `activity_demand` / ActivitySim and `traffic_assignment` / BEAM)**: Executes `run.supply_demand_iters` feedback iterations for the current travel year:
   - **ActivitySim**: Runs `activitysim_preprocess`, `activitysim_run`, and `activitysim_postprocess`, publishing demand outputs (`beam_plans_asim_out`, `households_asim_out`, `persons_asim_out`, `tours_asim_out`, `trips_asim_out`) and snapshotting the Consist DB at the ActivitySim boundary before BEAM begins.
   - **BEAM**: Runs `beam_preprocess`, compiles the iteration launch config (`_compile_beam_launch_config`), executes `beam_run`, and runs `beam_postprocess` (plus optional `beam_full_skim`) to publish `zarr_skims`, `final_skims_omx`, `linkstats`, and split event tables.
4. **Postprocessing (`postprocessing`)**: Runs output validation checks when enabled at the end of the simulation year.

Every step executes through `execute_step()` (`pilates/workflows/step_execution.py`), which resolves semantic inputs once, invokes `scenario.run()` with Consist's native `RunCompletionArchivePolicy` and `cache_hydration="outputs-requested"`, projects `RunResult.outputs` into validated `StepOutputsBase` instances, and updates `run_state.yaml`.

```mermaid
flowchart LR
    Inputs["Resolved Semantic Inputs"] --> Outputs["Consist scenario.run()\n+ RunCompletionArchivePolicy"]
    Outputs --> Projector["project_outputs()\n(StepOutputsBase)"]
    Projector --> Coupler["scenario.coupler"]
```

At the end of each stage and outer year, the launcher snapshots `.consist/provenance.duckdb` to `.consist/snapshots/latest/`.

## 4. Restart and Resume

PILATES coordinates stage-level restart policy through `WorkflowState` (`run_state.yaml`), while Consist manages artifact identity, coupler snapshots, cache admission, and output hydration.

### Normal Stage and Year Frontier Resume

To resume an interrupted run, pass the archived `run_state.yaml` path via `-S/--stage`:

```bash
python run.py -c path/to/settings.yaml -S /path/to/archive_run_dir/run_state.yaml
```

Outside a committed BEAM checkpoint, restart resumes at the next incomplete stage or year recorded in `run_state.yaml`. If a run stopped mid-ActivitySim before a committed BEAM checkpoint, `_rewind_uncommitted_activitysim_restart_to_provider_boundary()` rewinds `WorkflowState` to the earliest enabled upstream stage for that year so ordinary native step execution and Consist cache hits republish ActivitySim's required inputs.

### Rewind Resume (`--allow-rewind-resume`)

By default, `pilates/runtime/restart.py` blocks restarts when `run.start_year` in the YAML config is earlier than the `current_year` recorded in `run_state.yaml`. Pass `--allow-rewind-resume` on the command line when you want to rewind an archived state file to an earlier year frontier and re-run from that year.

### The `beam_run_completed -> beam_postprocess` Mid-Stage Boundary

BEAM travel execution is split into separate Consist `@define_step` steps to protect multi-hour container runs from postprocessing failures:

1. `beam_run`: Runs the BEAM container and commits verified raw outputs (archived via Consist's `RunCompletionArchivePolicy` when `enable_archive_copy` is enabled).
2. `beam_postprocess`: Merges raw iteration skims into the cumulative Zarr skims store and extracts link statistics and events.

If a job fails or hits a walltime limit after `beam_run` commits but before `beam_postprocess` finishes:

- Restarting with `-S <archive_run_dir>/run_state.yaml` cache-hits `beam_preprocess` and `beam_run`.
- If the node-local mutable workspace was wiped between jobs, Consist (`cache_hydration="outputs-requested"`) hydrates the required outputs from the archive (`consist-recovery/`).
- `beam_postprocess` runs without re-executing the BEAM container.

### `beam_postprocess` Separate Stores and Idempotency

`beam_postprocess` treats its input skims store (`zarr_skims`) as read-only and writes merged iteration skims into a dedicated output store (`<beam_output_dir>/.pilates-consist-outputs/beam_postprocess/year-<year>-iteration-<iteration>/skims.zarr`) in a single pass:

- If `beam_postprocess` crashes mid-execution, the upstream `zarr_skims` store remains unmodified.
- On restart, PILATES cache-hits `beam_run`, discards any partial output store at the destination path, and re-runs `beam_postprocess` against the intact input skims without double-merging trips.
- Subsequent iterations bind `ZARR_SKIMS` from the latest output store published to the coupler rather than reading a static in-place workspace path.

## 5. Shutdown

At the end of the simulation year loop (or on early exit), the launcher writes a final snapshot of `.consist/provenance.duckdb` into `.consist/snapshots/latest/` inside `archive_run_dir` via `snapshot_manager.final_snapshot()`. If the snapshot step fails, `mirror_consist_db_to_archive()` copies the local DuckDB file to the archive path so the archive directory retains the complete provenance record.

## Related Pages

- Check [CLI](cli.md) for invocation flags and Slurm submission commands.
- Consult [Configuration Reference](configuration_reference.md) for `run.restart_strict` and `run.consist_db_*` settings.
- Read [Architecture](../workflow/architecture.md) and [Step Contracts](../workflow/step_contracts.md) for coupler and step execution design.
