---
title: Model Boundaries
summary: Canonical public reference for what each model family requires, publishes, and hands off next.
---

# Model Boundaries

## How To Read This Page

This is the main public answer to:

- what each model family requires
- what it publishes across the workflow boundary
- what consumes those outputs next
- where it runs in the stage loop
- what changes when adjacent models are disabled

Use this page when you want the model-I/O picture without reading adapter code first.

For adjacent views:

- use [Simulation Logic by Stage](../workflow/simulation_logic_by_stage.md) for why the stages are ordered the way they are
- use [Artifact Flow](../workflow/artifact_flow.md) for the workflow-key and coupler view
- use [Model Integration Guide](../extend/model_integration_guide.md) when you are changing the implementation

## UrbanSim

### Role in the workflow

UrbanSim establishes the land-use-side regional state for the current forecast year. It is the first major scientific boundary in the yearly run and the source of the mutable datastore that later stages read.

### Stage and step placement

- Major stage: `land_use`
- Step family: `urbansim_preprocess`, `urbansim_run`, `urbansim_postprocess`

### Required inputs

- `USIM_DATASTORE_BASE_H5`
- `USIM_DATASTORE_CURRENT_H5`

### Optional inputs

- final skims or other carried-forward travel artifacts when the land-use path uses them

### Main published outputs

- `USIM_DATASTORE_CURRENT_H5` / `USIM_DATASTORE_H5`
- `USIM_FORECAST_OUTPUT`
- `USIM_INPUT_ARCHIVE_PREFIX` (`usim_input_archive_{year}`)
- `USIM_INPUT_MERGED_PREFIX` (`usim_input_merged_{year}`)

### Downstream consumers

- ATLAS reads the current/base UrbanSim datastore in the vehicle-ownership stage
- ActivitySim can read the current or population-source datastore directly when ATLAS is disabled

### Restart and archive-relevant artifacts

- current mutable datastore handle
- forecast-year datastore handle
- year-scoped archive and merged datastore families

### If adjacent models are disabled

- If ATLAS is disabled, the ActivitySim boundary falls back to UrbanSim-owned datastore handles.
- If land use is disabled entirely, downstream stages must start from previously staged or configured datastore inputs instead of a fresh UrbanSim handoff.

### Boundary owners

- Typed outputs: `UrbanSimPreprocessOutputs`, `UrbanSimRunOutputs`, `UrbanSimPostprocessOutputs`
- Trace it in code:
  - catalog: `pilates/workflows/catalog.py`
  - steps: `pilates/workflows/steps/urbansim_atlas.py`
  - binding / input policy: `pilates/workflows/binding.py`, `pilates/urbansim/inputs.py`
  - stage: `pilates/workflows/stages/land_use.py`

## ATLAS

### Role in the workflow

ATLAS refines the current regional state with vehicle-ownership outputs and chooses the UrbanSim datastore that ActivitySim should treat as the population source.

### Stage and step placement

- Major stage: `vehicle_ownership_model`
- Step family: `atlas_preprocess`, `atlas_run`, `atlas_postprocess`

### Required inputs

- `USIM_DATASTORE_CURRENT_H5`

### Optional inputs

- `USIM_DATASTORE_BASE_H5`
- travel skims from earlier years when the scenario uses them

### Main published outputs

- `householdv_{year}`
- `vehicles_{year}`
- `USIM_POPULATION_SOURCE_H5`
- `ATLAS_VEHICLES2_OUTPUT`

### Downstream consumers

- ActivitySim uses `USIM_POPULATION_SOURCE_H5` as its preferred population/datastore handoff
- BEAM preprocess can use `ATLAS_VEHICLES2_OUTPUT` as the staged vehicle input

### Restart and archive-relevant artifacts

- the selected population-source datastore handle
- ATLAS vehicle outputs that later travel steps can restage

### If adjacent models are disabled

- If land use is disabled, ATLAS must start from already available UrbanSim datastore handles.
- If ActivitySim is disabled, the population-source handoff may not be consumed immediately, but it remains the public vehicle-ownership boundary for downstream inspection and replay.

### Boundary owners

- Typed outputs: `AtlasPreprocessOutputs`, `AtlasRunOutputs`, `AtlasPostprocessOutputs`
- Trace it in code:
  - catalog: `pilates/workflows/catalog.py`
  - steps: `pilates/workflows/steps/urbansim_atlas.py`
  - binding / input policy: `pilates/workflows/binding.py`, `pilates/atlas/inputs.py`
  - stage: `pilates/workflows/stages/vehicle_ownership.py`

## ActivitySim

### Role in the workflow

ActivitySim converts the current regional state into staged demand-model inputs and then into demand outputs that BEAM can consume. It is the main population-and-travel-demand boundary in the supply-demand loop.

### Stage and step placement

- Major stage: `supply_demand_loop`
- Substage: `activity_demand`
- Step family: `activitysim_preprocess`, `activitysim_compile`, `activitysim_run`, `activitysim_postprocess`

### Required inputs

- `USIM_POPULATION_SOURCE_H5`

### Optional inputs

- `USIM_DATASTORE_CURRENT_H5` for postprocess/writeback-sensitive flows
- `ASIM_OMX_SKIMS` when compile uses OMX skims
- `ZARR_SKIMS` when compiled skims already exist

### Main published outputs

- staged inputs:
  - `ASIM_LAND_USE_IN`
  - `ASIM_HOUSEHOLDS_IN`
  - `ASIM_PERSONS_IN`
  - `ASIM_OMX_SKIMS`
- shared compile/runtime artifacts:
  - `ASIM_SHARROW_CACHE_DIR`
  - `ZARR_SKIMS`
- BEAM-facing demand outputs:
  - `beam_plans_asim_out`
  - `households_asim_out`
  - `persons_asim_out`
  - other standard ActivitySim outputs such as `tours_asim_out` and `trips_asim_out`
- UrbanSim-facing writeback output:
  - `USIM_DATASTORE_CURRENT_H5` / `USIM_DATASTORE_H5` when land-use writeback is active

### Downstream consumers

- BEAM preprocess uses the plans/households/persons handoff
- later land-use / ATLAS stages can consume the updated UrbanSim datastore when writeback is enabled
- archive/replay logic uses the archived ActivitySim inputs and outputs

### Restart and archive-relevant artifacts

- archived staged inputs such as households/persons/land_use/skims
- `ZARR_SKIMS`
- the standard ActivitySim run outputs
- any writeback datastore handed back to the workflow

### If adjacent models are disabled

- If ATLAS is disabled, ActivitySim falls back to UrbanSim-owned datastore handles for population-source selection.
- If land use is disabled, ActivitySim can still run from staged datastore inputs, but it should not assume a fresh UrbanSim update happened earlier in the year.
- If BEAM is disabled, ActivitySim still produces the standard demand-side outputs, but the plans/households/persons boundary is not consumed by traffic assignment in that run shape.

### Operational step variant: `activitysim_compile`

- Purpose: publish reusable compile products before the main run boundary
- Main outputs:
  - `ZARR_SKIMS`
  - `ASIM_SHARROW_CACHE_DIR`
- Why it matters operationally: compile artifacts affect restart, replay, and whether later ActivitySim/BEAM steps can reuse shared skim products instead of rebuilding them

### Boundary owners

- Typed outputs: `ActivitySimPreprocessOutputs`, `ActivitySimRunOutputs`, `ActivitySimPostprocessOutputs`
- Trace it in code:
  - catalog: `pilates/workflows/catalog.py`
  - steps: `pilates/workflows/steps/activitysim.py`
  - binding / input policy: `pilates/workflows/binding.py`, `pilates/activitysim/inputs.py`
  - stage: `pilates/workflows/stages/supply_demand_activity.py`

## BEAM

### Role in the workflow

BEAM turns demand-side plans and traveler inputs into traffic-assignment outputs, network-performance artifacts, and shared skims. It closes the main supply-demand loop.

### Stage and step placement

- Major stage: `supply_demand_loop`
- Substage: `traffic_assignment`
- Step family: `beam_preprocess`, `beam_run`, `beam_postprocess`, `beam_full_skim`

### Required inputs

- `BEAM_PLANS_IN`
- `BEAM_HOUSEHOLDS_IN`
- `BEAM_PERSONS_IN`

### Optional inputs

- `LINKSTATS_WARMSTART`
- `ATLAS_VEHICLES2_OUTPUT`
- shared skims handoff when BEAM merges or republishes them

### Main published outputs

- `LINKSTATS`
- `LINKSTATS_WARMSTART`
- `BEAM_PLANS_OUT`
- `ZARR_SKIMS` when BEAM republishes the shared skims handoff
- `FINAL_SKIMS_OMX`
- BEAM plan/XML output families used for warm starts and archive inspection

### Downstream consumers

- later supply-demand iterations can reuse warm-start linkstats and plan outputs
- later years can reuse BEAM-side restart artifacts
- UrbanSim or downstream analysis can consume `FINAL_SKIMS_OMX`
- archive/replay tooling can materialize BEAM input and warm-start families back into the workspace

### Restart and archive-relevant artifacts

- `LINKSTATS_WARMSTART`
- archived BEAM input families
- plan warm-start XML families
- BEAM output plan/XML artifacts

### If adjacent models are disabled

- If ActivitySim is disabled, BEAM must start from already available plans/households/persons inputs instead of fresh demand outputs.
- If ATLAS is disabled, the optional vehicle input is absent but the core BEAM plans/households/persons boundary is unchanged.

### Operational step variant: `beam_full_skim`

- Purpose: publish a dedicated full-skim output separate from the main BEAM run/postprocess pair
- Main output:
  - `BEAM_FULL_SKIMS`
- Why it matters operationally: this step is separately scheduled and can be enabled for runs that need full skims without changing the main traffic-assignment boundary

### Boundary owners

- Typed outputs: `BeamPreprocessOutputs`, `BeamRunOutputs`, `BeamPostprocessOutputs`, `BeamFullSkimOutputs`
- Trace it in code:
  - catalog: `pilates/workflows/catalog.py`
  - steps: `pilates/workflows/steps/beam.py`
  - binding / input policy: `pilates/workflows/binding.py`, `pilates/beam/preprocessor.py`
  - stage: `pilates/workflows/stages/supply_demand_beam.py`

## Recommended Reading Path

If you are new to the repo and want the shortest useful path:

1. [Workflow Primer](../workflow/workflow_primer.md)
2. [Architecture](../workflow/architecture.md)
3. this page
4. [Simulation Logic by Stage](../workflow/simulation_logic_by_stage.md)
5. [Artifact Flow](../workflow/artifact_flow.md)
6. [Model Integration Guide](../extend/model_integration_guide.md)
