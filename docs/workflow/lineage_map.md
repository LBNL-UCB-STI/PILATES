---
title: Lineage Map
summary: Detailed step-by-step lineage reference for the current PILATES workflow.
---

# Lineage Map

## How To Read This Page

- Each stage below shows what PILATES reads, what it publishes, and which keys matter for restart or replay.
- The list is derived from the current workflow catalog and step publication logic, not from older migration notes.
- The emphasis is on workflow-visible lineage: what later tracked steps can resolve through published keys, not every internal temporary file.

## Stage Map

### Bootstrap

What PILATES does:

- Creates the scenario and workspace context for the run.
- Seeds bootstrap-safe inputs that later tracked steps resolve through normal workflow lookup instead of direct filesystem discovery.

Why it matters:

- Restart bootstrap can re-enter the workflow through the same tracked path, so later stages do not need a special lineage model just because a run resumed.

### UrbanSim

`urbansim_preprocess`

- **Reads**
  - base datastore state and workspace-backed UrbanSim inputs
- **Publishes**
  - the prepared run-local UrbanSim input state needed by the run boundary
- **Why the next step cares**
  - it converts the base inventory into the exact runner inputs for the current year

`urbansim_run`

- **Reads**
  - the prepared UrbanSim input state from preprocess
- **Publishes**
  - the forecast datastore output for the current year
- **Why the next step cares**
  - postprocess needs the runner output to make it the public mutable datastore handoff

`urbansim_postprocess`

- **Reads**
  - the forecast/current UrbanSim datastore output
- **Publishes**
  - `USIM_DATASTORE_CURRENT_H5` / `USIM_DATASTORE_H5`
  - `USIM_DATASTORE_BASE_H5` at the stage boundary where the current workflow still exposes both base and current datastore handles
  - `USIM_INPUT_ARCHIVE_PREFIX` (`usim_input_archive_{year}`)
  - `USIM_INPUT_MERGED_PREFIX` (`usim_input_merged_{year}`)
- **Why the next stage cares**
  - ATLAS and later restart/archive logic read the published datastore roles rather than reconstructing them from the UrbanSim run directory

### ATLAS

`atlas_preprocess`

- **Reads**
  - `USIM_DATASTORE_CURRENT_H5` / `USIM_DATASTORE_H5`
  - `USIM_DATASTORE_BASE_H5`
  - optional skim inputs such as `FINAL_SKIMS_OMX` when available
- **Publishes**
  - the staged inputs ATLAS needs for the current year
- **Why the next step cares**
  - the ATLAS run boundary is driven from the current UrbanSim-side state, not from bootstrap inputs directly

`atlas_run`

- **Reads**
  - staged ATLAS inputs from preprocess
- **Publishes**
  - `householdv_{year}`
  - `vehicles_{year}`
- **Why the next step cares**
  - postprocess uses the ATLAS outputs plus the current datastore state to construct the downstream handoff

`atlas_postprocess`

- **Reads**
  - the current UrbanSim datastore role
  - ATLAS vehicle ownership outputs
- **Publishes**
  - `USIM_POPULATION_SOURCE_H5`
  - `USIM_H5_UPDATED` where the postprocess path republishes the updated datastore role
  - `ATLAS_VEHICLES2_OUTPUT`
- **Why the next stage cares**
  - ActivitySim preprocess resolves its population-side input from `USIM_POPULATION_SOURCE_H5`, and downstream consumers can read the ATLAS vehicle output through a stable workflow key

### ActivitySim

`activitysim_preprocess`

- **Reads**
  - `USIM_POPULATION_SOURCE_H5`
  - current skim inputs such as `FINAL_SKIMS_OMX` when that path is active
- **Publishes**
  - `ASIM_LAND_USE_IN`
  - `ASIM_HOUSEHOLDS_IN`
  - `ASIM_PERSONS_IN`
  - `ASIM_OMX_SKIMS`
- **Why the next step cares**
  - compile and run operate on staged ActivitySim inputs rather than re-deriving them from the datastore each time

`activitysim_compile`

- **Reads**
  - the staged ActivitySim inputs and shared skim state available to the compile path
- **Publishes**
  - `ASIM_SHARROW_CACHE_DIR` when the compile cache is materialized
  - `ZARR_SKIMS` when the compile path republishes the shared skims handoff used by ActivitySim and BEAM
- **Why the next step cares**
  - the run boundary and later BEAM/analysis paths can reuse these published compile products

`activitysim_run`

- **Reads**
  - `ASIM_LAND_USE_IN`
  - `ASIM_HOUSEHOLDS_IN`
  - `ASIM_PERSONS_IN`
  - `ASIM_OMX_SKIMS`
- **Publishes**
  - the standard ActivitySim run outputs for the current year and iteration
- **Why the next step cares**
  - postprocess archives the actual ActivitySim-side inputs and outputs that the workflow used

`activitysim_postprocess`

- **Reads**
  - ActivitySim run outputs and the current datastore context
- **Publishes**
  - archived ActivitySim input/output materials needed for inspection and replay
  - the `USIM_INPUT_MERGED_PREFIX` (`usim_input_merged_{year}`) family used by later archive and lineage queries
- **Why the next stage cares**
  - BEAM preprocess and archive analysis can resolve the exact demand-side state that was produced for this iteration

### BEAM

`beam_preprocess`

- **Reads**
  - `BEAM_PLANS_IN`
  - `BEAM_HOUSEHOLDS_IN`
  - `BEAM_PERSONS_IN`
  - BEAM config and optional warm-start inputs such as `LINKSTATS_WARMSTART`
- **Publishes**
  - the staged BEAM run inputs and any archived-input snapshots that replay logic needs to recover later
- **Why the next step cares**
  - BEAM run consumes the staged, workflow-resolved travel-demand inputs instead of ad hoc files from earlier step directories

`beam_run`

- **Reads**
  - staged BEAM inputs and optional warm-start materials
- **Publishes**
  - `LINKSTATS`
  - `BEAM_PLANS_OUT`
  - BEAM plan/XML outputs used as later restart sources
- **Why the next step cares**
  - postprocess and later iterations depend on the network-performance outputs and restart-capable XML publications

`beam_postprocess`

- **Reads**
  - BEAM run outputs, including linkstats and plan/XML publications
- **Publishes**
  - `ZARR_SKIMS`
  - `FINAL_SKIMS_OMX` when produced
  - linkstats families with structured year/iteration/phys-sim metadata
- **Why the next stage cares**
  - later demand or archive-analysis paths can resolve shared skim outputs and indexed network-performance artifacts from this boundary

`beam_full_skim`

- **Reads**
  - the BEAM state required for the separate full-skim production path
- **Publishes**
  - `BEAM_FULL_SKIMS`
- **Why it matters**
  - this is a tracked output family with its own producer step rather than an implicit side effect of `beam_postprocess`

### Postprocessing

What PILATES does:

- Runs the final postprocessing boundary from the workflow catalog after the main staged exchanges are already complete.

Why it matters:

- This stage can produce final derived products, but the main cross-model lineage has already been established by the UrbanSim, ATLAS, ActivitySim, and BEAM boundaries.

## Restart-Relevant Notes

- Restart queries use the workflow catalog to determine the producer step for a key.
- The current restart logic gives special treatment to `ZARR_SKIMS`, ActivitySim archive inputs, and BEAM archived inputs because those are the keys the replay-first path needs to restore.
- `USIM_INPUT_ARCHIVE_PREFIX` (`usim_input_archive_{year}`) and `USIM_INPUT_MERGED_PREFIX` (`usim_input_merged_{year}`) are also restart-relevant because they preserve the published UrbanSim datastore lineage for archive inspection and explicit recovery cases.
- The launcher now prefers replay plus cache hits. The older manual hydration helpers remain available only for explicit recovery cases and tests.

## Adjacent Pages

- Read [Artifact Flow](artifact_flow.md) first if you only need the boundary map.
- Pair this with [Artifact Semantics](artifact_semantics.md).
- For archived-run inspection, go to [Opening Archives](../analysis/opening_archives.md).
