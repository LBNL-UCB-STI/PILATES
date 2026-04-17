---
title: Artifact Flow
summary: Short map of the main workflow-facing artifact handoffs in PILATES.
---

# Artifact Flow

This page is the artifact-key view of the workflow. It is intentionally about coupler-visible handoffs, not the full “what does each model require and produce?” explanation.

If you need the per-model boundary story first, read [Model Boundaries](../reference/model_boundaries.md) before this page.

## The Main Flow

The flow below is about what PILATES publishes across step boundaries through the coupler and tracked outputs. It is not a full inventory of every temporary file created inside a model run directory.

### Bootstrap

- Bootstrap establishes the run-local workspace and seeds the first workflow-visible inputs before the staged model loop begins.
- The launcher records bootstrap-safe artifacts so later steps can resolve the base inventory without reconstructing it from scratch.
- This boundary matters because downstream steps consume published handles, not ad hoc filesystem discovery.

### Land Use

- `urbansim_preprocess` reads the base datastore handle and prepares mutable workspace inputs for the UrbanSim run boundary.
- `urbansim_run` publishes the forecast datastore handle produced by the runner.
- `urbansim_postprocess` republishes the datastore as the current mutable UrbanSim handoff for downstream stages and can also publish year-scoped archive and merged snapshot families.

Main published boundary roles:

- `USIM_DATASTORE_BASE_H5`
- `USIM_DATASTORE_CURRENT_H5` / `USIM_DATASTORE_H5`
- `USIM_INPUT_ARCHIVE_PREFIX` (`usim_input_archive_{year}`)
- `USIM_INPUT_MERGED_PREFIX` (`usim_input_merged_{year}`)

### Vehicle Ownership

- `atlas_preprocess` reads both required UrbanSim datastore handles (`USIM_DATASTORE_CURRENT_H5` and `USIM_DATASTORE_BASE_H5`) and, when available, skims produced by earlier travel steps.
- `atlas_run` publishes year-specific vehicle ownership outputs such as `householdv_{year}` and `vehicles_{year}`.
- `atlas_postprocess` republishes the UrbanSim datastore that PILATES wants ActivitySim to use as the population-source datastore and also publishes `ATLAS_VEHICLES2_OUTPUT`.

Main published boundary roles:

- `householdv_{year}`
- `vehicles_{year}`
- `USIM_POPULATION_SOURCE_H5`
- `ATLAS_VEHICLES2_OUTPUT`

### Activity Demand

- `activitysim_preprocess` reads the population-source datastore and the currently selected skims input, then stages the land-use, households, and persons tables that the ActivitySim run consumes.
- `activitysim_compile` may publish reusable compile products such as `ASIM_SHARROW_CACHE_DIR` and can republish `ZARR_SKIMS` when that shared skims handoff is part of the compile path.
- `activitysim_run` consumes the staged ActivitySim inputs and publishes the standard ActivitySim run outputs.
- `activitysim_postprocess` archives ActivitySim inputs and outputs, and republishes the UrbanSim-side datastore state that later workflow logic or archive inspection needs.

Main published boundary roles:

- `ASIM_LAND_USE_IN`
- `ASIM_HOUSEHOLDS_IN`
- `ASIM_PERSONS_IN`
- `ASIM_OMX_SKIMS`
- `ASIM_SHARROW_CACHE_DIR`
- `ZARR_SKIMS` as the shared skims handoff used by ActivitySim and BEAM

### Traffic Assignment

- `beam_preprocess` reads the staged travel-demand outputs, the selected `BEAM_CONFIG_FILE`, and any warm-start artifacts that have been selected for the run.
- `beam_run` publishes `LINKSTATS`, `BEAM_PLANS_OUT`, and the BEAM plan/XML outputs that can also serve as restart sources.
- `beam_postprocess` republishes skim products for later travel stages or archive analysis and may publish `FINAL_SKIMS_OMX`.
- `beam_full_skim` is a separate tracked step that publishes `BEAM_FULL_SKIMS`.

Main published boundary roles:

- `BEAM_PLANS_IN`
- `BEAM_HOUSEHOLDS_IN`
- `BEAM_PERSONS_IN`
- `BEAM_CONFIG_FILE`
- `LINKSTATS`
- `LINKSTATS_WARMSTART`
- `BEAM_PLANS_OUT`
- `ZARR_SKIMS` when BEAM republishes the shared skims handoff
- `FINAL_SKIMS_OMX`
- `BEAM_FULL_SKIMS`

### Postprocessing

- The postprocessing stage is tracked in the workflow catalog, but most cross-model handoffs have already been established by earlier stages.
- In practice, this stage is more about producing final derived outputs than defining the main simulation boundary between model families.

## What Moves Across The Public Boundary

- Published coupler keys and typed step outputs form the workflow-visible surface.
- Archive snapshots and restart-specific families are public when replay logic needs an explicit producer for them.
- Scratch files inside a model run directory stay internal unless a tracked step republishes them under a workflow key.

## Why Similar Files Can Have Different Keys

- PILATES names artifacts by workflow role, not by path alone.
- A datastore may appear as a base handle, a canonical current mutable handle, a population-source handle, and a year-scoped `USIM_INPUT_ARCHIVE_PREFIX` or `USIM_INPUT_MERGED_PREFIX` family over the course of one run.
- Skims can appear under live cross-model handoff keys and under final or archived output families, depending on which later step needs them.

## Why This Matters

The coupler schema is the public list of keys the workflow declares. The lineage map below shows how those keys move through the current stage graph. If a handoff is not listed here, treat it as internal unless the catalog and tests say otherwise.

## Adjacent Pages

- Read [Model Boundaries](../reference/model_boundaries.md) for the per-model requirements and consumers.
- Read [Artifact Semantics](artifact_semantics.md) for key meanings.
- Use [Lineage Map](lineage_map.md) for the detailed per-step reference.
- Use [Step Contracts](step_contracts.md) for the semantic contract layer.
