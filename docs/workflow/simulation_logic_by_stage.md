---
title: Simulation Logic by Stage
summary: Science-facing description of what each stage is logically doing in the overall simulation.
---

# Simulation Logic by Stage

## Reading Rule

- This page explains what PILATES is doing with the staged workflow state.
- It is intentionally lighter on model-internal science and heavier on why the workflow is ordered the way it is.
- For exact published keys and step boundaries, use [Artifact Flow](artifact_flow.md) and [Lineage Map](lineage_map.md).

## Stage Logic

### Bootstrap

Bootstrap creates the run-local environment that the tracked workflow will use. In logical terms, this is where PILATES turns scenario configuration and workspace inputs into a concrete run context with stable paths, mounts, and initial workflow-visible handles.

This stage is not part of the scientific simulation itself. Its job is to establish the initial state that later stages can consume without guessing where inputs live or which files are authoritative for the run.

### Land Use

The land-use stage advances the UrbanSim-side representation of the region for the current forecast year. In workflow terms, PILATES starts from a base datastore, runs the land-use model path, and republishes a current mutable datastore that later stages can treat as the authoritative regional state.

This is the first major scientific handoff in the run because later stages do not consume raw bootstrap inputs directly. They consume the updated land-use-side state that this stage publishes.

### Vehicle Ownership

The vehicle-ownership stage refines the regional state with ATLAS outputs tied to the current year. PILATES uses this stage to derive year-scoped vehicle ownership artifacts and to decide which UrbanSim datastore should become the population source for the next demand-model boundary.

Logically, this stage sits between land use and activity demand because it updates the traveler and vehicle context that downstream demand generation will read. The important workflow fact is not the internal ATLAS science; it is that PILATES turns the current land-use-side datastore into a population-source handoff plus year-specific ATLAS outputs.

### Activity Demand

The activity-demand stage converts the current regional state into staged ActivitySim inputs and then into demand-side outputs that BEAM can use. PILATES reads the chosen population-source datastore, selects the current skims input, and stages concrete inputs such as land use, households, persons, and skims for the ActivitySim run boundary.

Logically, this is where the workflow turns regional population and land-use context into explicit travel-demand inputs and outputs. It also republishes shared skim products and archive-relevant inputs so the next traffic-assignment stage and later archive analysis can resolve the demand-side state that was actually used.

### Traffic Assignment

The traffic-assignment stage takes the demand-side outputs and pushes them through BEAM. PILATES stages the inputs BEAM needs, executes the traffic-assignment path, and then republishes network-performance and skim artifacts that later iterations or later years may consume.

This stage closes the main supply-demand loop. Linkstats, plans outputs, warm-start materials, and skim products are the workflow-visible record of how the network side responded to the demand-side state produced earlier in the year or iteration.

### Postprocessing

The postprocessing stage is where PILATES computes or republishes final derived products after the main cross-model handoffs are already established. In logical terms, it is downstream of the main simulation exchange rather than a peer stage in the main feedback loop.

That distinction matters for reading the workflow: most of the substantive cross-model state transitions have already happened before postprocessing begins. Postprocessing is where PILATES makes those results easier to inspect, export, or analyze.

## Where The Main Handoffs Happen

- The land-use stage establishes the current mutable regional datastore.
- The vehicle-ownership stage chooses and republishes the population-source datastore for demand modeling.
- The activity-demand stage turns regional state plus skims into staged ActivitySim demand inputs.
- The traffic-assignment stage turns those demand outputs into linkstats, skims, and restart-relevant BEAM outputs.
- The postprocessing stage mostly works on already-published results instead of defining the primary handoff between model families.

## Adjacent Pages

- Read [Stages and Steps](stages_and_steps.md) for the runtime execution model.
- Use [Artifact Flow](artifact_flow.md) for the concrete handoffs.
- Use [Model Stack](../reference/model_stack.md) for the role of each model family.
